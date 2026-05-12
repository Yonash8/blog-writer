"""Per-session in-memory run buffer.

Holds an append-only, monotonically-numbered event log for each in-flight
agent run, keyed by ``session_id``. Multiple SSE handlers can attach to the
same buffer at different ``from_seq`` positions and stream live; once the
run finishes, the buffer is retained for a short grace period so late
reconnects still see the terminal events.

Design notes
------------
- One ordered list (not separate status/token lists like the previous
  ``_stream_chat_generator``) preserves the exact interleaving the agent
  emits, which matters for replay.
- All append/finish/fail operations take ``_lock``. Setting ``finished=True``
  and appending the terminal event happen in the same critical section so a
  reader can't observe finished=False after the terminal event was queued.
- Readers carry their own ``last_seq`` cursor and poll the buffer every
  50ms (same cadence as the old generator). Polling beats a Condition
  variable here because there's no asyncio<->threading bridge to fight.
- ``last_tool`` carries the human-readable name of the currently-running
  tool (or ``None`` between tool calls). The sidebar reads this off the
  snapshot event to show "what is session X doing right now."
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

# Grace period after a run finishes during which the buffer is kept around so
# a delayed reconnect (page reload, slow tab) can still see the terminal events.
_RETENTION_SEC = 300.0
_POLL_INTERVAL_SEC = 0.05
# Maximum time a single attach call will hold the SSE connection open while
# waiting for new events. Clients reconnect on timeout — this is mostly a
# defense against proxies that kill idle streams anyway.
_MAX_TAIL_WAIT_SEC = 25.0


@dataclass
class RunBuffer:
    session_id: str
    run_id: str
    started_at: float
    events: list[dict] = field(default_factory=list)
    seq: int = 0
    finished: bool = False
    finished_at: Optional[float] = None
    cancel_requested: bool = False
    last_tool: Optional[str] = None
    last_status: Optional[str] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def append(self, event_type: str, **fields) -> int:
        with self._lock:
            self.seq += 1
            ev = {"seq": self.seq, "type": event_type, "ts": time.time(), **fields}
            self.events.append(ev)
            if event_type == "status":
                self.last_status = fields.get("text")
            elif event_type == "tool_start":
                self.last_tool = fields.get("name")
            elif event_type == "tool_end":
                self.last_tool = None
            return self.seq

    def finish(self, final: dict) -> int:
        """Append a terminal `done` event and mark finished atomically."""
        with self._lock:
            self.seq += 1
            ev = {"seq": self.seq, "type": "done", "ts": time.time(), **final}
            self.events.append(ev)
            self.finished = True
            self.finished_at = time.time()
            self.last_tool = None
            return self.seq

    def fail(self, error_message: str) -> int:
        with self._lock:
            self.seq += 1
            ev = {
                "seq": self.seq,
                "type": "error",
                "ts": time.time(),
                "message": error_message,
            }
            self.events.append(ev)
            self.finished = True
            self.finished_at = time.time()
            self.last_tool = None
            return self.seq

    def request_cancel(self) -> None:
        with self._lock:
            self.cancel_requested = True

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "session_id": self.session_id,
                "run_id": self.run_id,
                "started_at": self.started_at,
                "last_seq": self.seq,
                "running": not self.finished,
                "cancel_requested": self.cancel_requested,
                "last_tool": self.last_tool,
                "last_status": self.last_status,
            }


# Module-level registry, keyed by session_id.
_runs: dict[str, RunBuffer] = {}
_runs_lock = threading.Lock()


def start_run(session_id: str, run_id: str) -> Optional[RunBuffer]:
    """Create a buffer for a fresh run on this session.

    Returns ``None`` if there's already an active (un-finished) buffer for the
    session — caller should respond with 409 in that case. A finished buffer
    still in the retention window is replaced.
    """
    now = time.time()
    with _runs_lock:
        existing = _runs.get(session_id)
        if existing and not existing.finished:
            return None
        # Either no buffer or one that has finished — replace it.
        buf = RunBuffer(session_id=session_id, run_id=run_id, started_at=now)
        _runs[session_id] = buf
        return buf


def get_run(session_id: str) -> Optional[RunBuffer]:
    with _runs_lock:
        buf = _runs.get(session_id)
    if not buf:
        return None
    if buf.finished and buf.finished_at and (time.time() - buf.finished_at) > _RETENTION_SEC:
        with _runs_lock:
            _runs.pop(session_id, None)
        return None
    return buf


def discard_run(session_id: str) -> None:
    """Drop the buffer entirely (used on session delete)."""
    with _runs_lock:
        _runs.pop(session_id, None)


def replay_and_tail(buf: RunBuffer, from_seq: int = 0) -> Iterator[dict]:
    """Yield buffered events from ``from_seq`` onwards, then live-tail.

    Always emits a synthetic ``snapshot`` event first so the client can pick
    up the current running/tool state without having to parse the whole
    event log. Returns when the run has finished AND the consumer has caught
    up, OR when the connection idles past ``_MAX_TAIL_WAIT_SEC``.
    """
    # Lead with a snapshot so the UI can render current state immediately.
    yield {"seq": from_seq, "type": "snapshot", **buf.snapshot()}

    cursor = from_seq
    idle_started: Optional[float] = None
    while True:
        # Pull any new events under the lock.
        with buf._lock:
            new_events = [e for e in buf.events if e["seq"] > cursor]
            finished = buf.finished
        if new_events:
            for ev in new_events:
                yield ev
                cursor = ev["seq"]
            idle_started = None
            if finished:
                # Drained all events including the terminal one — done.
                return
            continue
        if finished:
            return
        # Wait for more events.
        if idle_started is None:
            idle_started = time.time()
        elif time.time() - idle_started > _MAX_TAIL_WAIT_SEC:
            # Tell the client we're still alive but pause this connection;
            # client will reconnect with from_seq=cursor.
            yield {"seq": cursor, "type": "idle", "running": not finished}
            return
        time.sleep(_POLL_INTERVAL_SEC)


def gc_finished_runs(older_than_sec: float = _RETENTION_SEC) -> int:
    """Drop finished buffers older than the retention window. Returns count removed."""
    now = time.time()
    removed = 0
    with _runs_lock:
        for sid, buf in list(_runs.items()):
            if buf.finished and buf.finished_at and (now - buf.finished_at) > older_than_sec:
                _runs.pop(sid, None)
                removed += 1
    return removed

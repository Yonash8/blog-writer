"""In-flight task tracking for status queries without interrupting long-running tasks.

Keys are opaque strings. WhatsApp uses ``f"{channel}:{channel_user_id}"`` via the
legacy tuple-argument API; the web flow keys per-session via ``f"session:{id}"``
through the ``_by_key`` siblings. Both paths share the same store.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TaskState:
    """State of an in-flight task."""

    status_text: str
    started_at: float
    updated_at: float
    current_step: Optional[str] = None
    cancel_requested: bool = False


# Key: opaque string (e.g. "whatsapp:972...", "web:default", "session:<uuid>")
_task_store: dict[str, TaskState] = {}
_lock = threading.Lock()
_TASK_STALE_TIMEOUT_SEC = int(os.getenv("TASK_STALE_TIMEOUT_SEC", "900"))
_CANCEL_GRACE_SEC = int(os.getenv("TASK_CANCEL_GRACE_SEC", "20"))

# Phrases that indicate a status-style question (don't start new agent run)
STATUS_PHRASES = (
    "how's it going",
    "hows it going",
    "how is it going",
    "still working",
    "what step",
    "what's the status",
    "whats the status",
    "what is the status",
    "status",
    "progress",
    "how long",
    "almost done",
    "done yet",
)

# Phrases that indicate the user wants to cancel the current task
CANCEL_PHRASES = (
    "stop",
    "cancel",
    "abort",
    "nevermind",
    "never mind",
    "forget it",
    "cancel that",
    "stop that",
)


def _normalize(text: str) -> str:
    return (text or "").strip().lower()


def is_status_question(text: str) -> bool:
    """Return True if the message looks like a status/progress question."""
    t = _normalize(text)
    if len(t) < 3:
        return False
    for phrase in STATUS_PHRASES:
        if phrase in t:
            return True
    return False


def _make_key(channel: str, channel_user_id: str) -> str:
    return f"{channel}:{channel_user_id}"


# --- Key-based API (preferred for new code) -----------------------------------

def set_task_by_key(key: str, status_text: str, current_step: Optional[str] = None) -> None:
    """Register or update an in-flight task by opaque key."""
    now = time.monotonic()
    with _lock:
        existing = _task_store.get(key)
        if existing:
            existing.status_text = status_text
            existing.current_step = current_step
            existing.updated_at = now
        else:
            _task_store[key] = TaskState(
                status_text=status_text,
                started_at=now,
                updated_at=now,
                current_step=current_step,
            )
        logger.debug("[TASK_STATE] set %s: %r", key[:40], status_text[:80])


def _is_stale(task: TaskState, now: float) -> bool:
    """Treat very old tasks as stale so users don't get blocked forever."""
    age = now - task.started_at
    idle = now - task.updated_at
    if task.cancel_requested and idle >= _CANCEL_GRACE_SEC:
        return True
    return age >= _TASK_STALE_TIMEOUT_SEC


def get_task_by_key(key: str) -> Optional[TaskState]:
    with _lock:
        task = _task_store.get(key)
        if not task:
            return None
        now = time.monotonic()
        if _is_stale(task, now):
            _task_store.pop(key, None)
            logger.warning("[TASK_STATE] stale task auto-cleared for %s", key[:40])
            return None
        return task


def clear_task_by_key(key: str) -> None:
    with _lock:
        _task_store.pop(key, None)
        logger.debug("[TASK_STATE] cleared %s", key[:40])


def has_task_by_key(key: str) -> bool:
    return get_task_by_key(key) is not None


def request_cancel_by_key(key: str) -> bool:
    with _lock:
        task = _task_store.get(key)
        if task:
            task.cancel_requested = True
            task.updated_at = time.monotonic()
            logger.info("[TASK_STATE] Cancel requested for %s", key[:40])
            return True
        return False


def is_cancel_requested_by_key(key: str) -> bool:
    task = get_task_by_key(key)
    return task.cancel_requested if task else False


# --- Legacy tuple API (WhatsApp + existing call sites) ------------------------
# Thin wrappers around the key-based functions so we don't break callers.

def set_task(channel: str, channel_user_id: str, status_text: str, current_step: Optional[str] = None) -> None:
    set_task_by_key(_make_key(channel, channel_user_id), status_text, current_step)


def get_task(channel: str, channel_user_id: str) -> Optional[TaskState]:
    return get_task_by_key(_make_key(channel, channel_user_id))


def clear_task(channel: str, channel_user_id: str) -> None:
    clear_task_by_key(_make_key(channel, channel_user_id))


def has_task(channel: str, channel_user_id: str) -> bool:
    return has_task_by_key(_make_key(channel, channel_user_id))


def request_cancel(channel: str, channel_user_id: str) -> bool:
    return request_cancel_by_key(_make_key(channel, channel_user_id))


def is_cancel_requested(channel: str, channel_user_id: str) -> bool:
    return is_cancel_requested_by_key(_make_key(channel, channel_user_id))


def is_cancel_message(text: str) -> bool:
    """Return True if the message looks like a cancellation request."""
    t = _normalize(text)
    if len(t) < 2:
        return False
    for phrase in CANCEL_PHRASES:
        if t == phrase or t.startswith(phrase + " ") or t.endswith(" " + phrase):
            return True
    return False

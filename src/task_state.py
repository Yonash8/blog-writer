"""In-flight task tracking for status queries without interrupting long-running tasks."""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
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


# Key: (channel, channel_user_id) e.g. ("whatsapp", "972546678582@c.us") or ("web", "default")
_task_store: dict[tuple[str, str], TaskState] = {}
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


def set_task(channel: str, channel_user_id: str, status_text: str, current_step: Optional[str] = None) -> None:
    """Register or update an in-flight task."""
    now = time.monotonic()
    with _lock:
        key = (channel, channel_user_id)
        existing = _task_store.get(key)
        if existing:
            # Preserve started_at + cancel_requested across status updates.
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
        logger.debug("[TASK_STATE] set %s/%s: %r", channel, channel_user_id[:20], status_text[:80])


def _is_stale(task: TaskState, now: float) -> bool:
    """Treat very old tasks as stale so users don't get blocked forever."""
    age = now - task.started_at
    idle = now - task.updated_at
    if task.cancel_requested and idle >= _CANCEL_GRACE_SEC:
        return True
    return age >= _TASK_STALE_TIMEOUT_SEC


def get_task(channel: str, channel_user_id: str) -> Optional[TaskState]:
    """Get current task state for a channel/user. Returns None if no task in flight."""
    with _lock:
        key = (channel, channel_user_id)
        task = _task_store.get(key)
        if not task:
            return None
        now = time.monotonic()
        if _is_stale(task, now):
            _task_store.pop(key, None)
            logger.warning("[TASK_STATE] stale task auto-cleared for %s/%s", channel, channel_user_id[:20])
            return None
        return task


def clear_task(channel: str, channel_user_id: str) -> None:
    """Clear the in-flight task when it completes."""
    with _lock:
        _task_store.pop((channel, channel_user_id), None)
        logger.debug("[TASK_STATE] cleared %s/%s", channel, channel_user_id[:20] if channel_user_id else "")


def has_task(channel: str, channel_user_id: str) -> bool:
    """Return True if there is an in-flight task for this channel/user."""
    return get_task(channel, channel_user_id) is not None


def is_cancel_message(text: str) -> bool:
    """Return True if the message looks like a cancellation request."""
    t = _normalize(text)
    if len(t) < 2:
        return False
    for phrase in CANCEL_PHRASES:
        if t == phrase or t.startswith(phrase + " ") or t.endswith(" " + phrase):
            return True
    return False


def request_cancel(channel: str, channel_user_id: str) -> bool:
    """Request cancellation of the in-flight task. Returns True if a task was found."""
    with _lock:
        task = _task_store.get((channel, channel_user_id))
        if task:
            task.cancel_requested = True
            task.updated_at = time.monotonic()
            logger.info("[TASK_STATE] Cancel requested for %s/%s", channel, channel_user_id[:20])
            return True
        return False


def is_cancel_requested(channel: str, channel_user_id: str) -> bool:
    """Check if cancellation has been requested for the in-flight task."""
    task = get_task(channel, channel_user_id)
    return task.cancel_requested if task else False

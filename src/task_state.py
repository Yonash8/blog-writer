"""In-flight task tracking for status queries without interrupting long-running tasks."""

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TaskState:
    """State of an in-flight task."""

    status_text: str
    started_at: float
    current_step: Optional[str] = None
    cancel_requested: bool = False


# Key: (channel, channel_user_id) e.g. ("whatsapp", "972546678582@c.us") or ("web", "default")
_task_store: dict[tuple[str, str], TaskState] = {}
_lock = threading.Lock()

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
    import time

    with _lock:
        _task_store[(channel, channel_user_id)] = TaskState(
            status_text=status_text,
            started_at=time.monotonic(),
            current_step=current_step,
        )
        logger.debug("[TASK_STATE] set %s/%s: %r", channel, channel_user_id[:20], status_text[:80])


def get_task(channel: str, channel_user_id: str) -> Optional[TaskState]:
    """Get current task state for a channel/user. Returns None if no task in flight."""
    with _lock:
        return _task_store.get((channel, channel_user_id))


def clear_task(channel: str, channel_user_id: str) -> None:
    """Clear the in-flight task when it completes."""
    with _lock:
        _task_store.pop((channel, channel_user_id), None)
        logger.debug("[TASK_STATE] cleared %s/%s", channel, channel_user_id[:20] if channel_user_id else "")


def has_task(channel: str, channel_user_id: str) -> bool:
    """Return True if there is an in-flight task for this channel/user."""
    with _lock:
        return (channel, channel_user_id) in _task_store


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
            logger.info("[TASK_STATE] Cancel requested for %s/%s", channel, channel_user_id[:20])
            return True
        return False


def is_cancel_requested(channel: str, channel_user_id: str) -> bool:
    """Check if cancellation has been requested for the in-flight task."""
    with _lock:
        task = _task_store.get((channel, channel_user_id))
        return task.cancel_requested if task else False

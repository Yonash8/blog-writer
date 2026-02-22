"""Thread manager: maps WhatsApp users to LangGraph article threads.

One LangGraph thread per article. Stores active_article_id per user in Supabase.
thread_id format: "{whatsapp_user_id}:{article_id}"
"""

import logging
import uuid
from typing import Optional

logger = logging.getLogger(__name__)


def _get_client():
    from src.db import get_client
    return get_client()


def get_active_thread(whatsapp_user_id: str) -> Optional[dict]:
    """Get the active article thread for a WhatsApp user.

    Returns {whatsapp_user_id, article_id, thread_id} or None.
    """
    client = _get_client()
    try:
        r = (
            client.table("active_article_threads")
            .select("*")
            .eq("whatsapp_user_id", whatsapp_user_id)
            .limit(1)
            .execute()
        )
        if r.data and len(r.data) > 0:
            return r.data[0]
    except Exception as e:
        logger.warning("[THREAD_MGR] Failed to get active thread: %s", e)
    return None


def set_active_thread(whatsapp_user_id: str, thread_id: str, article_id: Optional[str] = None) -> None:
    """Set the active article thread for a WhatsApp user (upsert).
    article_id can be None for placeholder threads (before an article exists).
    """
    client = _get_client()
    try:
        data = {"whatsapp_user_id": whatsapp_user_id, "thread_id": thread_id}
        if article_id is not None:
            data["article_id"] = article_id
        client.table("active_article_threads").upsert(
            data,
            on_conflict="whatsapp_user_id",
        ).execute()
        logger.info("[THREAD_MGR] Set active thread: user=%s thread=%s", whatsapp_user_id[:20], thread_id[:40])
    except Exception as e:
        logger.warning("[THREAD_MGR] Failed to set active thread: %s", e)


def clear_active_thread(whatsapp_user_id: str) -> None:
    """Clear the active thread for a user."""
    client = _get_client()
    try:
        client.table("active_article_threads").delete().eq(
            "whatsapp_user_id", whatsapp_user_id
        ).execute()
    except Exception as e:
        logger.warning("[THREAD_MGR] Failed to clear active thread: %s", e)


def _looks_like_new_article(text: str) -> bool:
    """Check if the user message looks like a new article request."""
    t = text.strip().lower()
    new_signals = (
        "write an article", "write article", "new article",
        "write about", "create an article", "start article",
        "/new",
    )
    for sig in new_signals:
        if sig in t:
            return True
    return False


def get_or_create_thread_id(whatsapp_user_id: str, user_text: str) -> str:
    """Resolve or create the thread_id for a user based on their message.

    - New article requests: always creates a new thread
    - Other messages: use existing active thread, or create new one
    """
    if _looks_like_new_article(user_text):
        article_id = str(uuid.uuid4())
        thread_id = f"{whatsapp_user_id}:{article_id}"
        set_active_thread(whatsapp_user_id, thread_id, article_id=None)
        logger.info("[THREAD_MGR] New thread for new article: %s", thread_id[:40])
        return thread_id

    # Try to resume existing thread
    active = get_active_thread(whatsapp_user_id)
    if active and active.get("thread_id"):
        logger.info("[THREAD_MGR] Resuming thread: %s", active["thread_id"][:40])
        return active["thread_id"]

    # No active thread; create a placeholder (don't store non-existent article_id)
    thread_id = f"{whatsapp_user_id}:{uuid.uuid4()}"
    set_active_thread(whatsapp_user_id, thread_id, article_id=None)
    logger.info("[THREAD_MGR] Created new thread (no active): %s", thread_id[:40])
    return thread_id

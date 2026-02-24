from __future__ import annotations
"""Shared helpers for graph nodes."""

import logging
import re

logger = logging.getLogger(__name__)


def resolve_article_from_params(whatsapp_user_id: str, intent_params: dict) -> tuple[None, None] | tuple[str, dict]:
    """Resolve an article from intent params (topic, article_title, article_id).

    Tries in order: article_id (exact/prefix), article_title, topic.
    Returns (article_id, article) or (None, None) if not found.
    """
    params = intent_params or {}
    channel = "whatsapp"

    # 1. Try by article_id (hex ID or prefix)
    article_id_param = params.get("article_id", "") or params.get("article_ref", "")
    if article_id_param and isinstance(article_id_param, str) and len(article_id_param.strip()) >= 8:
        from src.db import get_article_by_id_or_prefix
        article = get_article_by_id_or_prefix(
            article_id_param.strip(),
            channel=channel,
            channel_user_id=whatsapp_user_id,
        )
        if article:
            logger.info("[RESOLVER] Found article by ID %r: %s", article_id_param[:16], article["id"][:8])
            return article["id"], article

    # 2. Try by article_title
    article_title = params.get("article_title", "") or params.get("title", "")
    if article_title and isinstance(article_title, str) and article_title.strip():
        from src.db import list_articles
        arts = list_articles(
            channel=channel,
            channel_user_id=whatsapp_user_id,
            title_query=article_title.strip(),
            limit=3,
        )
        if arts:
            article = arts[0]
            logger.info("[RESOLVER] Found article by title %r: %s", article_title[:40], article["id"][:8])
            return article["id"], article

    # 3. Try by topic (also use article_ref as topic when it looks like subject/title, not an ID)
    topic = params.get("topic", "")
    if not topic and article_id_param and " " in str(article_id_param):
        topic = str(article_id_param).strip()
    if topic and isinstance(topic, str) and topic.strip():
        from src.db import list_articles
        arts = list_articles(
            channel=channel,
            channel_user_id=whatsapp_user_id,
            title_query=topic.strip(),
            limit=3,
        )
        if arts:
            article = arts[0]
            logger.info("[RESOLVER] Found article by topic %r: %s", topic[:40], article["id"][:8])
            return article["id"], article

    return (None, None)


def is_approval(text: str) -> bool:
    """Return True if the user's message indicates approval."""
    t = (text or "").strip().lower()
    approval_phrases = (
        "approve", "approved", "yes", "looks good", "lgtm", "ok", "okay",
        "go ahead", "continue", "good", "great", "perfect", "ship it",
        "publish", "do it", "fine", "nice", "love it", "accept",
    )
    for phrase in approval_phrases:
        if t == phrase or t.startswith(phrase + " ") or t.startswith(phrase + ","):
            return True
    # Also accept thumbs-up emoji and similar
    if t in ("👍", "✅", "🤝", "👌"):
        return True
    return False


def log_action(action: str, details: str = "") -> list[dict]:
    """Create an actions_log entry list for state update."""
    from datetime import datetime, timezone
    return [{"action": action, "details": details, "ts": datetime.now(timezone.utc).isoformat()}]

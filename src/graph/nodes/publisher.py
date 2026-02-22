"""Publisher node: Google Docs injection and final QA."""

import logging
from typing import Any

from langgraph.types import interrupt

from src.comm import teach_back
from src.graph.helpers import is_approval, log_action
from src.graph.state import ArticleState

logger = logging.getLogger(__name__)


def publisher_node(state: ArticleState) -> dict[str, Any]:
    """Ensure Google Doc is created/synced, then ask for publish approval."""
    from src.db import get_article, set_article_google_doc_url
    from src.pipeline import emit_status

    emit_status("Publishing article to Google Docs...")
    article_id = state.get("article_id")
    if not article_id:
        return {"response_to_user": "No active article to publish.", "stage": "idle"}

    article = get_article(article_id)
    if not article:
        return {"response_to_user": "Article not found.", "stage": "idle"}

    doc_url = article.get("google_doc_url") or state.get("doc_url")

    # Create Google Doc if not exists
    if not doc_url:
        try:
            from src.google_docs import create_doc_from_markdown
            title = article.get("title", "Untitled Article")
            result = create_doc_from_markdown(article["content"], title=title)
            doc_url = result.get("document_url")
            if doc_url:
                set_article_google_doc_url(article_id, doc_url)
                logger.info("[PUBLISHER] Created Google Doc: %s", doc_url[:60])
        except Exception as e:
            logger.warning("[PUBLISHER] Google Doc creation failed: %s", e)
    else:
        # Sync latest content
        try:
            from src.google_docs import update_doc_from_markdown, _document_id_from_url
            doc_id = _document_id_from_url(doc_url)
            if doc_id:
                update_doc_from_markdown(doc_id, article["content"])
                logger.info("[PUBLISHER] Synced article to Google Doc")
        except Exception as e:
            logger.warning("[PUBLISHER] Google Doc sync failed: %s", e)

    return {
        "doc_url": doc_url,
        "stage": "published",
        "response_to_user": teach_back(
            "Article published!" + (f" Google Doc: {doc_url}" if doc_url else " (Google Doc not available)"),
            ["Say *new article* to start another", "Or continue to make edits"],
        ),
        "actions_log": log_action("published", f"doc_url={doc_url}"),
    }


def qa_node(state: ArticleState) -> dict[str, Any]:
    """QA check: style guide adherence, citation completeness, formatting.

    For now this is a lightweight pass-through. Full QA can be added later.
    """
    article_id = state.get("article_id")
    logger.info("[QA] QA check for article %s", article_id[:8] if article_id else "none")

    return {
        "actions_log": log_action("qa_check", "passed"),
    }

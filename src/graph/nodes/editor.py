"""Editor node: targeted edits to the current draft."""

import logging
from typing import Any

from src.comm import teach_back
from src.graph.helpers import log_action
from src.graph.state import ArticleState

logger = logging.getLogger(__name__)


def editor_node(state: ArticleState) -> dict[str, Any]:
    """Apply edits to the current article draft."""
    from src.tools import improve_article
    from src.pipeline import emit_status

    emit_status("Revising article based on your feedback...")
    article_id = state.get("article_id")
    if not article_id:
        return {
            "response_to_user": "No active article to edit.",
            "stage": "idle",
        }

    user_message = state.get("user_message", "")
    feedback = state.get("intent_params", {}).get("feedback", user_message)

    logger.info("[EDITOR] Editing article %s with feedback: %s", article_id[:8], feedback[:100])

    result = improve_article(
        article_id=article_id,
        feedback=feedback,
        changelog_entry=f"Edited: {feedback[:60]}",
    )

    if result.get("success") is False:
        return {
            "response_to_user": f"Edit failed: {result.get('error', 'unknown error')}",
            "stage": "awaiting_draft_approval",
        }

    new_version = state.get("draft_version", 1) + 1
    doc_url = result.get("google_doc_url") or state.get("doc_url")

    return {
        "draft_version": new_version,
        "doc_url": doc_url,
        "stage": "awaiting_draft_approval",
        "response_to_user": teach_back(
            f"Draft updated (v{new_version})." + (f" Google Doc: {doc_url}" if doc_url else ""),
            ["Reply *approve* to continue", "Or describe more changes"],
        ),
        "actions_log": log_action("editor", f"v{new_version} feedback={feedback[:60]}"),
    }

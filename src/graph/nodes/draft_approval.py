"""Draft approval gate: pauses for user to approve the draft or request edits."""

import logging
from typing import Any

from langgraph.types import interrupt

from src.graph.helpers import is_approval, log_action
from src.graph.state import ArticleState

logger = logging.getLogger(__name__)


def draft_approval_gate(state: ArticleState) -> dict[str, Any]:
    """Pause and ask the user to approve the draft or describe changes."""
    draft_version = state.get("draft_version", 1)
    doc_url = state.get("doc_url")

    user_response = interrupt({
        "type": "draft_approval",
        "message": (
            f"Draft v{draft_version} is ready."
            + (f"\nGoogle Doc: {doc_url}" if doc_url else "")
            + "\nReply *approve* to continue to hero image, or describe changes."
        ),
        "draft_version": draft_version,
    })

    user_text = str(user_response)
    logger.info("[DRAFT_APPROVAL] User response: %s", user_text[:100])

    if is_approval(user_text):
        return {
            "approvals": {"draft": True},
            "stage": "awaiting_hero_description",
            "response_to_user": "Draft approved! Moving to hero image...",
            "actions_log": log_action("draft_approved", f"v{draft_version}"),
        }
    else:
        # User wants edits
        return {
            "intent": "edit_section",
            "user_message": user_text,
            "stage": "editing",
            "actions_log": log_action("draft_edit_requested", user_text[:80]),
        }

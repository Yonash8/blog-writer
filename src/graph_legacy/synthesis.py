"""Synthesis node: build outline + claim/citation plan from research."""

import logging
from typing import Any

from src.graph.helpers import log_action
from src.graph.state import ArticleState

logger = logging.getLogger(__name__)


def synthesis_node(state: ArticleState) -> dict[str, Any]:
    """Synthesize research into an outline.

    For now this is a pass-through that moves to drafting.
    The PromptLayer SEO workflow handles synthesis internally.
    We keep this node as a placeholder for future outline-level control.
    """
    from src.pipeline import emit_status

    emit_status("Building outline from research...")
    sources = state.get("sources", [])
    topic = state.get("intent_params", {}).get("topic", state.get("user_message", ""))

    logger.info("[SYNTHESIS] Passing through to writer with %d sources", len(sources))

    return {
        "stage": "drafting",
        "response_to_user": f"Outline ready. Writing draft for \"{topic[:60]}\"...",
        "actions_log": log_action("synthesis", f"sources={len(sources)}"),
    }

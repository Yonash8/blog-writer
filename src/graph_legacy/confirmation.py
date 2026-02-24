"""Confirmation gate: CRM core communication cycle.

Receives message -> ensures understanding -> user says yes -> then runs.
For start_article: interrupts with "I understood: ... Reply *yes* to proceed."
Sub-steps (research, Tavily, writer) run without approval - status only.
"""

import logging
from typing import Any

from langgraph.types import interrupt

from src.comm import confirmation_message, is_long_running
from src.graph.helpers import is_approval, log_action
from src.graph.state import ArticleState

logger = logging.getLogger(__name__)


def confirmation_gate(state: ArticleState) -> dict[str, Any]:
    """Pause and confirm understanding before running long work. User must reply *yes*."""
    intent = state.get("intent", "")
    params = state.get("intent_params") or {}

    if not is_long_running(intent):
        # Should not reach here; router sends long-running to this gate
        return {"stage": "researching"}

    msg = confirmation_message(intent, params)
    user_response = interrupt({
        "type": "confirmation",
        "message": msg,
    })

    user_text = str(user_response).strip()
    logger.info("[CONFIRMATION] User response: %s", user_text[:100])

    if is_approval(user_text):
        return {
            "stage": "researching",
            "response_to_user": "Proceeding...",
            "actions_log": log_action("confirmation_approved", f"intent={intent}"),
        }

    # User did not say yes - treat as cancel
    return {
        "stage": "idle",
        "response_to_user": (
            "Cancelled. Say *write article* and your topic to start again, "
            "or *help* for options."
        ),
        "actions_log": log_action("confirmation_cancelled", user_text[:80]),
    }

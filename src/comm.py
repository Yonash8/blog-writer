"""CRM (Crew Resource Management) closed-loop communication methods.

Provides call-out (immediate acknowledgment), check-back (status updates),
and teach-back (confirmation with next steps) for agent-user interaction.
"""

from typing import Optional

# Long-running intents that warrant an immediate call-out before processing
LONG_RUNNING_INTENTS = frozenset({
    "start_article",
    "edit_section",
    "edit_last_section",
    "rewrite_with_constraints",
    "regenerate_draft",
    "generate_hero",
    "regenerate_hero",
    "generate_infographic",
    "regenerate_infographic",
    "generate_images",
})


def is_long_running(intent: str) -> bool:
    """True if intent typically involves multi-minute processing."""
    return intent in LONG_RUNNING_INTENTS


def confirmation_message(intent: str, params: dict) -> str:
    """Return 'I understood' confirmation text. User must reply *yes* to proceed."""
    topic = (params.get("topic") or "").strip()
    feedback = (params.get("feedback") or "").strip()[:80]

    if intent == "start_article":
        if topic:
            return (
                f"I understood: you want me to write an article about *{topic}*. "
                "I'll research it, gather links, and write the draft. "
                "Reply *yes* to proceed."
            )
        return (
            "I understood: you want me to write an article. "
            "I'll research, gather links, and write the draft. "
            "Reply *yes* to proceed."
        )

    if intent in ("edit_section", "edit_last_section", "rewrite_with_constraints", "regenerate_draft"):
        if feedback:
            return f"I understood: I'll revise the draft based on your feedback. Reply *yes* to proceed."
        return "I understood: I'll revise the draft. Reply *yes* to proceed."

    if intent in ("generate_hero", "regenerate_hero"):
        return "I understood: you want a hero image. Reply *yes* to proceed."

    if intent in ("generate_infographic", "regenerate_infographic"):
        return "I understood: you want an infographic. I'll analyze the article and generate it. Reply *yes* to proceed."

    if intent == "generate_images":
        return "I understood: you want images for the article. Reply *yes* to proceed."

    return "I understood. Reply *yes* to proceed."


def call_out(intent: str, params: dict) -> str:
    """Return immediate acknowledgment (for status display). Use confirmation_message for interrupt."""
    topic = (params.get("topic") or "").strip()
    if intent == "start_article" and topic:
        return f"Researching *{topic}* and writing the article..."
    return confirmation_message(intent, params)


def teach_back(delivery: str, next_steps: Optional[list[str]] = None) -> str:
    """Format final confirmation with delivery summary and next options.

    Args:
        delivery: What was delivered (e.g. "Draft ready. Google Doc: {url}")
        next_steps: Bulleted options for user (e.g. ["Reply *approve* to continue", "Or describe changes"])
    """
    if not next_steps:
        return delivery.strip()
    bullets = "\n".join(f"• {s}" for s in next_steps)
    return f"{delivery.strip()}\n\nNext:\n{bullets}"

"""Pre-router: lightweight classification that runs BEFORE entering the graph.

Determines whether a message is a general question (handled outside the graph),
or an article-related intent (enters the graph via thread lookup).
"""

import logging
import os
import re
from typing import Literal, Optional

logger = logging.getLogger(__name__)

PreIntent = Literal["general_question", "article_intent", "help", "list_articles"]

# Fast keyword-based heuristics (no LLM call needed for obvious cases)
ARTICLE_KEYWORDS = (
    "write", "article", "draft", "edit", "change", "rewrite", "improve",
    "approve", "hero", "infographic", "publish", "google doc",
    "section", "title", "headline", "intro", "conclusion", "paragraph",
    "status", "outline", "cancel", "stop", "new article",
    "image", "images", "generate", "regenerate",
    "visual", "visuals", "photo", "photos", "illustration",
    "picture", "graphic", "banner",
    "try again", "retry", "list", "find", "show", "recent", "my articles",
)

GENERAL_SIGNALS = (
    "what is", "what are", "how does", "how do", "explain", "tell me about",
    "why", "can you", "define", "difference between",
)

# Stages where ANY non-meta message is almost certainly a response to the workflow
_INTERACTIVE_STAGES = frozenset({
    "awaiting_draft_approval",
    "awaiting_hero_description",
    "awaiting_hero_approval",
    "awaiting_infographic_approval",
    "generating_hero",
    "generating_infographic",
})


def classify_pre_intent(
    text: str,
    active_stage: Optional[str] = None,
) -> PreIntent:
    """Classify whether a message is general or article-related.

    Uses fast keyword heuristics first. If *active_stage* is provided
    (from an active article thread), messages are strongly biased toward
    article_intent when the workflow is waiting for user input.
    """
    t = text.strip().lower()

    # Trivial cases
    if not t:
        return "general_question"
    if t in ("help", "/help", "?"):
        return "help"
    if t in ("list", "list articles", "my articles", "/list"):
        return "list_articles"

    # Stage-aware: if the workflow is waiting for input, almost any message
    # is a response to the current step (hero description, approval, etc.)
    if active_stage and active_stage in _INTERACTIVE_STAGES:
        # Only peel off explicit meta commands
        if t in ("help", "/help", "?"):
            return "help"
        if t in ("list", "list articles", "my articles", "/list"):
            return "list_articles"
        logger.info("[PRE_ROUTER] Active stage %s -> article_intent (stage-aware)", active_stage)
        return "article_intent"

    # Check for strong article signals
    for kw in ARTICLE_KEYWORDS:
        if kw in t:
            return "article_intent"

    # Check for strong general signals
    for sig in GENERAL_SIGNALS:
        if t.startswith(sig):
            return "general_question"

    # Short messages that look like approval/feedback -> article intent
    if len(t) < 30:
        approval_words = {"yes", "no", "ok", "okay", "approve", "approved", "looks good", "lgtm", "👍"}
        if t in approval_words:
            return "article_intent"

    # Ambiguous: default to article_intent if there's an active article,
    # otherwise general_question. The caller should check active thread.
    # For safety, default to general_question.
    return "general_question"

"""Router node: classifies user intent from the closed enum.

Uses a cheap/fast model for intent classification. Output is validated
against the Intent enum -- anything outside the allowed set maps to UNKNOWN.
"""

import json
import logging
import os
from typing import Any

from src.graph.intents import ALL_INTENT_VALUES, Intent
from src.graph.state import ArticleState

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """\
You are an intent classifier for a blog-writing assistant. Given the user's message \
and the current workflow stage, classify the intent into exactly one of the allowed intents.

Allowed intents:
{intent_list}

Current workflow stage: {stage}
Active article ID: {article_id}

## Intent rules

- write a new article -> start_article
- edit/change something in the current article -> edit_section or edit_last_section
- "approve", "looks good", "yes" + stage awaiting_draft_approval -> approve_draft
- "approve" + stage awaiting_hero_approval -> approve_hero
- "approve" + stage awaiting_infographic_approval -> approve_infographic
- provides a description + stage awaiting_hero_description -> generate_hero
- create/generate a hero image -> generate_hero
- feedback on a hero image -> regenerate_hero
- create/generate an infographic -> generate_infographic
- "add visuals", "do visuals for X", "images for X" -> generate_visuals
- feedback on an infographic -> regenerate_infographic
- unrelated question -> general_question
- see current status -> show_status
- list articles / see articles -> list_articles
- info about a specific article (visuals count, word count, etc.) -> show_article_info
- "try again", "retry" after failure -> list_articles
- publish or create Google Doc -> publish_article or inject_to_docs
- switch to another article -> switch_article
- cancel -> cancel_article
- "help" -> help
- unclear -> unknown

## Stage-aware guidance

- After draft approved (stage awaiting_hero_description or generating_hero) -> bias toward generate_hero
- After hero approved (stage analyzing_infographic or generating_infographic) -> bias toward generate_infographic
- When stage is idle and user mentions visuals/images for an article -> generate_visuals

## CRITICAL: Always extract article references

For EVERY intent (except general_question, help, start_article), if the user references an article, \
you MUST extract the reference into params so the system can find it. Use these param fields:

- "topic": when user refers to a subject (e.g. "opus vs sonnet", "solar energy", "kubernetes")
- "article_title": when user refers to a title or description (e.g. "the solar article", "my opus piece")
- "article_id": when user gives a hex ID (e.g. "e799b28b")
- "feedback": for edit/feedback intents

Examples:
- "let's do visuals for opus vs sonnet" -> {{"intent": "generate_visuals", "params": {{"topic": "opus vs sonnet"}}}}
- "edit the kubernetes article" -> {{"intent": "edit_section", "params": {{"article_title": "kubernetes"}}}}
- "publish e799b28b" -> {{"intent": "publish_article", "params": {{"article_id": "e799b28b"}}}}
- "how many images in the AI article?" -> {{"intent": "show_article_info", "params": {{"article_title": "AI"}}}}
- "approve" -> {{"intent": "approve_draft", "params": {{}}}}

Respond with ONLY a JSON object: {{"intent": "<intent_value>", "params": {{}}, "confidence": 0.0-1.0}}
"""


def router_node(state: ArticleState) -> dict[str, Any]:
    """Classify the user's intent and extract parameters."""
    from anthropic import Anthropic
    from src.pipeline import emit_status

    emit_status("Understanding your request...")
    user_message = state.get("user_message", "")
    stage = state.get("stage", "idle")
    article_id = state.get("article_id")

    intent_list = "\n".join(f"- {v}" for v in sorted(ALL_INTENT_VALUES))
    system = ROUTER_SYSTEM_PROMPT.format(
        intent_list=intent_list,
        stage=stage,
        article_id=article_id or "none",
    )

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is required")

    client = Anthropic(api_key=api_key)

    # Build messages with recent conversation context so the router can
    # understand short replies like "1", "the first one", image descriptions, etc.
    recent = state.get("recent_messages", [])
    router_messages = []
    for m in recent[-4:]:
        router_messages.append({"role": m["role"], "content": m["content"]})
    router_messages.append({"role": "user", "content": user_message})

    # Use a cheap model for routing (configurable; claude-3-5-haiku may return 404 on some accounts)
    from src.config import get_config
    router_model = get_config("router_model", "claude-sonnet-4-20250514")
    response = client.messages.create(
        model=router_model,
        max_tokens=256,
        system=system,
        messages=router_messages,
    )

    raw = response.content[0].text.strip()
    logger.info("[ROUTER] Raw output: %s", raw[:300])

    # Parse JSON response
    try:
        # Handle markdown code fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[ROUTER] Failed to parse JSON, defaulting to unknown: %s", raw[:200])
        parsed = {"intent": "unknown", "params": {}, "confidence": 0.0}

    intent_value = parsed.get("intent", "unknown")
    params = parsed.get("params", {})
    confidence = parsed.get("confidence", 0.0)

    # Validate against enum
    if intent_value not in ALL_INTENT_VALUES:
        logger.warning("[ROUTER] Invalid intent %r, mapping to unknown", intent_value)
        intent_value = Intent.UNKNOWN.value

    logger.info(
        "[ROUTER] intent=%s, confidence=%.2f, params=%s",
        intent_value, confidence, {k: str(v)[:80] for k, v in params.items()},
    )

    return {
        "intent": intent_value,
        "intent_params": params,
        "actions_log": [{"action": "router", "details": f"intent={intent_value} conf={confidence:.2f}"}],
    }

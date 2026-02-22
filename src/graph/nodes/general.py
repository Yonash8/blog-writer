"""General question answerer -- no article state mutation."""

import logging
import os
from typing import Any

from src.graph.state import ArticleState

logger = logging.getLogger(__name__)


def general_question_node(state: ArticleState) -> dict[str, Any]:
    """Answer a general question without touching article state."""
    from anthropic import Anthropic
    from src.config import get_config

    user_message = state.get("user_message", "")
    recent = state.get("recent_messages", [])

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is required")

    client = Anthropic(api_key=api_key)
    model = get_config("agent_model", "claude-sonnet-4-5")

    messages = []
    for m in recent[-10:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=(
            "You are a blog-writing assistant. Answer the user's question concisely.\n\n"
            "## Your capabilities\n"
            "You can help with:\n"
            "- *Writing articles*: deep research + SEO optimization → full article + Google Doc\n"
            "- *Hero images*: generated via Gemini AI image generation\n"
            "- *Infographics*: AI-analyzed and generated for articles\n"
            "- *Article editing*: revise, improve, rewrite sections\n"
            "- *Publishing*: create/update Google Docs\n"
            "- *Web search*: find facts and citations\n\n"
            "## Redirecting users\n"
            "You are the general Q&A node — you CANNOT directly generate images, write articles, or modify articles. "
            "When the user wants to perform an action, redirect them to the correct command:\n"
            "- To list articles: 'Say *list articles* to see your articles.'\n"
            "- To generate a hero image: 'Say *generate hero for [article name]* to create a hero image.'\n"
            "- To generate an infographic: 'Say *generate infographic for [article name]*'\n"
            "- To write an article: 'Say *write an article about [topic]*'\n"
            "- To edit an article: 'Say *edit [article name]*'\n"
            "- To add visuals: 'Say *add visuals for [article name]*'\n\n"
            "CRITICAL: Do NOT describe what an image or article would look like. "
            "Do NOT pretend to fulfill action requests. If the user is asking you to generate, create, "
            "or modify something, tell them the exact command to use. Be brief.\n\n"
            "If they ask a pure knowledge question unrelated to article actions, answer it directly."
        ),
        messages=messages,
    )

    answer = response.content[0].text.strip()
    logger.info("[GENERAL] Answered question, len=%d", len(answer))

    return {
        "response_to_user": answer,
        "actions_log": [{"action": "general_question", "details": f"answer_len={len(answer)}"}],
    }

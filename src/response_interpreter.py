"""Agent that translates raw API responses and errors into user-friendly messages."""

import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)

INTERPRETER_PROMPT = """You translate technical API errors and responses into clear, helpful messages for end users.

Context: The user is chatting with an article-writing assistant (drafting articles, topics, images, web search). Something went wrong on our side.

Raw response/error from the system:
---
{raw_content}
---

Your task: Explain what happened in plain language. 1-2 sentences. No technical jargon. If there is a concrete fix (e.g. "set X in settings", "try again later"), mention it briefly. Be helpful and concise.
"""


def explain_to_user(raw_content: str, context: str = "") -> str:
    """
    Pass raw API error/response to an LLM that translates it into a user-friendly message.
    Falls back to a generic message if the interpreter fails.
    """
    if not raw_content or not raw_content.strip():
        return "Something went wrong. Please try again."
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        prompt = INTERPRETER_PROMPT.format(raw_content=raw_content.strip())
        if context:
            prompt = f"Additional context: {context}\n\n{prompt}"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        out = (response.choices[0].message.content or "").strip()
        if out:
            logger.info("[RESPONSE_INTERPRETER] Translated %d chars -> %d chars", len(raw_content), len(out))
            return out
    except Exception as e:
        logger.warning("[RESPONSE_INTERPRETER] Failed to translate: %s", e)
    return "Something went wrong on our side. Please try again in a moment."

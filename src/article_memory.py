from __future__ import annotations

import logging
import os
from typing import Optional

from anthropic import Anthropic

from src.config import get_config

logger = logging.getLogger(__name__)

SUMMARY_MARKER = "[Article summary cached]"


def _fallback_summary(content: str) -> str:
    """Deterministic fallback summary when LLM summarization is unavailable."""
    snippet = ""
    for raw_line in (content or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        snippet = line
        break
    if not snippet:
        snippet = (content or "").strip().replace("\n", " ")
    if len(snippet) > 240:
        snippet = snippet[:240].rstrip() + "..."
    return snippet or "No summary available."


def summarize_with_sonnet(title: str, content: str) -> str:
    """Create a compact, retrieval-friendly article summary using Sonnet."""
    if not (content or "").strip():
        return "Empty article."

    model = os.getenv("ARTICLE_SUMMARY_MODEL") or get_config("agent_model") or "claude-sonnet-4-5"
    if "sonnet" not in model.lower():
        model = "claude-sonnet-4-5"

    body = (content or "").strip()
    # Keep request bounded; enough text for robust summarization.
    truncated = body[:12000]
    was_truncated = len(body) > len(truncated)

    prompt = (
        "Summarize this article for future conversational memory.\n"
        "Return plain text only (no markdown, no bullets), max 2 sentences, max 420 characters.\n"
        "Include core thesis and key scope.\n\n"
        f"Title: {title or 'Untitled'}\n"
        f"Truncated: {was_truncated}\n\n"
        "Article:\n"
        f"{truncated}"
    )

    try:
        client = Anthropic()
        resp = client.messages.create(
            model=model,
            max_tokens=220,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "\n".join(b.text for b in resp.content if hasattr(b, "text")).strip()
        if not text:
            return _fallback_summary(content)
        if len(text) > 420:
            text = text[:420].rstrip() + "..."
        return text
    except Exception as e:
        logger.warning("[ARTICLE_MEMORY] Sonnet summary failed, using fallback: %s", e)
        return _fallback_summary(content)


def format_summary_memory(
    *,
    title: str,
    article_id: str,
    google_doc_url: Optional[str],
    content: str,
    summary: str,
) -> str:
    """Format the memory payload stored in messages for later retrieval."""
    return (
        f"{SUMMARY_MARKER}\n"
        f"title={title}\n"
        f"article_id={article_id}\n"
        f"google_doc_url={google_doc_url or 'none'}\n"
        f"content_chars={len(content or '')}\n"
        f"summary={summary}"
    )


def has_cached_summary(messages: list[dict], article_id: str) -> bool:
    """True if a cached summary for article_id already exists in memory messages."""
    needle = f"article_id={article_id}"
    for m in messages or []:
        content = str(m.get("content", ""))
        if SUMMARY_MARKER in content and needle in content:
            return True
    return False


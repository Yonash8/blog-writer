"""Read-only nodes: show_status, show_outline, list_articles, help."""

import logging
from typing import Any

from src.graph.state import ArticleState

logger = logging.getLogger(__name__)


def show_status_node(state: ArticleState) -> dict[str, Any]:
    """Return current workflow status."""
    stage = state.get("stage", "idle")
    article_id = state.get("article_id")
    draft_version = state.get("draft_version", 0)
    approvals = state.get("approvals", {})

    lines = [f"*Current status*: {stage}"]
    if article_id:
        lines.append(f"Article ID: {article_id[:8]}...")
    if draft_version:
        lines.append(f"Draft version: {draft_version}")
    if approvals:
        for k, v in approvals.items():
            lines.append(f"  {k}: {'approved' if v else 'pending'}")

    return {"response_to_user": "\n".join(lines)}


def list_articles_node(state: ArticleState) -> dict[str, Any]:
    """List recent articles for this user."""
    from src.db import list_articles

    user_id = state.get("whatsapp_user_id", "")
    articles = list_articles(channel="whatsapp", channel_user_id=user_id, limit=10)

    if not articles:
        return {"response_to_user": "No articles found."}

    lines = ["*Your articles:*"]
    for a in articles:
        title = a.get("title", "Untitled")
        status = a.get("status", "draft")
        aid = a.get("id", "?")[:8]
        doc = a.get("google_doc_url", "")
        line = f"- {title} ({status}) [{aid}]"
        if doc:
            line += f" {doc}"
        lines.append(line)

    return {"response_to_user": "\n".join(lines)}


def _looks_like_article_id(s: str) -> bool:
    """True if the string looks like an article ID (UUID or short hex prefix), not a title."""
    if not s or len(s) < 8:
        return False
    t = s.strip()
    # Full UUID format: 8-4-4-4-12 with hex and dashes
    if len(t) >= 32 and "-" in t and all(c in "0123456789abcdefABCDEF-" for c in t):
        return True
    # Short hex prefix (e.g. e799b28b) - 8+ hex chars, no spaces
    if len(t) >= 8 and " " not in t and all(c in "0123456789abcdefABCDEF" for c in t):
        return True
    return False


def show_article_info_node(state: ArticleState) -> dict[str, Any]:
    """Answer questions about a specific article (e.g. how many images, visuals)."""
    from src.db import get_article, get_article_images

    # article_resolver should have set article_id; use it if available
    article_id = state.get("article_id")
    article = None
    if article_id:
        article = get_article(article_id)

    if not article:
        return {"response_to_user": "No article found. Please specify the article by name or ID."}

    article_id = article["id"]
    images = get_article_images(article_id)
    hero_count = sum(1 for i in images if i.get("image_type") == "hero")
    infographic_count = sum(1 for i in images if i.get("image_type") == "infographic")
    generic_count = sum(1 for i in images if i.get("image_type") == "generic")
    total = len(images)

    lines = [f"*{article.get('title', 'Untitled')}*"]
    lines.append(f"Total visuals: {total}")
    if hero_count:
        lines.append(f"  - Hero image: {hero_count}")
    if infographic_count:
        lines.append(f"  - Infographic(s): {infographic_count}")
    if generic_count:
        lines.append(f"  - Other images: {generic_count}")

    return {"response_to_user": "\n".join(lines)}


def help_node(state: ArticleState) -> dict[str, Any]:
    """Return help text."""
    return {
        "response_to_user": (
            "*Available commands:*\n"
            "- \"Write an article about X\" - start a new article\n"
            "- \"Change/edit X\" - modify the current article\n"
            "- \"Create a hero image\" - generate hero with your description\n"
            "- \"Add an infographic\" - auto-analyze and generate\n"
            "- \"Publish\" or \"Create Google Doc\" - publish the article\n"
            "- \"Status\" - see current workflow status\n"
            "- \"List articles\" - see your articles\n"
            "- \"Cancel\" - cancel current operation\n"
            "- Any question - I'll answer without modifying your article"
        ),
    }

from __future__ import annotations
"""Hero image sub-flow: description gate -> generator -> approval gate."""

import logging
import re
from typing import Any

from langgraph.types import interrupt

from src.comm import teach_back
from src.graph.helpers import is_approval, log_action
from src.graph.state import ArticleState

logger = logging.getLogger(__name__)


def hero_description_gate(state: ArticleState) -> dict[str, Any]:
    """Ask the user to describe the hero image. Pauses via interrupt."""
    description = interrupt({
        "type": "hero_description_request",
        "message": (
            "Draft approved! Describe what the hero image should show "
            "(e.g. 'a sunrise over solar panels', 'a person walking into a modern office'). "
            "Hero images are generated with Gemini."
        ),
    })

    logger.info("[HERO] User provided description: %s", str(description)[:100])

    return {
        "hero_description": str(description),
        "hero_feedback": None,
        "hero_attempt": 1,
        "stage": "generating_hero",
        "actions_log": log_action("hero_description", str(description)[:80]),
    }


def hero_generator_node(state: ArticleState) -> dict[str, Any]:
    """Generate a hero image using template + user description + style refs."""
    from src.images import generate_hero_image, upload_to_supabase
    from src.db import add_article_image
    from src.pipeline import emit_status

    emit_status("Generating hero image...")
    article_id = state.get("article_id")

    # article_resolver should have set article_id; fallback to active_thread just in case
    if not article_id:
        whatsapp_user_id = state.get("whatsapp_user_id")
        if whatsapp_user_id:
            from src.graph.thread_manager import get_active_thread
            active = get_active_thread(whatsapp_user_id)
            article_id = (active or {}).get("article_id")
            if article_id:
                logger.info("[HERO] Recovered article_id from active_thread: %s", article_id[:8])

    if not article_id:
        logger.error("[HERO] No article_id in state — cannot generate hero image")
        return {"response_to_user": "No active article for hero image. Please select or create an article first.", "stage": "idle"}

    # Strip "via gemini" etc. at end – hero images always use Gemini
    raw_desc = state.get("hero_description", "")
    description = re.sub(
        r"[,.\s]*(?:via|using|with)\s+gemini\.?\s*$",
        "",
        raw_desc,
        flags=re.IGNORECASE,
    ).strip()
    if not description:
        description = raw_desc
    feedback = state.get("hero_feedback")
    previous_url = state.get("hero_image_url")
    attempt = state.get("hero_attempt", 1)

    logger.info(
        "[HERO] Generating hero image: attempt=%d, desc=%s, feedback=%s",
        attempt, description[:80], (feedback or "")[:80],
    )

    try:
        img_bytes, prompt_used = generate_hero_image(
            description=description,
            feedback=feedback,
            previous_image_url=previous_url if feedback else None,
        )
    except Exception as e:
        logger.error("[HERO] Generation failed: %s", e)
        return {
            "response_to_user": f"Hero image generation failed: {e}\nPlease try again with a different description.",
            "stage": "awaiting_hero_description",
        }

    url = upload_to_supabase(img_bytes, "hero.png")

    image_rec = add_article_image(
        article_id=article_id,
        url=url,
        position=-1,
        alt_text=f"Hero: {description[:100]}",
        prompt_used=prompt_used,
        status="pending_approval",
        image_type="hero",
        metadata={"description": description, "feedback": feedback, "attempt": attempt},
    )

    logger.info("[HERO] Image generated and uploaded: %s", url[:80])

    return {
        "hero_image_id": image_rec["id"],
        "hero_image_url": url,
        "stage": "awaiting_hero_approval",
        "actions_log": log_action("hero_generate", f"attempt={attempt} url={url[:60]}"),
    }


def hero_approval_gate(state: ArticleState) -> dict[str, Any]:
    """Show hero preview and ask for approval. Pauses via interrupt."""
    image_url = state.get("hero_image_url", "")
    attempt = state.get("hero_attempt", 1)

    user_response = interrupt({
        "type": "hero_approval",
        "message": f"Here's the hero image (attempt {attempt}). Reply *approve* or describe what to change.",
        "image_url": image_url,
    })

    user_text = str(user_response)
    logger.info("[HERO] User response to approval: %s", user_text[:100])

    if is_approval(user_text):
        # Approve and inject into article
        _approve_and_inject_hero(state.get("article_id"), state.get("hero_image_id"))
        return {
            "approvals": {"hero": True},
            "hero_feedback": None,
            "stage": "analyzing_infographic",
            "response_to_user": teach_back(
                "Hero image approved and embedded in the article.",
                ["Moving to infographic..."],
            ),
            "actions_log": log_action("hero_approved", f"image_id={state.get('hero_image_id')}"),
        }
    else:
        # Re-iterate with feedback
        return {
            "hero_feedback": user_text,
            "hero_attempt": attempt + 1,
            "stage": "generating_hero",
            "actions_log": log_action("hero_feedback", user_text[:80]),
        }


def _approve_and_inject_hero(article_id: str | None, image_id: str | None) -> None:
    """Approve hero image and inject into article markdown."""
    if not article_id or not image_id:
        logger.warning("[HERO] Cannot approve: article_id=%s, image_id=%s", article_id, image_id)
        return

    from src.db import (
        get_article, get_article_image, update_article_image_status,
        update_article, get_client,
    )
    from src.images import inject_hero_into_markdown

    article = get_article(article_id)
    image = get_article_image(image_id)
    if not article or not image:
        logger.warning("[HERO] Article or image not found")
        return

    update_article_image_status(image_id, "approved")

    updated_content = inject_hero_into_markdown(
        article["content"],
        image["url"],
        alt_text=image.get("alt_text", "Hero image"),
    )

    update_article(article_id, updated_content, changelog_action="Added hero image")
    get_client().table("articles").update({"hero_image_url": image["url"]}).eq("id", article_id).execute()

    # Sync to Google Doc
    try:
        from src.google_docs import update_doc_from_markdown, _document_id_from_url
        doc_url = article.get("google_doc_url")
        if doc_url:
            doc_id = _document_id_from_url(doc_url)
            if doc_id:
                update_doc_from_markdown(doc_id, updated_content)
                logger.info("[HERO] Synced hero image to Google Doc")
    except Exception as e:
        logger.warning("[HERO] Failed to sync to Google Doc: %s", e)

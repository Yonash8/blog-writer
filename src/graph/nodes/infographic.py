"""Infographic sub-flow: analysis -> generator -> approval gate."""

import logging
from typing import Any

from langgraph.types import interrupt

from src.comm import teach_back
from src.graph.helpers import is_approval, log_action
from src.graph.state import ArticleState

logger = logging.getLogger(__name__)


def infographic_analysis_node(state: ArticleState) -> dict[str, Any]:
    """Analyze article to find best infographic placement and type."""
    from src.images import analyze_infographic_placement
    from src.db import get_article
    from src.pipeline import emit_status

    emit_status("Analyzing article for infographic placement...")
    article_id = state.get("article_id")

    # article_resolver should have set article_id; fallback to active_thread just in case
    if not article_id:
        whatsapp_user_id = state.get("whatsapp_user_id")
        if whatsapp_user_id:
            from src.graph.thread_manager import get_active_thread
            active = get_active_thread(whatsapp_user_id)
            article_id = (active or {}).get("article_id")
            if article_id:
                logger.info("[INFOGRAPHIC] Recovered article_id from active_thread: %s", article_id[:8])

    if not article_id:
        return {"response_to_user": "No active article for infographic. Please select or create an article first.", "stage": "idle"}

    article = get_article(article_id)
    if not article or not article.get("content"):
        return {"response_to_user": "Article has no content for infographic analysis.", "stage": "idle"}

    logger.info("[INFOGRAPHIC] Analyzing article %s for infographic placement", article_id[:8])

    try:
        analysis = analyze_infographic_placement(article["content"])
    except Exception as e:
        logger.error("[INFOGRAPHIC] Analysis failed: %s", e)
        return {
            "response_to_user": f"Infographic analysis failed: {e}",
            "stage": "publishing",
        }

    logger.info(
        "[INFOGRAPHIC] Analysis: type=%s, title=%s, section=%s",
        analysis.get("infographic_type"),
        analysis.get("title"),
        analysis.get("section_name"),
    )

    return {
        "infographic_analysis": analysis,
        "infographic_feedback": None,
        "infographic_attempt": 1,
        "stage": "generating_infographic",
        "response_to_user": (
            f"Infographic plan: *{analysis.get('title', 'Infographic')}* "
            f"({analysis.get('infographic_type', 'chart')}) "
            f"after \"{analysis.get('section_name', 'N/A')}\" section. Generating..."
        ),
        "actions_log": log_action("infographic_analysis", f"type={analysis.get('infographic_type')} title={analysis.get('title')}"),
    }


def infographic_generator_node(state: ArticleState) -> dict[str, Any]:
    """Generate infographic using refs from the 'infographics' bucket."""
    from src.images import generate_infographic, upload_to_supabase
    from src.db import add_article_image, get_article
    from src.pipeline import emit_status

    emit_status("Generating infographic...")
    article_id = state.get("article_id")

    if not article_id:
        whatsapp_user_id = state.get("whatsapp_user_id")
        if whatsapp_user_id:
            from src.graph.thread_manager import get_active_thread
            active = get_active_thread(whatsapp_user_id)
            article_id = (active or {}).get("article_id")
            if article_id:
                logger.info("[INFOGRAPHIC] Recovered article_id from active_thread: %s", article_id[:8])

    if not article_id:
        return {"response_to_user": "No active article.", "stage": "idle"}

    article = get_article(article_id)
    if not article or not article.get("content"):
        return {"response_to_user": "Article has no content.", "stage": "idle"}

    analysis = state.get("infographic_analysis", {})
    feedback = state.get("infographic_feedback")
    previous_url = state.get("infographic_image_url")
    attempt = state.get("infographic_attempt", 1)

    logger.info(
        "[INFOGRAPHIC] Generating: attempt=%d, type=%s, feedback=%s",
        attempt, analysis.get("infographic_type"), (feedback or "")[:80],
    )

    try:
        img_bytes, prompt_used, updated_analysis = generate_infographic(
            article["content"],
            feedback=feedback,
            infographic_type=analysis.get("infographic_type"),
            previous_image_url=previous_url if feedback else None,
        )
    except Exception as e:
        logger.error("[INFOGRAPHIC] Generation failed: %s", e)
        return {
            "response_to_user": f"Infographic generation failed: {e}",
            "stage": "publishing",
        }

    url = upload_to_supabase(img_bytes, "infographic.png")

    image_rec = add_article_image(
        article_id=article_id,
        url=url,
        position=0,
        alt_text=f"Infographic: {analysis.get('title', '')}",
        prompt_used=prompt_used,
        status="pending_approval",
        image_type="infographic",
        metadata={
            "position_after": updated_analysis.get("position_after"),
            "infographic_type": updated_analysis.get("infographic_type"),
            "title": updated_analysis.get("title"),
            "description": updated_analysis.get("description"),
            "section_name": updated_analysis.get("section_name"),
            "feedback": feedback,
            "attempt": attempt,
        },
    )

    logger.info("[INFOGRAPHIC] Generated and uploaded: %s", url[:80])

    return {
        "infographic_image_id": image_rec["id"],
        "infographic_image_url": url,
        "infographic_analysis": updated_analysis,
        "stage": "awaiting_infographic_approval",
        "actions_log": log_action("infographic_generate", f"attempt={attempt} url={url[:60]}"),
    }


def infographic_approval_gate(state: ArticleState) -> dict[str, Any]:
    """Show infographic preview and ask for approval."""
    analysis = state.get("infographic_analysis", {})
    image_url = state.get("infographic_image_url", "")
    attempt = state.get("infographic_attempt", 1)

    user_response = interrupt({
        "type": "infographic_approval",
        "message": (
            f"Infographic: \"{analysis.get('title', 'Infographic')}\" "
            f"({analysis.get('infographic_type', 'chart')}), "
            f"placed after \"{analysis.get('section_name', 'N/A')}\" section "
            f"(attempt {attempt}). Reply *approve* or describe changes."
        ),
        "image_url": image_url,
    })

    user_text = str(user_response)
    logger.info("[INFOGRAPHIC] User response: %s", user_text[:100])

    if is_approval(user_text):
        _approve_and_inject_infographic(state.get("article_id"), state.get("infographic_image_id"))
        return {
            "approvals": {"infographic": True},
            "infographic_feedback": None,
            "stage": "publishing",
            "response_to_user": teach_back(
                f"Infographic \"{analysis.get('title', '')}\" approved and embedded in the article.",
                ["Moving to publish..."],
            ),
            "actions_log": log_action("infographic_approved", f"image_id={state.get('infographic_image_id')}"),
        }
    else:
        return {
            "infographic_feedback": user_text,
            "infographic_attempt": attempt + 1,
            "stage": "generating_infographic",
            "actions_log": log_action("infographic_feedback", user_text[:80]),
        }


def _approve_and_inject_infographic(article_id: str | None, image_id: str | None) -> None:
    """Approve infographic and inject into article at analyzed position."""
    if not article_id or not image_id:
        logger.warning("[INFOGRAPHIC] Cannot approve: article_id=%s, image_id=%s", article_id, image_id)
        return

    from src.db import (
        get_article, get_article_image, update_article_image_status,
        update_article, get_client,
    )
    from src.images import inject_infographic_into_markdown

    article = get_article(article_id)
    image = get_article_image(image_id)
    if not article or not image:
        logger.warning("[INFOGRAPHIC] Article or image not found")
        return

    meta = image.get("metadata") or {}
    position_after = meta.get("position_after", "")

    update_article_image_status(image_id, "approved")

    updated_content = inject_infographic_into_markdown(
        article["content"],
        position_after=position_after,
        image_url=image["url"],
        alt_text=image.get("alt_text", "Infographic"),
    )

    changelog = f"Added infographic: {meta.get('title', 'Infographic')}"
    update_article(article_id, updated_content, changelog_action=changelog)
    get_client().table("articles").update({"infographic_url": image["url"]}).eq("id", article_id).execute()

    # Sync to Google Doc
    try:
        from src.google_docs import update_doc_from_markdown, _document_id_from_url
        doc_url = article.get("google_doc_url")
        if doc_url:
            doc_id = _document_id_from_url(doc_url)
            if doc_id:
                update_doc_from_markdown(doc_id, updated_content)
                logger.info("[INFOGRAPHIC] Synced to Google Doc")
    except Exception as e:
        logger.warning("[INFOGRAPHIC] Failed to sync to Google Doc: %s", e)

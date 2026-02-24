"""Writer node: wraps PromptLayer SEO pipeline to generate a draft."""

import logging
from typing import Any

from src.comm import teach_back
from src.graph.helpers import log_action
from src.graph.state import ArticleState

logger = logging.getLogger(__name__)


def writer_node(state: ArticleState) -> dict[str, Any]:
    """Generate an article draft via the existing pipeline."""
    from src.pipeline import run_promptlayer_agent, emit_status
    from src.db import create_article, get_or_create_topic_for_article

    emit_status("Writing article with PromptLayer...")
    topic = state.get("intent_params", {}).get("topic", state.get("user_message", ""))
    research_text = state.get("research_text", "")

    if not research_text:
        logger.warning("[WRITER] No research found, using topic directly")
        research_text = f"Write a comprehensive article about: {topic}"

    logger.info("[WRITER] Calling PromptLayer, topic=%s, research_len=%d", topic[:80], len(research_text))
    try:
        pl_result = run_promptlayer_agent(topic, research_text)
        article_content = pl_result.get("article", "")
    except Exception as e:
        from src.pipeline import _parse_promptlayer_error_for_user
        parsed = _parse_promptlayer_error_for_user(str(e))
        if parsed:
            msg = f"*What went wrong:* {parsed['error']}\n\n*Plan:* {parsed['plan']}"
            logger.warning("[WRITER] PromptLayer token overflow: %s", parsed["error"])
            return {
                "article_id": None,
                "draft_id": None,
                "draft_version": 0,
                "doc_id": None,
                "doc_url": None,
                "stage": "idle",
                "response_to_user": msg,
                "actions_log": log_action("writer", f"failed: token overflow"),
            }
        raise
    logger.info("[WRITER] Draft generated, len=%d", len(article_content))

    # Save to Supabase
    user_id = state.get("whatsapp_user_id", "")
    topic_rec = get_or_create_topic_for_article(topic)
    article = create_article(
        channel="whatsapp",
        channel_user_id=user_id,
        topic_id=topic_rec["id"],
        content=article_content,
        version=1,
        title=topic,
        changelog_entry="Initial draft via LangGraph pipeline",
    )

    article_id = article["id"]
    logger.info("[WRITER] Article saved: id=%s", article_id)

    # Persist article_id in active_article_threads so it survives checkpointer resets
    if user_id:
        from src.graph.thread_manager import get_active_thread, set_active_thread
        active = get_active_thread(user_id)
        if active and active.get("thread_id"):
            set_active_thread(user_id, active["thread_id"], article_id=article_id)
            logger.info("[WRITER] Updated active_article_threads with article_id=%s", article_id[:8])

    # Create Google Doc
    doc_url = None
    try:
        from src.google_docs import create_doc_from_markdown
        from src.db import set_article_google_doc_url
        result = create_doc_from_markdown(article_content, title=topic)
        doc_url = result.get("document_url")
        if doc_url:
            set_article_google_doc_url(article_id, doc_url)
            logger.info("[WRITER] Google Doc created: %s", doc_url[:60])
    except Exception as e:
        logger.warning("[WRITER] Google Doc creation failed: %s", e)

    return {
        "article_id": article_id,
        "draft_id": article_id,
        "draft_version": 1,
        "doc_id": doc_url,
        "doc_url": doc_url,
        "stage": "awaiting_draft_approval",
        "response_to_user": teach_back(
            f"Draft ready for \"{topic[:60]}\"." + (f" Google Doc: {doc_url}" if doc_url else ""),
            ["Reply *approve* to continue", "Or describe what to change"],
        ),
        "actions_log": log_action("writer", f"article_id={article_id} len={len(article_content)}"),
    }

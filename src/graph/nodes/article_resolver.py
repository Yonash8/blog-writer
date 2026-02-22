"""Article resolver node: searches DB for referenced articles and confirms with user.

Runs after the router for any intent that needs an article.
Uses the user's message + intent_params to find candidate articles,
then confirms with the user via interrupt before loading into state.
"""

import json
import logging
import os
from typing import Any

from langgraph.types import interrupt

from src.graph.helpers import log_action
from src.graph.state import ArticleState

logger = logging.getLogger(__name__)

# Intents that skip the resolver (they don't need an article, or create one)
SKIP_RESOLVER_INTENTS = frozenset({
    "general_question", "help", "list_articles", "start_article", "unknown",
})

# Intents where the resolver should run but the article is optional (not fatal if missing)
ARTICLE_OPTIONAL_INTENTS = frozenset({
    "show_status", "show_outline", "show_article_info", "cancel_article",
    "switch_article",
})


def article_resolver_node(state: ArticleState) -> dict[str, Any]:
    """Resolve which article the user is referring to.

    Strategy:
    1. If article_id already in state and valid -> skip (article already loaded)
    2. Extract any article reference from user message + intent_params
    3. Search DB: by ID/prefix, then by title, then by topic/keywords
    4. If 1 strong match -> confirm with user ("Did you mean '[title]'?")
    5. If multiple matches -> show options and ask user to pick
    6. If no matches -> tell user (or skip if article is optional for this intent)
    """
    intent = state.get("intent", "unknown")
    article_id = state.get("article_id")
    user_id = state.get("whatsapp_user_id", "")

    # Skip for intents that don't need article resolution
    if intent in SKIP_RESOLVER_INTENTS:
        logger.info("[RESOLVER] Skipping for intent=%s", intent)
        return {}

    # If article_id already set and valid, skip resolution
    if article_id:
        from src.db import get_article
        if get_article(article_id):
            logger.info("[RESOLVER] article_id already in state: %s", article_id[:8])
            return {}
        else:
            logger.warning("[RESOLVER] article_id %s in state but not found in DB, re-resolving", article_id[:8])

    # Check active_thread for article_id
    if not article_id and user_id:
        from src.graph.thread_manager import get_active_thread
        active = get_active_thread(user_id)
        active_article_id = (active or {}).get("article_id")
        if active_article_id:
            from src.db import get_article
            if get_article(active_article_id):
                logger.info("[RESOLVER] Recovered article_id from active_thread: %s", active_article_id[:8])
                return {"article_id": active_article_id}

    # Search for articles matching the user's references
    candidates = _find_candidate_articles(state)

    if not candidates:
        # No matches found
        if intent in ARTICLE_OPTIONAL_INTENTS:
            logger.info("[RESOLVER] No article found, but intent=%s is optional", intent)
            return {}

        # For required intents, ask user to specify
        logger.info("[RESOLVER] No article candidates found for intent=%s", intent)
        return {
            "response_to_user": (
                "I couldn't find which article you're referring to. "
                "Please specify the article by name or ID.\n"
                "Say *list articles* to see your articles."
            ),
            "stage": "idle",
            "intent": "unknown",
        }

    if len(candidates) == 1:
        article = candidates[0]
        title = article.get("title", "Untitled")
        aid = article["id"][:8]

        # Confirm with user
        user_response = interrupt({
            "type": "article_confirmation",
            "message": f"I found *{title}* [{aid}]. Is this the article you mean? (yes/no)",
        })

        if _is_yes(str(user_response)):
            logger.info("[RESOLVER] User confirmed article: %s (%s)", title[:40], aid)
            # Update active thread to point to this article
            from src.graph.thread_manager import set_active_thread
            thread_id = f"{user_id}:{article['id']}"
            set_active_thread(user_id, thread_id, article_id=article["id"])
            return {
                "article_id": article["id"],
                "resolved_article_title": title,
                "actions_log": log_action("article_resolved", f"confirmed {title[:40]} [{aid}]"),
            }
        else:
            # User said no -- show alternatives or ask to specify
            logger.info("[RESOLVER] User rejected article: %s", title[:40])
            return {
                "response_to_user": (
                    "No problem. Please specify the article by name or ID.\n"
                    "Say *list articles* to see your articles."
                ),
                "stage": "idle",
                "intent": "unknown",
            }

    # Multiple candidates -- show top 3 and ask user to pick
    options = []
    for i, a in enumerate(candidates[:5], 1):
        title = a.get("title", "Untitled")
        aid = a["id"][:8]
        status = a.get("status", "draft")
        options.append(f"{i}. *{title}* ({status}) [{aid}]")

    options_text = "\n".join(options)
    user_response = interrupt({
        "type": "article_selection",
        "message": (
            f"I found multiple articles that could match:\n{options_text}\n\n"
            "Reply with the number (1-5) or article name."
        ),
    })

    selected = _parse_selection(str(user_response), candidates)
    if selected:
        sel_title = selected.get("title", "Untitled")
        logger.info("[RESOLVER] User selected article: %s", sel_title[:40])
        from src.graph.thread_manager import set_active_thread
        thread_id = f"{user_id}:{selected['id']}"
        set_active_thread(user_id, thread_id, article_id=selected["id"])
        return {
            "article_id": selected["id"],
            "resolved_article_title": sel_title,
            "actions_log": log_action("article_resolved", f"selected {sel_title[:40]} [{selected['id'][:8]}]"),
        }
    else:
        return {
            "response_to_user": (
                "I couldn't match your response to an article. "
                "Please specify the article by name or ID."
            ),
            "stage": "idle",
            "intent": "unknown",
        }


def _find_candidate_articles(state: ArticleState) -> list[dict]:
    """Search DB for articles matching the user's message and intent params.

    Uses a multi-strategy approach:
    1. Exact ID or prefix from intent_params
    2. Title search from intent_params (article_title, title, topic)
    3. Individual keyword search (split multi-word params into single words)
    4. LLM keyword extraction from user_message
    5. Semantic matching: fetch all user articles and let LLM pick the best match
    """
    from src.db import get_article_by_id_or_prefix, list_articles

    user_id = state.get("whatsapp_user_id", "")
    params = state.get("intent_params") or {}
    user_message = state.get("user_message", "")
    channel = "whatsapp"
    found = {}

    # Strategy 1: ID from params
    article_id_param = params.get("article_id", "") or ""
    if article_id_param and isinstance(article_id_param, str) and len(article_id_param.strip()) >= 4:
        art = get_article_by_id_or_prefix(article_id_param.strip(), channel=channel, channel_user_id=user_id)
        if art:
            found[art["id"]] = art

    # Strategy 2: Full phrase title search from params
    search_terms = []
    for key in ("article_title", "title", "topic"):
        val = params.get(key, "")
        if val and isinstance(val, str) and val.strip() and len(val.strip()) >= 2:
            search_terms.append(val.strip())
            arts = list_articles(channel=channel, channel_user_id=user_id, title_query=val.strip(), limit=5)
            for a in arts:
                found.setdefault(a["id"], a)

    # Strategy 3: Split multi-word terms into individual significant words
    if not found and search_terms:
        stop_words = {"the", "a", "an", "is", "vs", "for", "to", "and", "or", "my", "about", "of", "in", "on", "with"}
        for term in search_terms:
            words = [w for w in term.lower().split() if w not in stop_words and len(w) >= 3]
            for word in words:
                arts = list_articles(channel=channel, channel_user_id=user_id, title_query=word, limit=5)
                for a in arts:
                    found.setdefault(a["id"], a)

    # Strategy 4: LLM keyword extraction from user message
    if not found and user_message:
        keywords = _extract_article_keywords(user_message, params)
        if keywords:
            for kw in keywords[:3]:
                arts = list_articles(channel=channel, channel_user_id=user_id, title_query=kw, limit=5)
                for a in arts:
                    found.setdefault(a["id"], a)
            # Also split LLM keywords into individual words
            if not found:
                stop_words = {"the", "a", "an", "is", "vs", "for", "to", "and", "or", "my", "about", "of", "in", "on", "with"}
                for kw in keywords[:3]:
                    words = [w for w in kw.lower().split() if w not in stop_words and len(w) >= 3]
                    for word in words:
                        arts = list_articles(channel=channel, channel_user_id=user_id, title_query=word, limit=5)
                        for a in arts:
                            found.setdefault(a["id"], a)

    # Strategy 5: Semantic matching - fetch all user articles and let LLM pick
    if not found and user_message:
        all_articles = list_articles(channel=channel, channel_user_id=user_id, limit=20)
        if all_articles:
            matched = _semantic_match_articles(user_message, all_articles)
            for a in matched:
                found.setdefault(a["id"], a)

    return list(found.values())


def _extract_article_keywords(user_message: str, params: dict) -> list[str]:
    """Use LLM to extract potential article-identifying keywords from the user message.

    Returns a list of search terms to try against article titles.
    """
    try:
        from anthropic import Anthropic
        from src.config import get_config

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return []

        client = Anthropic(api_key=api_key)
        router_model = get_config("router_model", "claude-3-haiku-20240307")

        response = client.messages.create(
            model=router_model,
            max_tokens=128,
            system=(
                "Extract article-identifying keywords from the user message. "
                "The user is referencing a blog article they previously created. "
                "Return a JSON array of 1-3 short search terms that could match "
                "the article's title or topic. Examples:\n"
                "- 'do visuals for the opus vs sonnet piece' -> [\"opus vs sonnet\"]\n"
                "- 'edit my solar energy article' -> [\"solar energy\"]\n"
                "- 'publish the one about kubernetes' -> [\"kubernetes\"]\n"
                "- 'add hero image to my latest AI post' -> [\"AI\"]\n"
                "Return ONLY a JSON array. If no article reference found, return []."
            ),
            messages=[{"role": "user", "content": user_message}],
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        keywords = json.loads(raw)
        if isinstance(keywords, list):
            logger.info("[RESOLVER] Extracted keywords: %s", keywords[:3])
            return [str(k).strip() for k in keywords if k and str(k).strip()]
    except Exception as e:
        logger.warning("[RESOLVER] Keyword extraction failed: %s", e)

    return []


def _semantic_match_articles(user_message: str, articles: list[dict]) -> list[dict]:
    """Use LLM to semantically match the user's message to one of their articles.

    This is a last-resort strategy when keyword search fails because the user
    describes the article differently from its title (e.g. "opus vs sonnet"
    when the title is "Is Opus Smarter Than Sonnet").
    """
    try:
        from anthropic import Anthropic
        from src.config import get_config

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return []

        article_list = "\n".join(
            f"{i + 1}. [{a.get('id', '?')[:8]}] {a.get('title', 'Untitled')}"
            for i, a in enumerate(articles)
        )

        client = Anthropic(api_key=api_key)
        router_model = get_config("router_model", "claude-3-haiku-20240307")

        response = client.messages.create(
            model=router_model,
            max_tokens=64,
            system=(
                "The user is referring to one of their blog articles. "
                "Given their message and the list of articles below, return the NUMBER "
                "(1-based index) of the article they most likely mean. "
                "If none match, return 0.\n\n"
                f"Articles:\n{article_list}\n\n"
                "Respond with ONLY a single integer."
            ),
            messages=[{"role": "user", "content": user_message}],
        )

        raw = response.content[0].text.strip()
        idx = int(raw) - 1
        if 0 <= idx < len(articles):
            logger.info("[RESOLVER] Semantic match: #%d -> %s", idx + 1, articles[idx].get("title", "?")[:40])
            return [articles[idx]]
    except Exception as e:
        logger.warning("[RESOLVER] Semantic matching failed: %s", e)

    return []


def _is_yes(text: str) -> bool:
    """Check if user confirmed."""
    t = text.strip().lower()
    yes_words = (
        "yes", "yep", "yeah", "yea", "y", "correct", "right", "that's it",
        "thats it", "exactly", "si", "da", "ok", "okay", "sure", "confirm",
        "that one", "this one",
    )
    for w in yes_words:
        if t == w or t.startswith(w + " ") or t.startswith(w + ",") or t.startswith(w + "."):
            return True
    if t in ("👍", "✅", "👌"):
        return True
    return False


def _parse_selection(text: str, candidates: list[dict]) -> dict | None:
    """Parse user's article selection from their response."""
    t = text.strip()

    # Try numeric selection (1, 2, 3, etc.)
    if t.isdigit():
        idx = int(t) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]

    # Try matching by title keyword
    t_lower = t.lower()
    for c in candidates:
        title = (c.get("title") or "").lower()
        if title and (t_lower in title or title in t_lower):
            return c

    # Try matching by ID prefix
    for c in candidates:
        if c.get("id", "").lower().startswith(t_lower):
            return c

    return None

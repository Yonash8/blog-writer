from __future__ import annotations
"""Tool implementations for the autonomous master agent.

General-purpose tools:
  - db: Full database access (select, insert, update, delete, sql)
  - google_docs_tool: Create/update Google Docs
  - web_search: Web search via Tavily
  - send_image_tool: Send an image to the user (WhatsApp media or web)

Pipeline tools (complex orchestration):
  - write_article: Deep research + Tavily + PromptLayer → article
  - improve_article: LLM-based article revision
  - generate_and_place_images: AI image generation + placement
  - generate_hero_image_tool / approve_hero_image_tool
  - generate_infographic_tool / approve_infographic_tool
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

from src.observability import observe_agent_call, observe_sub_agent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context holders: set by the agent runner to pass channel info to tools
# ---------------------------------------------------------------------------

import contextvars

_current_channel = contextvars.ContextVar("_current_channel", default=None)
_current_chat_id = contextvars.ContextVar("_current_chat_id", default=None)
_current_quoted_message_id = contextvars.ContextVar("_current_quoted_message_id", default=None)


# ---------------------------------------------------------------------------
# General-purpose tool: db
# ---------------------------------------------------------------------------

def db(
    action: str,
    table: Optional[str] = None,
    columns: str = "*",
    filters: Optional[dict] = None,
    data: Optional[dict] = None,
    order_by: Optional[str] = None,
    limit: Optional[int] = None,
    sql: Optional[str] = None,
) -> dict[str, Any]:
    """General-purpose database tool.

    Actions:
      select  – query rows from a table (with optional filters, ordering, limit)
      insert  – insert a row into a table
      update  – update rows matching filters
      delete  – delete rows matching filters
      sql     – execute a read-only SQL query (SELECT/WITH only) for aggregations, JOINs, etc.
    """
    from src.db import get_client, execute_sql

    logger.info("[TOOL] db: action=%s, table=%s, filters=%s", action, table, filters)

    try:
        if action == "sql":
            if not sql:
                return {"success": False, "error": "sql parameter is required for action='sql'"}
            rows = execute_sql(sql)
            return {"success": True, "rows": rows, "count": len(rows)}

        if not table:
            return {"success": False, "error": "table parameter is required for CRUD actions"}

        client = get_client()

        if action == "select":
            q = client.table(table).select(columns)
            for k, v in (filters or {}).items():
                q = q.eq(k, v)
            if order_by:
                desc = order_by.startswith("-")
                col = order_by.lstrip("-")
                q = q.order(col, desc=desc)
            if limit:
                q = q.limit(limit)
            r = q.execute()
            rows = r.data or []
            return {"success": True, "rows": rows, "count": len(rows)}

        elif action == "insert":
            if not data:
                return {"success": False, "error": "data parameter is required for insert"}
            r = client.table(table).insert(data).execute()
            if not r.data or len(r.data) == 0:
                return {"success": False, "error": f"Insert into {table} returned no data"}
            return {"success": True, "row": r.data[0]}

        elif action == "update":
            if not data:
                return {"success": False, "error": "data parameter is required for update"}
            if not filters:
                return {"success": False, "error": "filters are required for update (to prevent accidental full-table updates)"}
            q = client.table(table).update(data)
            for k, v in filters.items():
                q = q.eq(k, v)
            r = q.execute()
            rows = r.data or []
            return {"success": True, "rows": rows, "count": len(rows)}

        elif action == "delete":
            if not filters:
                return {"success": False, "error": "filters are required for delete (to prevent accidental full-table deletes)"}
            q = client.table(table).delete()
            for k, v in filters.items():
                q = q.eq(k, v)
            r = q.execute()
            deleted = r.data or []
            return {"success": True, "deleted": len(deleted)}

        else:
            return {"success": False, "error": f"Unknown action: {action}. Use select/insert/update/delete/sql."}

    except Exception as e:
        logger.exception("[TOOL] db EXCEPTION: %s", e)
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# General-purpose tool: google_docs_tool
# ---------------------------------------------------------------------------

def google_docs_tool(
    action: str,
    article_id: Optional[str] = None,
    markdown: Optional[str] = None,
    title: Optional[str] = None,
    document_id: Optional[str] = None,
    document_url: Optional[str] = None,
) -> dict[str, Any]:
    """General-purpose Google Docs tool.

    Actions:
      create – Create a new Google Doc from an article (by article_id) or raw markdown.
               Saves the google_doc_url back to the article in DB if article_id is provided.
      update – Update an existing Google Doc with new content.
               Provide document_id (or article_id to auto-resolve), plus article_id or markdown.
      fetch  – Read content from a Google Doc. Provide document_url or document_id.
               Returns content, title. Doc must be shared with the service account/OAuth user.
    """
    logger.info("[TOOL] google_docs: action=%s, article_id=%s, document_id=%s, document_url=%s", action, article_id, document_id, document_url[:50] if document_url else None)

    try:
        if action == "fetch":
            from src.google_docs import fetch_doc_content

            doc_ref = document_url or document_id
            if not doc_ref:
                return {"success": False, "error": "Provide document_url or document_id for fetch action."}
            result = fetch_doc_content(doc_ref)
            return result

        if action == "create":
            if article_id:
                from src.google_docs import create_google_doc_from_article
                from src.db import set_article_google_doc_url
                result = create_google_doc_from_article(article_id, title=title)
                if "error" in result:
                    return {"success": False, "error": result["error"]}
                doc_url = result["document_url"]
                set_article_google_doc_url(article_id, doc_url)
                return {
                    "success": True,
                    "document_id": result["document_id"],
                    "document_url": doc_url,
                    "article_id": article_id,
                }
            elif markdown:
                from src.google_docs import create_doc_from_markdown
                result = create_doc_from_markdown(markdown, title=title or "Untitled")
                return {
                    "success": True,
                    "document_id": result["document_id"],
                    "document_url": result["document_url"],
                }
            else:
                return {"success": False, "error": "Provide article_id or markdown for create action."}

        elif action == "update":
            from src.google_docs import update_doc_from_markdown, _document_id_from_url

            # Resolve document_id
            doc_id = document_id
            if not doc_id and article_id:
                from src.db import get_article
                article = get_article(article_id)
                if not article:
                    return {"success": False, "error": f"Article {article_id} not found"}
                url = article.get("google_doc_url")
                if url:
                    doc_id = _document_id_from_url(url)

            if not doc_id:
                return {"success": False, "error": "Could not resolve document_id. Provide document_id or an article_id with a google_doc_url."}

            # Resolve content
            content = markdown
            if not content and article_id:
                from src.db import get_article
                article = get_article(article_id)
                if article:
                    content = article.get("content", "")

            if not content:
                return {"success": False, "error": "No content to update. Provide markdown or an article_id with content."}

            result = update_doc_from_markdown(doc_id, content)
            return {
                "success": True,
                "document_id": result["document_id"],
                "document_url": result["document_url"],
            }

        else:
            return {"success": False, "error": f"Unknown action: {action}. Use create/update/fetch."}

    except Exception as e:
        logger.exception("[TOOL] google_docs EXCEPTION: %s", e)
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# General-purpose tool: web_search
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 10) -> dict[str, Any]:
    """Web search via Tavily."""
    from tavily import TavilyClient

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {"success": False, "error": "TAVILY_API_KEY not set"}

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
        )
        results = []
        for r in response.get("results", []):
            results.append({
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "content": r.get("content", ""),
            })
        return {"success": True, "results": results, "count": len(results)}
    except Exception as e:
        logger.exception("[TOOL] web_search EXCEPTION: %s", e)
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# General-purpose tool: send_image
# ---------------------------------------------------------------------------

def send_image_tool(
    image_url: str,
    caption: Optional[str] = None,
) -> dict[str, Any]:
    """Send an image to the current user as a media message.

    On WhatsApp: sends via Green API sendFileByUrl (user sees the image inline).
    On web: returns the URL (frontend handles display).
    The agent should call this whenever it wants the user to SEE an image
    (previews, approved images, etc.) rather than just pasting a URL in text.
    """
    channel = _current_channel.get()
    chat_id = _current_chat_id.get()

    # Ensure https
    secure_url = image_url.replace("http://", "https://", 1) if image_url.startswith("http://") else image_url

    if channel == "whatsapp" and chat_id:
        try:
            from src.channels.whatsapp import send_image_by_url
            ext = secure_url.rsplit(".", 1)[-1].split("?")[0]
            fname = f"image.{ext}" if ext in ("png", "jpg", "jpeg", "webp") else "image.png"
            result = send_image_by_url(chat_id, secure_url, filename=fname, caption=caption)
            logger.info("[TOOL] send_image: sent media to WhatsApp %s", chat_id[:20])
            return {"success": True, "sent_as": "whatsapp_media", "chat_id": chat_id}
        except Exception as e:
            logger.warning("[TOOL] send_image: WhatsApp media failed (%s), returning URL", e)
            return {"success": True, "sent_as": "url_fallback", "image_url": secure_url, "note": f"Media send failed: {e}. Include the URL in your text reply instead."}

    # Web or unknown channel: return the URL for the frontend to display
    return {"success": True, "sent_as": "url", "image_url": secure_url}


# ---------------------------------------------------------------------------
# Pipeline tool: write_article
# ---------------------------------------------------------------------------

def write_article(
    topic: str,
    channel: Optional[str] = None,
    channel_user_id: Optional[str] = None,
    include_tavily: bool = True,
    tavily_max_results: int = 20,
    changelog_entry: Optional[str] = None,
) -> dict[str, Any]:
    """Full pipeline: deep research + Tavily + PromptLayer SEO → save to DB → create Google Doc.

    Returns dict with article_id, content, google_doc_url.
    """
    # Fall back to ContextVars when the LLM doesn't pass channel explicitly
    if not channel:
        channel = _current_channel.get(None)
    if not channel_user_id:
        channel_user_id = _current_chat_id.get(None)

    logger.info("[TOOL] write_article: topic=%r, channel=%s, channel_user_id=%s, include_tavily=%s, tavily_max_results=%d",
                topic, channel, channel_user_id, include_tavily, tavily_max_results)
    from src.pipeline import write_article as _write
    from src.db import create_article, add_message, get_or_create_topic_for_article, set_article_google_doc_url

    if not channel or not channel_user_id:
        return {"success": False, "error": "channel and channel_user_id are required", "retry_hint": "This should be set automatically by the agent."}

    from src.pipeline import emit_status

    try:
        pl_result = _write(topic, include_tavily=include_tavily, tavily_max_results=tavily_max_results)
    except Exception as e:
        from src.pipeline import _parse_promptlayer_error_for_user
        parsed = _parse_promptlayer_error_for_user(str(e))
        if parsed:
            return {
                "success": False,
                "error": parsed["error"],
                "plan": parsed["plan"],
                "retry_hint": parsed["retry_hint"],
            }
        raise

    content = pl_result.get("article", "")
    metadata = pl_result.get("metadata")

    emit_status("Saving article to database...")
    topic_rec = get_or_create_topic_for_article(topic)
    article = create_article(
        channel=channel,
        channel_user_id=channel_user_id,
        topic_id=topic_rec["id"],
        content=content,
        version=1,
        title=topic,
        changelog_entry=changelog_entry,
    )
    emit_status("Creating Google Doc...")
    google_doc_url = None
    try:
        from src.google_docs import create_doc_from_markdown
        result = create_doc_from_markdown(content, title=topic)
        google_doc_url = result.get("document_url")
        if google_doc_url:
            set_article_google_doc_url(article["id"], google_doc_url)
            logger.info("[TOOL] write_article: created Google Doc %s", google_doc_url[:60])
    except Exception as e:
        logger.warning("[TOOL] write_article: Google Doc creation failed (%s), returning content only", e)

    add_message(channel, channel_user_id, "user", f"Write an article about: {topic}")
    add_message(channel, channel_user_id, "assistant", f"[Article created]\n\n{content}")
    logger.info("[TOOL] write_article DONE: article_id=%s, content_len=%d", article["id"], len(content))
    return {
        "article_id": article["id"],
        "content": content,
        "topic": topic,
        "google_doc_url": google_doc_url,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Pipeline tool: improve_article
# ---------------------------------------------------------------------------

def _sync_article_to_google_doc(article_id: str, content: str) -> Optional[str]:
    """If article has a google_doc_url, update the doc with new content. Returns url or None."""
    from src.db import get_article
    from src.google_docs import update_doc_from_markdown, _document_id_from_url

    article = get_article(article_id)
    if not article:
        return None
    url = article.get("google_doc_url")
    if not url:
        return None
    doc_id = _document_id_from_url(url)
    if not doc_id:
        return None
    try:
        update_doc_from_markdown(doc_id, content)
        logger.info("[TOOL] Synced article %s to Google Doc", article_id[:8])
        return url
    except Exception as e:
        logger.warning("[TOOL] Failed to sync article to Google Doc: %s", e)
        return None


def _revise_with_promptlayer(content: str, feedback: str) -> Optional[str]:
    """Call PromptLayer revise agent (if configured). Returns revised content or None."""
    import httpx

    api_key = os.getenv("PROMPTLAYER_API_KEY")
    workflow = os.getenv("PROMPTLAYER_REVISE_WORKFLOW", "revise agent")
    if not api_key:
        raise ValueError("PROMPTLAYER_API_KEY not set")

    start_ts = datetime.now(timezone.utc).isoformat()
    start_perf = time.perf_counter()
    with httpx.Client() as client:
        r = client.post(
            f"https://api.promptlayer.com/workflows/{workflow.replace(' ', '%20')}/run",
            headers={"Content-Type": "application/json", "X-API-KEY": api_key},
            json={
                "input_variables": {"article": content, "feedback": feedback},
                "return_all_outputs": True,
            },
            timeout=300.0,
        )
        r.raise_for_status()
        data = r.json()
        if not data.get("success"):
            raise RuntimeError(data.get("message", "PromptLayer revise failed"))
        exec_id = data.get("workflow_version_execution_id")
        if not exec_id:
            raise RuntimeError("No execution ID")

        import time
        from src.pipeline import emit_status
        for poll_i in range(60):
            res = client.get(
                "https://api.promptlayer.com/workflow-version-execution-results",
                headers={"X-API-KEY": api_key},
                params={"workflow_version_execution_id": exec_id},
                timeout=30.0,
            )
            if res.status_code == 202:
                if poll_i > 0 and poll_i % 4 == 0:
                    elapsed_min = (poll_i + 1) * 15 // 60
                    emit_status(f"Still improving... ({elapsed_min}m)")
                time.sleep(15)
                continue
            if res.status_code != 200:
                raise RuntimeError(f"Failed to get results: {res.text}")
            results = res.json()
            for v in results.values() if isinstance(results, dict) else []:
                if isinstance(v, dict) and "value" in v:
                    revised = v["value"]
                    if isinstance(revised, str) and len(revised) > 100:
                        latency_ms = (time.perf_counter() - start_perf) * 1000
                        observe_sub_agent(
                            name="promptlayer_revise",
                            input_keys=["article", "feedback"],
                            output_size=len(revised),
                            latency_ms=latency_ms,
                            status="success",
                            provider="promptlayer",
                        )
                        return revised
        latency_ms = (time.perf_counter() - start_perf) * 1000
        observe_sub_agent(
            name="promptlayer_revise",
            input_keys=["article", "feedback"],
            output_size=0,
            latency_ms=latency_ms,
            status="failure",
            provider="promptlayer",
        )
        raise RuntimeError("PromptLayer revise timed out")


def improve_article(
    article_id: str,
    feedback: str,
    use_promptlayer: bool = False,
    links: Optional[list[dict[str, str]]] = None,
    changelog_entry: Optional[str] = None,
) -> dict[str, Any]:
    """Revise an existing article based on user feedback. Uses OpenAI by default.

    Supports: text edits (rephrase, add, remove), full rewrites (use_promptlayer=True),
    and injecting Markdown links.
    """
    from src.db import get_article, update_article

    article = get_article(article_id)
    if not article:
        return {"success": False, "error": f"Article {article_id} not found"}

    current_content = article["content"]
    changelog_action = changelog_entry.strip() if changelog_entry and changelog_entry.strip() else None

    # Use Google Docs as source of truth — humans may have edited the doc directly
    google_doc_url = article.get("google_doc_url")
    if google_doc_url:
        from src.google_docs import fetch_doc_content, _document_id_from_url
        doc_id = _document_id_from_url(google_doc_url)
        if doc_id:
            doc_result = fetch_doc_content(doc_id)
            if doc_result.get("success") and doc_result.get("content"):
                current_content = doc_result["content"]

    if use_promptlayer:
        try:
            revised = _revise_with_promptlayer(current_content, feedback)
            if revised:
                update_article(article_id, revised, changelog_action=changelog_action)
                _sync_article_to_google_doc(article_id, revised)
            article = get_article(article_id)
            return {"article_id": article_id, "content": article["content"], "google_doc_url": article.get("google_doc_url")}
        except Exception as e:
            err_str = str(e)
            retry_hint = "Try without use_promptlayer, or use shorter feedback."
            if "token" in err_str.lower() or "limit" in err_str.lower():
                retry_hint = "Content too long for PromptLayer. Try without use_promptlayer for focused edits."
            return {"success": False, "error": err_str, "retry_hint": retry_hint}

    # Anthropic path: edits + link injection via Claude Sonnet
    from src.prompts_loader import get_prompt, get_prompt_llm_kwargs
    import anthropic as _anthropic

    links_section = ""
    if links:
        lines = ["Links to inject (add as [anchor](url) in the article where relevant):"]
        for i, lnk in enumerate(links[:10], 1):  # cap at 10
            url = lnk.get("url") or lnk.get("link", "")
            title = lnk.get("title") or lnk.get("anchor_text", "")
            if url:
                lines.append(f"  {i}. {url}" + (f" — use as: {title}" if title else ""))
        links_section = "\n" + "\n".join(lines)

    anthropic_client = _anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    improve_prompt = get_prompt("improve_article")
    prompt = improve_prompt.format(
        feedback=feedback,
        links_section=links_section,
        article=current_content,
    )

    llm_kwargs = get_prompt_llm_kwargs("improve_article")
    model = llm_kwargs.get("model") or "claude-sonnet-4-5"
    max_tokens = llm_kwargs.get("max_tokens") or 8192
    start_ts = datetime.now(timezone.utc).isoformat()
    start_perf = time.perf_counter()
    response = anthropic_client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    end_ts = datetime.now(timezone.utc).isoformat()
    duration_ms = (time.perf_counter() - start_perf) * 1000
    revised = response.content[0].text.strip()

    usage = response.usage
    tokens = {
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "total": usage.input_tokens + usage.output_tokens,
    }
    observe_agent_call(
        name="improve_article",
        provider="anthropic",
        model=model,
        prompt={"prompt_len": len(prompt), "messages_count": 1},
        response={"content_preview": revised[:500] + ("..." if len(revised) > 500 else "")},
        tokens=tokens,
        span={"start_ts": start_ts, "end_ts": end_ts, "duration_ms": round(duration_ms, 2)},
    )
    # Remove markdown code blocks if wrapped
    if revised.startswith("```"):
        lines = revised.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        revised = "\n".join(lines)

    # Always save revised article to DB on every change
    update_article(article_id, revised, changelog_action=changelog_action)
    google_doc_url = _sync_article_to_google_doc(article_id, revised)
    return {"article_id": article_id, "content": revised, "google_doc_url": google_doc_url}


# ---------------------------------------------------------------------------
# Tool: inject_links — surgical link injection via Google Docs API
# ---------------------------------------------------------------------------

def inject_links(
    article_id: str,
    links: list[dict],
) -> dict[str, Any]:
    """Inject hyperlinks into an article surgically.

    For articles with a Google Doc: Claude picks anchor phrases from the current doc text,
    then link styles are applied directly via the Docs API — no rewrite.

    For articles without a Google Doc: applies [phrase](url) replacements in DB markdown.
    """
    from src.db import get_article, update_article
    from src.google_docs import (
        fetch_doc_content,
        inject_links_into_doc,
        _document_id_from_url,
    )

    article = get_article(article_id)
    if not article:
        return {"success": False, "error": f"Article {article_id} not found"}

    google_doc_url = article.get("google_doc_url")
    document_id = _document_id_from_url(google_doc_url) if google_doc_url else None

    # 1. Get current article text — prefer Google Docs as source of truth
    current_text = article.get("content", "")
    if document_id:
        doc_result = fetch_doc_content(document_id)
        if doc_result.get("success") and doc_result.get("content"):
            current_text = doc_result["content"]

    # 2. Ask Claude to map each URL → exact anchor phrase from the article text
    import anthropic as _anthropic
    import json as _json

    links_lines = []
    for i, lnk in enumerate(links[:10], 1):
        line = f"  {i}. URL: {lnk.get('url', '')}"
        if lnk.get("anchor_hint"):
            line += f"\n     Anchor hint: {lnk['anchor_hint']}"
        if lnk.get("context_hint"):
            line += f"\n     Place near: {lnk['context_hint']}"
        links_lines.append(line)

    placement_prompt = (
        "You are a link placement assistant. Given an article and a list of URLs, "
        "return a JSON array of placement decisions.\n\n"
        "For each URL, find the best SHORT phrase (3-8 words) in the article that "
        "should become the clickable anchor text.\n\n"
        "RULES:\n"
        "- The phrase MUST be an exact verbatim substring of the article text below.\n"
        "- Choose descriptive, natural phrases — not generic words like 'here' or 'this'.\n"
        "- If anchor_hint is given, find the closest exact match in the text.\n"
        "- Prefer the first occurrence of a phrase.\n"
        "- Do NOT invent text that is not in the article.\n\n"
        "Return ONLY valid JSON — no markdown, no explanation:\n"
        '[{"url": "https://...", "phrase": "exact phrase from article"}, ...]\n\n'
        f"Links:\n{chr(10).join(links_lines)}\n\n"
        f"Article text:\n---\n{current_text[:6000]}\n---"
    )

    _client = _anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = _client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": placement_prompt}],
    )
    raw = resp.content[0].text.strip()
    # Strip markdown code block if present
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
        raw = raw.strip()

    try:
        placements = _json.loads(raw)
    except _json.JSONDecodeError:
        return {"success": False, "error": "Failed to parse placement decisions from Claude", "raw": raw}

    # 3a. Google Doc path — inject directly via Docs API, no rewrite
    if document_id:
        result = inject_links_into_doc(document_id, placements)
        placed = result.get("placed", [])
        not_placed = result.get("not_placed", [])
        # Update DB changelog only; GDoc is source of truth for content
        update_article(
            article_id,
            article.get("content", ""),
            changelog_action=f"Injected {len(placed)} link(s) into Google Doc",
        )
        return {
            "success": True,
            "placed": placed,
            "not_placed": not_placed,
            "google_doc_url": google_doc_url,
        }

    # 3b. No Google Doc — apply [phrase](url) replacements in DB markdown
    content = article.get("content", "")
    placed = []
    not_placed = []
    for p in placements:
        phrase = (p.get("phrase") or "").strip()
        url = (p.get("url") or "").strip()
        if phrase and url and phrase in content:
            content = content.replace(phrase, f"[{phrase}]({url})", 1)
            placed.append({"phrase": phrase, "url": url})
        else:
            not_placed.append({"phrase": phrase, "url": url, "reason": "phrase not found in content"})

    update_article(article_id, content, changelog_action=f"Injected {len(placed)} link(s)")
    new_doc_url = _sync_article_to_google_doc(article_id, content)
    return {"success": True, "placed": placed, "not_placed": not_placed, "google_doc_url": new_doc_url}


# ---------------------------------------------------------------------------
# Pipeline tool: generate_and_place_images
# ---------------------------------------------------------------------------

def generate_and_place_images(
    article_id: str,
    max_images: int = 4,
    save_to_db: bool = True,
    approval_mode: bool = False,
    changelog_entry: Optional[str] = None,
) -> dict[str, Any]:
    """Generate AI images and inject them into an article.

    If approval_mode=True: generates images but does NOT inject; returns image URLs for user approval.
    """
    from src.db import get_article, update_article, add_article_image

    article = get_article(article_id)
    if not article:
        return {"success": False, "error": f"Article {article_id} not found"}

    from src.images import generate_and_inject_images

    updated_content, image_records = generate_and_inject_images(
        article["content"],
        max_images=max_images,
        use_supabase=save_to_db,
        inject=not approval_mode,
    )

    if not approval_mode:
        update_article(article_id, updated_content, changelog_action=changelog_entry)
        if save_to_db:
            for rec in image_records:
                add_article_image(
                    article_id=article_id,
                    url=rec["url"],
                    position=rec["position"],
                    alt_text=rec.get("alt_text"),
                    prompt_used=rec.get("prompt_used"),
                )
        google_doc_url = _sync_article_to_google_doc(article_id, updated_content)
        return {
            "article_id": article_id,
            "content": updated_content,
            "images_added": len(image_records),
            "google_doc_url": google_doc_url,
        }
    # Approval mode: return image URLs for user to approve
    return {
        "article_id": article_id,
        "content": article["content"],
        "images_added": 0,
        "images_for_approval": [
            {"position": r["position"], "url": r["url"], "alt_text": r.get("alt_text")}
            for r in image_records
        ],
        "message": f"Generated {len(image_records)} images for approval.",
    }


# ---------------------------------------------------------------------------
# Pipeline tool: generate_hero_image (with approval)
# ---------------------------------------------------------------------------

def generate_hero_image_tool(
    article_id: str,
    description: str,
    feedback: Optional[str] = None,
    changelog_entry: Optional[str] = None,
) -> dict[str, Any]:
    """Generate a hero image for an article using style references.

    Generates the image and saves it with status='pending_approval'.
    Does NOT embed the image yet -- returns a preview URL for the user to approve.

    When feedback is provided (regeneration/refinement), the most recent hero
    image for this article is used as a primary reference so the model can see
    what it is modifying, instead of generating from scratch with random refs.
    """
    from src.db import get_article, add_article_image, get_pending_article_images, get_article_images, update_article_image_status
    from src.images import generate_hero_image, upload_to_supabase

    article = get_article(article_id)
    if not article:
        return {"success": False, "error": f"Article {article_id} not found"}

    logger.info("[TOOL] generate_hero_image: article_id=%s, description=%r, feedback=%r",
                article_id[:8], description[:80], (feedback or "")[:80])

    # Reject any previously pending hero images so only one pending exists at a time.
    # This prevents the agent from accidentally approving a stale image_id from an
    # earlier iteration (the most common cause of the wrong image being injected).
    previous_image_url = None
    existing_pending = get_pending_article_images(article_id, image_type="hero")
    if existing_pending:
        previous_image_url = existing_pending[0].get("url")
        for old_img in existing_pending:
            try:
                update_article_image_status(old_img["id"], "rejected")
                logger.info("[TOOL] generate_hero_image: superseded pending image %s", old_img["id"][:8])
            except Exception as e:
                logger.warning("[TOOL] generate_hero_image: could not supersede image %s: %s", old_img["id"][:8], e)

    if not previous_image_url and feedback:
        # Fall back to any hero image (including approved ones) for visual reference
        all_images = get_article_images(article_id)
        hero_images = [i for i in all_images if i.get("image_type") == "hero"]
        if hero_images:
            previous_image_url = hero_images[-1].get("url")

    if previous_image_url and feedback:
        logger.info("[TOOL] generate_hero_image: using previous image as reference for refinement")

    try:
        img_bytes, prompt_used = generate_hero_image(
            description, feedback=feedback, previous_image_url=previous_image_url if feedback else None,
        )
    except Exception as e:
        return {"success": False, "error": f"Hero image generation failed: {e}"}

    url = upload_to_supabase(img_bytes, "hero.png")

    # Save as pending approval
    image_rec = add_article_image(
        article_id=article_id,
        url=url,
        position=-1,  # -1 denotes hero (above title)
        alt_text=f"Hero image: {description[:100]}",
        prompt_used=prompt_used,
        status="pending_approval",
        image_type="hero",
        metadata={"description": description, "feedback": feedback},
    )

    return {
        "article_id": article_id,
        "image_id": image_rec["id"],
        "image_url": url,
        "prompt_used": prompt_used,
        "status": "pending_approval",
        "message": "Hero image generated. Send the preview AND the prompt_used to the user. Ask for approval.",
    }


# ---------------------------------------------------------------------------
# Pipeline tool: approve_hero_image
# ---------------------------------------------------------------------------

def approve_hero_image_tool(
    article_id: str,
    image_id: str,
    changelog_entry: Optional[str] = None,
) -> dict[str, Any]:
    """Approve a pending hero image: embed it into the article above the title and sync to Google Doc.

    Content-safe: only prepends the hero image markdown above the existing
    content. Validates that existing content is preserved after injection.
    """
    from src.db import get_article, get_article_image, get_pending_article_images, update_article_image_status, update_article
    from src.images import inject_hero_into_markdown

    article = get_article(article_id)
    if not article:
        return {"success": False, "error": f"Article {article_id} not found"}

    original_content = (article.get("content") or "").strip()
    if not original_content:
        return {"success": False, "error": "Article has no content. Cannot embed hero image in an empty article."}

    image = get_article_image(image_id)
    if not image:
        return {"success": False, "error": f"Image {image_id} not found"}

    # Fallback: if the requested image is no longer pending (e.g. it was superseded by a
    # later iteration), automatically use the most recent pending hero image instead.
    if image.get("status") != "pending_approval":
        latest_pending = get_pending_article_images(article_id, image_type="hero")
        if latest_pending:
            logger.warning(
                "[TOOL] approve_hero_image: image %s is not pending (status=%s); "
                "falling back to latest pending hero image %s",
                image_id[:8], image.get("status"), latest_pending[0]["id"][:8],
            )
            image_id = latest_pending[0]["id"]
            image = latest_pending[0]
        else:
            return {"success": False, "error": f"Image is not pending approval (status: {image.get('status')}) and no other pending hero image found"}

    if image.get("article_id") != article_id:
        return {"success": False, "error": "Image does not belong to this article"}

    # Mark approved
    update_article_image_status(image_id, "approved")

    # Inject hero into markdown (prepend only — does NOT modify existing content)
    updated_content = inject_hero_into_markdown(
        article["content"],
        image["url"],
        alt_text=image.get("alt_text", "Hero image"),
    )

    # Safeguard: verify the original content is fully preserved after injection
    if original_content not in updated_content:
        logger.error(
            "[TOOL] approve_hero_image: content validation FAILED — original content not found in updated content! "
            "original_len=%d, updated_len=%d", len(original_content), len(updated_content),
        )
        return {"success": False, "error": "Internal error: content validation failed after hero injection. Article was NOT modified."}

    # Save updated article + hero_image_url on the article row
    update_article(article_id, updated_content, changelog_action=changelog_entry or "Added hero image")
    from src.db import get_client
    get_client().table("articles").update({"hero_image_url": image["url"]}).eq("id", article_id).execute()

    # Sync to Google Doc
    google_doc_url = _sync_article_to_google_doc(article_id, updated_content)

    logger.info("[TOOL] approve_hero_image: done for article %s (content: %d -> %d chars)",
                article_id[:8], len(original_content), len(updated_content))
    return {
        "article_id": article_id,
        "image_id": image_id,
        "image_url": image["url"],
        "content": updated_content,
        "google_doc_url": google_doc_url,
        "message": "Hero image approved and embedded above the title.",
    }


# ---------------------------------------------------------------------------
# Pipeline tool: generate_infographic (with approval)
# ---------------------------------------------------------------------------

def generate_infographic_tool(
    article_id: str,
    feedback: Optional[str] = None,
    infographic_type: Optional[str] = None,
    changelog_entry: Optional[str] = None,
) -> dict[str, Any]:
    """Generate an infographic for an article using style references.

    Analyzes the article to find the best position and type, generates the image,
    and saves with status='pending_approval'. Does NOT inject -- returns preview.

    When feedback is provided (regeneration/refinement), the most recent infographic
    for this article is used as a primary reference so the model can see what it is
    modifying.
    """
    from src.db import get_article, add_article_image, get_pending_article_images, get_article_images, update_article_image_status
    from src.images import generate_infographic, upload_to_supabase

    article = get_article(article_id)
    if not article:
        return {"success": False, "error": f"Article {article_id} not found"}

    logger.info("[TOOL] generate_infographic: article_id=%s, type=%s, feedback=%r",
                article_id[:8], infographic_type, (feedback or "")[:80])

    # Reject any previously pending infographic images so only one pending exists at a time.
    # This prevents the agent from accidentally approving a stale image_id from an
    # earlier iteration (the most common cause of the wrong image being injected).
    previous_image_url = None
    existing_pending = get_pending_article_images(article_id, image_type="infographic")
    if existing_pending:
        previous_image_url = existing_pending[0].get("url")
        for old_img in existing_pending:
            try:
                update_article_image_status(old_img["id"], "rejected")
                logger.info("[TOOL] generate_infographic: superseded pending image %s", old_img["id"][:8])
            except Exception as e:
                logger.warning("[TOOL] generate_infographic: could not supersede image %s: %s", old_img["id"][:8], e)

    if not previous_image_url and feedback:
        # Fall back to any infographic image (including approved ones) for visual reference
        all_images = get_article_images(article_id)
        infographic_images = [i for i in all_images if i.get("image_type") == "infographic"]
        if infographic_images:
            previous_image_url = infographic_images[-1].get("url")

    if previous_image_url and feedback:
        logger.info("[TOOL] generate_infographic: using previous image as reference for refinement")

    try:
        img_bytes, prompt_used, analysis = generate_infographic(
            article["content"],
            feedback=feedback,
            infographic_type=infographic_type,
            previous_image_url=previous_image_url if feedback else None,
        )
    except Exception as e:
        return {"success": False, "error": f"Infographic generation failed: {e}"}

    url = upload_to_supabase(img_bytes, "infographic.png")

    # Save as pending approval with analysis metadata
    image_rec = add_article_image(
        article_id=article_id,
        url=url,
        position=0,
        alt_text=f"Infographic: {analysis.get('title', 'Infographic')}",
        prompt_used=prompt_used,
        status="pending_approval",
        image_type="infographic",
        metadata={
            "position_after": analysis.get("position_after"),
            "infographic_type": analysis.get("infographic_type"),
            "title": analysis.get("title"),
            "description": analysis.get("description"),
            "section_name": analysis.get("section_name"),
            "feedback": feedback,
        },
    )

    return {
        "article_id": article_id,
        "image_id": image_rec["id"],
        "image_url": url,
        "prompt_used": prompt_used,
        "infographic_type": analysis.get("infographic_type"),
        "infographic_title": analysis.get("title"),
        "suggested_section": analysis.get("section_name"),
        "status": "pending_approval",
        "message": (
            f"Infographic generated: \"{analysis.get('title')}\" "
            f"({analysis.get('infographic_type')}) — suggested placement after "
            f"\"{analysis.get('section_name', 'N/A')}\" section. "
            "Send the preview AND the prompt_used to the user. Ask for approval."
        ),
    }


# ---------------------------------------------------------------------------
# Pipeline tool: approve_infographic
# ---------------------------------------------------------------------------

def approve_infographic_tool(
    article_id: str,
    image_id: str,
    changelog_entry: Optional[str] = None,
) -> dict[str, Any]:
    """Approve a pending infographic: inject into article at the analyzed position and sync to Google Doc.

    Content-safe: only inserts the infographic markdown at the determined position.
    Validates that existing content is preserved after injection.
    """
    from src.db import get_article, get_article_image, get_pending_article_images, update_article_image_status, update_article
    from src.images import inject_infographic_into_markdown

    article = get_article(article_id)
    if not article:
        return {"success": False, "error": f"Article {article_id} not found"}

    original_content = (article.get("content") or "").strip()
    if not original_content:
        return {"success": False, "error": "Article has no content. Cannot embed infographic in an empty article."}

    image = get_article_image(image_id)
    if not image:
        return {"success": False, "error": f"Image {image_id} not found"}

    # Fallback: if the requested image is no longer pending (e.g. it was superseded by a
    # later iteration), automatically use the most recent pending infographic instead.
    if image.get("status") != "pending_approval":
        latest_pending = get_pending_article_images(article_id, image_type="infographic")
        if latest_pending:
            logger.warning(
                "[TOOL] approve_infographic: image %s is not pending (status=%s); "
                "falling back to latest pending infographic %s",
                image_id[:8], image.get("status"), latest_pending[0]["id"][:8],
            )
            image_id = latest_pending[0]["id"]
            image = latest_pending[0]
        else:
            return {"success": False, "error": f"Image is not pending approval (status: {image.get('status')}) and no other pending infographic found"}

    if image.get("article_id") != article_id:
        return {"success": False, "error": "Image does not belong to this article"}

    meta = image.get("metadata") or {}
    position_after = meta.get("position_after", "")

    # Mark approved
    update_article_image_status(image_id, "approved")

    # Inject infographic into markdown at the determined position (insert only — does NOT modify existing content)
    updated_content = inject_infographic_into_markdown(
        article["content"],
        position_after=position_after,
        image_url=image["url"],
        alt_text=image.get("alt_text", "Infographic"),
    )

    # Safeguard: verify the updated content is at least as long as the original
    # (injection only adds — it should never shrink the content)
    if len(updated_content) < len(original_content):
        logger.error(
            "[TOOL] approve_infographic: content validation FAILED — content shrunk! "
            "original_len=%d, updated_len=%d", len(original_content), len(updated_content),
        )
        return {"success": False, "error": "Internal error: content validation failed after infographic injection. Article was NOT modified."}

    # Save updated article + infographic_url on the article row
    changelog = changelog_entry or f"Added infographic: {meta.get('title', 'Infographic')}"
    update_article(article_id, updated_content, changelog_action=changelog)
    from src.db import get_client
    get_client().table("articles").update({"infographic_url": image["url"]}).eq("id", article_id).execute()

    # Sync to Google Doc
    google_doc_url = _sync_article_to_google_doc(article_id, updated_content)

    logger.info("[TOOL] approve_infographic: done for article %s (content: %d -> %d chars)",
                article_id[:8], len(original_content), len(updated_content))
    return {
        "article_id": article_id,
        "image_id": image_id,
        "image_url": image["url"],
        "infographic_title": meta.get("title"),
        "content": updated_content,
        "google_doc_url": google_doc_url,
        "message": f"Infographic \"{meta.get('title', '')}\" approved and embedded in the article.",
    }


# ---------------------------------------------------------------------------
# Pipeline tool: generate_seo_metadata
# ---------------------------------------------------------------------------

def generate_seo_metadata(article_id: str) -> dict[str, Any]:
    """Call PromptLayer metadata agent on the article content and save to DB.

    Sends the current article content to the PromptLayer metadata workflow
    (PROMPTLAYER_METADATA_WORKFLOW, default "metadata agent"), stores the result
    in the articles.seo_metadata column, and returns it.
    """
    from src.db import get_article, set_article_seo_metadata
    from src.pipeline import run_metadata_agent, emit_status

    logger.info("[TOOL] generate_seo_metadata: article_id=%s", article_id)

    article = get_article(article_id)
    if not article:
        return {"success": False, "error": f"Article {article_id} not found"}

    content = article.get("content", "")
    if not content:
        return {"success": False, "error": "Article has no content"}

    try:
        metadata, pl_exec_id = run_metadata_agent(content)
    except Exception as e:
        logger.exception("[TOOL] generate_seo_metadata EXCEPTION: %s", e)
        return {"success": False, "error": str(e)}

    # Normalise metadata: parse strings, resolve key aliases, normalise tags.
    # This ensures the DB always has clean canonical keys that create_ghost_draft
    # can use directly, regardless of what naming convention the metadata agent uses.
    from src.ghost import _normalize_seo_metadata
    metadata_to_store = _normalize_seo_metadata(metadata)

    # If normalisation returned empty (e.g. completely unparseable), keep raw for debugging
    if not metadata_to_store:
        if isinstance(metadata, str):
            metadata_to_store = {"raw": metadata}
        elif isinstance(metadata, dict):
            metadata_to_store = dict(metadata)
        else:
            metadata_to_store = {}

    logger.info("[TOOL] generate_seo_metadata: normalised keys=%s", list(metadata_to_store.keys()))

    # Record exec ID in SQLite (separate from seo_metadata — clean separation)
    if pl_exec_id:
        try:
            from src.message_cache import add_pl_execution
            add_pl_execution(
                exec_id=pl_exec_id,
                workflow=os.getenv("PROMPTLAYER_METADATA_WORKFLOW", "metadata agent"),
                article_id=article_id,
                channel=_current_channel.get(None),
                channel_user_id=_current_chat_id.get(None),
            )
        except Exception as e:
            logger.warning("[TOOL] generate_seo_metadata: failed to cache exec_id: %s", e)

    emit_status("Saving SEO metadata...")
    try:
        set_article_seo_metadata(article_id, metadata_to_store)
    except Exception as e:
        logger.warning("[TOOL] generate_seo_metadata: failed to save to DB: %s", e)
        return {"success": False, "error": f"Metadata generated but DB save failed: {e}", "metadata": metadata_to_store}

    logger.info("[TOOL] generate_seo_metadata DONE: article_id=%s, pl_exec_id=%s", article_id, pl_exec_id)
    return {
        "article_id": article_id,
        "metadata": metadata_to_store,
        "pl_exec_id": pl_exec_id,
    }


# ---------------------------------------------------------------------------
# Tool: push_to_ghost
# ---------------------------------------------------------------------------


def push_to_ghost(article_id: str) -> dict[str, Any]:
    """Push the approved article to Ghost CMS as a draft.

    Steps:
    1. Fetch article from DB (content + hero_image_url + seo_metadata).
    2. If seo_metadata is absent, run the metadata agent first and save it.
    3. Call src.ghost.create_ghost_draft() which strips the hero image from the
       inline markdown, converts MD to HTML, and POSTs to Ghost Admin API.
    4. Return the Ghost editor URL and post ID.
    """
    from src.db import get_article
    from src.ghost import create_ghost_draft
    from src.pipeline import emit_status

    logger.info("[TOOL] push_to_ghost: article_id=%s", article_id)

    article = get_article(article_id)
    if not article:
        return {"success": False, "error": f"Article {article_id} not found"}

    content = article.get("content", "")
    if not content:
        return {"success": False, "error": "Article has no content"}

    # Ensure SEO metadata exists — generate if absent
    metadata = article.get("seo_metadata") or {}
    if not metadata:
        emit_status("Generating SEO metadata before pushing to Ghost...")
        meta_result = generate_seo_metadata(article_id)
        if not meta_result.get("metadata"):
            return {"success": False, "error": "Failed to generate SEO metadata for Ghost draft"}
        metadata = meta_result["metadata"]

    hero_url = article.get("hero_image_url")
    title = article.get("title", "Untitled")

    try:
        emit_status("Creating Ghost draft...")
        result = create_ghost_draft(
            title=title,
            content_md=content,
            hero_image_url=hero_url,
            metadata=metadata,
        )
    except Exception as e:
        logger.exception("[TOOL] push_to_ghost EXCEPTION: %s", e)
        return {"success": False, "error": str(e)}

    logger.info("[TOOL] push_to_ghost DONE: ghost_id=%s", result.get("id"))
    return {
        "article_id": article_id,
        "ghost_id": result["id"],
        "ghost_url": result["url"],
        "ghost_editor_url": result.get("editor_url", result["url"]),
        "message": f"Ghost draft created successfully. Open in editor: {result.get('editor_url', result['url'])}",
    }


# ---------------------------------------------------------------------------
# Tool: fetch_promptlayer_execution
# ---------------------------------------------------------------------------

def fetch_promptlayer_execution(
    execution_id: Optional[str] = None,
    article_id: Optional[str] = None,
    last_n: Optional[int] = None,
) -> dict[str, Any]:
    """Fetch PromptLayer workflow execution result(s).

    Modes:
      - execution_id: fetch a specific execution by its workflow_version_execution_id
      - article_id: look up the stored _pl_exec_id from the article's seo_metadata and fetch it
      - last_n: query the DB for the N most recent articles with a stored _pl_exec_id,
                return their stored metadata (no re-fetch needed; PL result is already saved)

    Note: PromptLayer's API only supports fetching by ID. last_n is served from our own DB.
    """
    import httpx as _httpx

    api_key = os.getenv("PROMPTLAYER_API_KEY")

    def _fetch_exec(exec_id: str) -> dict:
        if not api_key:
            return {"error": "PROMPTLAYER_API_KEY not set"}
        with _httpx.Client() as client:
            res = client.get(
                "https://api.promptlayer.com/workflow-version-execution-results",
                headers={"X-API-KEY": api_key},
                params={"workflow_version_execution_id": exec_id, "return_all_outputs": True},
                timeout=30.0,
            )
        if res.status_code == 202:
            return {"exec_id": exec_id, "status": "still_running"}
        if res.status_code != 200:
            return {"exec_id": exec_id, "error": f"HTTP {res.status_code}: {res.text[:200]}"}
        return {"exec_id": exec_id, "status": "done", "results": res.json()}

    # --- Mode 1: direct execution ID ---
    if execution_id:
        logger.info("[TOOL] fetch_promptlayer_execution: by exec_id=%s", execution_id)
        return _fetch_exec(execution_id)

    # --- Mode 2: lookup via article (reads from SQLite) ---
    if article_id:
        from src.message_cache import get_recent_pl_executions
        rows = get_recent_pl_executions(limit=20)  # search across all users
        match = next((r for r in rows if r.get("article_id") == article_id), None)
        if not match:
            return {"success": False, "error": "No PromptLayer execution ID cached for this article. Run generate_seo_metadata first."}
        exec_id = match["exec_id"]
        logger.info("[TOOL] fetch_promptlayer_execution: article_id=%s -> exec_id=%s", article_id[:8], exec_id)
        return _fetch_exec(exec_id)

    # --- Mode 3: last N from SQLite cache ---
    if last_n:
        from src.message_cache import get_recent_pl_executions
        channel = _current_channel.get(None)
        channel_user_id = _current_chat_id.get(None)
        rows = get_recent_pl_executions(channel=channel, channel_user_id=channel_user_id, limit=min(last_n, 20))
        logger.info("[TOOL] fetch_promptlayer_execution: last_n=%d -> found %d records", last_n, len(rows))
        return {"count": len(rows), "executions": rows}

    return {"success": False, "error": "Provide execution_id, article_id, or last_n."}


def fetch_history(before_timestamp: str, limit: int = 20, **_) -> dict:
    """Fetch older conversation messages from SQLite cache (or Supabase fallback)."""
    from src.tools import _current_channel, _current_chat_id
    channel = _current_channel.get(None)
    channel_user_id = _current_chat_id.get(None)
    if not channel or not channel_user_id:
        return {"error": "No channel context available.", "messages": []}
    limit = min(int(limit), 50)
    try:
        from src.message_cache import get_before, count
        cached = count(channel, channel_user_id)
        if cached > 0:
            msgs = get_before(channel, channel_user_id, before_timestamp=before_timestamp, limit=limit)
        else:
            # Fallback: query Supabase directly
            from src.db import get_client
            client = get_client()
            r = (
                client.table("messages")
                .select("role, content, created_at")
                .eq("channel", channel)
                .eq("channel_user_id", channel_user_id)
                .lt("created_at", before_timestamp)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            msgs = list(reversed(r.data or []))
        logger.info("[TOOL] fetch_history: returned %d messages before %s", len(msgs), before_timestamp)
        return {
            "messages": [{"role": m["role"], "content": m["content"], "created_at": m.get("created_at")} for m in msgs],
            "count": len(msgs),
        }
    except Exception as e:
        logger.error("[TOOL] fetch_history error: %s", e)
        return {"error": str(e), "messages": []}


def manage_optimization(
    action: str,
    remove_ids: Optional[list] = None,
    **_,
) -> dict[str, Any]:
    """View, modify, or deploy the pending self-optimization session.

    Actions:
    - list: Show action items in the pending session
    - remove_items: Remove specific item IDs from the pending list before deploying
    - deploy: Publish all pending prompt_change items to PromptLayer, mark session deployed
    - reject: Discard the pending session without deploying
    """
    from src.db import get_pending_optimization_session, update_optimization_session

    session = get_pending_optimization_session()
    if not session:
        return {"success": False, "error": "No pending optimization session found."}

    session_id = session["id"]
    action_items = session.get("action_items") or []
    if isinstance(action_items, str):
        import json as _json
        action_items = _json.loads(action_items)
    pending_ids = session.get("pending_item_ids") or [i["id"] for i in action_items if i.get("auto_deployable")]

    if action == "list":
        deployable = [i for i in action_items if i["id"] in pending_ids]
        skipped    = [i for i in action_items if i["id"] not in pending_ids]
        return {
            "success": True,
            "session_id": session_id[:8],
            "created_at": session.get("created_at", "")[:16],
            "to_deploy": [{"id": i["id"], "goal": i["goal"], "title": i["title"], "impact": i.get("impact", "")} for i in deployable],
            "skipped":   [{"id": i["id"], "goal": i["goal"], "title": i["title"]} for i in skipped],
            "note": "Reply with remove_items to remove specific IDs, or deploy to publish.",
        }

    elif action == "remove_items":
        ids_to_remove = set(remove_ids or [])
        updated_ids = [i for i in pending_ids if i not in ids_to_remove]
        update_optimization_session(session_id, {"pending_item_ids": updated_ids})
        remaining = [i for i in action_items if i["id"] in updated_ids]
        return {
            "success": True,
            "removed": sorted(ids_to_remove),
            "remaining_to_deploy": [{"id": i["id"], "title": i["title"]} for i in remaining],
        }

    elif action == "deploy":
        import httpx as _httpx
        import re as _re
        import os as _os

        PL_API_KEY = _os.environ.get("PROMPTLAYER_API_KEY", "")
        PL_PREFIX  = _os.getenv("PROMPTLAYER_PROMPT_PREFIX", "blog_writer")
        PL_BASE    = "https://api.promptlayer.com/rest"
        analysis   = session.get("analysis_text", "")

        pending_items = [i for i in action_items if i["id"] in pending_ids]
        prompt_items  = [i for i in pending_items if i.get("type") == "prompt_change"]
        config_items  = [i for i in pending_items if i.get("type") == "config_change"]
        deployed_prompts, deployed_configs, failed = [], [], []

        # Deploy prompt changes to PromptLayer
        for item in prompt_items:
            prompt_key = item.get("prompt_key")
            if not prompt_key:
                continue
            section_label = f"{prompt_key} (optimized)"
            pat = rf"#### {_re.escape(section_label)}\n(.*?)(?=\n####|\n###|\Z)"
            m = _re.search(pat, analysis, _re.DOTALL)
            if not m:
                failed.append(f"{prompt_key} (section not found)")
                continue
            content = m.group(1).strip()
            if len(content) < 200:
                failed.append(f"{prompt_key} (content too short)")
                continue
            if prompt_key == "master_system_core" and "{{PLAYBOOKS}}" not in content:
                failed.append(f"{prompt_key} ({{{{PLAYBOOKS}}}} placeholder missing)")
                continue
            input_vars = list(dict.fromkeys(_re.findall(r"\{(\w+)\}", content)))
            prompt_template: dict = {
                "messages": [{"role": "system", "content": content}],
                "input_variables": input_vars,
            }
            # Preserve llm_kwargs for master_system_core; apply model_override if specified
            if prompt_key == "master_system_core":
                from src.prompts_loader import get_prompt_llm_kwargs
                existing_kwargs = dict(get_prompt_llm_kwargs("master_system_core"))
                if item.get("model_override"):
                    existing_kwargs["model"] = item["model_override"]
                if existing_kwargs:
                    prompt_template["llm_kwargs"] = existing_kwargs
            body = {
                "prompt_name": f"{PL_PREFIX}/{prompt_key}" if PL_PREFIX else prompt_key,
                "prompt_template": prompt_template,
                "tags": ["self-optimized"],
                "api_key": PL_API_KEY,
            }
            try:
                r = _httpx.post(f"{PL_BASE}/publish-prompt-template", json=body, timeout=15.0)
                r.raise_for_status()
                data = r.json()
                if data.get("id") or data.get("success") is not False:
                    deployed_prompts.append(prompt_key)
                else:
                    failed.append(f"{prompt_key} ({data})")
            except Exception as exc:
                failed.append(f"{prompt_key} ({exc})")

        # Deploy config changes to agent_config table
        if config_items:
            from src.db import upsert_agent_config
            from src.config import invalidate_config_cache
            for item in config_items:
                cfg_key = item.get("config_key")
                cfg_val = item.get("config_value")
                if not cfg_key or cfg_val is None:
                    failed.append(f"config:{cfg_key or '?'} (missing key or value)")
                    continue
                try:
                    upsert_agent_config(cfg_key, str(cfg_val))
                    deployed_configs.append(f"{cfg_key}={cfg_val!r}")
                except Exception as exc:
                    failed.append(f"config:{cfg_key} ({exc})")
            if deployed_configs:
                try:
                    invalidate_config_cache()
                except Exception:
                    pass  # non-fatal

        update_optimization_session(session_id, {"status": "deployed"})
        logger.info(
            "[TOOL] manage_optimization deploy: prompts=%s configs=%s failed=%s",
            deployed_prompts, deployed_configs, failed,
        )
        notes = []
        if deployed_prompts:
            notes.append(f"PromptLayer updated: {', '.join(deployed_prompts)}. Roll back at dashboard.promptlayer.com.")
        if deployed_configs:
            notes.append(f"Agent config updated: {', '.join(deployed_configs)}. Takes effect on next request.")
        if not notes:
            notes.append("Nothing deployed (all items failed validation).")
        return {
            "success": True,
            "deployed_prompts": deployed_prompts,
            "deployed_configs": deployed_configs,
            "failed": failed,
            "note": " ".join(notes),
        }

    elif action == "reject":
        update_optimization_session(session_id, {"status": "rejected"})
        return {"success": True, "note": "Optimization session discarded."}

    return {"success": False, "error": f"Unknown action: {action!r}. Use list, remove_items, deploy, or reject."}

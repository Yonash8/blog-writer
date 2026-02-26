from __future__ import annotations
"""Master agent: autonomous agent with tool calling. Uses Claude Opus by default."""

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from anthropic import Anthropic

_PL_PREFIX = os.getenv("PROMPTLAYER_PROMPT_PREFIX", "blog_writer")

_PL_ENABLED = False
try:
    from promptlayer import PromptLayer as _PromptLayer
    _pl = _PromptLayer(api_key=os.getenv("PROMPTLAYER_API_KEY", ""))
    _AnthropicClient = _pl.anthropic.Anthropic
    _PL_ENABLED = True
except Exception:
    _AnthropicClient = Anthropic


def _pl_track_prompt(request_id: int | str, prompt_name: str) -> None:
    """Link a PromptLayer request_id to a prompt template. Best-effort, non-blocking."""
    import httpx as _httpx
    try:
        _httpx.post(
            "https://api.promptlayer.com/rest/track-prompt",
            json={
                "request_id": request_id,
                "prompt_name": prompt_name,
                "api_key": os.getenv("PROMPTLAYER_API_KEY", ""),
            },
            timeout=5.0,
        )
    except Exception:
        pass

from src.config import get_config, get_config_int
from src.prompts_loader import get_prompt, get_master_system_prompt, get_prompt_llm_kwargs
from src.observability import (
    get_trace_id,
    init_trace_payload,
    log_event,
    observe_agent_call,
    observe_tool,
    set_trace_id,
)
from src.pipeline import status_callback as pipeline_status_callback, cancel_check as pipeline_cancel_check, TaskCancelledError

logger = logging.getLogger(__name__)

# System prompt loaded from DB via get_master_system_prompt()


# ---------------------------------------------------------------------------
# Tool definitions (6 tools: 3 infrastructure + 3 pipeline)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "db",
            "description": (
                "Query or modify the Postgres database. "
                "Actions: 'select' for simple reads (with filters, ordering, limit), "
                "'insert'/'update'/'delete' for writes, "
                "'sql' for complex read-only queries (COUNT, GROUP BY, JOINs, subqueries). "
                "You write the queries yourself. Filters use equality matching."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["select", "insert", "update", "delete", "sql"],
                        "description": "The database action to perform",
                    },
                    "table": {
                        "type": "string",
                        "description": "Table name (required for select/insert/update/delete). One of: articles, topics, messages, article_images",
                    },
                    "columns": {
                        "type": "string",
                        "description": "Columns to select (default: *). Comma-separated, e.g. 'id, title, status'",
                        "default": "*",
                    },
                    "filters": {
                        "type": "object",
                        "description": "Equality filters as {column: value}, e.g. {\"channel\": \"whatsapp\", \"status\": \"draft\"}",
                        "additionalProperties": True,
                    },
                    "data": {
                        "type": "object",
                        "description": "Row data for insert or fields to update. e.g. {\"title\": \"My Topic\", \"status\": \"posted\"}",
                        "additionalProperties": True,
                    },
                    "order_by": {
                        "type": "string",
                        "description": "Column to sort by. Prefix with '-' for descending, e.g. '-created_at'",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max rows to return",
                    },
                    "sql": {
                        "type": "string",
                        "description": "Read-only SQL query for action='sql'. Only SELECT/WITH allowed. e.g. \"SELECT status, COUNT(*) as count FROM articles GROUP BY status\"",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "google_docs",
            "description": (
                "Create, update, or read Google Docs. "
                "action='create': create a new doc from article_id (loads content from DB) or raw markdown. Saves URL back to article. "
                "action='update': sync an existing doc with updated content (provide document_id or article_id to auto-resolve). "
                "action='fetch': read content from a Google Doc. Provide document_url (full URL) or document_id. Returns content and title. Doc must be shared with service account."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "update", "fetch"],
                        "description": "create, update, or fetch (read)",
                    },
                    "article_id": {
                        "type": "string",
                        "description": "Article UUID. For create: loads content from DB. For update: resolves google_doc_url to get document_id.",
                    },
                    "markdown": {
                        "type": "string",
                        "description": "Raw markdown content (alternative to article_id for create, or new content for update)",
                    },
                    "title": {
                        "type": "string",
                        "description": "Document title (for create). Defaults to article title.",
                    },
                    "document_id": {
                        "type": "string",
                        "description": "Google Doc document ID (for update/fetch, if not using document_url)",
                    },
                    "document_url": {
                        "type": "string",
                        "description": "Full Google Docs URL (e.g. https://docs.google.com/document/d/ID/edit). For fetch action.",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for facts, examples, citations, or up-to-date information via Tavily.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Max results (default 10)", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_image",
            "description": (
                "Send an image to the user as a visible media message (not a URL link). "
                "On WhatsApp, the image appears inline in the chat. "
                "Use this to show image previews, generated heroes, infographics, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_url": {"type": "string", "description": "Public URL of the image to send"},
                    "caption": {"type": "string", "description": "Optional caption to display with the image"},
                },
                "required": ["image_url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_article",
            "description": (
                "Full article pipeline: PromptLayer SEO writer (research is handled inside PromptLayer). "
                "Optionally enriches with Tavily web sources — ASK the user first before setting include_tavily=True. "
                "Auto-saves to DB and creates Google Doc. Returns article_id, content, google_doc_url, metadata."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "The topic to write about"},
                    "include_tavily": {"type": "boolean", "description": "Include Tavily web source enrichment (default False). Ask the user first.", "default": False},
                    "tavily_max_results": {"type": "integer", "description": "Max Tavily sources (5-20)", "default": 20},
                    "changelog_entry": {"type": "string", "description": "Brief changelog entry, e.g. 'Created draft on topic X'"},
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "improve_article",
            "description": (
                "Revise an existing article based on feedback. Supports text edits, rephrasing, "
                "adding/removing content, injecting Markdown links, and full rewrites (use_promptlayer=True). "
                "Auto-saves and syncs to Google Doc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "article_id": {"type": "string", "description": "Article UUID (resolve from CURRENT CONTEXT)"},
                    "feedback": {"type": "string", "description": "What to change, add, remove, or rephrase"},
                    "links": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"url": {"type": "string"}, "title": {"type": "string"}},
                            "required": ["url"],
                        },
                        "description": "URLs to inject as [anchor](url) links. Use with web_search results.",
                    },
                    "use_promptlayer": {"type": "boolean", "description": "Use PromptLayer for full rewrite (default False)", "default": False},
                    "changelog_entry": {"type": "string", "description": "Brief changelog entry, e.g. 'Rephrased intro'"},
                },
                "required": ["article_id", "feedback"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inject_links",
            "description": (
                "Surgically inject hyperlinks into an article without rewriting its prose. "
                "Fetches the latest content from Google Docs (source of truth — humans may have edited it). "
                "Claude picks the best anchor phrase for each URL from the existing text; "
                "links are applied directly via the Google Docs API (no doc rewrite). "
                "Use for citations, references, and internal links. "
                "Prefer this over improve_article when the only change is adding links."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "article_id": {"type": "string", "description": "Article UUID (resolve from CURRENT CONTEXT)"},
                    "links": {
                        "type": "array",
                        "description": "URLs to inject as hyperlinks. Up to 10.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string", "description": "URL to inject"},
                                "anchor_hint": {
                                    "type": "string",
                                    "description": "Suggested anchor text phrase (optional). Claude will find it in the article.",
                                },
                                "context_hint": {
                                    "type": "string",
                                    "description": "Which section or topic this link belongs to (optional).",
                                },
                            },
                            "required": ["url"],
                        },
                    },
                },
                "required": ["article_id", "links"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_images",
            "description": (
                "Generate AI images and inject into an article. Auto-saves and syncs to Google Doc. "
                "Use approval_mode=True to preview images before injecting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "article_id": {"type": "string", "description": "Article UUID (resolve from CURRENT CONTEXT)"},
                    "max_images": {"type": "integer", "description": "Max images to generate (default 4)", "default": 4},
                    "approval_mode": {"type": "boolean", "description": "If true, return image URLs for approval before injecting", "default": False},
                    "changelog_entry": {"type": "string", "description": "Brief changelog entry, e.g. 'Added 3 images'"},
                },
                "required": ["article_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_hero_image",
            "description": (
                "Generate a hero image for an article. Returns a preview URL — does NOT embed. "
                "Use 'feedback' to refine an existing image (the tool automatically uses the previous image as reference). "
                "Omit 'feedback' for a fresh generation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "article_id": {"type": "string", "description": "Article UUID (resolve from CURRENT CONTEXT)"},
                    "description": {"type": "string", "description": "What the hero should depict, e.g. 'walking into a futuristic data center'"},
                    "feedback": {"type": "string", "description": "Optional refinement instructions when regenerating, e.g. 'make it more vibrant' or 'add a laptop'"},
                    "changelog_entry": {"type": "string", "description": "Brief changelog entry"},
                },
                "required": ["article_id", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "approve_hero_image",
            "description": (
                "Approve a pending hero image and embed it in the article above the title. "
                "Call this after the user approves the hero image preview. Auto-syncs to Google Doc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "article_id": {"type": "string", "description": "Article UUID"},
                    "image_id": {"type": "string", "description": "Image UUID from generate_hero_image result"},
                    "changelog_entry": {"type": "string", "description": "Brief changelog entry"},
                },
                "required": ["article_id", "image_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_infographic",
            "description": (
                "Generate an infographic for an article (auto-detects best type and position). "
                "Returns a preview URL — does NOT inject. "
                "Use 'feedback' to refine an existing infographic (automatically uses previous image as reference). "
                "Use 'infographic_type' to pick a specific type. Omit both for a fresh auto-analyzed generation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "article_id": {"type": "string", "description": "Article UUID (resolve from CURRENT CONTEXT)"},
                    "feedback": {"type": "string", "description": "Optional refinement instructions, e.g. 'try a flowchart instead' or 'include more statistics'"},
                    "infographic_type": {
                        "type": "string",
                        "description": "Override auto-detected type. One of: comparison_table, flowchart, bar_chart, pie_chart, timeline, process_diagram, statistics_highlight, checklist",
                    },
                    "changelog_entry": {"type": "string", "description": "Brief changelog entry"},
                },
                "required": ["article_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "approve_infographic",
            "description": (
                "Approve a pending infographic and inject it into the article at the suggested position. "
                "Call this after the user approves the infographic preview. Auto-syncs to Google Doc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "article_id": {"type": "string", "description": "Article UUID"},
                    "image_id": {"type": "string", "description": "Image UUID from generate_infographic result"},
                    "changelog_entry": {"type": "string", "description": "Brief changelog entry"},
                },
                "required": ["article_id", "image_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_seo_metadata",
            "description": (
                "Call the PromptLayer metadata agent on the current article content and store the result in the DB. "
                "Use this AFTER the user has approved the article (before finishing). "
                "Returns SEO metadata (title, description, slug, tags, etc.) and saves it to the article record."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "article_id": {"type": "string", "description": "Article UUID (resolve from CURRENT CONTEXT)"},
                },
                "required": ["article_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "push_to_ghost",
            "description": (
                "Push the approved article to Ghost CMS as a draft. "
                "Automatically runs the metadata agent if SEO metadata is not yet generated. "
                "Strips the hero image from the inline content and sets it as Ghost's feature image. "
                "Always creates a draft (never publishes). "
                "Returns the Ghost editor URL so the user can review and publish manually."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "article_id": {"type": "string", "description": "Article UUID to push to Ghost (resolve from CURRENT CONTEXT)"},
                },
                "required": ["article_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_promptlayer_execution",
            "description": (
                "Fetch PromptLayer workflow execution results. Three modes:\n"
                "1. execution_id — fetch a specific execution by its ID directly from PromptLayer.\n"
                "2. article_id — look up the stored execution ID from that article's seo_metadata and fetch it.\n"
                "3. last_n — return the last N articles that have stored SEO metadata (with their exec IDs and metadata), served from our DB. Use this for 'show me the last 5 runs'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "execution_id": {"type": "string", "description": "PromptLayer workflow_version_execution_id to fetch directly"},
                    "article_id": {"type": "string", "description": "Article UUID — fetches the execution linked to this article's seo_metadata"},
                    "last_n": {"type": "integer", "description": "Return the last N articles with stored metadata from our DB (max 20)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_history",
            "description": (
                "Fetch older conversation messages beyond the current context window. "
                "Use when the user refers to something not visible in recent context "
                "(e.g. 'that article from last week', 'what did I say earlier'). "
                "Returns messages in chronological order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "before_timestamp": {
                        "type": "string",
                        "description": "ISO timestamp — fetch messages older than this. Use the created_at of the oldest message currently in context.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max messages to return (default 20, max 50)",
                        "default": 20,
                    },
                },
                "required": ["before_timestamp"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "manage_optimization",
            "description": (
                "View, modify, or deploy the pending self-optimization session. "
                "Use when the user wants to see, approve, refine, or skip the daily agent analysis. "
                "Actions: 'list' (show pending items), 'remove_items' (drop specific item IDs before deploy), "
                "'deploy' (publish approved prompt changes to PromptLayer), 'reject' (discard session)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "remove_items", "deploy", "reject"],
                        "description": "What to do with the pending optimization session.",
                    },
                    "remove_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of action item IDs to remove from the deploy list (use with remove_items or before deploy).",
                    },
                },
                "required": ["action"],
            },
        },
    },
]


def apply_tool_description_overrides() -> None:
    """Load tool descriptions from PromptLayer and override hardcoded defaults in TOOL_DEFINITIONS.

    Called once at app startup (after load_all_prompts). Safe to call multiple times —
    re-applies current PromptLayer values. No-op if tool_descriptions is not set in PromptLayer.
    """
    from src.prompts_loader import get_tool_descriptions, invalidate_prompts_cache
    invalidate_prompts_cache()
    overrides = get_tool_descriptions()
    if not overrides:
        logger.debug("[AGENT] No tool_descriptions overrides in PromptLayer — using defaults")
        return
    applied = []
    for tool_def in TOOL_DEFINITIONS:
        name = tool_def.get("function", {}).get("name", "")
        if name in overrides:
            tool_def["function"]["description"] = overrides[name]
            applied.append(name)
    logger.info("[AGENT] Applied tool description overrides for: %s", applied)


def _openai_tools_to_anthropic() -> list[dict]:
    """Convert OpenAI tool format to Anthropic input_schema format."""
    tools = []
    for t in TOOL_DEFINITIONS:
        fn = t.get("function", {})
        params = fn.get("parameters", {})
        # Anthropic expects name, description, input_schema
        tools.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": params,
        })
    return tools


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

EXPENSIVE_TOOLS = {"write_article", "generate_images", "generate_hero_image", "generate_infographic"}
EXPENSIVE_DB_ACTIONS = {"delete"}


def _execute_tool(name: str, arguments: dict) -> Any:
    """Execute a tool by name with given arguments."""
    from src.tools import (
        db as _db,
        google_docs_tool as _google_docs,
        web_search as _web_search,
        send_image_tool as _send_image,
        write_article as _write_article,
        improve_article as _improve_article,
        inject_links as _inject_links,
        generate_and_place_images as _generate_images,
        generate_hero_image_tool as _generate_hero_image,
        approve_hero_image_tool as _approve_hero_image,
        generate_infographic_tool as _generate_infographic,
        approve_infographic_tool as _approve_infographic,
        generate_seo_metadata as _generate_seo_metadata,
        push_to_ghost as _push_to_ghost,
        fetch_promptlayer_execution as _fetch_pl_execution,
        fetch_history as _fetch_history,
        manage_optimization as _manage_optimization,
    )

    # Guard: log a warning for expensive operations so we have an audit trail
    is_expensive = name in EXPENSIVE_TOOLS or (
        name == "db" and arguments.get("action") in EXPENSIVE_DB_ACTIONS
    )
    if is_expensive:
        logger.warning(
            "[AGENT] EXPENSIVE tool call: name=%s, args=%s",
            name,
            {k: (str(v)[:80] + "..." if isinstance(v, str) and len(str(v)) > 80 else v) for k, v in arguments.items()},
        )
    else:
        logger.info(
            "[AGENT] Tool call: name=%s, args=%s",
            name,
            {k: (str(v)[:80] + "..." if isinstance(v, str) and len(str(v)) > 80 else v) for k, v in arguments.items()},
        )
    tools = {
        "db": _db,
        "google_docs": _google_docs,
        "web_search": _web_search,
        "send_image": _send_image,
        "write_article": _write_article,
        "improve_article": _improve_article,
        "inject_links": _inject_links,
        "generate_images": _generate_images,
        "generate_hero_image": _generate_hero_image,
        "approve_hero_image": _approve_hero_image,
        "generate_infographic": _generate_infographic,
        "approve_infographic": _approve_infographic,
        "generate_seo_metadata": _generate_seo_metadata,
        "push_to_ghost": _push_to_ghost,
        "fetch_promptlayer_execution": _fetch_pl_execution,
        "fetch_history": _fetch_history,
        "manage_optimization": _manage_optimization,
    }
    fn = tools.get(name)
    if not fn:
        logger.error("[AGENT] Unknown tool: %s", name)
        return {"success": False, "error": f"Unknown tool: {name}", "retry_hint": "Use a valid tool name."}
    start = time.perf_counter()
    try:
        result = fn(**arguments)
        latency_ms = (time.perf_counter() - start) * 1000
        has_error = isinstance(result, dict) and ("error" in result or result.get("success") is False)
        observe_tool(name, arguments, result, latency_ms, error=result.get("error") if has_error else None)
        if has_error:
            logger.error("[AGENT] Tool %s returned error: %s", name, result.get("error"))
            if "success" not in result:
                result["success"] = False
            if "retry_hint" not in result and "error" in result:
                result["retry_hint"] = "Check the error and try again with different parameters."
        else:
            if isinstance(result, dict) and "success" not in result:
                result = dict(result)
                result["success"] = True
            preview = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
            logger.info("[AGENT] Tool %s OK: result=%s", name, preview)
        return result
    except TaskCancelledError:
        # Don't swallow cancellation — let it propagate to the agent loop
        raise
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        observe_tool(name, arguments, None, latency_ms, error=str(e))
        logger.exception("[AGENT] Tool %s EXCEPTION: %s", name, e)
        err_str = str(e)
        retry_hint = "Try again or use a different tool."
        if "token" in err_str.lower() or "limit" in err_str.lower():
            retry_hint = "Try improve_article for smaller edits, or use a shorter topic with write_article."
        return {"success": False, "error": err_str, "retry_hint": retry_hint}


# ---------------------------------------------------------------------------
# Tool call parsing (for <tool_calls> XML emitted by the PromptLayer LLM)
# ---------------------------------------------------------------------------

def _parse_tool_calls(text: str) -> list[dict]:
    """Parse <tool_calls>/<tool_call> XML blocks from LLM response text."""
    tool_calls = []
    for match in re.findall(r"<tool_calls>(.*?)</tool_calls>", text, re.DOTALL):
        for call_json in re.findall(r"<tool_call>(.*?)</tool_call>", match, re.DOTALL):
            try:
                tc = json.loads(call_json.strip())
                if isinstance(tc, dict) and "name" in tc:
                    tool_calls.append(tc)
            except json.JSONDecodeError:
                logger.warning("[AGENT] Failed to parse tool_call JSON: %r", call_json[:200])
    return tool_calls


def _strip_tool_calls(text: str) -> str:
    """Remove all <tool_calls>…</tool_calls> blocks from text."""
    return re.sub(r"<tool_calls>.*?</tool_calls>", "", text, flags=re.DOTALL).strip()




# ---------------------------------------------------------------------------
# Agent run loop
# ---------------------------------------------------------------------------

def run_agent(
    user_message: str,
    history: Optional[list[dict]] = None,
    channel: Optional[str] = None,
    channel_user_id: Optional[str] = None,
    max_turns: int = 10,
    format_for_whatsapp: bool = False,
    on_status: Optional[Callable[[str], None]] = None,
    trace_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Run the master agent with tool-calling loop.
    Returns {message, article?}.
    If on_status is provided, it is called with progress updates during article writing.
    """
    if trace_id:
        set_trace_id(trace_id)
        init_trace_payload(user_message, channel, channel_user_id)
    log_event(
        "agent_run_start",
        history_len=len(history) if history else 0,
        format_for_whatsapp=format_for_whatsapp,
    )
    logger.info("[AGENT] Run started: message_len=%d, history_len=%d, channel=%s, channel_user_id=%s",
                len(user_message), len(history) if history else 0, channel, channel_user_id)

    token = None
    cancel_token = None
    if on_status is not None:
        token = pipeline_status_callback.set(on_status)
        logger.info("[AGENT] Status callback enabled")

    # Set up cancellation check if channel/user provided
    if channel and channel_user_id:
        from src.task_state import is_cancel_requested
        cancel_token = pipeline_cancel_check.set(
            lambda: is_cancel_requested(channel, channel_user_id)
        )
        logger.info("[AGENT] Cancel check enabled for %s/%s", channel, channel_user_id[:20] if channel_user_id else "")

    # Set channel context so tools (like send_image) know where to send
    channel_token = None
    chat_id_token = None
    if channel:
        from src.tools import _current_channel, _current_chat_id
        channel_token = _current_channel.set(channel)
        chat_id_token = _current_chat_id.set(channel_user_id)

    try:
        # Build dynamic context for the workflow
        context_parts = []

        if format_for_whatsapp:
            wp = get_prompt("whatsapp_format")
            if wp:
                context_parts.append(wp)

        if os.getenv("GHOST_ADMIN_URL"):
            context_parts.append(
                "## Ghost Publishing\n"
                "When the user asks to push, publish, send, or create a draft in Ghost "
                "(e.g. 'push to ghost', 'send to ghost', 'ghost draft', 'publish to ghost'), "
                "ALWAYS call the `push_to_ghost` tool immediately with the current article_id. "
                "Do NOT narrate or describe the action — just call the tool."
            )

        # Fetch article context + history in parallel
        history_limit = get_config_int("agent_history_limit", 10)
        if channel and channel_user_id:
            context_parts.append(f"CURRENT CONTEXT:\nchannel={channel}\nchannel_user_id={channel_user_id}")
            from concurrent.futures import ThreadPoolExecutor
            from src.db import get_latest_article_for_user, list_articles, get_pending_article_images
            from src.message_cache import get_recent_with_backfill

            with ThreadPoolExecutor(max_workers=2) as pool:
                f_article = pool.submit(get_latest_article_for_user, channel, channel_user_id)
                f_history = pool.submit(get_recent_with_backfill, channel, channel_user_id, limit=history_limit)
                latest = f_article.result()
                history = f_history.result()

            try:
                if latest:
                    ctx = f"Current article: article_id={latest['id']}, title=\"{latest.get('title', 'Untitled')}\", version={latest.get('version', 1)}, status={latest.get('status', 'draft')}."
                    if latest.get("google_doc_url"):
                        ctx += f"\nGoogle Doc: {latest['google_doc_url']}"
                        ctx += (
                            "\nSOURCE OF TRUTH: The Google Doc is the authoritative version of the article content — "
                            "humans may have edited it directly. When the user asks what the article says, shows, or contains, "
                            "always use the google_docs tool (action='fetch') to read the current version. "
                            "Use the database only for metadata (status, title, article_id, etc.)."
                        )
                    ctx += "\nWhen the user says \"the article\", \"it\", or refers to the draft - use this article_id. Do NOT ask for article_id."
                    try:
                        pending = get_pending_article_images(latest["id"])
                        if pending:
                            ctx += "\nPending images (use these IDs for approve tools):"
                            for img in pending[:5]:
                                ctx += f"\n- image_id={img['id']}, type={img.get('image_type', 'generic')}"
                    except Exception:
                        pass
                    context_parts.append(ctx)
                else:
                    articles = list_articles(channel=channel, channel_user_id=channel_user_id, limit=3)
                    if articles:
                        ctx = "Recent articles:"
                        for a in articles:
                            ctx += f"\n- article_id={a['id']}, title=\"{a.get('title', 'Untitled')}\", status={a.get('status', 'draft')}"
                            if a.get("google_doc_url"):
                                ctx += f", doc={a['google_doc_url']}"
                        ctx += "\nUse the most recent article_id when the user refers to \"the article\" or \"it\"."
                        context_parts.append(ctx)
                    else:
                        context_parts.append("No articles yet for this user.")
            except Exception as ctx_err:
                logger.warning("[AGENT] Failed to load article context: %s", ctx_err)

            # Inject pending optimization session if one exists
            try:
                from src.db import get_pending_optimization_session as _get_pending_opt
                pending_opt = _get_pending_opt(channel_user_id)
                if pending_opt:
                    items = pending_opt.get("action_items") or []
                    n_items = len(items) if isinstance(items, list) else 0
                    created = str(pending_opt.get("created_at", ""))[:16]
                    context_parts.append(
                        f"## Pending Self-Optimization (Session {str(pending_opt.get('id',''))[:8]})\n"
                        f"Analysis from {created}. {n_items} action items found.\n"
                        "Use the `manage_optimization` tool when the user wants to view, approve, modify, or skip it.\n"
                        "User commands: 'deploy all', 'remove [ids], deploy', 'show optimization', 'skip optimization'."
                    )
            except Exception:
                pass

        # Build system prompt: static core (cacheable) + dynamic context (uncached)
        context_str = "\n\n".join(context_parts)
        static_system = get_master_system_prompt()

        # Tag the trace with a hash of the static system prompt (tracks prompt versions,
        # independent of per-session context which changes every call).
        import hashlib as _hashlib
        from src.observability import get_trace_payload as _get_payload
        _payload = _get_payload()
        if _payload is not None:
            _payload["prompt_version"] = _hashlib.md5(static_system.encode()).hexdigest()[:8]

        # Build system as a list so Anthropic can cache the large static block.
        # The ephemeral cache_control on the static block saves ~90% on repeat reads.
        system: list[dict] | str
        if context_str.strip():
            system = [
                {"type": "text", "text": static_system, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": context_str},
            ]
        else:
            system = [
                {"type": "text", "text": static_system, "cache_control": {"type": "ephemeral"}},
            ]

        # Build Anthropic messages from history
        messages: list[dict] = []
        for m in (history or [])[-history_limit:]:
            role = m["role"]
            content = m["content"] if isinstance(m["content"], str) else str(m["content"])
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_message.strip() or "(no message)"})

        tools = _openai_tools_to_anthropic()
        # Turn 0 uses Haiku for fast routing / conversational replies.
        # Turn 1+ (after tool execution) uses Sonnet for synthesis and reasoning.
        haiku_model = get_config("agent_haiku_model") or "claude-haiku-4-5-20251001"
        sonnet_model = (
            get_prompt_llm_kwargs("master_system_core").get("model")
            or get_config("agent_model")
            or "claude-sonnet-4-5"
        )
        client = _AnthropicClient()
        result_text = ""
        _pl_tags = [channel or "web", "blog-writer"]
        _pl_prompt_name = f"{_PL_PREFIX}/master_system_core" if _PL_PREFIX else "master_system_core"

        logger.info("[AGENT] Calling Anthropic directly: haiku=%s, sonnet=%s, messages=%d, tools=%d",
                    haiku_model, sonnet_model, len(messages), len(tools))

        for turn in range(max_turns):
            current_model = haiku_model if turn == 0 else sonnet_model
            max_tokens = 2048 if turn == 0 else 8096
            _pl_id = None
            if _PL_ENABLED:
                resp, _pl_id = client.messages.create(
                    model=current_model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                    tools=tools,
                    pl_tags=_pl_tags,
                    return_pl_id=True,
                )
            else:
                resp = client.messages.create(
                    model=current_model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                    tools=tools,
                )
            logger.info("[AGENT] model=%s (turn %d)", current_model, turn + 1)

            # Link this request to the master_system_core template version in PL (async)
            if _pl_id:
                import threading as _threading
                _threading.Thread(
                    target=_pl_track_prompt,
                    args=(_pl_id, _pl_prompt_name),
                    daemon=True,
                ).start()

            # Extract text content
            result_text = "\n".join(
                b.text for b in resp.content if hasattr(b, "text")
            ).strip()
            logger.info("[AGENT] response: stop_reason=%s, text_len=%d",
                        resp.stop_reason, len(result_text))

            if resp.stop_reason != "tool_use":
                break

            # Serialize assistant message (text + tool_use blocks)
            assistant_content = []
            for block in resp.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
            messages.append({"role": "assistant", "content": assistant_content})

            # Execute each tool and collect results
            tool_results = []
            for tu in [b for b in resp.content if b.type == "tool_use"]:
                logger.info("[AGENT] Executing tool (turn %d): %s", turn + 1, tu.name)
                result = _execute_tool(tu.name, tu.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps(result, default=str),
                })
            messages.append({"role": "user", "content": tool_results})
        else:
            logger.warning("[AGENT] Reached max_turns=%d without a final answer", max_turns)

        return {
            "message": result_text,
            "article": None,
        }
    except TaskCancelledError:
        logger.info("[AGENT] Task cancelled by user")
        return {
            "message": "Cancelled.",
            "article": None,
        }
    finally:
        if token is not None:
            pipeline_status_callback.reset(token)
        if cancel_token is not None:
            pipeline_cancel_check.reset(cancel_token)
        if channel_token is not None:
            _current_channel.reset(channel_token)
        if chat_id_token is not None:
            _current_chat_id.reset(chat_id_token)
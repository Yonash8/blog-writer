"""Master agent: autonomous agent with tool calling. Uses Anthropic Claude Sonnet by default."""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from anthropic import Anthropic

from src.config import get_config, get_config_int
from src.prompts_loader import get_master_system_prompt, get_prompt
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
                "Full article pipeline: deep research + Tavily enrichment + PromptLayer SEO writer. "
                "Auto-saves to DB and creates Google Doc. Returns article_id, content, google_doc_url. "
                "Use tavily_max_results 5-10 for quick drafts, 20 for thorough research."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "The topic to write about"},
                    "include_tavily": {"type": "boolean", "description": "Include Tavily enrichment (default True)", "default": True},
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
]


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
        generate_and_place_images as _generate_images,
        generate_hero_image_tool as _generate_hero_image,
        approve_hero_image_tool as _approve_hero_image,
        generate_infographic_tool as _generate_infographic,
        approve_infographic_tool as _approve_infographic,
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
        "generate_images": _generate_images,
        "generate_hero_image": _generate_hero_image,
        "approve_hero_image": _approve_hero_image,
        "generate_infographic": _generate_infographic,
        "approve_infographic": _approve_infographic,
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
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required. Add it to your .env file.")
        client = Anthropic(api_key=api_key)

        system = get_master_system_prompt() + (get_prompt("whatsapp_format") if format_for_whatsapp else "")

        # Inject current context: channel, user, and article info
        if channel and channel_user_id:
            system += f"\n\nCURRENT CONTEXT:\nchannel={channel}\nchannel_user_id={channel_user_id}"
            try:
                from src.db import get_latest_article_for_user, list_articles, get_pending_article_images
                latest = get_latest_article_for_user(channel, channel_user_id)
                if latest:
                    system += f"\nCurrent article: article_id={latest['id']}, title=\"{latest.get('title', 'Untitled')}\", version={latest.get('version', 1)}, status={latest.get('status', 'draft')}."
                    if latest.get("google_doc_url"):
                        system += f"\nGoogle Doc: {latest['google_doc_url']}"
                    system += "\nWhen the user says \"the article\", \"it\", or refers to the draft - use this article_id. Do NOT ask for article_id."
                    # Include pending image IDs for approve/reject tool calls
                    try:
                        pending = get_pending_article_images(latest["id"])
                        if pending:
                            system += "\nPending images (use these IDs for approve tools):"
                            for img in pending[:5]:
                                system += f"\n- image_id={img['id']}, type={img.get('image_type', 'generic')}"
                    except Exception:
                        pass
                else:
                    articles = list_articles(channel=channel, channel_user_id=channel_user_id, limit=3)
                    if articles:
                        system += "\nRecent articles:"
                        for a in articles:
                            system += f"\n- article_id={a['id']}, title=\"{a.get('title', 'Untitled')}\", status={a.get('status', 'draft')}"
                            if a.get("google_doc_url"):
                                system += f", doc={a['google_doc_url']}"
                        system += "\nUse the most recent article_id when the user refers to \"the article\" or \"it\"."
                    else:
                        system += "\nNo articles yet for this user."
            except Exception as ctx_err:
                logger.warning("[AGENT] Failed to load article context: %s", ctx_err)

        # Note: We do NOT duplicate recent exchanges into the system prompt.
        # The full conversation history is already passed as messages below,
        # and the "Conversation Context" section in the system prompt instructs
        # the model how to use it for resolving references like "it", "which one", etc.

        # Anthropic message format: list of {role, content}; system is separate
        messages: list[dict] = []
        if history:
            for m in history[-get_config_int("agent_history_limit", 20):]:
                messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": user_message})

        article = None
        turns = 0
        anthropic_tools = _openai_tools_to_anthropic()
        model = get_config("agent_model", "claude-sonnet-4-5")

        while turns < max_turns:
            start_ts = datetime.now(timezone.utc).isoformat()
            start_perf = time.perf_counter()
            response = client.messages.create(
                model=model,
                max_tokens=8192,
                system=system,
                messages=messages,
                tools=anthropic_tools,
                tool_choice={"type": "auto"},
            )
            end_ts = datetime.now(timezone.utc).isoformat()
            duration_ms = (time.perf_counter() - start_perf) * 1000

            # Build agent_call metadata for observability
            usage = getattr(response, "usage", None)
            tokens = None
            if usage:
                tokens = {
                    "input": getattr(usage, "input_tokens", None),
                    "output": getattr(usage, "output_tokens", None),
                    "total": (getattr(usage, "input_tokens", 0) or 0) + (getattr(usage, "output_tokens", 0) or 0),
                }
            thinking_blocks = []
            text_parts_for_obs = []
            tool_calls_for_obs = []
            from src.observability import OBSERVABILITY_SAVE_PROMPTS
            _preview_len = 2000 if OBSERVABILITY_SAVE_PROMPTS else 500
            _thinking_len = 1500 if OBSERVABILITY_SAVE_PROMPTS else 500
            for b in (response.content or []):
                t = getattr(b, "type", None) or (b.get("type") if isinstance(b, dict) else None)
                if t == "thinking":
                    txt = getattr(b, "text", "") or b.get("text", "") if isinstance(b, dict) else ""
                    thinking_blocks.append(txt[:_thinking_len] + ("..." if len(txt) > _thinking_len else ""))
                elif t == "text":
                    txt = getattr(b, "text", "") or b.get("text", "") if isinstance(b, dict) else ""
                    text_parts_for_obs.append(txt)
                elif t == "tool_use":
                    nm = getattr(b, "name", "") or b.get("name", "")
                    bid = getattr(b, "id", "") or b.get("id", "")
                    # Capture the actual tool call input (arguments the agent chose)
                    raw_input = getattr(b, "input", {}) or (b.get("input", {}) if isinstance(b, dict) else {})
                    tc_entry = {"name": nm, "id": bid}
                    if isinstance(raw_input, dict):
                        # Sanitize + truncate each arg value for safe logging
                        tc_entry["input"] = {
                            k: (str(v)[:300] + "..." if len(str(v)) > 300 else str(v))
                            for k, v in list(raw_input.items())[:10]
                        }
                    else:
                        tc_entry["input"] = str(raw_input)[:500]
                    tool_calls_for_obs.append(tc_entry)
            # Build prompt metadata (include last user message when SAVE_PROMPTS)
            prompt_obj = {
                "system_len": len(system),
                "messages_count": len(messages),
                "tools_count": len(anthropic_tools),
            }
            if OBSERVABILITY_SAVE_PROMPTS and messages:
                # Store the last user/tool_result message for context
                last_msg = messages[-1]
                last_content = last_msg.get("content", "")
                if isinstance(last_content, str):
                    prompt_obj["last_message_preview"] = last_content[:500] + ("..." if len(last_content) > 500 else "")
                elif isinstance(last_content, list):
                    # tool_result messages are lists
                    parts = []
                    for item in last_content[:3]:
                        if isinstance(item, dict):
                            c = item.get("content", item.get("text", ""))
                            parts.append(str(c)[:200])
                    prompt_obj["last_message_preview"] = " | ".join(parts)[:500]
                prompt_obj["last_message_role"] = last_msg.get("role", "?")
            response_obj = {
                "content_preview": ("".join(text_parts_for_obs))[:_preview_len] + ("..." if len("".join(text_parts_for_obs)) > _preview_len else ""),
                "thinking": thinking_blocks if thinking_blocks else None,
                "tool_calls": tool_calls_for_obs,
                "stop_reason": getattr(response, "stop_reason", None),
            }
            observe_agent_call(
                name=f"agent_turn_{turns + 1}",
                provider="anthropic",
                model=model,
                prompt=prompt_obj,
                response=response_obj,
                tokens=tokens,
                span={"start_ts": start_ts, "end_ts": end_ts, "duration_ms": round(duration_ms, 2)},
            )

            # Extract text and tool_use blocks from content
            text_parts = []
            tool_use_blocks = []
            for block in (response.content or []):
                if hasattr(block, "type"):
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "tool_use":
                        tool_use_blocks.append(block)
                elif isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tool_use_blocks.append(block)

            # No tool use: return the text response
            if response.stop_reason != "tool_use" or not tool_use_blocks:
                logger.info("[AGENT] No tool calls - returning final message (turn=%d)", turns + 1)
                return {
                    "message": "".join(text_parts).strip() or "",
                    "article": article,
                }

            logger.info("[AGENT] Turn %d: %d tool call(s)", turns + 1, len(tool_use_blocks))

            # Append assistant message — preserve ALL content blocks (text + tool_use)
            assistant_content = []
            for b in response.content or []:
                if hasattr(b, "type"):
                    if b.type == "text":
                        assistant_content.append({"type": "text", "text": b.text})
                    elif b.type == "tool_use":
                        assistant_content.append({
                            "type": "tool_use",
                            "id": b.id,
                            "name": b.name,
                            "input": b.input,
                        })
                elif isinstance(b, dict):
                    if b.get("type") == "text":
                        assistant_content.append({"type": "text", "text": b.get("text", "")})
                    elif b.get("type") == "tool_use":
                        assistant_content.append(b)
            messages.append({"role": "assistant", "content": assistant_content})

            # Execute each tool and build tool_result blocks
            tool_results = []
            for block in tool_use_blocks:
                bid = block.id if hasattr(block, "id") else block.get("id")
                name = block.name if hasattr(block, "name") else block.get("name")
                args = block.input if hasattr(block, "input") else block.get("input", {})
                if not isinstance(args, dict):
                    args = {}

                if channel and channel_user_id and name in ("write_article",):
                    args.setdefault("channel", channel)
                    args.setdefault("channel_user_id", channel_user_id)

                result = _execute_tool(name, args)
                if isinstance(result, dict) and ("content" in result or "article_id" in result) and "article_id" in str(result):
                    article = result

                content_str = json.dumps(result) if not isinstance(result, str) else result
                tool_results.append({"type": "tool_result", "tool_use_id": bid, "content": content_str})

            messages.append({"role": "user", "content": tool_results})
            turns += 1

        logger.warning("[AGENT] Max turns (%d) reached", max_turns)
        return {
            "message": "Maximum turns reached. Please try again.",
            "article": article,
        }
    except TaskCancelledError:
        logger.info("[AGENT] Task cancelled by user")
        return {
            "message": "Cancelled.",
            "article": article,
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
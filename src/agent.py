from __future__ import annotations
"""Master agent: autonomous agent with tool calling. Uses Claude Opus by default."""

import json
import logging
import os
import re
import time
import textwrap
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from anthropic import Anthropic

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
trace_logger = logging.getLogger("devchat.trace")

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
            "name": "send_interactive",
            "description": (
                "Send WhatsApp interactive options to the user. "
                "Use mode='buttons' for 1-3 quick choices, or mode='poll' for 2-12 options. "
                "If WhatsApp interactive delivery fails, fallback text options are returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["buttons", "poll"], "description": "Interactive type to send"},
                    "body": {"type": "string", "description": "Main text shown above options"},
                    "choices": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Option labels. Buttons: 1-3, Poll: 2-12.",
                    },
                    "header": {"type": "string", "description": "Optional heading (buttons mode)"},
                    "footer": {"type": "string", "description": "Optional footer (buttons mode)"},
                    "multiple_answers": {"type": "boolean", "description": "Allow multiple answers (poll mode)", "default": False},
                },
                "required": ["mode", "body", "choices"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_article",
            "description": (
                "Full article pipeline: PromptLayer SEO writer (research is handled inside PromptLayer). "
                "Requires explicit user approval in the tool call via approved=true before execution. "
                "Auto-saves to DB and creates Google Doc. Returns article_id, content, google_doc_url, metadata."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "The topic to write about"},
                    "approved": {"type": "boolean", "description": "Set true only after explicit user approval to run write_article."},
                    "changelog_entry": {"type": "string", "description": "Brief changelog entry, e.g. 'Created draft on topic X'"},
                },
                "required": ["topic", "approved"],
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
    {
        "type": "function",
        "function": {
            "name": "clean_memory",
            "description": (
                "Clear the conversation cache for the current user. "
                "Call this when the user says: clean memory, clear memory, forget, reset chat, "
                "clear history, start fresh, wipe conversation, etc. "
                "Removes all stored messages so the next turn starts with no prior context. "
                "Always call this tool—do not just explain; actually run it."
            ),
            "parameters": {"type": "object", "properties": {}},
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


# Action-oriented requests are more sensitive to routing misses.
# Prefer Sonnet on turn 0 for these, keep Haiku for simple conversational turns.
_ACTION_FIRST_PAT = re.compile(
    r"\b("
    r"write|create|generate|improve|edit|revise|add|inject|approve|publish|push|"
    r"ghost|seo|metadata|image|hero|infographic|optimi[sz]e|deploy|run|"
    r"clean|forget|reset|wipe"
    r")\b",
    re.IGNORECASE,
)

_ARTICLE_WORK_PAT = re.compile(
    r"\b("
    r"article|draft|doc|google doc|outline|edit|revise|improve|rewrite|"
    r"hero|infographic|seo|publish|ghost|title|headline|intro|conclusion|"
    r"source|citation|this one|that one|it"
    r")\b",
    re.IGNORECASE,
)

_SESSION_SUMMARY_ATTEMPTS: set[tuple[str, str, str]] = set()
_MISSION_TOOL_MAP = {
    "generate_hero_image": "Generate hero image for the current article.",
    "approve_hero_image": "Finalize hero image in article.",
    "generate_infographic": "Generate infographic for the current article.",
    "approve_infographic": "Finalize infographic placement in article.",
    "generate_seo_metadata": "Generate SEO metadata for the current article.",
    "improve_article": "Revise the current article draft.",
    "push_to_ghost": "Publish current article to Ghost.",
}
_APPROVAL_REQUIRED_TOOLS = {
    "improve_article",
    "approve_hero_image",
    "approve_infographic",
    "push_to_ghost",
}
_ALLOWED_REACTION_REPLIES = ("👍", "🔥", "✅")


def _should_use_sonnet_first_turn(user_message: str) -> bool:
    text = (user_message or "").strip()
    if not text:
        return False
    if len(text) <= 16 and text.lower() in {"ok", "yes", "no", "thanks", "thx", "why", "how"}:
        return False
    return bool(_ACTION_FIRST_PAT.search(text))


def _should_attempt_session_article_summary(user_message: str, history: Optional[list[dict]]) -> bool:
    """Run summary caching once when a likely article work session starts."""
    text = (user_message or "").strip().lower()
    if not text:
        return False
    if len(history or []) <= 2:
        return True
    return bool(_ARTICLE_WORK_PAT.search(text))


def _infer_mission_from_user_message(user_message: str) -> Optional[str]:
    text = (user_message or "").strip().lower()
    if not text:
        return None
    if re.search(r"\b(hero|cover image|thumbnail)\b", text):
        return "Create and approve a hero image for the current article."
    if re.search(r"\b(infographic|diagram|chart)\b", text):
        return "Create and place an infographic in the current article."
    if re.search(r"\b(seo|metadata|slug|description|tags)\b", text):
        return "Generate SEO metadata for the current article."
    if re.search(r"\b(publish|post|ship|push to ghost|ghost)\b", text):
        return "Publish the current article to Ghost."
    if re.search(r"\b(improve|edit|revise|rewrite|fix|polish)\b", text) and "article" in text:
        return "Revise the current article draft."
    if re.search(r"\b(write|create|draft)\b", text) and "article" in text:
        return "Write a new article draft."
    if re.search(r"\b(memory|forget|reset|clear)\b", text):
        return "Reset conversation memory."
    return None


def _is_tool_approval_required(name: str, arguments: Optional[dict]) -> bool:
    """Return True when tool execution requires explicit user approval."""
    if name in _APPROVAL_REQUIRED_TOOLS:
        return True
    if name == "manage_optimization":
        action = str((arguments or {}).get("action", "")).strip().lower()
        return action == "deploy"
    return False


def _has_explicit_human_approval(user_message: str) -> bool:
    """Conservative check for explicit approval phrasing in latest user turn."""
    text = (user_message or "").strip().lower()
    if not text:
        return False
    if re.search(r"\b(do not|don't|stop|wait|hold|cancel|no)\b", text):
        return False
    return bool(
        re.search(
            r"\b(yes|approve|approved|go ahead|go-ahead|proceed|ship it|deploy|publish|run it|do it|yalla)\b",
            text,
        )
    )


def _normalize_response_style(text: str) -> str:
    """Allow strict standalone reactions; never allow reaction emojis inside text."""
    out = (text or "").strip()
    if not out:
        return out

    # If response is only allowed reaction emojis/spaces, reduce to one emoji.
    if re.fullmatch(r"[\s👍🔥✅]+", out):
        for ch in out:
            if ch in _ALLOWED_REACTION_REPLIES:
                return ch
        return out

    # Otherwise, remove allowed reaction emojis from inside regular text.
    for emo in _ALLOWED_REACTION_REPLIES:
        out = out.replace(emo, "")
    out = re.sub(r"[ \t]{2,}", " ", out).strip()
    return out


def _wrap_for_log(text: str, width: int = 60) -> str:
    """Wrap long prompt lines for readable terminal logs."""
    lines = []
    for ln in (text or "").splitlines():
        if not ln.strip():
            lines.append("")
            continue
        lines.append(textwrap.fill(ln, width=width, break_long_words=False, break_on_hyphens=False))
    return "\n".join(lines)


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
        send_interactive_tool as _send_interactive,
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
        clean_memory_tool as _clean_memory,
        manage_optimization as _manage_optimization,
    )

    # Guard: log a warning for expensive operations so we have an audit trail
    is_expensive = name in EXPENSIVE_TOOLS or (
        name == "db" and arguments.get("action") in EXPENSIVE_DB_ACTIONS
    )
    arg_preview = {
        k: (str(v)[:80] + "..." if isinstance(v, str) and len(str(v)) > 80 else v)
        for k, v in arguments.items()
    }
    if is_expensive:
        trace_logger.warning("tool_call | expensive:%s args=%s", name, arg_preview)
    else:
        trace_logger.info("tool_call | %s args=%s", name, arg_preview)
    tools = {
        "db": _db,
        "google_docs": _google_docs,
        "web_search": _web_search,
        "send_image": _send_image,
        "send_interactive": _send_interactive,
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
        "clean_memory": _clean_memory,
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
            trace_logger.error(
                "tool_result | %s error runtime_ms=%.1f error=%s",
                name,
                latency_ms,
                result.get("error"),
            )
            if "success" not in result:
                result["success"] = False
            if "retry_hint" not in result and "error" in result:
                result["retry_hint"] = "Check the error and try again with different parameters."
        else:
            if isinstance(result, dict) and "success" not in result:
                result = dict(result)
                result["success"] = True
            preview = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
            trace_logger.info(
                "tool_result | %s ok runtime_ms=%.1f result=%s",
                name,
                latency_ms,
                preview,
            )
        return result
    except TaskCancelledError:
        # Don't swallow cancellation — let it propagate to the agent loop
        raise
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        observe_tool(name, arguments, None, latency_ms, error=str(e))
        trace_logger.exception("tool_result | %s exception runtime_ms=%.1f error=%s", name, latency_ms, e)
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
    run_start = time.perf_counter()

    if trace_id:
        set_trace_id(trace_id)
        init_trace_payload(user_message, channel, channel_user_id)
    log_event(
        "agent_run_start",
        history_len=len(history) if history else 0,
        format_for_whatsapp=format_for_whatsapp,
    )
    logger.info(
        "[AGENT] Run started: message_len=%d, history_len=%d, channel=%s, channel_user_id=%s",
        len(user_message),
        len(history) if history else 0,
        channel,
        channel_user_id,
    )

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

        context_parts.append(
            "## Operating Mode\n"
            "Be autonomous and conversational. Infer intent from context and execute tools when intent is clear. "
            "Avoid unnecessary clarification loops. For short follow-ups and pronouns, resolve from current context/history. "
            "If multiple actions are clearly requested, do them in sequence without asking for step-by-step confirmation. "
            "Only ask a brief clarifying question when proceeding could target the wrong resource or create irreversible side effects."
        )
        context_parts.append(
            "## Reply Style\n"
            "Keep replies short and casual, like the user's tone. Prefer concise wording over formal phrasing. "
            "You may reply with a standalone reaction only: 👍 or 🔥 or ✅. "
            "Never place these reaction emojis inside a normal text message."
        )

        # Fetch article context + history in parallel
        history_limit = get_config_int("agent_history_limit", 10)
        if channel and channel_user_id:
            context_parts.append(f"CURRENT CONTEXT:\nchannel={channel}\nchannel_user_id={channel_user_id}")
            from concurrent.futures import ThreadPoolExecutor
            from src.db import get_latest_article_for_user, list_articles, get_pending_article_images
            from src.message_cache import get_recent_with_backfill, get_current_mission, set_current_mission

            with ThreadPoolExecutor(max_workers=3) as pool:
                f_article = pool.submit(get_latest_article_for_user, channel, channel_user_id)
                f_history = pool.submit(get_recent_with_backfill, channel, channel_user_id, limit=history_limit)
                f_mission = pool.submit(get_current_mission, channel, channel_user_id)
                latest = f_article.result()
                history = f_history.result()
                current_mission = f_mission.result()

            inferred_mission = _infer_mission_from_user_message(user_message)
            if inferred_mission:
                existing = (current_mission or {}).get("mission", "")
                if inferred_mission != existing:
                    set_current_mission(
                        channel=channel,
                        channel_user_id=channel_user_id,
                        mission=inferred_mission,
                        article_id=(latest or {}).get("id"),
                        status="active",
                    )
                    current_mission = get_current_mission(channel, channel_user_id)
                    trace_logger.info("mission | updated from user intent")

            if current_mission and current_mission.get("status") == "active":
                mission_ctx = (
                    "CURRENT MISSION:\n"
                    f"- mission={current_mission.get('mission', '')}\n"
                    f"- article_id={current_mission.get('article_id') or 'none'}\n"
                    f"- updated_at={current_mission.get('updated_at', '')}"
                )
                context_parts.append(mission_ctx)

            # Lazy cache: summarize article once at the start of a work session.
            if latest and latest.get("id"):
                key = (channel, channel_user_id, latest["id"])
                if key not in _SESSION_SUMMARY_ATTEMPTS and _should_attempt_session_article_summary(user_message, history):
                    _SESSION_SUMMARY_ATTEMPTS.add(key)
                    try:
                        from src.article_memory import has_cached_summary, summarize_with_sonnet, format_summary_memory
                        from src.db import add_message as _add_message

                        if not has_cached_summary(history or [], latest["id"]):
                            summary_text = summarize_with_sonnet(
                                latest.get("title", "Untitled"),
                                latest.get("content") or "",
                            )
                            summary_msg = format_summary_memory(
                                title=latest.get("title", "Untitled"),
                                article_id=latest["id"],
                                google_doc_url=latest.get("google_doc_url"),
                                content=latest.get("content") or "",
                                summary=summary_text,
                            )
                            _add_message(channel, channel_user_id, "assistant", summary_msg)
                            history = (history or []) + [{"role": "assistant", "content": summary_msg}]
                            # Keep memory logs concise; avoid dumping memory payloads.
                            trace_logger.info("memory | cached_summary")
                    except Exception as e:
                        logger.warning("[AGENT] Failed to cache session article summary: %s", e)

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
                    ctx += (
                        "\nWhen the user says \"the article\" or \"it\", treat this as the default target article unless the user explicitly switches."
                    )
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

        # In dev chat, print the full prompts for debugging.
        if channel == "dev":
            trace_logger.info("master_core | static_system_full |\n%s", _wrap_for_log(static_system, width=60))
            if context_str.strip():
                trace_logger.info("master_core | dynamic_context_full |\n%s", _wrap_for_log(context_str, width=60))

        # Build Anthropic messages from history
        messages: list[dict] = []
        for m in (history or [])[-history_limit:]:
            role = m["role"]
            content = m["content"] if isinstance(m["content"], str) else str(m["content"])
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_message.strip() or "(no message)"})

        tools = _openai_tools_to_anthropic()
        # Default: Haiku for quick turn-0 routing, Sonnet after tool execution.
        # For explicit action requests, start with Sonnet to reduce routing/context misses.
        haiku_model = get_config("agent_haiku_model") or "claude-haiku-4-5-20251001"
        sonnet_model = (
            get_prompt_llm_kwargs("master_system_core").get("model")
            or get_config("agent_model")
            or "claude-sonnet-4-5"
        )
        first_turn_model = sonnet_model if _should_use_sonnet_first_turn(user_message) else haiku_model
        client = Anthropic()
        result_text = ""

        logger.info("[AGENT] Calling Anthropic directly: haiku=%s, sonnet=%s, messages=%d, tools=%d",
                    haiku_model, sonnet_model, len(messages), len(tools))

        for turn in range(max_turns):
            turn_start = time.perf_counter()
            current_model = first_turn_model if turn == 0 else sonnet_model
            max_tokens = 2048 if turn == 0 else 8096
            resp = client.messages.create(
                model=current_model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                tools=tools,
            )
            # Extract text content
            result_text = "\n".join(
                b.text for b in resp.content if hasattr(b, "text")
            ).strip()
            turn_ms = (time.perf_counter() - turn_start) * 1000
            trace_logger.info(
                "agent_turn | turn=%d model=%s stop_reason=%s text_len=%d runtime_ms=%.1f",
                turn + 1,
                current_model,
                resp.stop_reason,
                len(result_text),
                turn_ms,
            )

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
                if _is_tool_approval_required(tu.name, tu.input) and not _has_explicit_human_approval(user_message):
                    result = {
                        "success": False,
                        "needs_approval": True,
                        "error": f"Explicit user approval required before running {tu.name}.",
                        "approval_instruction": (
                            "Ask for explicit approval first (e.g., 'yes approve', 'go ahead', "
                            "'publish now'), then retry the same tool."
                        ),
                        "retry_hint": "Wait for explicit approval in the latest user message, then retry.",
                    }
                    trace_logger.info("approval_gate | blocked tool=%s waiting_for_human_approval", tu.name)
                else:
                    result = _execute_tool(tu.name, tu.input)
                # Master agent owns mission updates; tools only provide execution signals.
                if channel and channel_user_id and isinstance(result, dict) and result.get("success") is not False:
                    try:
                        from src.message_cache import set_current_mission
                        next_mission = _MISSION_TOOL_MAP.get(tu.name)
                        if next_mission:
                            mission_status = "done" if tu.name == "push_to_ghost" else "active"
                            latest_article_id = latest["id"] if "latest" in locals() and isinstance(latest, dict) and latest.get("id") else None
                            mission_article_id = (
                                result.get("article_id")
                                or (tu.input or {}).get("article_id")
                                or latest_article_id
                            )
                            set_current_mission(
                                channel=channel,
                                channel_user_id=channel_user_id,
                                mission=next_mission,
                                article_id=mission_article_id,
                                status=mission_status,
                            )
                            trace_logger.info(
                                "mission | updated from tool %s status=%s",
                                tu.name,
                                mission_status,
                            )
                    except Exception as mission_err:
                        logger.warning("[AGENT] Mission update failed after tool %s: %s", tu.name, mission_err)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps(result, default=str),
                })
            messages.append({"role": "user", "content": tool_results})
        else:
            logger.warning("[AGENT] Reached max_turns=%d without a final answer", max_turns)

        final_text = _normalize_response_style(result_text)
        return {
            "message": final_text,
            "article": None,
        }
    except TaskCancelledError:
        logger.info("[AGENT] Task cancelled by user")
        return {
            "message": "Cancelled.",
            "article": None,
        }
    finally:
        total_ms = (time.perf_counter() - run_start) * 1000
        trace_logger.info("agent_run | finished runtime_ms=%.1f", total_ms)
        if token is not None:
            pipeline_status_callback.reset(token)
        if cancel_token is not None:
            pipeline_cancel_check.reset(cancel_token)
        if channel_token is not None:
            _current_channel.reset(channel_token)
        if chat_id_token is not None:
            _current_chat_id.reset(chat_id_token)
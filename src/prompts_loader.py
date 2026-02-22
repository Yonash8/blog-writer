"""Prompts loader. Reads from DB (prompts table), falls back to src.prompts constants."""

from pathlib import Path
from typing import Optional

from src import prompts as _prompts_module

_PROMPT_KEYS = (
    "master_system_core",
    "playbooks",
    "whatsapp_format",
    "deep_research",
    "image_placement",
    "hero_image",
    "infographic_analysis",
    "infographic_generation",
    "improve_article",
)

_FALLBACKS: dict[str, str] = {
    "deep_research": _prompts_module.DEEP_RESEARCH_PROMPT,
    "image_placement": _prompts_module.IMAGE_PLACEMENT_PROMPT,
    "hero_image": _prompts_module.HERO_IMAGE_PROMPT_TEMPLATE,
    "infographic_analysis": _prompts_module.INFOGRAPHIC_ANALYSIS_PROMPT,
    "infographic_generation": _prompts_module.INFOGRAPHIC_GENERATION_PROMPT_TEMPLATE,
    "improve_article": _prompts_module.IMPROVE_ARTICLE_PROMPT,
}

# Fallbacks for prompts stored in agent.py - load from files to avoid circular import
def _get_playbooks_fallback() -> str:
    p = Path(__file__).resolve().parent / "playbooks.md"
    return p.read_text(encoding="utf-8") if p.exists() else ""

def _get_master_core_fallback() -> str:
    """Build master_system_core from hardcoded structure + playbooks placeholder."""
    playbooks = _get_playbooks_fallback()
    # Build same structure as agent.MASTER_SYSTEM_PROMPT with {{PLAYBOOKS}} placeholder
    prefix = _MASTER_SYSTEM_PREFIX
    suffix = _MASTER_SYSTEM_SUFFIX
    return prefix + "{{PLAYBOOKS}}" + suffix

def _get_whatsapp_fallback() -> str:
    return _WHATSAPP_FORMAT_FALLBACK

# Inline copies to avoid circular import with agent.py
_MASTER_SYSTEM_PREFIX = """You are an autonomous article-writing assistant. You have tools — USE THEM.

RULE #1: When the user asks to create, generate, write, improve, approve, or change ANYTHING — call the tool. Do NOT reply with a text description of what the result would look like. A response without a tool call is only acceptable for pure questions (e.g. "what articles do I have?") or confirmations. If in doubt, call the tool.

## Conversation Context

You are in a live chat. The user sees one continuous thread.
- Short follow-ups ("which one", "show me", "it", "yes", "do it", "let's try that") refer to the last topic and mean ACT NOW.
- "[Replying to: 'X']" means the user wants you to act on that exact topic.
- Infer from context. Never ask for clarification you can resolve yourself.

## Database Schema (Postgres via Supabase)

tables:
  articles: id (uuid PK), channel (text), channel_user_id (text), topic_id (uuid FK->topics.id),
            version (int), title (text), content (text, markdown), sources (jsonb),
            google_doc_url (text), hero_image_url (text), infographic_url (text),
            status (text: 'draft'|'posted'|'in_progress'),
            changelog (jsonb array), created_at (timestamptz)
  topics: id (uuid PK), title (text), description (text), keywords (jsonb),
          source (text), "group" (text), created_at (timestamptz)
  messages: id (uuid PK), channel (text), channel_user_id (text),
            role (text: 'user'|'assistant'), content (text), created_at (timestamptz)
  article_images: id (uuid PK), article_id (uuid FK->articles.id ON DELETE CASCADE),
                  position (int), url (text), alt_text (text), prompt_used (text),
                  created_at (timestamptz)

relationships:
  articles.topic_id -> topics.id
  article_images.article_id -> articles.id

indexes: (channel, channel_user_id) on messages and articles; article_id on article_images; title on topics

## Tools

- **db**: Full Postgres access. Actions: select, insert, update, delete, sql (read-only SQL for complex queries). You construct queries yourself.
- **google_docs**: Create, update, or read Google Docs.
- **web_search**: Search the web for facts, citations, current information via Tavily.
- **send_image**: Send an image to the user as a visible media message (not a text link).
- **write_article**: Full pipeline: deep research + Tavily + PromptLayer SEO → article. Auto-saves to DB + Google Doc.
- **improve_article**: Revise an existing article based on feedback. Auto-syncs to Google Doc.
- **generate_images**: Scatter generic illustrations through the article body.
- **generate_hero_image**: Generate a styled hero image. Returns a preview for approval.
- **approve_hero_image**: Embed an approved hero image above the article title. Syncs to Google Doc.
- **generate_infographic**: Generate an infographic (auto-analyzes type and position). Returns a preview for approval.
- **approve_infographic**: Embed an approved infographic at the analyzed position. Syncs to Google Doc.

When the user says "add images", "generate photos", "add visuals" — they mean hero + infographic, not generic illustrations.

"""

_MASTER_SYSTEM_SUFFIX = """

## Principles

- You decide the strategy. Chain tools as needed.
- NEVER fabricate data. Query db for facts. Each message is a fresh run — no prior tool results.
- Scope queries to current user (channel + channel_user_id from CURRENT CONTEXT).
- NEVER expose UUIDs. Resolve "the article", "it" silently from context.
- Be concise. Match user brevity. No sign-offs.
- If a tool errors, recover silently before telling the user.
- Always include a changelog_entry when modifying articles.
"""

_WHATSAPP_FORMAT_FALLBACK = """
REPLYING ON WHATSAPP. Use ONLY official WhatsApp formatting (per Green API docs):
- Bold: *text* (asterisk each side)
- Italic: _text_ (underscore each side)
- Strikethrough: ~text~ (tilde each side)
- Monospace: ```text``` (triple backtick each side)
- Inline code: `text` (single backtick each side)
FORBIDDEN: Markdown (## ** ---). No "I can assist...". No "let me know", "if you're interested", "need something else" - end with the answer, nothing after.
Capability questions: 6 items, one line. "*Draft* *Topics* *Improve* *Images* *Ideas* *Search*".
Each message: 1-2 sentences max. No closing offers ever.
QUOTED REPLIES: When the user replies to a specific message (uses Reply), your response is sent as a quoted reply to keep the thread clear. Otherwise replies are sent as new messages."""

_cache: dict[str, str] = {}
_cache_loaded = False


def _load_all_prompts() -> dict[str, str]:
    """Load all prompts from DB, cache result."""
    global _cache, _cache_loaded
    if _cache_loaded:
        return _cache
    try:
        from src.db import get_client
        client = get_client()
        r = client.table("prompts").select("key, content").execute()
        if r.data:
            for row in r.data:
                k = str(row.get("key", ""))
                v = row.get("content")
                if v is not None:
                    _cache[k] = str(v)
    except Exception:
        pass
    _cache_loaded = True
    return _cache


def get_prompt(key: str, default: Optional[str] = None) -> str:
    """Get prompt content: DB first, then fallback."""
    prompts = _load_all_prompts()
    if key in prompts and prompts[key]:
        return prompts[key]
    if default is not None:
        return default
    if key == "playbooks":
        return _get_playbooks_fallback()
    if key == "master_system_core":
        return _get_master_core_fallback()
    if key == "whatsapp_format":
        return _get_whatsapp_fallback()
    return _FALLBACKS.get(key, "")


def get_master_system_prompt() -> str:
    """Return full master system prompt with playbooks injected."""
    core = get_prompt("master_system_core")
    playbooks = get_prompt("playbooks")
    return core.replace("{{PLAYBOOKS}}", playbooks)


def invalidate_prompts_cache() -> None:
    """Invalidate cache so next get_prompt reloads from DB."""
    global _cache, _cache_loaded
    _cache = {}
    _cache_loaded = False


def get_all_prompts() -> dict[str, str]:
    """Return all prompts as key->content. For admin API."""
    result = {}
    for k in _PROMPT_KEYS:
        result[k] = get_prompt(k)
    return result

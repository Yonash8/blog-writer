from __future__ import annotations
"""Prompts loader. Single source of truth: PromptLayer registry."""

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# PromptLayer prompt name prefix (e.g. "blog_writer/master_system_core")
_PL_PREFIX = os.getenv("PROMPTLAYER_PROMPT_PREFIX", "blog_writer")

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
    "tool_descriptions",   # JSON map: {tool_name: description_string} — editable by optimizer
)

_pl_cache: dict[str, str] = {}


def _fetch_from_promptlayer(key: str) -> Optional[str]:
    """Fetch a prompt template from PromptLayer registry. Returns content string or None."""
    api_key = os.getenv("PROMPTLAYER_API_KEY")
    if not api_key:
        logger.error("[PROMPTS] PROMPTLAYER_API_KEY not set — cannot load prompts")
        return None
    prompt_name = f"{_PL_PREFIX}/{key}" if _PL_PREFIX else key
    if prompt_name in _pl_cache:
        return _pl_cache[prompt_name] or None
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(
                "https://api.promptlayer.com/rest/get-prompt-template",
                headers={"X-API-KEY": api_key},
                params={"prompt_name": prompt_name},
            )
        if r.status_code != 200:
            logger.warning("[PROMPTS] PromptLayer returned %s for %r", r.status_code, prompt_name)
            _pl_cache[prompt_name] = ""
            return None
        data = r.json()
        template = data.get("prompt_template") or data.get("template") or {}
        messages = template.get("messages") or []
        if messages:
            parts = []
            for m in messages:
                # PromptLayer LangChain format: {"role": ..., "prompt": {"template": "..."}}
                prompt_obj = m.get("prompt") or {}
                text = prompt_obj.get("template") or m.get("content") or ""
                if text:
                    parts.append(text)
            content = "\n\n".join(parts)
            if content:
                _pl_cache[prompt_name] = content
                logger.debug("[PROMPTS] Loaded %r from PromptLayer", prompt_name)
                return content
        # Fallback: raw string template
        raw = template.get("template") or template.get("content") or ""
        if raw:
            _pl_cache[prompt_name] = raw
            return raw
        logger.warning("[PROMPTS] Empty content from PromptLayer for %r", prompt_name)
        _pl_cache[prompt_name] = ""
        return None
    except Exception as e:
        logger.error("[PROMPTS] PromptLayer fetch failed for %r: %s", prompt_name, e)
        _pl_cache[prompt_name] = ""
        return None


def get_prompt(key: str, default: Optional[str] = None) -> str:
    """Get prompt content from PromptLayer registry."""
    val = _fetch_from_promptlayer(key)
    if val:
        return val
    if default is not None:
        return default
    logger.warning("[PROMPTS] No prompt found for %r", key)
    return ""


def get_master_system_prompt() -> str:
    """Return full master system prompt with playbooks injected."""
    core = get_prompt("master_system_core")
    playbooks = get_prompt("playbooks")
    return core.replace("{{PLAYBOOKS}}", playbooks)


def load_all_prompts() -> None:
    """Eagerly fetch all prompts from PromptLayer into memory. Call at app startup."""
    from concurrent.futures import ThreadPoolExecutor
    keys = list(_PROMPT_KEYS)
    with ThreadPoolExecutor(max_workers=len(keys)) as pool:
        pool.map(_fetch_from_promptlayer, keys)
    loaded = [k for k in keys if _pl_cache.get(f"{_PL_PREFIX}/{k}" if _PL_PREFIX else k)]
    logger.info("[PROMPTS] Loaded %d/%d prompts at startup: %s", len(loaded), len(keys), loaded)


def invalidate_prompts_cache() -> None:
    """Invalidate cache so next get_prompt reloads from PromptLayer."""
    global _pl_cache
    _pl_cache = {}


def get_all_prompts() -> dict[str, str]:
    """Return all prompts as key->content. For admin API."""
    result = {}
    for k in _PROMPT_KEYS:
        result[k] = get_prompt(k)
    return result


def get_tool_descriptions() -> dict[str, str]:
    """Return the PromptLayer tool_descriptions map as {tool_name: description}.

    Falls back to empty dict if not set (hardcoded defaults in agent.py are used).
    The JSON stored in PromptLayer is expected to be: {"tool_name": "description", ...}
    """
    import json as _json
    raw = get_prompt("tool_descriptions")
    if not raw:
        return {}
    try:
        data = _json.loads(raw)
        if isinstance(data, dict):
            return {k: str(v) for k, v in data.items() if k and v}
    except Exception as e:
        logger.warning("[PROMPTS] Failed to parse tool_descriptions JSON: %s", e)
    return {}

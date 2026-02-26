"""Central agent config loader. Reads from DB (agent_config table), falls back to env, then defaults."""

import os
from typing import Optional

# Env key mapping for fallback when DB has no row
_ENV_KEYS: dict[str, str] = {
    "agent_model": "AGENT_MODEL",
    "agent_history_limit": "AGENT_HISTORY_LIMIT",
    "deep_research_model": "DEEP_RESEARCH_MODEL",
    "image_model_generic": "IMAGE_MODEL_GENERIC",
    "image_model_hero": "IMAGE_MODEL_HERO",
    "infographic_analysis_model": "INFOGRAPHIC_ANALYSIS_MODEL",
    "article_write_model": "ARTICLE_WRITE_MODEL",
    "image_placement_model": "IMAGE_PLACEMENT_MODEL",
}

_DEFAULTS: dict[str, str] = {
    "agent_model": "claude-sonnet-4-5",
    "agent_history_limit": "20",
    "deep_research_model": "o3-deep-research",
    "image_model_generic": "imagen-3.0-generate-002",
    "image_model_hero": "gemini-2.5-flash-image",
    "infographic_analysis_model": "claude-sonnet-4-20250514",
    "article_write_model": "gpt-4o",
    "image_placement_model": "claude-haiku-4-5-20251001",
}

_cache: dict[str, str] = {}
_cache_loaded = False


def _load_all_config() -> dict[str, str]:
    """Load all config from DB, cache result."""
    global _cache, _cache_loaded
    if _cache_loaded:
        return _cache
    try:
        from src.db import get_client
        client = get_client()
        r = client.table("agent_config").select("key, value").execute()
        if r.data:
            for row in r.data:
                _cache[str(row.get("key", ""))] = str(row.get("value", ""))
    except Exception:
        pass
    _cache_loaded = True
    return _cache


def get_config(key: str, default: Optional[str] = None) -> str:
    """Get config value: DB first, then env, then default."""
    config = _load_all_config()
    if key in config and config[key]:
        return config[key]
    env_key = _ENV_KEYS.get(key)
    if env_key:
        val = os.getenv(env_key)
        if val is not None and val != "":
            return val
    if default is not None:
        return default
    return _DEFAULTS.get(key, "")


def get_config_int(key: str, default: int = 0) -> int:
    """Get config as int (e.g. agent_history_limit)."""
    try:
        return int(get_config(key, str(default)))
    except ValueError:
        return default


def invalidate_config_cache() -> None:
    """Invalidate cache so next get_config reloads from DB. Call after admin update."""
    global _cache, _cache_loaded
    _cache = {}
    _cache_loaded = False


def get_all_config() -> dict[str, str]:
    """Return all config as key->value. For admin API. DB overrides env over defaults."""
    result = {}
    for k in _DEFAULTS:
        result[k] = get_config(k, _DEFAULTS[k])
    return result

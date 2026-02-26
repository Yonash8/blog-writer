#!/usr/bin/env python3
"""
Seed model metadata into PromptLayer prompts so the app reads model from PL.

Usage:
    python3 scripts/seed_pl_models.py

Publishes a new version of each prompt with metadata.model set.
Run after adding new prompts or when models need to be updated in PL.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

PL_API_KEY = os.getenv("PROMPTLAYER_API_KEY")
PREFIX = os.getenv("PROMPTLAYER_PROMPT_PREFIX", "blog_writer")
PL_GET = "https://api.promptlayer.com/rest/get-prompt-template"
PL_PUBLISH = "https://api.promptlayer.com/rest/publish-prompt-template"

# prompt_key -> metadata.model (provider, name, parameters)
MODEL_CONFIG: dict[str, dict] = {
    "master_system_core": {
        "provider": "anthropic",
        "name": "claude-sonnet-4-5",
        "parameters": {"max_tokens": 8192},
    },
    "improve_article": {
        "provider": "anthropic",
        "name": "claude-sonnet-4-5",
        "parameters": {"max_tokens": 8192},
    },
    "image_placement": {
        "provider": "anthropic",
        "name": "claude-haiku-4-5-20251001",
        "parameters": {"max_tokens": 1024},
    },
    "infographic_analysis": {
        "provider": "anthropic",
        "name": "claude-sonnet-4-5",
        "parameters": {"max_tokens": 1024},
    },
    "hero_image": {
        "provider": "google",
        "name": "gemini-2.5-flash-image",
        "parameters": {},
    },
    "infographic_generation": {
        "provider": "google",
        "name": "gemini-2.5-flash-image",
        "parameters": {},
    },
}


def _simplify_messages(messages: list) -> list:
    """Convert PL format to simple {role, content} for republish."""
    out = []
    for m in messages:
        role = m.get("role", "system")
        content = ""
        if "prompt" in m:
            p = m["prompt"]
            content = p.get("template", p.get("content", ""))
        else:
            content = m.get("content", "")
        out.append({"role": role, "content": content})
    return out


def seed_prompt(key: str, metadata: dict) -> bool:
    """Publish a new version with metadata.model. Returns True on success."""
    prompt_name = f"{PREFIX}/{key}" if PREFIX else key
    try:
        r = httpx.get(
            PL_GET,
            headers={"X-API-KEY": PL_API_KEY},
            params={"prompt_name": prompt_name},
            timeout=10,
        )
        if r.status_code != 200:
            print(f"  SKIP {key}: not found ({r.status_code})")
            return False

        data = r.json()
        tmpl = data.get("prompt_template") or {}
        messages = tmpl.get("messages") or []
        input_vars = tmpl.get("input_variables") or []

        simple_messages = _simplify_messages(messages)
        if not any(m.get("content") for m in simple_messages):
            print(f"  SKIP {key}: no content")
            return False

        body = {
            "prompt_name": prompt_name,
            "prompt_template": {
                "messages": simple_messages,
                "input_variables": input_vars,
            },
            "metadata": {"model": metadata},
            "tags": ["blog-writer", "model-config"],
            "api_key": PL_API_KEY,
        }

        r2 = httpx.post(PL_PUBLISH, json=body, timeout=15)
        r2.raise_for_status()
        resp = r2.json()
        if resp.get("id") or resp.get("success") is not False:
            print(f"  OK  {key:30s}  model={metadata['name']}")
            return True
        print(f"  ERR {key}: {resp}")
        return False
    except Exception as e:
        print(f"  ERR {key}: {e}")
        return False


def main():
    if not PL_API_KEY:
        print("ERROR: PROMPTLAYER_API_KEY not set in .env")
        sys.exit(1)

    print(f"Seeding model metadata to PromptLayer (prefix: {PREFIX!r})")
    print("-" * 60)

    ok = 0
    for key, meta in MODEL_CONFIG.items():
        if seed_prompt(key, meta):
            ok += 1

    print(f"\nDone: {ok}/{len(MODEL_CONFIG)} prompts updated")
    print("Prompts without models (playbooks, whatsapp_format, etc.) are text-only — no model needed.")


if __name__ == "__main__":
    main()

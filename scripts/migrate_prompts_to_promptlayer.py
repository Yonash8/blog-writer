#!/usr/bin/env python3
"""
Seed prompts into PromptLayer from local source files.

Usage:
    python3 scripts/migrate_prompts_to_promptlayer.py

Reads prompts from src/prompts.py and src/playbooks.md, publishes them
to PromptLayer under "blog_writer/<key>". Re-running pushes a new version.

After initial seeding, PromptLayer is the single source of truth —
edit prompts directly in the PromptLayer dashboard.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

PROMPTLAYER_API_KEY = os.getenv("PROMPTLAYER_API_KEY")
PREFIX = os.getenv("PROMPTLAYER_PROMPT_PREFIX", "blog_writer")
PL_PUBLISH_URL = "https://api.promptlayer.com/rest/publish-prompt-template"


def publish(name: str, content: str, tags: list[str] | None = None,
            llm_kwargs: dict | None = None) -> bool:
    """Publish a prompt template to PromptLayer. Returns True on success."""
    if not content.strip():
        print(f"  SKIP {name!r} (empty content)")
        return False

    template: dict = {
        "messages": [{"role": "system", "content": content}],
        "input_variables": list(dict.fromkeys(re.findall(r"\{(\w+)\}", content))),
    }
    if llm_kwargs:
        template["llm_kwargs"] = llm_kwargs

    payload = {
        "prompt_name": f"{PREFIX}/{name}" if PREFIX else name,
        "prompt_template": template,
        "tags": tags or ["blog-writer"],
        "api_key": PROMPTLAYER_API_KEY,
    }
    try:
        r = httpx.post(PL_PUBLISH_URL, json=payload, timeout=15.0)
        r.raise_for_status()
        data = r.json()
        if data.get("id") or data.get("success") is not False:
            print(f"  OK  {PREFIX}/{name}")
            return True
        print(f"  ERR {PREFIX}/{name}: {data}")
        return False
    except Exception as e:
        print(f"  ERR {PREFIX}/{name}: {e}")
        return False


def _get_prompts() -> dict[str, str]:
    """Collect all prompts from local source files."""
    prompts: dict[str, str] = {}

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import prompts as _p

    # Pipeline prompts from src/prompts.py
    prompts["deep_research"] = _p.DEEP_RESEARCH_PROMPT
    prompts["image_placement"] = _p.IMAGE_PLACEMENT_PROMPT
    prompts["hero_image"] = _p.HERO_IMAGE_PROMPT_TEMPLATE
    prompts["infographic_analysis"] = _p.INFOGRAPHIC_ANALYSIS_PROMPT
    prompts["infographic_generation"] = _p.INFOGRAPHIC_GENERATION_PROMPT_TEMPLATE
    prompts["improve_article"] = _p.IMPROVE_ARTICLE_PROMPT

    # Playbooks from src/playbooks.md
    playbooks_path = Path(__file__).resolve().parent.parent / "src" / "playbooks.md"
    if playbooks_path.exists():
        prompts["playbooks"] = playbooks_path.read_text(encoding="utf-8")

    return prompts


def _get_tool_descriptions() -> str:
    """Extract top-level tool descriptions from TOOL_DEFINITIONS in agent.py.

    Returns a JSON string: {"tool_name": "description", ...}
    This becomes the initial seed for the PromptLayer tool_descriptions prompt.
    The optimizer can then rewrite individual descriptions to improve tool invocation.
    """
    import json
    from src.agent import TOOL_DEFINITIONS

    descriptions: dict[str, str] = {}
    for tool_def in TOOL_DEFINITIONS:
        fn = tool_def.get("function", {})
        name = fn.get("name", "")
        desc = fn.get("description", "")
        if name and desc:
            # Normalise multi-line strings to a single clean string
            descriptions[name] = " ".join(str(desc).split())

    return json.dumps(descriptions, indent=2, ensure_ascii=False)


def main():
    if not PROMPTLAYER_API_KEY:
        print("ERROR: PROMPTLAYER_API_KEY not set in .env")
        sys.exit(1)

    print(f"Seeding prompts to PromptLayer (prefix: {PREFIX!r})")
    print("-" * 60)

    prompts = _get_prompts()

    # Add tool_descriptions as a special JSON prompt
    print("\nExtracting tool descriptions from TOOL_DEFINITIONS...")
    tool_desc_json = _get_tool_descriptions()
    prompts["tool_descriptions"] = tool_desc_json

    print(f"\nFound {len(prompts)} prompts to publish:\n")

    # llm_kwargs to seed alongside master_system_core prompt text
    _MASTER_LLM_KWARGS = {"model": "claude-sonnet-4-5", "max_tokens": 8192}

    ok = 0
    fail = 0
    for key, content in sorted(prompts.items()):
        kw = _MASTER_LLM_KWARGS if key == "master_system_core" else None
        if publish(key, content, tags=["blog-writer", "seeded"], llm_kwargs=kw):
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} published, {fail} failed/skipped")
    print(f"\nNOTE: master_system_core and whatsapp_format are managed directly")
    print(f"in PromptLayer — edit them at https://dashboard.promptlayer.com/registry")
    print(f"\ntool_descriptions is now seeded. The optimizer can rewrite individual")
    print(f"tool descriptions; changes take effect on the next server restart.")


if __name__ == "__main__":
    main()

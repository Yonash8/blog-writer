#!/usr/bin/env python3
"""
Fetch the orchestrator prompt template from PromptLayer and add the
{{context}}, {{history}}, and {{user}} variable placeholders.

Usage:
    python3 scripts/update_orchestrator_variables.py [--dry-run]

Fetches the current blog_writer/master_system_core prompt, appends
{{context}} to the system message, and sets the user message to:

    {{history}}

    {{user}}

Then publishes the updated version back to PromptLayer.
Use --dry-run to preview changes without publishing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_KEY = os.getenv("PROMPTLAYER_API_KEY")
PREFIX = os.getenv("PROMPTLAYER_PROMPT_PREFIX", "blog_writer")
PROMPT_NAME = f"{PREFIX}/master_system_core"

DRY_RUN = "--dry-run" in sys.argv


def fetch_template(name: str) -> dict:
    with httpx.Client(timeout=10.0) as client:
        r = client.get(
            "https://api.promptlayer.com/rest/get-prompt-template",
            headers={"X-API-KEY": API_KEY},
            params={"prompt_name": name},
        )
    if r.status_code != 200:
        print(f"ERROR fetching {name!r}: {r.status_code} {r.text[:200]}")
        sys.exit(1)
    return r.json()


def publish_template(name: str, messages: list[dict]) -> None:
    input_vars = []
    for m in messages:
        content = m.get("content") or m.get("prompt", {}).get("template", "")
        import re
        input_vars += re.findall(r"\{\{(\w+)\}\}", content)
    input_vars = list(dict.fromkeys(input_vars))  # deduplicate, preserve order

    body = {
        "prompt_name": name,
        "prompt_template": {
            "type": "chat",
            "messages": messages,
            "input_variables": input_vars,
        },
    }
    if DRY_RUN:
        import json
        print("\n--- DRY RUN: would publish ---")
        print(json.dumps(body, indent=2))
        return

    with httpx.Client(timeout=10.0) as client:
        r = client.post(
            "https://api.promptlayer.com/rest/publish-prompt-template",
            headers={"X-API-KEY": API_KEY},
            json=body,
        )
    if r.status_code not in (200, 201):
        print(f"ERROR publishing: {r.status_code} {r.text[:300]}")
        sys.exit(1)
    print(f"OK: published new version of {name!r}")


if not API_KEY:
    print("ERROR: PROMPTLAYER_API_KEY not set in .env")
    sys.exit(1)

print(f"Fetching {PROMPT_NAME!r} from PromptLayer...")
data = fetch_template(PROMPT_NAME)

template = data.get("prompt_template") or data.get("template") or {}
messages = template.get("messages") or []

if not messages:
    print(f"ERROR: no messages found in template for {PROMPT_NAME!r}")
    print("Raw response:", data)
    sys.exit(1)

# Extract current system message text
system_text = ""
for m in messages:
    if m.get("role") == "system":
        prompt_obj = m.get("prompt") or {}
        system_text = prompt_obj.get("template") or m.get("content") or ""
        break

if not system_text:
    print("ERROR: no system message found in template")
    sys.exit(1)

print(f"Current system message: {len(system_text)} chars")

# Append {{context}} if not already present
if "{{context}}" not in system_text:
    system_text = system_text.rstrip() + "\n\n{{context}}"
    print("Added {{context}} to system message")
else:
    print("{{context}} already present in system message")

# Build user message with {{history}} + {{user}}
user_text = "{{history}}\n\n{{user}}"

new_messages = [
    {"role": "system", "content": system_text},
    {"role": "user", "content": user_text},
]

print(f"New user message: {user_text!r}")
print(f"Input variables: user, context, history")
print()

publish_template(PROMPT_NAME, new_messages)

#!/usr/bin/env python3
"""Optimize from Fly.io logs — No traces

Fetches the last 15 minutes of Fly logs, analyzes them with Claude,
and produces findings. No Supabase traces, no WhatsApp.

Usage:
    python scripts/optimize_from_fly_logs.py
    python scripts/optimize_from_fly_logs.py --minutes 30
    python scripts/optimize_from_fly_logs.py --dry-run
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

# ── CLI args ────────────────────────────────────────────────────────────────
DRY_RUN = "--dry-run" in sys.argv
MINUTES = int(next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--minutes"), 15))

# ── Config ──────────────────────────────────────────────────────────────────
PL_API_KEY = os.environ.get("PROMPTLAYER_API_KEY", "")
PL_PREFIX = os.getenv("PROMPTLAYER_PROMPT_PREFIX", "blog_writer")
PL_BASE = "https://api.promptlayer.com/rest"
OPTIMIZER_MODEL = os.getenv("OPTIMIZER_MODEL", "claude-opus-4-6")
FLY_APP = os.getenv("FLY_APP", "blog-writer")
OUTPUT_FILE = ROOT / "data" / "fly_log_analysis.md"

# Max log chars to send to model (avoid token overflow)
MAX_LOG_CHARS = 80_000

_PROMPT_KEYS = [
    "master_system_core",
    "playbooks",
]

_PROMPT_LIMITS = {
    "master_system_core": 4000,
    "playbooks": 3000,
}

LOG_ANALYSIS_PROMPT = """\
You are an AI performance engineer analyzing application logs from a blog writing agent.

Below are production logs from Fly.io for the last {minutes} minutes. Look for:
- **Errors and exceptions** — tool failures, API timeouts, stack traces
- **Inefficiencies** — redundant calls, wasted work, slow paths
- **Anomalies** — unexpected behavior, retries, fallbacks
- **Accuracy failures** — agent confused, refuses to execute, loses context:
  - User asks for infographic/hero/image but agent asks "which article?" instead of calling generate_infographic/generate_hero_image (agent had article_id in context)
  - Agent asks clarifying questions when it could resolve from context (e.g. "the article", current article)
  - Agent fails to understand current mission and responds with text instead of calling the tool
  - Context loss: article_id, pending images, or conversation state dropped between turns

Be specific. Only report real patterns you can see in the logs.

---

## Fly Logs (last {minutes} minutes, {n} lines)

```
{logs}
```

## Changeable Prompts (for context; only suggest prompt changes if logs support it)
{master_system}
{playbooks}

---

## Your Output Format

### ACCURACY FINDINGS
(Issues causing failures, misrouting, or requiring repeats)
For each issue — or "No accuracy issues found." if none:
**[N]** Title
- Data: what in the logs shows this
- Root cause: one sentence
- Fix type: prompt_change | code_change | investigate

### EFFICIENCY FINDINGS
(Wasted work, redundant calls, slow paths)
For each — or "No efficiency issues found." if none:
**[N]** Title
- Data: log evidence
- Root cause: one sentence
- Fix type: prompt_change | code_change | investigate

### ACTION ITEMS JSON
```json
[
  {{"id": 1, "goal": "accuracy"|"efficiency", "title": "...", "impact": "...", "type": "prompt_change"|"code_change"|"investigate", "prompt_key": "..."}}
]
```
"""


def fetch_fly_logs(minutes: int) -> list[dict]:
    """Fetch Fly logs via flyctl, filter to last N minutes."""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    cutoff_str = cutoff.isoformat().replace("+00:00", "Z")

    try:
        proc = subprocess.run(
            ["flyctl", "logs", "-a", FLY_APP, "--no-tail", "-j"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=ROOT,
        )
    except FileNotFoundError:
        print("  flyctl not found. Install: https://fly.io/docs/hands-on/install-flyctl/")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("  flyctl logs timed out")
        sys.exit(1)

    if proc.returncode != 0:
        print(f"  flyctl logs failed: {proc.stderr}")
        sys.exit(1)

    lines = []
    for raw in proc.stdout.strip().split("\n"):
        raw = raw.strip()
        if not raw:
            continue
        try:
            entry = json.loads(raw)
            ts_str = entry.get("timestamp") or entry.get("time") or ""
            if ts_str and ts_str >= cutoff_str:
                lines.append(entry)
        except json.JSONDecodeError:
            lines.append({"raw": raw})

    return sorted(lines, key=lambda e: e.get("timestamp", "") or e.get("raw", ""))


def format_logs(entries: list[dict], max_chars: int) -> str:
    """Format log entries into a readable string, truncating if needed."""
    lines = []
    for e in entries:
        ts = (e.get("timestamp") or "").replace("Z", "Z").split(".")[0]
        lvl = e.get("level", "")
        msg = e.get("message", e.get("raw", ""))
        inst = e.get("instance", "")
        region = e.get("region", "")
        line = f"[{ts}] {lvl} "
        if inst:
            line += f"[{inst[:8]}] "
        if region:
            line += f"({region}) "
        line += msg
        lines.append(line)

    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[: max_chars - 200] + "\n\n... [truncated]"
    return out


def fetch_prompt(key: str) -> str:
    if not PL_API_KEY:
        return ""
    prompt_name = f"{PL_PREFIX}/{key}" if PL_PREFIX else key
    try:
        r = httpx.get(
            f"{PL_BASE}/get-prompt-template",
            headers={"X-API-KEY": PL_API_KEY},
            params={"prompt_name": prompt_name},
            timeout=10.0,
        )
        r.raise_for_status()
        data = r.json()
        template = data.get("prompt_template") or data.get("template") or {}
        messages = template.get("messages") or []
        parts = []
        for m in messages:
            text = (m.get("prompt") or {}).get("template") or m.get("content") or ""
            if text:
                parts.append(text)
        return "\n\n".join(parts) or template.get("template") or ""
    except Exception as e:
        print(f"  WARN could not fetch {key!r}: {e}")
        return ""


def run_analysis(minutes: int, logs_text: str, prompts: dict[str, str]) -> str:
    from anthropic import Anthropic

    master = (prompts.get("master_system_core") or "(not fetched)")[: _PROMPT_LIMITS["master_system_core"]]
    playbooks = (prompts.get("playbooks") or "(not fetched)")[: _PROMPT_LIMITS["playbooks"]]

    prompt = LOG_ANALYSIS_PROMPT.format(
        minutes=minutes,
        n=logs_text.count("\n") + 1,
        logs=logs_text,
        master_system=master or "(none)",
        playbooks=playbooks or "(none)",
    )

    client = Anthropic()
    resp = client.messages.create(
        model=OPTIMIZER_MODEL,
        max_tokens=8096,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def main():
    print(f"[1/3] Fetching Fly logs (last {MINUTES} min)...")
    entries = fetch_fly_logs(MINUTES)
    if not entries:
        print("  No logs in this window.")
        sys.exit(1)
    logs_text = format_logs(entries, MAX_LOG_CHARS)
    print(f"  {len(entries)} log entries, {len(logs_text):,} chars")

    print("[2/3] Fetching prompts (optional)...")
    prompts = {}
    if PL_API_KEY:
        for k in _PROMPT_KEYS:
            prompts[k] = fetch_prompt(k)
    else:
        print("  PROMPTLAYER_API_KEY not set — skipping prompts")

    print(f"[3/3] Running analysis with {OPTIMIZER_MODEL}...")
    analysis = run_analysis(MINUTES, logs_text, prompts)

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = f"# Fly Log Analysis\nGenerated: {ts} | Window: last {MINUTES} min | {len(entries)} lines\n\n---\n\n"
    OUTPUT_FILE.write_text(header + analysis, encoding="utf-8")
    print(f"  Saved → {OUTPUT_FILE}")

    if DRY_RUN:
        print("  --dry-run: no WhatsApp (this mode never sends anyway)")
    print("\nDone.")


if __name__ == "__main__":
    main()

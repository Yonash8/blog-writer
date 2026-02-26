#!/usr/bin/env python3
"""Blog Writer Self-Optimizer — Production

Daily analysis of agent traces. Identifies inefficiencies across 3 goals:
accuracy, capabilities, and cost. Saves a session to Supabase and sends a
WhatsApp summary with numbered action items for user approval.

Usage:
    python scripts/self_optimize.py              # 24h analysis + WhatsApp notify
    python scripts/self_optimize.py --dry-run    # analysis only, no DB save or notify
    python scripts/self_optimize.py --window 48  # last 48 hours (default: 24)
    python scripts/self_optimize.py --n 200      # max traces to load (default: 200)
"""
from __future__ import annotations

import json
import os
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

# ── CLI args ────────────────────────────────────────────────────────────────
DRY_RUN     = "--dry-run" in sys.argv
WINDOW_H    = int(next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--window"), 24))
MAX_TRACES  = int(next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--n"), 200))

# ── Config ──────────────────────────────────────────────────────────────────
PL_API_KEY  = os.environ["PROMPTLAYER_API_KEY"]
PL_PREFIX   = os.getenv("PROMPTLAYER_PROMPT_PREFIX", "blog_writer")
PL_BASE     = "https://api.promptlayer.com/rest"
OPTIMIZER_MODEL = os.getenv("OPTIMIZER_MODEL", "claude-opus-4-6")
OUTPUT_FILE = ROOT / "data" / "optimized_prompts.md"

# WhatsApp (optional — skip notify if not configured)
WA_CHAT_ID    = os.getenv("WHATSAPP_ALLOWED_CHAT_ID", "")
WA_INSTANCE   = os.getenv("GREEN_API_INSTANCE_ID", "")
WA_TOKEN      = os.getenv("GREEN_API_TOKEN", "")

# ── Analysis prompt ──────────────────────────────────────────────────────────

ANALYSIS_PROMPT = """\
You are an AI performance engineer tasked with improving a blog writing agent.
You have access to real production telemetry from the last {window_h} hours.

Analyze this data across THREE goals: accuracy, capabilities, and cost.
Be specific and data-driven. Only report real patterns you can see in the data.

---

## Production Metrics (last {window_h}h, {n} sessions)
- Average cost per session: ${avg_cost:.4f}
- P90 cost per session: ${p90_cost:.4f}
- Average agent turns per session: {avg_steps:.1f}
- Average input tokens per session: {avg_input_tokens:,.0f}
- Tool error rate: {tool_error_rate:.1%} of tool calls failed
- Tools called: {tools_summary}
- Models used: {models_summary}

## Tool Failure Details
{tool_failures}

## Conversation Patterns (repeat requests = accuracy failures)
{conv_patterns}

## Prompt Version Performance
{version_perf}

## Previous Optimization History
{prev_session}

## Changeable Prompts (all stored in PromptLayer — all can be rewritten)

### master_system_core (agent personality, rules, DB schema, tool list)
```
{master_system}
```

### playbooks (step-by-step workflows)
```
{playbooks}
```

### improve_article (article revision instructions)
```
{improve_article}
```

### hero_image (hero image generation prompt)
```
{hero_image}
```

### infographic_analysis (infographic type detection)
```
{infographic_analysis}
```

### infographic_generation (infographic generation prompt)
```
{infographic_generation}
```

### image_placement (image placement logic)
```
{image_placement}
```

### whatsapp_format (WhatsApp formatting rules)
```
{whatsapp_format}
```

### tool_descriptions (JSON map of tool_name → description shown to the LLM)
```json
{tool_descriptions}
```

## Agent Config (stored in DB — values can be changed as config_change items)
{agent_config}

## Architecture Facts
- System prompt re-sent every agent turn (up to 10 turns per session)
- Turn 0 = claude-haiku (routing), turns 1+ = claude-sonnet (execution)
- Tool descriptions from `tool_descriptions` prompt override hardcoded defaults at startup
- PromptLayer SEO workflow handles article writing (separate black box)
- Research injected into 5 variables in PromptLayer workflow (known duplication)
- No Anthropic prompt caching in use

---

## Your Output Format

Produce your analysis in EXACTLY this structure:

### ACCURACY FINDINGS
(Issues causing the agent to fail, misroute, or require the user to repeat themselves)
For each issue found — or "No accuracy issues found in this window." if none:
**[N]** Title of issue
- Data: what in the traces shows this
- Root cause: one sentence
- Fix type: prompt_change | code_change | config_change | investigate
- If prompt_change: which prompt key

### CAPABILITY FINDINGS
(Gaps where the agent can't do things it should be able to)
For each issue found — or "No capability issues found in this window." if none:
**[N]** Title of issue
- Data: evidence from traces
- Root cause: one sentence
- Fix type: prompt_change | code_change | investigate
- If prompt_change: which prompt key

### COST FINDINGS
(Inefficiencies wasting tokens or money)
For each issue found — or "No cost issues found in this window." if none:
**[N]** Title of issue
- Data: estimated token/cost impact
- Root cause: one sentence
- Fix type: prompt_change | code_change | config_change
- If config_change: which key and proposed value

### REWRITTEN PROMPTS
Only include sections for prompt_change items. Use exact heading format below.

Available prompt keys: master_system_core, playbooks, improve_article, hero_image,
infographic_analysis, infographic_generation, image_placement, whatsapp_format, tool_descriptions

For master_system_core: {{PLAYBOOKS}} placeholder MUST be preserved exactly.
For tool_descriptions: output the FULL updated JSON object (all tools, not just changed ones).
Mark anything removed with [REMOVED: reason].

#### master_system_core (optimized)
[only if this prompt is being changed]

#### playbooks (optimized)
[only if this prompt is being changed]

#### improve_article (optimized)
[only if this prompt is being changed]

#### hero_image (optimized)
[only if this prompt is being changed]

#### infographic_analysis (optimized)
[only if this prompt is being changed]

#### infographic_generation (optimized)
[only if this prompt is being changed]

#### image_placement (optimized)
[only if this prompt is being changed]

#### whatsapp_format (optimized)
[only if this prompt is being changed]

#### tool_descriptions (optimized)
[only if any tool description is being changed — output FULL JSON with all tools]

### ACTION ITEMS JSON
Output a JSON array. Each item must have:
- "id": sequential int starting at 1
- "goal": "accuracy" | "capabilities" | "cost"
- "title": short string
- "impact": estimated impact string
- "type": "prompt_change" | "code_change" | "config_change" | "investigate"
- For prompt_change: "prompt_key" (one of the keys above)
- For prompt_change on master_system_core: optionally "model_override": "claude-sonnet-4-6" to also change the model (stored in llm_kwargs alongside the prompt text in PromptLayer)
- For config_change: "config_key" (DB key name) and "config_value" (new value as string)
- "auto_deployable": true if type is prompt_change or config_change (and rewrite/value provided)

```json
[...]
```

### FORECAST
Estimated total improvement from auto-deployable items: X% fewer input tokens, \
$Y savings per 100 sessions.
"""


# ── Data loading ─────────────────────────────────────────────────────────────

def load_traces(window_hours: int, max_n: int) -> tuple[list[dict], dict]:
    """Load traces from the last N hours with aggregate stats."""
    from src.db import execute_sql

    rows = execute_sql(
        "SELECT trace_id, created_at, user_message, "
        "(payload -> 'summary' ->> 'steps')::int AS steps, "
        "(payload -> 'summary' ->> 'total_input_tokens')::int AS input_tokens, "
        "(payload -> 'summary' ->> 'total_cost_usd')::float AS cost_usd, "
        "payload -> 'summary' -> 'tools_used' AS tools_used, "
        "payload -> 'summary' -> 'models_used' AS models_used, "
        "payload -> 'events' AS events "
        "FROM observability_traces "
        f"WHERE created_at > now() - interval '{window_hours} hours' "
        "AND payload -> 'summary' IS NOT NULL "
        f"ORDER BY created_at DESC LIMIT {max_n}"
    )

    costs, steps_list, inputs = [], [], []
    all_tools: list[str] = []
    all_models: set[str] = set()
    tool_calls_total = 0
    tool_errors_total = 0

    for r in rows:
        if r.get("cost_usd") is not None:
            costs.append(float(r["cost_usd"]))
        if r.get("steps") is not None:
            steps_list.append(int(r["steps"]))
        if r.get("input_tokens") is not None:
            inputs.append(int(r["input_tokens"]))

        tools = r.get("tools_used") or []
        if isinstance(tools, str):
            tools = json.loads(tools)
        all_tools.extend(tools)

        models = r.get("models_used") or []
        if isinstance(models, str):
            models = json.loads(models)
        all_models.update(models)

        # Count tool errors from events
        events = r.get("events") or []
        if isinstance(events, str):
            events = json.loads(events)
        for ev in events:
            if ev.get("event_type") == "tool_result":
                tool_calls_total += 1
                if not ev.get("success", True):
                    tool_errors_total += 1

    def _p90(lst: list) -> float:
        return sorted(lst)[int(0.9 * len(lst))] if lst else 0.0

    from collections import Counter
    tool_counts = Counter(all_tools)

    stats = {
        "n_traces":         len(rows),
        "avg_cost_usd":     statistics.mean(costs) if costs else 0.0,
        "p90_cost_usd":     _p90(costs),
        "avg_steps":        statistics.mean(steps_list) if steps_list else 0.0,
        "avg_input_tokens": statistics.mean(inputs) if inputs else 0.0,
        "tool_error_rate":  (tool_errors_total / tool_calls_total) if tool_calls_total else 0.0,
        "tools_summary":    ", ".join(f"{t}×{c}" for t, c in tool_counts.most_common(8)) or "none",
        "models_summary":   ", ".join(sorted(all_models)) or "none",
        "total_cost":       sum(costs),
    }
    return rows, stats


def load_tool_failures(window_hours: int) -> str:
    """Return a summary of failed tool calls in the window."""
    from src.db import execute_sql

    rows = execute_sql(
        "SELECT "
        "  event ->> 'tool_name' AS tool_name, "
        "  event ->> 'error' AS error, "
        "  COUNT(*) AS cnt "
        "FROM observability_traces, "
        "     jsonb_array_elements(payload -> 'events') AS event "
        f"WHERE created_at > now() - interval '{window_hours} hours' "
        "  AND event ->> 'event_type' = 'tool_result' "
        "  AND (event ->> 'success')::boolean = false "
        "GROUP BY tool_name, error "
        "ORDER BY cnt DESC "
        "LIMIT 10"
    )
    if not rows:
        return "No tool failures in this window."
    lines = []
    for r in rows:
        lines.append(f"- {r.get('tool_name','?')} failed {r.get('cnt','?')}x: {str(r.get('error',''))[:120]}")
    return "\n".join(lines)


def load_conversation_patterns(window_hours: int) -> str:
    """Detect users who sent repeated similar messages — signals accuracy failures."""
    from src.db import execute_sql

    rows = execute_sql(
        "SELECT channel_user_id, COUNT(*) AS msg_count "
        "FROM messages "
        f"WHERE created_at > now() - interval '{window_hours} hours' "
        "  AND role = 'user' "
        "GROUP BY channel_user_id "
        "HAVING COUNT(*) > 2 "
        "ORDER BY msg_count DESC LIMIT 5"
    )
    if not rows:
        return "No unusual conversation patterns detected."
    lines = ["Users with high message count (may indicate repeated requests):"]
    for r in rows:
        uid = str(r.get("channel_user_id", "?"))[-8:]
        lines.append(f"- user ...{uid}: {r.get('msg_count',0)} messages in {window_hours}h")
    return "\n".join(lines)


def load_version_performance() -> str:
    """Show cost comparison across prompt_version hashes if data exists."""
    from src.db import execute_sql

    rows = execute_sql(
        "SELECT "
        "  payload ->> 'prompt_version' AS version, "
        "  COUNT(*) AS runs, "
        "  ROUND(AVG((payload -> 'summary' ->> 'total_cost_usd')::float)::numeric, 4) AS avg_cost "
        "FROM observability_traces "
        "WHERE payload ->> 'prompt_version' IS NOT NULL "
        "  AND payload -> 'summary' IS NOT NULL "
        "GROUP BY version "
        "ORDER BY MIN(created_at) DESC "
        "LIMIT 5"
    )
    if not rows:
        return "No prompt_version data yet (will appear after next agent run)."
    lines = []
    for r in rows:
        lines.append(f"- version {r.get('version','?')}: {r.get('runs',0)} runs, avg cost ${r.get('avg_cost',0)}")
    return "\n".join(lines)


def load_previous_session() -> str:
    """Load the last deployed optimization session for loop closure."""
    from src.db import execute_sql

    rows = execute_sql(
        "SELECT id, created_at, status, action_items "
        "FROM optimization_sessions "
        "ORDER BY created_at DESC LIMIT 1"
    )
    if not rows:
        return "No previous optimization sessions."
    s = rows[0]
    items = s.get("action_items") or []
    if isinstance(items, str):
        items = json.loads(items)
    deployed = [i for i in items if i.get("type") == "prompt_change"]
    investigate = [i for i in items if i.get("type") != "prompt_change"]
    parts = [
        f"Last session: {s.get('created_at','?')[:10]} (status: {s.get('status','?')})",
    ]
    if deployed:
        parts.append(f"Previously deployed: {', '.join(i.get('title','?') for i in deployed)}")
    if investigate:
        parts.append(f"Previously flagged for investigation: {', '.join(i.get('title','?') for i in investigate)}")
    return "\n".join(parts)


# ── Prompt fetching ──────────────────────────────────────────────────────────

_PROMPT_KEYS = [
    "master_system_core",
    "playbooks",
    "improve_article",
    "hero_image",
    "infographic_analysis",
    "infographic_generation",
    "image_placement",
    "whatsapp_format",
    "tool_descriptions",
]

# Max chars per prompt in the analysis context (keeps total prompt manageable)
_PROMPT_LIMITS: dict[str, int] = {
    "master_system_core": 8000,
    "playbooks": 4000,
    "improve_article": 2000,
    "hero_image": 1500,
    "infographic_analysis": 1500,
    "infographic_generation": 1500,
    "image_placement": 1500,
    "whatsapp_format": 1000,
    "tool_descriptions": 3000,
}


def fetch_prompt(key: str) -> str:
    """Fetch prompt template text from PromptLayer. Returns empty string if not found."""
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
        print(f"  WARN could not fetch {prompt_name!r}: {e}")
        return ""


def load_all_prompts() -> dict[str, str]:
    """Fetch all PromptLayer prompts for the optimizer. Missing keys return ''."""
    prompts = {}
    for key in _PROMPT_KEYS:
        text = fetch_prompt(key)
        prompts[key] = text
        status = f"{len(text):,} chars" if text else "not found"
        print(f"  {key}: {status}")
    return prompts


def load_agent_config_text() -> str:
    """Return the current agent_config table values as a formatted string."""
    try:
        from src.config import get_all_config
        cfg = get_all_config()
        if not cfg:
            return "No agent_config entries found."
        lines = []
        for k, v in sorted(cfg.items()):
            lines.append(f"- {k}: {v!r}")
        return "\n".join(lines)
    except Exception as e:
        return f"Could not load agent_config: {e}"


# ── Analysis ─────────────────────────────────────────────────────────────────

def run_analysis(
    window_hours: int,
    stats: dict,
    tool_failures: str,
    conv_patterns: str,
    version_perf: str,
    prev_session: str,
    prompts: dict[str, str],
    agent_config: str,
) -> str:
    """Call Claude Opus to analyze traces and produce findings + action items."""
    from anthropic import Anthropic

    # Prompt key → ANALYSIS_PROMPT template variable name (most match 1:1)
    _KEY_TO_VAR = {
        "master_system_core": "master_system",  # prompt var is {master_system} for brevity
    }

    # Build format kwargs — truncate each prompt to its limit
    fmt: dict[str, object] = dict(
        window_h=window_hours,
        n=stats["n_traces"],
        avg_cost=stats["avg_cost_usd"],
        p90_cost=stats["p90_cost_usd"],
        avg_steps=stats["avg_steps"],
        avg_input_tokens=stats["avg_input_tokens"],
        tool_error_rate=stats["tool_error_rate"],
        tools_summary=stats["tools_summary"],
        models_summary=stats["models_summary"],
        tool_failures=tool_failures,
        conv_patterns=conv_patterns,
        version_perf=version_perf,
        prev_session=prev_session,
        agent_config=agent_config,
    )
    for key in _PROMPT_KEYS:
        limit = _PROMPT_LIMITS.get(key, 2000)
        text = prompts.get(key, "") or "(not configured in PromptLayer)"
        var_name = _KEY_TO_VAR.get(key, key)  # rename where template variable differs
        fmt[var_name] = text[:limit]

    client = Anthropic()
    prompt = ANALYSIS_PROMPT.format(**fmt)
    resp = client.messages.create(
        model=OPTIMIZER_MODEL,
        max_tokens=8096,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def parse_action_items(analysis: str) -> list[dict]:
    """Extract the JSON action items array from the analysis text."""
    m = re.search(r"### ACTION ITEMS JSON\s*```json\s*(.*?)\s*```", analysis, re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group(1))
        return items if isinstance(items, list) else []
    except json.JSONDecodeError:
        return []


# ── PromptLayer publish ───────────────────────────────────────────────────────

def publish_prompt_change(action_item: dict, analysis: str) -> bool:
    """Deploy a single prompt_change action item by publishing to PromptLayer."""
    prompt_key = action_item.get("prompt_key")
    if not prompt_key:
        return False

    section_label = f"{prompt_key} (optimized)"
    pat = rf"#### {re.escape(section_label)}\n(.*?)(?=\n####|\n###|\Z)"
    m = re.search(pat, analysis, re.DOTALL)
    if not m:
        print(f"  WARN could not extract section {section_label!r}")
        return False

    content = m.group(1).strip()
    if len(content) < 200:
        print(f"  WARN extracted content too short ({len(content)} chars) for {prompt_key!r}")
        return False

    if prompt_key == "master_system_core" and "{{PLAYBOOKS}}" not in content:
        print(f"  WARN {{{{PLAYBOOKS}}}} placeholder missing — skipping")
        return False

    input_vars = list(dict.fromkeys(re.findall(r"\{(\w+)\}", content)))
    body = {
        "prompt_name": f"{PL_PREFIX}/{prompt_key}" if PL_PREFIX else prompt_key,
        "prompt_template": {
            "messages": [{"role": "system", "content": content}],
            "input_variables": input_vars,
        },
        "tags": ["self-optimized"],
        "api_key": PL_API_KEY,
    }
    try:
        r = httpx.post(f"{PL_BASE}/publish-prompt-template", json=body, timeout=15.0)
        r.raise_for_status()
        data = r.json()
        if data.get("id") or data.get("success") is not False:
            print(f"  OK  published {PL_PREFIX}/{prompt_key}")
            return True
        print(f"  ERR {PL_PREFIX}/{prompt_key}: {data}")
        return False
    except Exception as e:
        print(f"  ERR {PL_PREFIX}/{prompt_key}: {e}")
        return False


# ── WhatsApp notification ─────────────────────────────────────────────────────

def _wa_send(message: str) -> bool:
    """Send a WhatsApp message via Green API. Returns True on success."""
    if not all([WA_CHAT_ID, WA_INSTANCE, WA_TOKEN]):
        print("  WhatsApp not configured (GREEN_API_INSTANCE_ID / GREEN_API_TOKEN / WHATSAPP_ALLOWED_CHAT_ID)")
        return False
    url = f"https://api.green-api.com/waInstance{WA_INSTANCE}/sendMessage/{WA_TOKEN}"
    try:
        r = httpx.post(url, json={"chatId": WA_CHAT_ID, "message": message}, timeout=10.0)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"  WhatsApp send failed: {e}")
        return False


def send_whatsapp_notification(session: dict, action_items: list, stats: dict) -> None:
    """Send a WhatsApp summary of the optimization session."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    accuracy  = [i for i in action_items if i.get("goal") == "accuracy"]
    caps      = [i for i in action_items if i.get("goal") == "capabilities"]
    cost      = [i for i in action_items if i.get("goal") == "cost"]
    deployable = [i for i in action_items if i.get("auto_deployable")]

    lines = [
        f"🔍 *Agent Self-Analysis* ({ts})",
        f"Window: last {WINDOW_H}h | {stats['n_traces']} sessions | avg cost ${stats['avg_cost_usd']:.4f}",
        "",
    ]

    if accuracy:
        lines.append(f"*Accuracy ({len(accuracy)} issue{'s' if len(accuracy) > 1 else ''}):*")
        for i in accuracy:
            lines.append(f"[{i['id']}] {i['title']} ({i.get('impact','')})")
        lines.append("")

    if caps:
        lines.append(f"*Capabilities ({len(caps)} issue{'s' if len(caps) > 1 else ''}):*")
        for i in caps:
            lines.append(f"[{i['id']}] {i['title']} ({i.get('impact','')})")
        lines.append("")

    if cost:
        lines.append(f"*Cost ({len(cost)} saving{'s' if len(cost) > 1 else ''}):*")
        for i in cost:
            lines.append(f"[{i['id']}] {i['title']} ({i.get('impact','')})")
        lines.append("")

    if not action_items:
        lines.append("✅ No significant issues found.")
    elif deployable:
        deploy_ids = [str(i["id"]) for i in deployable]
        lines.append(f"Auto-deployable (prompt changes): [{', '.join(deploy_ids)}]")
        lines.append("")
        lines.append("Reply to me:")
        lines.append('• "deploy all" — apply all prompt changes')
        lines.append(f'• "remove {deploy_ids[0]}, deploy" — skip an item then deploy')
        lines.append('• "skip" — discard this analysis')
    else:
        lines.append("No auto-deployable items — all require investigation or code changes.")

    message = "\n".join(lines)
    if _wa_send(message):
        print("  WhatsApp notification sent.")


# ── Output / persist ──────────────────────────────────────────────────────────

def save_output(analysis: str, stats: dict, action_items: list) -> None:
    """Write the full analysis report to data/optimized_prompts.md."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = (
        f"# Self-Optimization Report\n"
        f"Generated: {ts} | Window: {WINDOW_H}h | Model: {OPTIMIZER_MODEL}\n\n"
        f"**Baseline metrics ({stats['n_traces']} sessions):**\n"
        f"- Avg cost/session:    ${stats['avg_cost_usd']:.4f}\n"
        f"- P90 cost/session:    ${stats['p90_cost_usd']:.4f}\n"
        f"- Avg steps:           {stats['avg_steps']:.1f}\n"
        f"- Avg input tokens:    {stats['avg_input_tokens']:,.0f}\n"
        f"- Tool error rate:     {stats['tool_error_rate']:.1%}\n"
        f"- Action items found:  {len(action_items)}\n\n"
        f"---\n\n"
    )
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    OUTPUT_FILE.write_text(header + analysis, encoding="utf-8")
    print(f"  Saved → {OUTPUT_FILE}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"[1/5] Loading traces from last {WINDOW_H}h (max {MAX_TRACES})...")
    rows, stats = load_traces(WINDOW_H, MAX_TRACES)
    if not rows:
        print("  No traces found in this window.")
        sys.exit(1)
    print(
        f"  n={stats['n_traces']}  avg_cost=${stats['avg_cost_usd']:.4f}  "
        f"p90=${stats['p90_cost_usd']:.4f}  err_rate={stats['tool_error_rate']:.1%}"
    )

    print("[2/5] Loading supporting data...")
    tool_failures = load_tool_failures(WINDOW_H)
    conv_patterns = load_conversation_patterns(WINDOW_H)
    version_perf  = load_version_performance()
    prev_session  = load_previous_session()
    agent_config  = load_agent_config_text()

    print("[3/5] Fetching all prompts from PromptLayer...")
    prompts = load_all_prompts()

    print(f"[4/5] Running analysis with {OPTIMIZER_MODEL}...")
    analysis = run_analysis(
        WINDOW_H, stats, tool_failures, conv_patterns,
        version_perf, prev_session, prompts, agent_config,
    )
    action_items = parse_action_items(analysis)
    print(f"  Found {len(action_items)} action items")

    print("[5/5] Saving and notifying...")
    save_output(analysis, stats, action_items)

    if DRY_RUN:
        print("  --dry-run: skipped DB save and WhatsApp notification")
        print(f"\nReview {OUTPUT_FILE}")
        print("Re-run without --dry-run to save session and notify via WhatsApp.")
        return

    # Save session to Supabase
    from src.db import create_optimization_session, update_optimization_session
    try:
        session = create_optimization_session(
            window_hours=WINDOW_H,
            trace_count=stats["n_traces"],
            analysis_text=analysis,
            action_items=action_items,
            channel_user_id=WA_CHAT_ID or None,
        )
        session_id = session["id"]
        print(f"  Session saved: {session_id[:8]}...")

        # Mark as notified after sending WhatsApp
        send_whatsapp_notification(session, action_items, stats)
        update_optimization_session(session_id, {"notified_at": datetime.now(timezone.utc).isoformat()})
    except Exception as e:
        print(f"  DB save failed: {e}")
        # Still send WhatsApp even if DB save fails
        send_whatsapp_notification({}, action_items, stats)

    print("\nDone. Reply to the WhatsApp message to approve, modify, or skip.")


if __name__ == "__main__":
    main()

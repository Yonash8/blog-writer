# Self-Improvement Loop — Architecture & Operations Guide

## What This Is

A daily automated process that reads the agent's own production logs, identifies
inefficiencies across three goals, and surfaces proposed changes for human review.
The human approves via WhatsApp. Approved changes are deployed automatically.

---

## The Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  1. OBSERVE          2. ANALYZE          3. APPROVE             │
│  ──────────          ────────            ────────               │
│  Supabase traces  →  Claude Opus      →  You (WhatsApp)         │
│  (last 24h)          analyzes 3 goals    review & decide        │
│                                                                 │
│  4. DEPLOY           5. MEASURE                                 │
│  ────────            ───────                                     │
│  PromptLayer      →  Next analysis                              │
│  updated             shows before/after                         │
│                      by prompt_version                          │
└─────────────────────────────────────────────────────────────────┘
```

**Trigger:** GitHub Actions at 6am UTC daily (or on-demand via `POST /api/admin/optimize`).

---

## Three Goals

**Accuracy** — Is the agent completing tasks correctly on the first try?
- Detects: tool failures, error responses, users who sent many messages in one session
- Example finding: "`write_article` failed in 2 sessions without retry"

**Capabilities** — Is the agent limited by unclear tool descriptions or missing workflows?
- Detects: tools that are never called despite being relevant, high-error tools
- Example finding: "`inject_links` called 0 times — tool description may be unclear"

**Cost** — Is the agent wasting tokens?
- Detects: prompt verbosity, redundant context, inefficient model choices
- Example finding: "Playbooks injected twice in system prompt — 650 tokens wasted/turn"

---

## What the Optimizer Can Change Automatically

### ✅ Prompt changes (auto-deployable)
These are deployed to **PromptLayer** when you say "deploy":

| Prompt | What it controls |
|--------|-----------------|
| `master_system_core` | Agent personality, rules, DB schema, tool descriptions |
| `playbooks` | Step-by-step workflows (write article, approve image, push to Ghost, etc.) |

PromptLayer keeps full version history. You can roll back any change at
`dashboard.promptlayer.com/registry` instantly.

### ⚠️ Config changes (semi-automatic)
The optimizer may recommend changing values in the `agent_config` Supabase table
(e.g., switch routing model, reduce history limit). These require you to apply via
the admin UI or directly in Supabase — the optimizer flags them but doesn't apply them.

### ❌ Code changes (manual only)
The optimizer identifies but **cannot apply** these:

- Python source changes (`src/agent.py`, `src/tools.py`, etc.)
- Adding Anthropic prompt caching headers to API calls
- Adding new tools
- Changing PromptLayer SEO workflow inputs (the 5x research duplication bug)
- DB migrations

These are reported in the action items with `"type": "code_change"` and won't appear
in the auto-deploy list.

---

## How a Session Works End to End

### Day 1 (6am UTC) — Analysis runs automatically

GitHub Actions runs `scripts/self_optimize.py --window 24 --n 200`.

The script:
1. Queries Supabase for the last 24h of traces
2. Queries `messages` table for conversation patterns
3. Fetches `master_system_core` + `playbooks` from PromptLayer
4. Calls `claude-opus-4-6` with all this data
5. Saves an `optimization_sessions` row in Supabase (status: `pending`)
6. Sends you a WhatsApp message like:

```
🔍 Agent Self-Analysis (2026-02-26)
Window: last 24h | 12 sessions | avg cost $0.044

Accuracy (1 issue):
[1] write_article failed twice without retry (~2 extra sessions wasted)

Capabilities (1 issue):
[2] inject_links never called — tool description may be unclear

Cost (3 savings):
[3] Playbooks duplicated in master_system_core (~650 tokens/turn)
[4] Ghost block sent every turn regardless of task
[5] Research injected 5x in PromptLayer workflow (~120k tokens/30k research)

Auto-deployable (prompt changes): [3, 4]
Items 1, 2, 5 need investigation or code changes.

Reply to me:
• "deploy all" — apply all prompt changes
• "remove 3, deploy" — skip an item then deploy
• "skip" — discard this analysis
```

### Day 1 — You reply via WhatsApp

The production agent is already running. When you reply, it checks for a pending
optimization session in its context and calls `manage_optimization`.

**"deploy all"** → agent deploys items 3 and 4 (prompt changes only), marks session deployed.

**"show me item 3"** → agent calls `manage_optimization(action='list')`, shows details.

**"remove 4, deploy"** → agent removes item 4 from the deploy list, deploys item 3 only.

**"skip"** → agent rejects the session, no changes made.

**"tell me more about item 2"** → agent describes the finding conversationally.
No tool call — just reading the injected context.

### Day 2 — Optimizer sees the improvement

The next analysis prompt includes:

```
## Prompt Version Performance
version a3f9b1c2: avg $0.044, 15 runs (before)
version b7d2e891: avg $0.031,  8 runs (after optimization)
Improvement: -29.5% ✅

## Previous Optimization History
Last session: 2026-02-26 (status: deployed)
Previously deployed: Playbooks duplication fix, Ghost block conditional
Previously flagged for investigation: write_article retry logic, inject_links description
```

The optimizer now knows its history and doesn't repeat the same findings.

---

## Data the Optimizer Reads

| Source | What it reads |
|--------|--------------|
| `observability_traces` | Cost, steps, tokens, tool names, tool errors per session |
| `observability_traces.payload.events` | Per-tool success/failure, latency, error messages |
| `observability_traces.payload.prompt_version` | Which prompt version produced each trace |
| `messages` | User message counts per session (detects repeated requests) |
| `optimization_sessions` | Previous analysis history (loop closure) |
| PromptLayer | Current text of `master_system_core` and `playbooks` |

---

## What You Need to Do (One-Time Setup)

### 1. Run the Supabase migration

```bash
# If using Supabase CLI:
supabase db push

# Or run manually in Supabase SQL editor:
# Copy contents of supabase/migrations/20260225000000_add_optimization_sessions.sql
```

This creates the `optimization_sessions` table.

### 2. Add GitHub repository secrets

Go to your repo → Settings → Secrets and variables → Actions → New repository secret.

Add all of these:

| Secret | Where to find it |
|--------|-----------------|
| `SUPABASE_URL` | Supabase dashboard → Project Settings → API → Project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase dashboard → Project Settings → API → service_role key |
| `PROMPTLAYER_API_KEY` | promptlayer.com → Settings → API Keys |
| `ANTHROPIC_API_KEY` | console.anthropic.com → API Keys |
| `GREEN_API_INSTANCE_ID` | green-api.com → your instance ID |
| `GREEN_API_TOKEN` | green-api.com → your instance API token |
| `WHATSAPP_ALLOWED_CHAT_ID` | Your WhatsApp chat ID (same as in `.env`) |

Optional:
| Secret | Default |
|--------|---------|
| `OPTIMIZER_MODEL` | `claude-opus-4-6` |
| `PROMPTLAYER_PROMPT_PREFIX` | `blog_writer` |

### 3. Enable the GitHub Actions workflow

The workflow file is already at `.github/workflows/daily-optimize.yml`.
Push it to your main branch. GitHub will schedule it automatically at 6am UTC daily.

To trigger it manually: GitHub → Actions → "Daily Agent Self-Optimization" → Run workflow.

---

## What You Need to Do (Ongoing)

### Daily: review the WhatsApp notification

- Takes ~2 minutes
- Reply "deploy all", "remove X, deploy", or "skip"
- If no issues found, you'll get a message saying so (no action needed)

### When you get a `code_change` item: implement it manually

Example: the optimizer identifies that `write_article` should retry on failure.
It will say:

> [2] write_article doesn't retry on PromptLayer timeout
> Type: code_change
> Fix: In tools.py write_article(), add a retry loop with backoff when the pipeline returns a timeout error.

You implement this in `src/tools.py`, test it, and commit. The optimizer will check
next cycle whether the tool failure rate for `write_article` dropped.

### When you get a `config_change` item: update in Supabase or .env

Example: "Switch haiku routing model to claude-haiku-4-5".
Go to Supabase → `agent_config` table → update the `agent_model` key.
Or use `PATCH /api/admin/agent-config`.

---

## How Prompts Are Protected

Before deploying any prompt change, the `manage_optimization` tool checks:

1. **Content minimum length** — rewritten prompts must be at least 200 chars
   (catches parse failures where Claude's output was misread)

2. **`{{PLAYBOOKS}}` placeholder** — `master_system_core` must contain this
   exact string or the agent will lose all its playbooks

3. **Previous version preserved** — PromptLayer keeps every version.
   Roll back instantly at `dashboard.promptlayer.com/registry`.

If any check fails, that item is skipped and you're told why. Other items still deploy.

---

## Manual Trigger (Anytime)

Via API (requires `WEB_API_KEY` header):
```bash
curl -X POST https://your-app.fly.dev/api/admin/optimize \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"window_hours": 48, "dry_run": false}'
```

Via WhatsApp to the agent: *"run optimization"* (the agent will call the endpoint).

Via GitHub Actions UI: Actions → Daily Agent Self-Optimization → Run workflow.

Via local script:
```bash
python3 scripts/self_optimize.py --window 24 --dry-run   # preview only
python3 scripts/self_optimize.py --window 24              # full run + WhatsApp
```

---

## Limits of the System

| What it does | What it doesn't do |
|---|---|
| Analyzes prompts | Analyzes Python code (reads traces, not source) |
| Deploys prompt changes | Writes or commits code |
| Tracks cost per prompt version | Tracks article quality (no quality signal yet) |
| Detects tool failures | Understands why PromptLayer SEO workflow is slow |
| Closes the loop on past changes | Automatically validates changes before deploying |

The biggest missing piece is **output quality signal** — the optimizer can see that
the agent responded, but not whether the response was good. Adding a thumbs-up/thumbs-down
to the WhatsApp flow (saved in traces) would unlock accuracy analysis much richer
than what message-count patterns provide.

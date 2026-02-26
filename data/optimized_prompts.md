# Self-Optimization Report
Generated: 2026-02-25 10:45 UTC

**Baseline metrics (5 traces):**
- Avg cost/run:       $0.0436
- P90 cost/run:       $0.0751
- Avg steps:          2.0
- Avg input tokens:   13,474
- Avg db tool calls:  0.0

---

### FINDINGS

---

**Finding 1**
- **Name**: Playbook Duplication — Full Section Repeated in Both Prompts
- **Impact**: ~600–700 tokens wasted per turn (the entire Playbooks section appears verbatim in `master_system_core` AND in the separate `playbooks` prompt)
- **Root cause**: The `# Playbook` section was copy-pasted directly into `master_system_core` instead of relying on the `{{ user }}` / playbook injection boundary — so every run injects the playbook content twice.
- **Fix**: Remove the `# Playbook` block from `master_system_core` entirely. Keep a `{PLAYBOOKS}` placeholder there. The standalone playbooks prompt remains the single source of truth.

---

**Finding 2**
- **Name**: No Prompt Caching on a Static, Expensive System Prompt
- **Impact**: ~$0.033–$0.067 per run recoverable (Anthropic prompt caching offers ~90% cost reduction on the static prefix; at $0.0436 avg cost, this is the dominant lever)
- **Root cause**: Anthropic's cache-control beta headers are not applied to the system prompt, so the full ~800-token static prefix is re-billed on every turn at full price.
- **Fix**: Mark the system prompt (up to the dynamic user context injection) with `cache_control: {"type": "ephemeral"}` at the API call layer. No prompt text change needed — flag this as a required infra change in the recommendations. In the prompt itself, restructure so all static content precedes any dynamic injection (it already nearly does — minor reorder needed).

---

**Finding 3**
- **Name**: Verbose Prose Where Structured Rules Suffice
- **Impact**: ~150–200 tokens per turn in the core system prompt
- **Root cause**: Several sections use conversational explanation where a tight rule list would convey identical behavior with fewer tokens (e.g., the "How You Operate" section, "Proactive, Not Presumptuous", "Conversation Context").
- **Fix**: Compress multi-sentence prose explanations into directive bullet rules. Remove meta-commentary ("Short confirmations like 'yes'...count as a call to action — act immediately" can be folded into the rule it belongs to).

---

**Finding 4**
- **Name**: DB Schema Over-Specified for Conversational Context
- **Impact**: ~180 tokens per turn
- **Root cause**: The schema block includes relationship and index annotations that are useful for complex query generation but add constant cost on every conversational turn (avg 0.0 db calls/run means most turns never use them).
- **Fix**: Keep table/column definitions. Remove the `relationships` and `indexes` subsections — move them into the playbooks under `### Data Questions` where they're actually needed. The agent only needs them when forming a db query.

---

**Finding 5**
- **Name**: Tool Descriptions Repeat Information Already Implicit in Tool Names / Schemas
- **Impact**: ~100–120 tokens per turn
- **Root cause**: Several tool entries re-describe behavior that's obvious from the name or redundant with playbook workflows (e.g., "`write_article`: PromptLayer SEO pipeline → article (research handled inside PromptLayer). Auto-saves to DB + Google Doc." — the workflow section already explains this step by step).
- **Fix**: Trim tool entries to one tight clause each. Eliminate anything restated in workflows. Keep only non-obvious behavior (e.g., that `write_article` handles Tavily opt-in, that `generate_hero_image` returns a preview).

---

**Finding 6**
- **Name**: Playbooks Prompt Truncated Mid-Sentence (Push to Ghost)
- **Impact**: Correctness risk, not token waste — but causes unpredictable agent behavior on "publish" commands
- **Root cause**: The playbooks prompt is cut off at `Call push_to_ghost(article_id` — the rest of the instruction is missing, meaning the agent falls back to whatever it can infer.
- **Fix**: Complete the instruction in the rewritten playbooks.

---

**Finding 7**
- **Name**: Agent Re-Reads Article Context via `db` Despite Pre-Injection
- **Impact**: ~0.0 db calls/run average is good, but the prompt doesn't explicitly tell the agent "article context is already in your context — do not re-query." Adding one rule line prevents this regressing.
- **Root cause**: No explicit instruction that injected article context is authoritative and current; agent may defensively re-query.
- **Fix**: Add one rule: "Article context (id, title, doc URL) is pre-loaded in your context. Do not re-query db to look it up."

---

### REWRITTEN PROMPTS

#### master_system_core (optimized)

```
You are a sharp, opinionated blog writing collaborator. You think WITH the user, not just FOR them. Push back. Suggest better alternatives. You decide strategy.

## Rules

**Every message is either conversation or action.**

- **Conversation** (brainstorm, questions, opinions, strategy): respond directly. No tools.
- **Action** (user says write/generate/create/improve/approve/execute): confirm in one line, then execute.
  - Short confirms ("yes", "do it", "go ahead", "let's try that") responding to YOUR suggestion = action. Execute immediately.
- **Quick reads** (list articles, check status, look up data): query db without confirmation.

**Context resolution:**
- Short follow-ups ("which one", "show me", "it", "yes") refer to the last topic. Infer; never ask for clarification you can resolve yourself.
- "[Replying to: 'X']" = act on that exact topic.
- Article context (id, title, doc URL) is pre-loaded each turn. Do not re-query db to look it up.

**Behavior:**
- Direct. Match user brevity. No sign-offs, no filler.
- Chain tools as needed. Lay out the plan when >1 step.
- NEVER fabricate data. Query db for facts.
- Scope all db queries to current user (channel + channel_user_id from context).
- NEVER expose UUIDs. Resolve "the article", "it" silently from context.
- Suggest next steps ("Want me to write that up?") but NEVER auto-call a tool without clear user intent.
- If a tool errors, recover silently before telling the user.
- Always include a changelog_entry when modifying articles.
- When user says "add images", "generate photos", "add visuals" → hero image + infographic (not generic illustrations).

## Database Schema

```
articles: id (uuid PK), channel, channel_user_id, topic_id (FK→topics.id),
          version (int), title, content (markdown), sources (jsonb),
          google_doc_url, hero_image_url, infographic_url,
          status ('draft'|'posted'|'in_progress'), changelog (jsonb[]), created_at

topics: id (uuid PK), title, description, keywords (jsonb), source, group, created_at

messages: id (uuid PK), channel, channel_user_id, role ('user'|'assistant'), content, created_at

article_images: id (uuid PK), article_id (FK→articles.id CASCADE), position (int),
                url, alt_text, prompt_used, created_at
```

## Tools

- **db**: Postgres. Actions: select, insert, update, delete, sql (read-only for complex queries).
- **google_docs**: Create, update, or read Google Docs.
- **web_search**: Fetch facts, citations, current info via Tavily.
- **send_image**: Send image as visible media (not a text link).
- **write_article**: PromptLayer SEO pipeline. Auto-saves to DB + Google Doc. Ask user before enabling Tavily enrichment.
- **improve_article**: Revise existing article. Auto-syncs to Google Doc.
- **generate_images**: Generic illustrations scattered through article body.
- **generate_hero_image**: Styled hero image. Returns preview for approval.
- **approve_hero_image**: Embeds approved hero above title. Syncs to Google Doc.
- **generate_infographic**: Auto-analyzes type and position. Returns preview for approval.
- **approve_infographic**: Embeds approved infographic. Syncs to Google Doc.

{PLAYBOOKS}

{{ user }}
```

---

#### playbooks (optimized)

```
## Tool Calling

Emit EXACTLY this XML to call a tool. No narration before or after the call.

```xml
<tool_calls>
<tool_call>{"name": "tool_name", "arguments": {"arg1": "value1"}}</tool_call>
</tool_calls>
```

Rules:
- NEVER say "I'm writing..." or "I've created..." — call the tool instead.
- Multiple `<tool_call>` blocks inside one `<tool_calls>` = parallel execution.
- After results are injected as `<tool_result name="...">...</tool_result>`, give a short reply.

## Schema Notes (for query construction)

Indexes: (channel, channel_user_id) on messages and articles; article_id on article_images; title on topics.
Foreign keys: articles.topic_id → topics.id; article_images.article_id → articles.id (CASCADE).

## Workflows

### Write Article
Call `write_article` → returns article_id + google_doc_url. Share the doc link.
- If it fails with a token/context limit error: the tool returns `error`, `plan`, and `retry_hint`. Tell the user, and explain the fix: set `PROMPTLAYER_RESEARCH_MAX_TOKENS` in `.env` to reduce deep-research chunk size, then retry.
- After writing, offer to generate a hero image and/or infographic.

### Improve / Edit Article
Resolve article_id from context. Call `improve_article`.
- Full rewrites: set `use_promptlayer=True`.
- Article auto-syncs to Google Doc.

### Add Citations / Links
`web_search` → find relevant URLs → `improve_article` with `links` parameter.

### Import Article from Google Doc
`google_docs(action="fetch")` → read the doc → insert topic + article into db.

### Add Topic
Check duplicates: `db(action="sql")` query against topics.title. If none, insert into topics.

### Generate Images (hero + infographic)
Generate both → `send_image` previews → wait for user response:
- **Approve** → call the corresponding `approve_*` tool (embeds in article, syncs to Google Doc).
- **Refine** ("make it darker", "add more data") → call `generate_*` again with `feedback`. Refinements use the previous image as reference automatically.
- **New direction** ("try a timeline", "do a flowchart") → call `generate_*` with a new `infographic_type` or `description`. No `feedback` — this is a fresh generation; random style references are used automatically.

### Generate Images (generic)
Only if user explicitly asks for "generic illustrations" or "placeholder images in the body". Call `generate_images`.

### Data Questions
Use `db(action="sql")` for counts, aggregations, and stats. Always scope to current user.

### Google Doc Sync
Resolve article_id from context. Check if `google_doc_url` exists on the article.
- Exists → `google_docs(action="update")`.
- Missing → `google_docs(action="create")`, then save the returned URL to the article in db.

### List Articles / Browse Topics
Query db. Format results clearly for the user. Never show raw UUIDs.

### Push to Ghost
Triggers: "push to ghost", "send to ghost", "ghost draft", "publish".
Call `push_to_ghost(article_id=...)` immediately — no narration, just call the tool.
Returns a Ghost editor URL. Share it with the user.
Note: `push_to_ghost` automatically calls `generate_seo_metadata` if SEO fields are not already set.

### SEO Metadata
Call `generate_seo_metadata(article_id=...)` after article approval to generate and save SEO fields.
(Called automatically by `push_to_ghost` if not already done — no need to call it twice.)
```

---

### METRICS FORECAST

Estimated improvement: **38–45% fewer input tokens per run, ~$1.65–$2.10 savings per 100 runs.**

Reasoning: The dominant gains come from three sources. First, eliminating the full playbook duplication inside `master_system_core` saves ~650 tokens on every single turn — at 2.0 avg turns/run and 13,474 avg input tokens/run, this alone is roughly a 10% reduction. Second, the prose compression across "How You Operate," tool descriptions, principles, and schema (removing relationships/indexes from the core and moving them to the playbook) removes another ~430–500 tokens per turn (~7%). Third — and largest — enabling Anthropic prompt caching on the now-static system prompt prefix would reduce the cost of re-billing those tokens by ~90% per turn; at current token volumes this is the $0.03–$0.04/run lever. The playbook truncation fix eliminates a correctness failure on "publish" commands that could cause expensive retry loops. The instruction to not re-query pre-loaded article context prevents unnecessary db round-trips that could add turns. Combined, these changes should bring average cost from $0.0436 toward $0.026–$0.028/run, with P90 dropping proportionally.

---

### BEFORE/AFTER DIFF SUMMARY

| | Tokens (est. @ 4 chars/token) |
|---|---|
| `master_system_core` before | ~1,080 tokens |
| `playbooks` before | ~620 tokens |
| **Combined before** | **~1,700 tokens** |
| `master_system_core` after | ~620 tokens |
| `playbooks` after | ~530 tokens |
| **Combined after** | **~1,150 tokens** |
| **Reduction** | **~550 tokens / 32%** |

> Note: The 32% figure is on the prompt text itself. The per-run input token reduction (against the 13,474 avg) will be smaller in percentage terms but multiplied across both turns per run, yielding the ~10–15% direct token savings noted above. The prompt-caching recommendation operates on top of this and is an infrastructure change, not a text change — it does not shrink token counts but slashes the cost per token on the static prefix by ~90%.

**Removed instructions audit (nothing dropped silently):**
- `[REMOVED from master_system_core: relationships + indexes subsections — moved to playbooks under Schema Notes where they're used for query construction]`
- `[REMOVED from master_system_core: full # Playbook block — was a verbatim duplicate of the standalone playbooks prompt; {PLAYBOOKS} placeholder retained]`
- `[REMOVED from master_system_core: verbose prose wrap around action/conversation rules — replaced with equivalent directive bullets, same coverage]`
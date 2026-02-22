# Make.com Scenario Analysis: Integration Tools, PromptLayer

## Overview

This is a **Make.com blueprint** that implements a research-to-content pipeline. It combines inputs from Google Docs and web search, runs them through a PromptLayer agent, and outputs a formatted Google Doc.

---

## Scenario Name

**Integration Tools, PromptLayer**

---

## Flow Summary

The scenario runs as a linear pipeline with these stages:

| Order | Module ID | Module Type | Purpose |
|-------|-----------|-------------|---------|
| 1 | 2 | Set Variables | Sets the topic: `"Why is this LLM failing intermittently?"` |
| 2 | 5 | Google Docs: List Documents | Lists docs from folder `PromptLayer/DeepResearch` |
| 3 | 7 | Basic Aggregator | Aggregates the list for iteration |
| 4 | 8 | Google Docs: Get a Document | Fetches content from each doc (filter: image) |
| 5 | 3 | Tavily Search | Web search for authoritative sources on the topic |
| 6 | 9 | JSON: Create JSON | Builds a JSON payload with topic + combined research |
| 7 | 1 | PromptLayer Agent | Runs the `"seo agent"` with the research payload |
| 8 | 11 | Markdown: Compile | Compiles agent output as GitHub Flavored Markdown |
| 9 | 10 | Google Docs: Create a Document | Creates a new doc with the markdown content |

---

## Data Flow

```
[Set topic] → [List docs from DeepResearch folder]
                    ↓
              [Aggregate docs]
                    ↓
[Get doc content] ← [Map: id, name, topic]
                    ↓
[Tavily Search] ────┴──→ [Combine into JSON: topic + llm_deep_research]
                                ↓
                    [PromptLayer: "seo agent"]
                                ↓
                    [Compile as Markdown]
                                ↓
                    [Create Google Doc in PromptLayer folder]
```

---

## Key Configuration Details

### Input
- **Topic**: `"Why is this LLM failing intermittently?"` (hardcoded in Set Variables)
- **Source docs**: Google Drive folder `PromptLayer/DeepResearch`
- **Output folder**: `PromptLayer` (root of My Drive)

### Connections Used
- **Google** (yonash8@gmail.com) – for Docs and Drive
- **Tavily** – for web search
- **PromptLayer** – for the SEO agent

### Tavily Search
- **Query**: `"Find authoritative and relevant sources explaining {{topic}}. Prefer high-quality articles, engineering blogs, academic papers, postmortems, and official documentation."`
- Search depth: Advanced  
- Max results: 20  
- Chunks per source: 3  

### JSON Payload to Agent
```json
{
  "topic": "{{topic}}",
  "llm_deep_reserach": "{{doc_text}}{{tavily_results}}"
}
```
*Note: `llm_deep_reserach` appears to be a typo for `llm_deep_research`.*

### PromptLayer Agent
- **Agent name**: `"seo agent"`
- **Return all outputs**: `false`
- Input: the JSON with topic and combined research

---

## What This Scenario Does (End-to-End)

1. **Collects research** from:
   - Existing docs in the DeepResearch folder
   - Web search via Tavily for authoritative sources on the topic
2. **Runs an agent** (PromptLayer “seo agent”) over this combined research
3. **Turns agent output into markdown** and creates a new Google Doc named after the topic in the PromptLayer folder

---

## Observations & Potential Gaps

1. **Document content type**: Google Docs “Get a Document” is configured with filter `"image"`. That suggests image extraction from docs; if the goal is full text, the filter may need adjustment.
2. **Typo**: `llm_deep_reserach` should likely be `llm_deep_research`.
3. **Topic flexibility**: Topic is fixed. For an agentic workflow, this could be parameterized (webhook, form, scheduler, etc.).
4. **Error handling**: Scenario allows up to 3 errors (`maxErrors: 3`), with auto-commit enabled.
5. **Execution**: One roundtrip per run, non-sequential processing.

---

## As a Start for an Agentic Workflow

This blueprint is a solid base for a research-to-content agentic workflow:

- ✅ Research gathering (Docs + web)
- ✅ LLM/agent orchestration (PromptLayer)
- ✅ Structured output (JSON → Markdown)
- ✅ Output storage (Google Docs)

Possible next steps:
- Parameterize the topic (e.g., webhook, form, or input from another scenario)
- Add loops or branching for multi-step or multi-document processing
- Introduce tools/actions for the agent (e.g., search, fact-check, citations)
- Add error handling and retries for robustness

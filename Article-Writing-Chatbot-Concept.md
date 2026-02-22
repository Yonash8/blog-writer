# Article-Writing Chatbot: Concept & Design

## Vision

A conversational AI app that writes articles for you. You give it a topic → it researches, drafts, and delivers. Then you chat to refine: "make it shorter," "add citations," "tone down the jargon," etc.

**Standalone build**—no Make.com. You own the full pipeline: research, drafting, chat, memory, and iteration.

---

## Core User Flow

```
User: "Write an article about why LLMs fail intermittently"
  ↓
Bot: [Researches → Drafts → Returns article]
     "Here's your draft. What would you like to change?"
  ↓
User: "Make it more technical, add 3 real-world examples"
  ↓
Bot: [Applies edits, pulls from memory/DB if needed]
     "Updated. Anything else?"
  ↓
...iterate until done...
```

---

## The Workflow (What to Build)

Your Make scenario gives you the blueprint. Here's the equivalent pipeline to implement:

| Step | Purpose | What to use |
|------|---------|-------------|
| 1. **Set topic** | From user message | Parse chat input |
| 2. **Web search** | Find authoritative sources | Tavily API (or Perplexity, SerpAPI, etc.) |
| 3. **Local docs** (optional) | Pull from your knowledge base | Google Docs API, Notion API, or local files |
| 4. **Combine research** | Build context for LLM | JSON: `{ topic, sources }` |
| 5. **Draft** | Generate article | LLM (OpenAI, Anthropic, PromptLayer agent, etc.) |
| 6. **Return + iterate** | Show article, take feedback | Chat loop with conversation history |
| 7. **Persist** | Save versions, export | DB + Google Docs / export APIs |

---

## Key Capabilities

### 1. **Topic → Article Pipeline**
- Web search (Tavily API)
- Doc/folder references (Google Docs, Notion, or file storage)
- LLM drafting (OpenAI, Anthropic, PromptLayer, or similar)
- Structured output (Markdown → Doc, PDF, etc.)

### 2. **Conversational Interface**
- Chat UI instead of "run scenario once"
- Turn-based: article → feedback → revised article → feedback
- Natural-language instructions: "shorter," "add sources," "fix the intro"

### 3. **Memory**
- **Session memory**: Remembers the article, prior edits, and context in this conversation
- **Long-term memory**: Past articles, preferences, style, recurring topics
- **Knowledge base**: Your docs, bookmarks, saved research you’ve approved

### 4. **Database Access**
- **Articles**: Store drafts, versions, published pieces
- **Sources**: Saved links, quotes, citations
- **User preferences**: Tone, length, citation style, target audience

### 5. **Improvement Loop**
- Diff-aware edits (show what changed)
- Version history (revert, compare)
- Explicit approval gates (e.g., "lock" before exporting)

---

## Architecture: How to Build It

### Stack Options

**Backend**
- **Python**: FastAPI + LangChain/LlamaIndex → great for agents, tools, RAG
- **Node**: Next.js API routes + Vercel AI SDK → good if you want full-stack JS
- **Framework**: LangChain, CrewAI, or raw LLM + function calling for orchestration

**APIs you'll need**
- **Search**: Tavily (you used this), Perplexity, or SerpAPI
- **LLM**: OpenAI, Anthropic, or PromptLayer (if you want to keep your SEO agent)
- **Storage**: Google Docs API, Notion, or local Markdown files
- **Auth** (if multi-user): Auth0, Clerk, NextAuth, or Supabase Auth

**Database**
- **Supabase** (Postgres): Articles, messages, sources, users—all in one
- **SQLite + Drizzle**: Simple if single-user / local-first
- **PlanetScale / Neon**: If you prefer managed Postgres elsewhere

**Frontend**
- **Next.js + shadcn**: Fast, good DX
- **React + Vercel AI SDK**: Built-in streaming, useChat hook, tool handling
- **Streamlit** (Python): Quick prototype if you want to stay in Python

### Pipeline Flow (Code-Level)

```
1. User sends message (topic or feedback)
2. Classify: "new article" vs "edit existing"
3. If new:
   - Call Tavily with query about topic
   - (Optional) Fetch from Google Docs / knowledge base
   - Build prompt: topic + research
   - Call LLM → get markdown
   - Save to DB (article + sources)
   - Stream response to chat
4. If edit:
   - Load current article from DB
   - Build prompt: article + user feedback
   - Call LLM → get revised markdown
   - Save new version
   - Stream response to chat
5. Repeat until user is done; export on demand
```

---

## Data Model (High Level)

```
Users
├── id, email, preferences (tone, length, citation_style)

Conversations
├── id, user_id, topic, created_at

Messages
├── id, conversation_id, role (user/assistant), content, article_snapshot_id?

Articles
├── id, conversation_id, version, content (markdown), status (draft/published)
├── metadata: sources_used, word_count, created_at

Sources
├── id, user_id, url, title, snippet, added_at, used_in_article_ids

Memory (for LLM context)
├── User-level: "prefers technical depth", "target audience: engineers"
├── Conversation-level: current article state, edit history
```

---

## Open Questions

1. **Hosting**: Web app, Slack/Discord bot, or both?
2. **Auth**: How do users log in? (Google, email, etc.)
3. **Export**: Google Docs only, or also Notion, Medium, Markdown download?
4. **Pricing model**: Per article, subscription, API credits?
5. **Research scope**: Only web + your docs, or also paid DBs (e.g., academic)?
6. **Collaboration**: Single user only, or shared drafts / team workspaces?

---

## Next Steps (Building from Scratch)

1. **Pick a stack**: Python (FastAPI + LangChain) or Node (Next.js + Vercel AI SDK)—whichever you're faster in.
2. **Tavily + LLM**: Wire topic → search → prompt → article. Get the core pipeline working in a script or simple API.
3. **Add chat**: Wrap it in a chat loop. User says topic → you return article → user says "revise X" → you return updated article.
4. **Persist**: Add DB for conversations, articles, versions. Load prior context when continuing a thread.
5. **Export**: Hook up Google Docs API or Markdown download when the user is satisfied.
6. **Refine**: Streaming, diffs, memory, preferences.

---

*This doc is a starting point—adjust as the idea evolves.*

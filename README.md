# Article-Writing Chatbot

A conversational AI that writes articles from topics using deep research, Tavily enrichment, and PromptLayer SEO agents.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

### Supabase Setup (CLI)

1. Install [Supabase CLI](https://supabase.com/docs/guides/cli/getting-started) (`npm install -g supabase` or scoop/brew)
2. Log in and link your project:
   ```bash
   supabase login
   supabase link --project-ref YOUR_PROJECT_ID
   ```
   Get `YOUR_PROJECT_ID` from your project URL: `https://supabase.com/dashboard/project/<project-id>`
3. Push migrations:
   ```bash
   supabase db push
   ```
4. Create storage bucket (run once):
   ```bash
   python scripts/create_storage_bucket.py
   ```
5. Add `SUPABASE_URL` and `SUPABASE_KEY` (anon key) to `.env`

## Run

```bash
# Start API server
uvicorn src.main:app --reload

# Seed topics (after DB setup)
python scripts/seed_topics.py

# Test pipeline locally
python scripts/test_pipeline.py
```

Open http://localhost:8000/docs for API documentation.

## Environment Variables

See `.env.example` for required keys: PromptLayer, OpenAI, Google, Perplexity, Tavily, Supabase.

## Architecture

- **Master agent** (OpenAI): Orchestrates five tools
- **Write article**: Deep research (Perplexity) + Tavily + PromptLayer SEO agent
- **Retrieve topics**: Topics database
- **Generate images**: Google Imagen + placement analysis
- **Improve draft**: OpenAI or PromptLayer revise agent
- **Web search**: Tavily

Target channels: Slack, WhatsApp, Telegram.

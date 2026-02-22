-- Run this in Supabase Dashboard > SQL Editor if migrations haven't been applied.
-- Required for LangGraph WhatsApp thread mapping (active article per user).

CREATE TABLE IF NOT EXISTS active_article_threads (
    whatsapp_user_id TEXT PRIMARY KEY,
    article_id UUID REFERENCES articles(id) ON DELETE SET NULL,
    thread_id TEXT NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_active_threads_user ON active_article_threads(whatsapp_user_id);

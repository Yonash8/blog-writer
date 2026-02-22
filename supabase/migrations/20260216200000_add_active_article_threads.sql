-- Active article thread mapping for LangGraph integration.
-- Maps each WhatsApp user to their currently active article thread.
-- thread_id format: "{whatsapp_user_id}:{article_id}"

CREATE TABLE IF NOT EXISTS active_article_threads (
    whatsapp_user_id TEXT PRIMARY KEY,
    article_id UUID REFERENCES articles(id) ON DELETE SET NULL,
    thread_id TEXT NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Index for quick lookups
CREATE INDEX IF NOT EXISTS idx_active_threads_user ON active_article_threads(whatsapp_user_id);

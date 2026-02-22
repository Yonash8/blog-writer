-- Add changelog column to articles (tracks edits: rewritten from scratch, added section, added image, etc.)
ALTER TABLE public.articles
  ADD COLUMN IF NOT EXISTS changelog JSONB DEFAULT '[]';

COMMENT ON COLUMN public.articles.changelog IS 'Array of change entries: [{"action": "...", "at": "ISO timestamp"}]';

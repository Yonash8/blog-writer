-- Remove conversations table; use channel + channel_user_id directly on messages and articles.
-- Migrates existing data from conversations table before dropping it.

-- 1. Add new columns to messages (nullable initially for backfill)
ALTER TABLE public.messages
  ADD COLUMN IF NOT EXISTS channel TEXT,
  ADD COLUMN IF NOT EXISTS channel_user_id TEXT;

-- 2. Add new columns to articles (nullable initially for backfill)
ALTER TABLE public.articles
  ADD COLUMN IF NOT EXISTS channel TEXT,
  ADD COLUMN IF NOT EXISTS channel_user_id TEXT;

-- 3. Backfill messages from conversations
UPDATE public.messages m
SET
  channel = COALESCE(c.channel, 'legacy'),
  channel_user_id = COALESCE(c.channel_user_id, c.id::text)
FROM public.conversations c
WHERE m.conversation_id = c.id;

-- 4. Backfill articles from conversations
UPDATE public.articles a
SET
  channel = COALESCE(c.channel, 'legacy'),
  channel_user_id = COALESCE(c.channel_user_id, c.id::text)
FROM public.conversations c
WHERE a.conversation_id = c.id;

-- 5. Set defaults for any orphaned rows (shouldn't exist with FK)
UPDATE public.messages SET channel = 'legacy', channel_user_id = 'orphaned' WHERE channel IS NULL;
UPDATE public.articles SET channel = 'legacy', channel_user_id = 'orphaned' WHERE channel IS NULL;

-- 6. Make columns NOT NULL
ALTER TABLE public.messages ALTER COLUMN channel SET NOT NULL;
ALTER TABLE public.messages ALTER COLUMN channel_user_id SET NOT NULL;
ALTER TABLE public.articles ALTER COLUMN channel SET NOT NULL;
ALTER TABLE public.articles ALTER COLUMN channel_user_id SET NOT NULL;

-- 7. Drop FK and conversation_id from messages
ALTER TABLE public.messages DROP CONSTRAINT IF EXISTS messages_conversation_id_fkey;
DROP INDEX IF EXISTS public.idx_messages_conversation;
ALTER TABLE public.messages DROP COLUMN IF EXISTS conversation_id;

-- 8. Drop FK and conversation_id from articles
ALTER TABLE public.articles DROP CONSTRAINT IF EXISTS articles_conversation_id_fkey;
DROP INDEX IF EXISTS public.idx_articles_conversation;
ALTER TABLE public.articles DROP COLUMN IF EXISTS conversation_id;

-- 9. Drop conversations table
DROP TABLE IF EXISTS public.conversations;

-- 10. New indexes
CREATE INDEX IF NOT EXISTS idx_messages_channel_user ON public.messages(channel, channel_user_id);
CREATE INDEX IF NOT EXISTS idx_articles_channel_user ON public.articles(channel, channel_user_id);

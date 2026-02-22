-- Add channel and channel_user_id to conversations for WhatsApp/other channel mapping
ALTER TABLE public.conversations
  ADD COLUMN IF NOT EXISTS channel TEXT,
  ADD COLUMN IF NOT EXISTS channel_user_id TEXT;

CREATE INDEX IF NOT EXISTS idx_conversations_channel_user
  ON public.conversations(channel, channel_user_id)
  WHERE channel IS NOT NULL AND channel_user_id IS NOT NULL;

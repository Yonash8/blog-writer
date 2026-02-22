-- Add group column to topics
ALTER TABLE public.topics ADD COLUMN IF NOT EXISTS "group" TEXT;

-- Add title (topic) to articles for easier display and browsing
ALTER TABLE public.articles ADD COLUMN IF NOT EXISTS title TEXT;

-- Add Google Doc URL to articles (link created when article is generated/edited)
ALTER TABLE public.articles ADD COLUMN IF NOT EXISTS google_doc_url TEXT;

-- Add status and topic_id to articles
-- Status: draft (default), posted, in_progress
-- Each article must reference a topic

ALTER TABLE public.articles ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'draft';
ALTER TABLE public.articles DROP CONSTRAINT IF EXISTS chk_article_status;
ALTER TABLE public.articles ADD CONSTRAINT chk_article_status CHECK (status IN ('draft', 'posted', 'in_progress'));
UPDATE public.articles SET status = 'draft' WHERE status IS NULL;

ALTER TABLE public.articles ADD COLUMN IF NOT EXISTS topic_id UUID REFERENCES public.topics(id);

-- Backfill existing articles: create default topic and assign
INSERT INTO public.topics (id, title, description)
VALUES ('a0000000-0000-0000-0000-000000000001'::uuid, 'Uncategorized', 'Default topic for migrated articles')
ON CONFLICT (id) DO NOTHING;

UPDATE public.articles SET topic_id = 'a0000000-0000-0000-0000-000000000001'::uuid WHERE topic_id IS NULL;

ALTER TABLE public.articles ALTER COLUMN topic_id SET NOT NULL;
ALTER TABLE public.articles ALTER COLUMN status SET NOT NULL;
ALTER TABLE public.articles ALTER COLUMN status SET DEFAULT 'draft';

CREATE INDEX IF NOT EXISTS idx_articles_topic ON public.articles(topic_id);
CREATE INDEX IF NOT EXISTS idx_articles_status ON public.articles(status);

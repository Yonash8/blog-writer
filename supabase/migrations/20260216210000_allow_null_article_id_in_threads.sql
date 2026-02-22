-- Allow article_id to be NULL in active_article_threads.
-- Used when we have a placeholder thread (e.g. before an article is created).

ALTER TABLE active_article_threads
ALTER COLUMN article_id DROP NOT NULL;

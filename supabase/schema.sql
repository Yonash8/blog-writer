-- Article-Writing Chatbot - Supabase Schema
-- Run this in Supabase SQL Editor to create tables

-- Users (optional - can use Supabase Auth later)
CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT UNIQUE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Topics database for article ideas
CREATE TABLE IF NOT EXISTS topics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  title TEXT NOT NULL,
  description TEXT,
  keywords JSONB DEFAULT '[]',
  source TEXT,
  "group" TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Messages per (channel, channel_user_id) - no separate conversations table
CREATE TABLE IF NOT EXISTS messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  channel TEXT NOT NULL,
  channel_user_id TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Articles (drafts and versions) - saved on generation and every change
CREATE TABLE IF NOT EXISTS articles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  channel TEXT NOT NULL,
  channel_user_id TEXT NOT NULL,
  topic_id UUID NOT NULL REFERENCES topics(id),
  version INT NOT NULL DEFAULT 1,
  title TEXT,
  content TEXT NOT NULL,
  sources JSONB DEFAULT '[]',
  google_doc_url TEXT,
  hero_image_url TEXT,
  infographic_url TEXT,
  status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'posted', 'in_progress')),
  changelog JSONB DEFAULT '[]',
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Article images (generated and placed)
CREATE TABLE IF NOT EXISTS article_images (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  article_id UUID NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
  position INT NOT NULL DEFAULT 0,
  url TEXT NOT NULL,
  alt_text TEXT,
  prompt_used TEXT,
  status TEXT NOT NULL DEFAULT 'approved' CHECK (status IN ('pending_approval', 'approved', 'rejected')),
  image_type TEXT NOT NULL DEFAULT 'generic' CHECK (image_type IN ('generic', 'hero', 'infographic')),
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_messages_channel_user ON messages(channel, channel_user_id);
CREATE INDEX IF NOT EXISTS idx_articles_channel_user ON articles(channel, channel_user_id);
CREATE INDEX IF NOT EXISTS idx_article_images_article ON article_images(article_id);
CREATE INDEX IF NOT EXISTS idx_topics_keywords ON topics USING GIN(keywords);
CREATE INDEX IF NOT EXISTS idx_topics_title ON topics(title);

-- RPC function for agent autonomy: execute read-only SQL
CREATE OR REPLACE FUNCTION execute_readonly_sql(query text)
RETURNS json
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
  result json;
BEGIN
  IF NOT (LOWER(TRIM(query)) LIKE 'select%' OR LOWER(TRIM(query)) LIKE 'with%') THEN
    RAISE EXCEPTION 'Only SELECT/WITH queries are allowed';
  END IF;
  EXECUTE format(
    'SELECT COALESCE(json_agg(row_to_json(t)), ''[]''::json) FROM (%s) t',
    query
  ) INTO result;
  RETURN result;
END;
$$;

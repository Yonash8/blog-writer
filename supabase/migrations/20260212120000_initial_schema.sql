-- Article-Writing Chatbot - Initial schema
-- Tables: users, topics, conversations, messages, articles, article_images

CREATE TABLE IF NOT EXISTS public.users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT UNIQUE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.topics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  title TEXT NOT NULL,
  description TEXT,
  keywords JSONB DEFAULT '[]',
  source TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.conversations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES public.users(id),
  topic TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES public.conversations(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.articles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES public.conversations(id) ON DELETE CASCADE,
  version INT NOT NULL DEFAULT 1,
  content TEXT NOT NULL,
  sources JSONB DEFAULT '[]',
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.article_images (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  article_id UUID NOT NULL REFERENCES public.articles(id) ON DELETE CASCADE,
  position INT NOT NULL DEFAULT 0,
  url TEXT NOT NULL,
  alt_text TEXT,
  prompt_used TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON public.messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_articles_conversation ON public.articles(conversation_id);
CREATE INDEX IF NOT EXISTS idx_article_images_article ON public.article_images(article_id);
CREATE INDEX IF NOT EXISTS idx_topics_keywords ON public.topics USING GIN(keywords);
CREATE INDEX IF NOT EXISTS idx_topics_title ON public.topics(title);

-- Enable RLS and grant access
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.topics ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.articles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.article_images ENABLE ROW LEVEL SECURITY;

-- Permissive policies for anon/service_role (MVP: backend uses service_role key)
CREATE POLICY "Allow all users" ON public.users FOR ALL TO anon, service_role USING (true) WITH CHECK (true);
CREATE POLICY "Allow all topics" ON public.topics FOR ALL TO anon, service_role USING (true) WITH CHECK (true);
CREATE POLICY "Allow all conversations" ON public.conversations FOR ALL TO anon, service_role USING (true) WITH CHECK (true);
CREATE POLICY "Allow all messages" ON public.messages FOR ALL TO anon, service_role USING (true) WITH CHECK (true);
CREATE POLICY "Allow all articles" ON public.articles FOR ALL TO anon, service_role USING (true) WITH CHECK (true);
CREATE POLICY "Allow all article_images" ON public.article_images FOR ALL TO anon, service_role USING (true) WITH CHECK (true);

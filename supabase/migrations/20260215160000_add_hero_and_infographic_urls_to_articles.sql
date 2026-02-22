-- Add hero_image_url and infographic_url columns to articles for quick access.
ALTER TABLE articles ADD COLUMN IF NOT EXISTS hero_image_url TEXT;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS infographic_url TEXT;

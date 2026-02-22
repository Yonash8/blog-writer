-- Add status column to article_images for approval workflow.
-- Values: 'pending_approval', 'approved', 'rejected'
ALTER TABLE article_images
  ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'approved'
  CHECK (status IN ('pending_approval', 'approved', 'rejected'));

-- Add image_type column to distinguish hero images from infographics and generic images.
-- Values: 'generic', 'hero', 'infographic'
ALTER TABLE article_images
  ADD COLUMN IF NOT EXISTS image_type TEXT NOT NULL DEFAULT 'generic'
  CHECK (image_type IN ('generic', 'hero', 'infographic'));

-- Store infographic analysis metadata (position_after snippet, infographic_type, etc.)
ALTER TABLE article_images
  ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}';

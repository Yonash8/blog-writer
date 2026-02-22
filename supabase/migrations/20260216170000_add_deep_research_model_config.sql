-- Add deep_research_model to agent_config
INSERT INTO agent_config (key, value) VALUES
  ('deep_research_model', 'o3-deep-research')
ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;

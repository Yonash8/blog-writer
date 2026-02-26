-- Stores self-optimization analysis sessions between when the analysis runs
-- and when the user approves/rejects/deploys the proposed changes.

CREATE TABLE IF NOT EXISTS optimization_sessions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at timestamptz DEFAULT now(),
  status text NOT NULL DEFAULT 'pending',   -- pending | deployed | rejected
  window_hours int NOT NULL DEFAULT 24,
  trace_count int,
  analysis_text text,                        -- full Claude analysis output
  action_items jsonb,                        -- structured list of findings with proposed changes
  pending_item_ids int[],                    -- subset of action_items to deploy (user can remove items)
  channel_user_id text,                      -- who triggered / who to notify
  notified_at timestamptz                    -- when WhatsApp notification was sent
);

CREATE INDEX IF NOT EXISTS idx_optimization_sessions_status ON optimization_sessions(status);
CREATE INDEX IF NOT EXISTS idx_optimization_sessions_created_at ON optimization_sessions(created_at DESC);

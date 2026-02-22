-- Observability traces table for agent run logs.
-- Stores hierarchical trace data (prompts, responses, tokens, tool calls, sub-agents).
CREATE TABLE IF NOT EXISTS observability_traces (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  trace_id text NOT NULL UNIQUE,
  channel text NOT NULL,
  channel_user_id text NOT NULL,
  user_message text NOT NULL,
  final_message text,
  payload jsonb NOT NULL,
  created_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_observability_traces_channel_user
  ON observability_traces(channel, channel_user_id);

CREATE INDEX IF NOT EXISTS idx_observability_traces_created
  ON observability_traces(created_at DESC);

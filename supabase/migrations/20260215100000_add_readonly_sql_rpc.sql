-- Migration: add execute_readonly_sql RPC for agent autonomy
-- Allows the agent to run arbitrary read-only SQL (SELECT/WITH) against the database.

CREATE OR REPLACE FUNCTION execute_readonly_sql(query text)
RETURNS json
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
  result json;
BEGIN
  -- Only allow SELECT and WITH (CTE) statements
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

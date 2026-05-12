import { authHeaders, getApiKey } from './auth'

export interface TraceListItem {
  trace_id: string
  created_at?: string
  channel?: string
  channel_user_id?: string
  final_message?: string
  payload?: Record<string, unknown>
  summary?: {
    steps?: number
    total_input_tokens?: number
    total_output_tokens?: number
    total_tokens?: number
    total_cost_usd?: number
    duration_ms?: number
    models_used?: string[]
    tools_used?: string[]
  }
}

export type TraceDetail = TraceListItem

export interface AgentConfigKey {
  key: string
  label: string
  type: string
  min?: number
  max?: number
}

export interface AgentDefinition {
  id: string
  name: string
  description: string
  model_type: 'chat' | 'image'
  config_keys: AgentConfigKey[]
  config: Record<string, string>
  prompts: string[]
  prompt_contents: Record<string, string>
  providers_models: Record<string, string[]>
  tools: string[]
  sub_agents: string[]
}

export interface AgentsResponse {
  agents: AgentDefinition[]
  model_to_provider: Record<string, string>
}

export async function listTraces(limit = 30): Promise<TraceListItem[]> {
  const r = await fetch(`/api/traces?limit=${limit}`, { headers: authHeaders() })
  if (!r.ok) throw new Error(`traces: ${r.status}`)
  const data = await r.json()
  return Array.isArray(data) ? data : (data.traces ?? [])
}

export async function getTrace(traceId: string): Promise<TraceDetail> {
  const r = await fetch(`/api/traces/${encodeURIComponent(traceId)}`, {
    headers: authHeaders(),
  })
  if (!r.ok) throw new Error(`trace: ${r.status}`)
  return r.json()
}

export async function getAgents(): Promise<AgentsResponse> {
  const r = await fetch('/api/admin/agents', { headers: authHeaders() })
  if (!r.ok) throw new Error(`agents: ${r.status}`)
  return r.json()
}

export async function patchAgentConfig(body: Record<string, unknown>): Promise<void> {
  const r = await fetch('/api/admin/agent-config', {
    method: 'PATCH',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`patch config: ${r.status}`)
}

export async function patchPrompts(body: Record<string, unknown>): Promise<void> {
  const r = await fetch('/api/admin/prompts', {
    method: 'PATCH',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`patch prompts: ${r.status}`)
}

export type ChatEvent =
  | { type: 'status'; text: string }
  | { type: 'token'; text: string }
  | { type: 'done'; message: string; article?: unknown; error?: boolean }

/**
 * POST /api/chat/stream — yields parsed SSE events until the stream ends.
 * EventSource doesn't support POST, so we use fetch + manual SSE parsing.
 */
export async function* chatStream(message: string, channelUserId?: string): AsyncGenerator<ChatEvent> {
  const r = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ message, channel_user_id: channelUserId ?? 'web-console' }),
  })
  if (!r.ok || !r.body) throw new Error(`chat: ${r.status}`)

  const reader = r.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    let idx: number
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const frame = buf.slice(0, idx)
      buf = buf.slice(idx + 2)
      for (const line of frame.split('\n')) {
        if (line.startsWith('data: ')) {
          try {
            yield JSON.parse(line.slice(6)) as ChatEvent
          } catch {
            // skip malformed
          }
        }
      }
    }
  }
}

/**
 * Subscribe to /api/events/stream — server-broadcast log_event firehose.
 * EventSource doesn't accept custom headers, so we forward the API key as a query param.
 * Returns an unsubscribe function.
 */
export function subscribeEvents(onEvent: (e: Record<string, unknown>) => void): () => void {
  const key = getApiKey()
  const url = key ? `/api/events/stream?key=${encodeURIComponent(key)}` : '/api/events/stream'
  const es = new EventSource(url)
  es.onmessage = (msg) => {
    try {
      onEvent(JSON.parse(msg.data))
    } catch {
      // skip
    }
  }
  es.onerror = () => {
    // EventSource auto-reconnects; nothing to do
  }
  return () => es.close()
}

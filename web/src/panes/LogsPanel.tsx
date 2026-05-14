import { useEffect, useRef, useState } from 'react'
import { subscribeEvents } from '../lib/api'
import { CloseButton } from '../lib/ui'

type EventItem = {
  ts: string
  event_type: string
  trace_id?: string
  rest: Record<string, unknown>
}

// Compact symbol per event type — keeps the live feed scannable
const SYMBOL: Record<string, { sym: string; color: string }> = {
  connected: { sym: '◉', color: 'var(--color-accent)' },
  request_start: { sym: '▶', color: 'var(--color-accent)' },
  request_done: { sym: '✓', color: 'var(--color-good)' },
  agent_run_start: { sym: '▸', color: 'var(--color-accent)' },
  agent_call: { sym: '·', color: 'var(--color-fg-dim)' },
  agent_turn: { sym: '↻', color: 'var(--color-fg-dim)' },
  tool_call: { sym: '⏵', color: 'var(--color-warn)' },
  tool_result: { sym: '◂', color: 'var(--color-good)' },
  tavily_research_started: { sym: '🔍', color: 'var(--color-warn)' },
  tavily_research_completed: { sym: '🔍', color: 'var(--color-good)' },
  error: { sym: '✗', color: 'var(--color-bad)' },
}

function fmtTime(iso: string): string {
  const d = new Date(iso)
  if (isNaN(d.getTime())) return iso
  return d.toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function fmtMs(v: unknown): string | null {
  const n = typeof v === 'number' ? v : Number(v)
  if (!isFinite(n)) return null
  if (n < 1000) return `${Math.round(n)}ms`
  return `${(n / 1000).toFixed(1)}s`
}

// Pick the most informative single-line summary from an event payload
function summarize(rest: Record<string, unknown>): string {
  if (rest.tool_name) {
    const args = rest.args_sanitized as Record<string, unknown> | undefined
    const argStr = args && typeof args === 'object'
      ? Object.entries(args).map(([k, v]) => `${k}=${truncate(String(v), 30)}`).join(' ')
      : ''
    const lat = fmtMs(rest.latency_ms)
    const tail = [lat, rest.success === false ? 'FAIL' : null].filter(Boolean).join(' · ')
    return `${rest.tool_name}${argStr ? ` ${argStr}` : ''}${tail ? `  ${tail}` : ''}`
  }
  if (rest.message_preview) return String(rest.message_preview)
  if (rest.model) {
    const lat = fmtMs(rest.latency_ms)
    return `${rest.model}${lat ? `  ${lat}` : ''}`
  }
  if (rest.total_latency_ms) {
    return `total ${fmtMs(rest.total_latency_ms)}`
  }
  if (rest.error) return String(rest.error)
  // Generic fallback: short JSON
  const keys = Object.keys(rest).filter((k) => k !== 'channel' && k !== 'channel_user_id')
  if (keys.length === 0) return ''
  const first = keys.slice(0, 2).map((k) => `${k}=${truncate(JSON.stringify(rest[k]), 40)}`).join(' ')
  return first
}

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n - 1) + '…' : s
}

export default function LogsPanel({ onClose }: { onClose: () => void }) {
  const [events, setEvents] = useState<EventItem[]>([])
  const [paused, setPaused] = useState(false)
  const [filter, setFilter] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)
  const pausedRef = useRef(paused)
  pausedRef.current = paused

  useEffect(() => {
    const unsub = subscribeEvents((raw) => {
      if (pausedRef.current) return
      const { event_type, ts, trace_id, ...rest } = raw as Record<string, unknown>
      setEvents((prev) => {
        const next = [
          ...prev,
          {
            ts: (ts as string) ?? new Date().toISOString(),
            event_type: (event_type as string) ?? 'unknown',
            trace_id: trace_id as string | undefined,
            rest,
          },
        ]
        return next.length > 500 ? next.slice(next.length - 500) : next
      })
    })
    return unsub
  }, [])

  useEffect(() => {
    if (!paused) scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight })
  }, [events, paused])

  const visible = filter
    ? events.filter((e) => e.event_type.toLowerCase().includes(filter.toLowerCase()))
    : events

  return (
    <div className="flex h-full flex-col border-l border-[var(--color-border)] bg-[var(--color-panel)]">
      <div className="flex items-center justify-between border-b border-[var(--color-border)] px-3 py-2 text-xs uppercase tracking-wider">
        <div className="flex items-center gap-2">
          <span className="text-[var(--color-accent)]">logs</span>
          <span className="text-[10px] text-[var(--color-fg-dim)]">
            live · {visible.length}
            {filter && `/${events.length}`}
            {paused && ' · paused'}
          </span>
        </div>
        <div className="flex items-center gap-2 normal-case">
          <input
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="filter type"
            className="w-28 rounded border border-[var(--color-border)] bg-[var(--color-bg)] px-2 py-0.5 text-[11px] text-[var(--color-fg)] focus:border-[var(--color-accent)] focus:outline-none"
          />
          <button
            onClick={() => setPaused((p) => !p)}
            title={paused ? 'resume' : 'pause'}
            className={
              'rounded border px-2 py-0.5 text-[11px] ' +
              (paused
                ? 'border-[var(--color-warn)] text-[var(--color-warn)]'
                : 'border-[var(--color-border)] text-[var(--color-fg-dim)] hover:text-[var(--color-fg)]')
            }
          >
            {paused ? '▶' : '❚❚'}
          </button>
          <button
            onClick={() => setEvents([])}
            title="clear"
            className="rounded border border-[var(--color-border)] px-2 py-0.5 text-[11px] text-[var(--color-fg-dim)] hover:text-[var(--color-fg)]"
          >
            ⌫
          </button>
          <CloseButton onClick={onClose} />
        </div>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-3 py-2">
        {visible.length === 0 && (
          <div className="text-xs text-[var(--color-fg-dim)]">
            waiting for events…{'\n'}trigger a chat to see live activity.
          </div>
        )}
        {visible.map((e, i) => {
          const meta = SYMBOL[e.event_type] ?? { sym: '·', color: 'var(--color-fg-dim)' }
          const summary = summarize(e.rest)
          return (
            <div key={i} className="group flex items-baseline gap-2 py-0.5 text-[11px] leading-relaxed">
              <span className="w-[60px] shrink-0 text-[var(--color-fg-dim)]">{fmtTime(e.ts)}</span>
              <span className="w-[14px] shrink-0 text-center" style={{ color: meta.color }}>
                {meta.sym}
              </span>
              <span className="w-[120px] shrink-0 truncate" style={{ color: meta.color }}>
                {e.event_type}
              </span>
              <span className="flex-1 truncate text-[var(--color-fg)] group-hover:whitespace-normal group-hover:break-all">
                {summary}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

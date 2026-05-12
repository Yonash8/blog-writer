import { useEffect, useState } from 'react'
import { listTraces, getTrace, type TraceListItem, type TraceDetail } from '../lib/api'
import { CloseButton } from '../lib/ui'

function fmtCost(c: number | undefined): string {
  if (c == null) return '—'
  return c < 0.01 ? `$${c.toFixed(4)}` : `$${c.toFixed(3)}`
}

function fmtMs(ms: number | undefined): string {
  if (ms == null) return '—'
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function fmtTime(iso: string | undefined): string {
  if (!iso) return '—'
  const d = new Date(iso)
  if (isNaN(d.getTime())) return iso
  return d.toLocaleString([], {
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

function fmtTokens(n: number | undefined): string {
  if (n == null) return '—'
  if (n < 1000) return String(n)
  return `${(n / 1000).toFixed(1)}k`
}

export default function TracesView({ onClose }: { onClose: () => void }) {
  const [traces, setTraces] = useState<TraceListItem[] | null>(null)
  const [selected, setSelected] = useState<TraceDetail | null>(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [channelFilter, setChannelFilter] = useState<string>('all')

  function refresh() {
    setLoading(true)
    listTraces(100)
      .then((data) => {
        setTraces(data)
        setErr(null)
      })
      .catch((e) => setErr(String(e)))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    refresh()
  }, [])

  async function openTrace(t: TraceListItem) {
    try {
      const detail = await getTrace(t.trace_id)
      setSelected(detail)
    } catch (e) {
      setErr(String(e))
    }
  }

  const channels = Array.from(new Set((traces ?? []).map((t) => t.channel).filter(Boolean) as string[]))
  const visible = channelFilter === 'all' ? traces ?? [] : (traces ?? []).filter((t) => t.channel === channelFilter)

  return (
    <div className="absolute inset-0 z-40 flex flex-col bg-[var(--color-bg)]">
      <div className="flex items-center justify-between border-b border-[var(--color-border)] bg-[var(--color-panel)] px-4 py-2 text-xs uppercase tracking-wider">
        <div className="flex items-center gap-3">
          <span className="font-bold text-[var(--color-accent)]">traces</span>
          <span className="text-[var(--color-fg-dim)]">{visible.length} runs</span>
          {loading && <span className="text-[var(--color-fg-dim)]">· loading…</span>}
        </div>
        <div className="flex items-center gap-2 normal-case">
          <select
            value={channelFilter}
            onChange={(e) => setChannelFilter(e.target.value)}
            className="rounded border border-[var(--color-border)] bg-[var(--color-bg)] px-2 py-0.5 text-[11px] text-[var(--color-fg)] focus:border-[var(--color-accent)] focus:outline-none"
          >
            <option value="all">all channels</option>
            {channels.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          <button
            onClick={refresh}
            className="rounded border border-[var(--color-border)] px-2 py-0.5 text-[11px] text-[var(--color-fg-dim)] hover:text-[var(--color-fg)]"
          >
            ↻ refresh
          </button>
          <CloseButton onClick={onClose} />
        </div>
      </div>

      {err && (
        <div className="border-b border-[var(--color-bad)] bg-[var(--color-bad)]/10 px-4 py-2 text-[var(--color-bad)]">
          {err}
        </div>
      )}

      <div className="flex flex-1 overflow-hidden">
        {/* Trace list — table view */}
        <div className="w-[55%] overflow-y-auto border-r border-[var(--color-border)]">
          <table className="w-full text-[12px]">
            <thead className="sticky top-0 z-10 bg-[var(--color-panel)] text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
              <tr className="border-b border-[var(--color-border)]">
                <th className="px-3 py-2 text-left">when</th>
                <th className="px-3 py-2 text-left">channel</th>
                <th className="px-3 py-2 text-left">message</th>
                <th className="px-3 py-2 text-right">steps</th>
                <th className="px-3 py-2 text-right">dur</th>
                <th className="px-3 py-2 text-right">tok</th>
                <th className="px-3 py-2 text-right">cost</th>
              </tr>
            </thead>
            <tbody>
              {visible.length === 0 && !loading && (
                <tr>
                  <td colSpan={7} className="px-4 py-6 text-center text-[var(--color-fg-dim)]">
                    no traces yet
                  </td>
                </tr>
              )}
              {visible.map((t) => {
                const isSel = selected?.trace_id === t.trace_id
                return (
                  <tr
                    key={t.trace_id}
                    onClick={() => openTrace(t)}
                    className={
                      'cursor-pointer border-b border-[var(--color-border)] transition-colors ' +
                      (isSel
                        ? 'bg-[var(--color-accent-dim)]/30'
                        : 'hover:bg-[var(--color-panel)]')
                    }
                  >
                    <td className="whitespace-nowrap px-3 py-2 text-[var(--color-fg-dim)]">
                      {fmtTime(t.created_at)}
                    </td>
                    <td className="px-3 py-2">
                      <span className="rounded bg-[var(--color-border)] px-1.5 py-0.5 text-[10px]">{t.channel ?? '—'}</span>
                    </td>
                    <td className="max-w-[300px] truncate px-3 py-2 text-[var(--color-fg)]">
                      {t.final_message?.slice(0, 80) || <span className="text-[var(--color-fg-dim)]">(no message)</span>}
                    </td>
                    <td className="px-3 py-2 text-right text-[var(--color-fg-dim)]">{t.summary?.steps ?? 0}</td>
                    <td className="px-3 py-2 text-right text-[var(--color-fg-dim)]">{fmtMs(t.summary?.duration_ms)}</td>
                    <td className="px-3 py-2 text-right text-[var(--color-fg-dim)]">{fmtTokens(t.summary?.total_tokens)}</td>
                    <td className="px-3 py-2 text-right text-[var(--color-fg-dim)]">{fmtCost(t.summary?.total_cost_usd)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        {/* Detail pane */}
        <div className="flex-1 overflow-y-auto">
          {!selected && (
            <div className="flex h-full items-center justify-center text-[var(--color-fg-dim)]">
              ← select a trace
            </div>
          )}
          {selected && <TraceDetail trace={selected} />}
        </div>
      </div>
    </div>
  )
}

function TraceDetail({ trace }: { trace: TraceDetail }) {
  const payload = (trace.payload ?? {}) as Record<string, unknown>
  const events = (payload.events as Array<Record<string, unknown>>) ?? []
  const summary = trace.summary ?? {}
  const userMessage = payload.user_message as string | undefined
  const modelsUsed = (summary.models_used ?? []) as string[]
  const toolsUsed = (summary.tools_used ?? []) as string[]
  const [expanded, setExpanded] = useState<Set<number>>(new Set())

  function toggle(i: number) {
    setExpanded((prev) => {
      const n = new Set(prev)
      if (n.has(i)) n.delete(i)
      else n.add(i)
      return n
    })
  }

  return (
    <div className="px-5 py-4">
      <div className="mb-2 flex items-center gap-2 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
        <span>trace</span>
        <code className="text-[var(--color-accent)]">{trace.trace_id}</code>
      </div>

      {userMessage && (
        <div className="mb-4 rounded border border-[var(--color-border)] bg-[var(--color-panel)] px-3 py-2">
          <div className="mb-1 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">request</div>
          <div className="whitespace-pre-wrap text-[var(--color-fg)]">{userMessage}</div>
        </div>
      )}

      {trace.final_message && (
        <div className="mb-4 rounded border border-[var(--color-border)] bg-[var(--color-panel)] px-3 py-2">
          <div className="mb-1 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">response</div>
          <div className="whitespace-pre-wrap text-[var(--color-fg)]">{trace.final_message}</div>
        </div>
      )}

      <div className="mb-4 grid grid-cols-2 gap-2 md:grid-cols-4">
        <Stat label="duration" value={fmtMs(summary.duration_ms)} />
        <Stat label="steps" value={String(summary.steps ?? 0)} />
        <Stat label="tokens" value={fmtTokens(summary.total_tokens)} />
        <Stat label="cost" value={fmtCost(summary.total_cost_usd)} />
      </div>

      {(modelsUsed.length > 0 || toolsUsed.length > 0) && (
        <div className="mb-4 grid grid-cols-1 gap-3 md:grid-cols-2">
          {modelsUsed.length > 0 && (
            <div>
              <div className="mb-1 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">models</div>
              <div className="flex flex-wrap gap-1">
                {modelsUsed.map((m) => (
                  <span key={m} className="rounded bg-[var(--color-border)] px-1.5 py-0.5 text-[10px]">
                    {m}
                  </span>
                ))}
              </div>
            </div>
          )}
          {toolsUsed.length > 0 && (
            <div>
              <div className="mb-1 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">tools</div>
              <div className="flex flex-wrap gap-1">
                {toolsUsed.map((t, i) => (
                  <span key={`${t}-${i}`} className="rounded bg-[var(--color-accent-dim)]/40 px-1.5 py-0.5 text-[10px]">
                    {t}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      <div className="mb-2 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
        events ({events.length})
      </div>
      <div className="space-y-1">
        {events.map((e, i) => {
          const isOpen = expanded.has(i)
          const otherFields = Object.fromEntries(
            Object.entries(e).filter(([k]) => k !== 'event_type' && k !== 'ts' && k !== 'trace_id'),
          )
          return (
            <div key={i} className="rounded border border-[var(--color-border)] bg-[var(--color-panel)]">
              <button
                onClick={() => toggle(i)}
                className="flex w-full items-center justify-between gap-2 px-3 py-1.5 text-left text-[11px] hover:bg-[var(--color-bg)]"
              >
                <div className="flex items-center gap-2">
                  <span className="text-[var(--color-fg-dim)]">{isOpen ? '▾' : '▸'}</span>
                  <span className="text-[var(--color-accent)]">{String(e.event_type)}</span>
                  <span className="truncate text-[var(--color-fg-dim)]">
                    {previewFields(otherFields)}
                  </span>
                </div>
                <span className="shrink-0 text-[10px] text-[var(--color-fg-dim)]">{fmtTime(e.ts as string)}</span>
              </button>
              {isOpen && (
                <pre className="max-h-64 overflow-auto border-t border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2 text-[10px] text-[var(--color-fg)]">
                  {JSON.stringify(otherFields, null, 2)}
                </pre>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

function previewFields(obj: Record<string, unknown>): string {
  const priority = ['tool_name', 'model', 'message_preview', 'latency_ms', 'success', 'error', 'total_latency_ms']
  const parts: string[] = []
  for (const k of priority) {
    if (obj[k] !== undefined) {
      let v = String(obj[k])
      if (v.length > 50) v = v.slice(0, 49) + '…'
      parts.push(`${k}=${v}`)
      if (parts.length >= 3) break
    }
  }
  return parts.join(' · ')
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-[var(--color-border)] bg-[var(--color-panel)] px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">{label}</div>
      <div className="text-base text-[var(--color-fg)]">{value}</div>
    </div>
  )
}

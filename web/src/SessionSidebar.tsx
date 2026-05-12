import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import {
  createSession,
  deleteSession,
  listSessions,
  type Session,
} from './lib/sessions'
import { clearSession, snapshot, subscribe } from './store/runs'

/**
 * Per-session live indicator. Subscribes to runs.ts so the "● running" dot
 * and the current tool/status line update as the agent works — even when
 * this session isn't the one the user is currently looking at.
 */
function SessionRow({
  session,
  active,
  onDelete,
}: {
  session: Session
  active: boolean
  onDelete: (id: string) => void
}) {
  const [snap, setSnap] = useState(() => snapshot(session.id))
  useEffect(() => {
    return subscribe(session.id, (s) => setSnap({ ...s }))
  }, [session.id])

  // Prefer the live in-process snapshot, fall back to the server-side meta
  // from the sessions list response. The latter is the source of truth for
  // sessions running on a different browser tab / device.
  const running = snap.running || session.running
  const tool = snap.lastTool || session.last_tool
  const status = snap.lastStatus || session.last_status
  const titleText = (session.title || '').trim() || 'Untitled'

  return (
    <Link
      to={`/session/${session.id}`}
      className={
        'group flex items-start gap-2 border-l-2 px-3 py-2 text-[13px] ' +
        (active
          ? 'border-[var(--color-accent)] bg-[var(--color-panel)] text-[var(--color-fg)]'
          : 'border-transparent text-[var(--color-fg-dim)] hover:border-[var(--color-accent-dim)] hover:bg-[var(--color-panel)] hover:text-[var(--color-fg)]')
      }
    >
      <span
        className={
          'mt-1 shrink-0 text-[11px] ' +
          (running ? 'text-[var(--color-warn)]' : 'text-[var(--color-fg-dim)]')
        }
        title={running ? 'running' : 'idle'}
      >
        {running ? '●' : '○'}
      </span>
      <div className="min-w-0 flex-1">
        <div className="truncate">{titleText}</div>
        {running && (tool || status) && (
          <div className="mt-0.5 truncate text-[11px] text-[var(--color-fg-dim)]">
            {tool ? `→ ${tool}` : status}
          </div>
        )}
      </div>
      <button
        onClick={(e) => {
          e.preventDefault()
          e.stopPropagation()
          if (confirm(`Delete session "${titleText}"? Any running work will be cancelled.`)) {
            onDelete(session.id)
          }
        }}
        title="delete session"
        className="invisible shrink-0 px-1 text-[var(--color-fg-dim)] hover:text-[var(--color-bad)] group-hover:visible"
      >
        ×
      </button>
    </Link>
  )
}

export default function SessionSidebar() {
  const navigate = useNavigate()
  const { id: activeId } = useParams<{ id: string }>()
  const [sessions, setSessions] = useState<Session[]>([])
  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)

  // Poll the sessions list every few seconds so the running indicator stays
  // fresh even for sessions whose runs we haven't subscribed to in this tab.
  useEffect(() => {
    let cancelled = false
    async function refresh() {
      try {
        const sx = await listSessions()
        if (!cancelled) setSessions(sx)
      } catch {
        // Surface auth/network failures elsewhere; sidebar stays stale.
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    refresh()
    const tick = setInterval(refresh, 4000)
    return () => {
      cancelled = true
      clearInterval(tick)
    }
  }, [])

  async function handleCreate() {
    setCreating(true)
    try {
      const s = await createSession()
      setSessions((prev) => [s, ...prev])
      navigate(`/session/${s.id}`)
    } finally {
      setCreating(false)
    }
  }

  async function handleDelete(id: string) {
    try {
      await deleteSession(id)
      clearSession(id)
      setSessions((prev) => prev.filter((s) => s.id !== id))
      if (id === activeId) {
        const remaining = sessions.filter((s) => s.id !== id)
        navigate(remaining[0] ? `/session/${remaining[0].id}` : '/')
      }
    } catch (e) {
      console.error('delete session failed', e)
    }
  }

  return (
    <div className="flex h-full w-[260px] shrink-0 flex-col border-r border-[var(--color-border)] bg-[var(--color-bg)]">
      <div className="flex shrink-0 items-center justify-between border-b border-[var(--color-border)] px-3 py-2 text-xs uppercase tracking-wider text-[var(--color-fg-dim)]">
        <span>sessions</span>
        <button
          onClick={handleCreate}
          disabled={creating}
          className="rounded border border-[var(--color-border)] px-2 py-0.5 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)] hover:border-[var(--color-accent-dim)] hover:text-[var(--color-fg)] disabled:opacity-50"
          title="new session"
        >
          + new
        </button>
      </div>
      <div className="flex-1 overflow-y-auto">
        {loading && (
          <div className="px-3 py-3 text-[12px] text-[var(--color-fg-dim)]">loading…</div>
        )}
        {!loading && sessions.length === 0 && (
          <div className="px-3 py-3 text-[12px] text-[var(--color-fg-dim)]">
            no sessions yet — hit + new to start
          </div>
        )}
        {sessions.map((s) => (
          <SessionRow
            key={s.id}
            session={s}
            active={s.id === activeId}
            onDelete={handleDelete}
          />
        ))}
      </div>
    </div>
  )
}

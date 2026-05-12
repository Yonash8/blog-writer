import { useCallback, useEffect, useState } from 'react'
import {
  BrowserRouter,
  Navigate,
  Route,
  Routes,
  useNavigate,
} from 'react-router-dom'
import ChatPane from './panes/ChatPane'
import LogsPanel from './panes/LogsPanel'
import TracesView from './views/TracesView'
import ConfigDrawer from './drawers/ConfigDrawer'
import SessionSidebar from './SessionSidebar'
import { useGlobalKeymap, type KeyAction } from './lib/keymap'
import {
  checkAuth,
  clearApiKey,
  getAuthToken,
  login,
  setApiKey,
} from './lib/auth'
import { createSession, listSessions } from './lib/sessions'

/**
 * Password gate. Talks to /api/auth/login to mint a signed bearer token —
 * the raw password fallback is still accepted by the server, but the login
 * flow gives us a real 30d-TTL token + a properly labeled UX.
 */
function LoginGate({ onAuthed }: { onAuthed: () => void }) {
  const [val, setVal] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function submit() {
    if (!val.trim()) return
    setBusy(true)
    setError(null)
    try {
      const ok = await login(val.trim())
      if (ok) {
        onAuthed()
        return
      }
      // Fall back to legacy raw-key behavior so an operator with the
      // CONSOLE_PASSWORD set as-is in localStorage can still get in.
      setApiKey(val.trim())
      const remoteOk = await checkAuth()
      if (remoteOk) {
        onAuthed()
        return
      }
      clearApiKey()
      setError('Invalid password.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex h-full items-center justify-center bg-[var(--color-bg)]">
      <div className="w-full max-w-md rounded border border-[var(--color-border)] bg-[var(--color-panel)] p-6">
        <div className="mb-4 text-xs uppercase tracking-wider text-[var(--color-fg-dim)]">
          blog-writer :: sign in
        </div>
        <div className="mb-4 text-[var(--color-fg)]">
          Enter the <code className="text-[var(--color-accent)]">CONSOLE_PASSWORD</code>{' '}
          to access this console. After successful sign-in, a 30-day bearer
          token is stored in your browser.
        </div>
        <input
          type="password"
          value={val}
          onChange={(e) => setVal(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && val.trim()) void submit()
          }}
          autoFocus
          placeholder="password"
          disabled={busy}
          className="w-full rounded border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2 text-[var(--color-fg)] focus:border-[var(--color-accent)] focus:outline-none disabled:opacity-50"
        />
        <button
          onClick={() => void submit()}
          disabled={busy || !val.trim()}
          className="mt-3 w-full rounded bg-[var(--color-accent-dim)] px-3 py-2 text-[var(--color-fg)] hover:bg-[var(--color-accent)] disabled:opacity-50"
        >
          {busy ? 'signing in…' : 'sign in'}
        </button>
        {error && (
          <div className="mt-3 text-[12px] text-[var(--color-bad)]">{error}</div>
        )}
      </div>
    </div>
  )
}

/** Redirect "/" to the most recent session, creating one if none exist. */
function SessionLanding() {
  const navigate = useNavigate()
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const sessions = await listSessions()
        if (cancelled) return
        const target = sessions[0] ?? (await createSession())
        navigate(`/session/${target.id}`, { replace: true })
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e))
      }
    })()
    return () => {
      cancelled = true
    }
  }, [navigate])
  return (
    <div className="flex h-full items-center justify-center text-[var(--color-fg-dim)]">
      {error ? `error: ${error}` : 'opening session…'}
    </div>
  )
}

function AuthedApp() {
  const [logsOpen, setLogsOpen] = useState(false)
  const [tracesOpen, setTracesOpen] = useState(false)
  const [configOpen, setConfigOpen] = useState(false)

  const handleAction = useCallback((action: KeyAction) => {
    if (action === 'close') {
      setLogsOpen(false)
      setTracesOpen(false)
      setConfigOpen(false)
      return
    }
    if (action === 'toggle-logs') setLogsOpen((v) => !v)
    if (action === 'toggle-traces') setTracesOpen((v) => !v)
    if (action === 'toggle-config') setConfigOpen((v) => !v)
  }, [])
  useGlobalKeymap(handleAction)

  useEffect(() => {
    document.title = 'blog-writer :: console'
  }, [])

  return (
    <div className="relative flex h-full w-full flex-col overflow-hidden">
      {/* Top bar */}
      <div className="flex shrink-0 items-center justify-between border-b border-[var(--color-border)] bg-[var(--color-panel)] px-4 py-2 text-xs uppercase tracking-wider text-[var(--color-fg-dim)]">
        <div className="flex items-center gap-3">
          <span className="font-bold text-[var(--color-accent)]">blog-writer</span>
          <span className="text-[var(--color-fg-dim)]">::</span>
          <span>console</span>
        </div>
        <div className="flex items-center gap-2">
          <NavButton label="logs" hotkey="⌘L" active={logsOpen} onClick={() => setLogsOpen((v) => !v)} />
          <NavButton label="traces" hotkey="⌘T" active={tracesOpen} onClick={() => setTracesOpen((v) => !v)} />
          <NavButton label="config" hotkey="⌘," active={configOpen} onClick={() => setConfigOpen((v) => !v)} />
          <button
            onClick={() => {
              clearApiKey()
              window.location.reload()
            }}
            className="px-2 py-1 text-[var(--color-fg-dim)] hover:text-[var(--color-bad)]"
            title="sign out"
          >
            logout
          </button>
        </div>
      </div>

      {/* Body — sidebar | chat | side-docked logs */}
      <div className="flex flex-1 overflow-hidden">
        <SessionSidebar />
        <div className="min-w-0 flex-1 transition-[flex-basis] duration-200">
          <Routes>
            <Route path="/" element={<SessionLanding />} />
            <Route path="/session/:id" element={<ChatPane />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
        {logsOpen && (
          <div className="drawer-right w-[42%] min-w-[380px] max-w-[640px] shrink-0 overflow-hidden">
            <LogsPanel onClose={() => setLogsOpen(false)} />
          </div>
        )}
      </div>

      {/* Full-page traces view sits on top when active */}
      {tracesOpen && <TracesView onClose={() => setTracesOpen(false)} />}

      {/* Config remains a bottom drawer */}
      {configOpen && <ConfigDrawer onClose={() => setConfigOpen(false)} />}
    </div>
  )
}

export default function App() {
  const [authState, setAuthState] = useState<'checking' | 'in' | 'out'>('checking')

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      // First: if the server isn't enforcing a password (local dev), skip the
      // login gate entirely. /api/auth/me returns
      // { authenticated: false, reason: 'no_password_configured' } in that
      // case — we treat it as 'in'.
      try {
        const r = await fetch('/api/auth/me')
        if (r.ok) {
          const data = (await r.json()) as { authenticated?: boolean; reason?: string }
          if (data.reason === 'no_password_configured') {
            if (!cancelled) setAuthState('in')
            return
          }
        }
      } catch {
        // network/auth probe failed — fall through to the credential check.
      }

      if (!getAuthToken() && !localStorage.getItem('blog-writer.api-key')) {
        if (!cancelled) setAuthState('out')
        return
      }
      const ok = await checkAuth()
      if (!cancelled) setAuthState(ok ? 'in' : 'out')
    })()
    return () => {
      cancelled = true
    }
  }, [])

  if (authState === 'checking') {
    return (
      <div className="flex h-full items-center justify-center bg-[var(--color-bg)] text-[var(--color-fg-dim)]">
        loading…
      </div>
    )
  }
  if (authState === 'out') {
    return <LoginGate onAuthed={() => setAuthState('in')} />
  }

  return (
    <BrowserRouter basename="/console">
      <AuthedApp />
    </BrowserRouter>
  )
}

function NavButton({
  label,
  hotkey,
  active,
  onClick,
}: {
  label: string
  hotkey: string
  active: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={
        'flex items-center gap-2 rounded border px-2 py-1 transition-colors ' +
        (active
          ? 'border-[var(--color-accent)] bg-[var(--color-accent-dim)] text-[var(--color-fg)]'
          : 'border-[var(--color-border)] text-[var(--color-fg-dim)] hover:border-[var(--color-accent-dim)] hover:text-[var(--color-fg)]')
      }
    >
      <span>{label}</span>
      <span className="text-[10px] opacity-60">{hotkey}</span>
    </button>
  )
}

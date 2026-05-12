import { useCallback, useEffect, useState } from 'react'
import ChatPane from './panes/ChatPane'
import LogsPanel from './panes/LogsPanel'
import TracesView from './views/TracesView'
import ConfigDrawer from './drawers/ConfigDrawer'
import { useGlobalKeymap, type KeyAction } from './lib/keymap'
import { getApiKey, setApiKey, clearApiKey } from './lib/auth'

function ApiKeyGate({ onSet }: { onSet: () => void }) {
  const [val, setVal] = useState('')
  return (
    <div className="flex h-full items-center justify-center bg-[var(--color-bg)]">
      <div className="w-full max-w-md rounded border border-[var(--color-border)] bg-[var(--color-panel)] p-6">
        <div className="mb-4 text-xs uppercase tracking-wider text-[var(--color-fg-dim)]">
          blog-writer :: auth
        </div>
        <div className="mb-4 text-[var(--color-fg)]">
          Paste your <code className="text-[var(--color-accent)]">WEB_API_KEY</code> to enter the
          console. It will be stored in <code className="text-[var(--color-accent)]">localStorage</code>{' '}
          and sent as <code className="text-[var(--color-accent)]">X-API-Key</code> on every request.
        </div>
        <input
          type="password"
          value={val}
          onChange={(e) => setVal(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && val.trim()) {
              setApiKey(val.trim())
              onSet()
            }
          }}
          autoFocus
          placeholder="api key"
          className="w-full rounded border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2 text-[var(--color-fg)] focus:border-[var(--color-accent)] focus:outline-none"
        />
        <button
          onClick={() => {
            if (val.trim()) {
              setApiKey(val.trim())
              onSet()
            }
          }}
          className="mt-3 w-full rounded bg-[var(--color-accent-dim)] px-3 py-2 text-[var(--color-fg)] hover:bg-[var(--color-accent)]"
        >
          enter
        </button>
        <div className="mt-3 text-xs text-[var(--color-fg-dim)]">
          If running locally without auth, paste any non-empty value.
        </div>
      </div>
    </div>
  )
}

export default function App() {
  const [logsOpen, setLogsOpen] = useState(false)
  const [tracesOpen, setTracesOpen] = useState(false)
  const [configOpen, setConfigOpen] = useState(false)
  const [authed, setAuthed] = useState<boolean>(!!getApiKey())

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

  if (!authed) return <ApiKeyGate onSet={() => setAuthed(true)} />

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
              setAuthed(false)
            }}
            className="px-2 py-1 text-[var(--color-fg-dim)] hover:text-[var(--color-bad)]"
            title="clear api key"
          >
            logout
          </button>
        </div>
      </div>

      {/* Body — chat + side-docked logs panel */}
      <div className="flex flex-1 overflow-hidden">
        <div className="min-w-0 flex-1 transition-[flex-basis] duration-200">
          <ChatPane />
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

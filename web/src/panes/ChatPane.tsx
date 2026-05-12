import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useParams } from 'react-router-dom'
import { cancelSession, getSession, getSessionMessages, type Session } from '../lib/sessions'
import {
  attachIfRunning,
  setTranscript,
  snapshot,
  startRun,
  subscribe,
  type Line,
  type RunState,
} from '../store/runs'

const PROMPT = '>'

function fmtElapsed(ms: number): string {
  const s = Math.floor(ms / 1000)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  const rem = s % 60
  if (m < 60) return rem === 0 ? `${m}m` : `${m}m ${rem}s`
  const h = Math.floor(m / 60)
  const mr = m % 60
  return mr === 0 ? `${h}h` : `${h}h ${mr}m`
}

function LinePrefix({ kind }: { kind: Line['kind'] }) {
  switch (kind) {
    case 'user':
      return <span className="select-none font-semibold text-[var(--color-accent)]">Me:&nbsp;</span>
    case 'assistant':
      return <span className="select-none font-semibold text-[var(--color-good)]">Agent:&nbsp;</span>
    case 'status':
      return <span className="select-none text-[var(--color-fg-dim)]">·&nbsp;</span>
    case 'error':
      return <span className="select-none font-semibold text-[var(--color-bad)]">Error:&nbsp;</span>
    case 'system':
      return <span className="select-none text-[var(--color-fg-dim)]">#&nbsp;</span>
  }
}

function Markdown({ text }: { text: string }) {
  return (
    <div className="md-body inline">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ children, ...props }) => (
            <a {...props} target="_blank" rel="noreferrer" className="text-[var(--color-accent)] underline hover:text-[var(--color-fg)]">
              {children}
            </a>
          ),
          code: ({ children, className }) => {
            const isBlock = /language-/.test(className || '')
            if (isBlock) {
              return (
                <pre className="my-2 overflow-x-auto rounded border border-[var(--color-border)] bg-[var(--color-panel)] px-3 py-2 text-[12px]">
                  <code>{children}</code>
                </pre>
              )
            }
            return (
              <code className="rounded bg-[var(--color-border)] px-1 py-0.5 text-[var(--color-accent)]">{children}</code>
            )
          },
          ul: ({ children }) => <ul className="my-1 ml-4 list-disc">{children}</ul>,
          ol: ({ children }) => <ol className="my-1 ml-4 list-decimal">{children}</ol>,
          li: ({ children }) => <li className="my-0.5">{children}</li>,
          h1: ({ children }) => <h1 className="my-2 text-base font-bold text-[var(--color-fg)]">{children}</h1>,
          h2: ({ children }) => <h2 className="my-2 text-sm font-bold text-[var(--color-fg)]">{children}</h2>,
          h3: ({ children }) => <h3 className="my-1 text-sm font-semibold text-[var(--color-fg)]">{children}</h3>,
          p: ({ children }) => <p className="my-1">{children}</p>,
          strong: ({ children }) => <strong className="text-[var(--color-fg)]">{children}</strong>,
          em: ({ children }) => <em className="text-[var(--color-fg)]">{children}</em>,
          blockquote: ({ children }) => (
            <blockquote className="my-1 border-l-2 border-[var(--color-accent-dim)] pl-2 text-[var(--color-fg-dim)]">
              {children}
            </blockquote>
          ),
          hr: () => <hr className="my-2 border-[var(--color-border)]" />,
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  )
}

function transcriptFromServerMessages(msgs: { role: string; content: string }[]): Line[] {
  return msgs.map((m) => {
    if (m.role === 'user') return { kind: 'user', text: m.content }
    if (m.role === 'assistant') {
      // The backend writes `[cancelled]` / `[error: ...]` stubs when a run
      // fails so the transcript reflects what was attempted — tag them.
      if (m.content.startsWith('[cancelled]')) return { kind: 'system', text: '[cancelled]' }
      if (m.content.startsWith('[error:')) return { kind: 'error', text: m.content.slice(1, -1) }
      return { kind: 'assistant', text: m.content }
    }
    return { kind: 'system', text: m.content }
  })
}

export default function ChatPane() {
  const { id: sessionId } = useParams<{ id: string }>()
  const [state, setState] = useState<RunState | null>(
    sessionId ? snapshot(sessionId) : null,
  )
  const [session, setSession] = useState<Session | null>(null)
  const [input, setInput] = useState('')
  const [now, setNow] = useState<number>(Date.now())
  const scrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const tailRef = useRef<HTMLDivElement>(null)

  // Re-bind the subscription whenever the route switches to a new session.
  useEffect(() => {
    if (!sessionId) return
    setState({ ...snapshot(sessionId) })
    const unsub = subscribe(sessionId, (s) => setState({ ...s }))
    return unsub
  }, [sessionId])

  // Fetch session metadata (mainly: title) on mount + whenever a run ends,
  // so the Haiku-generated title appears automatically after the first
  // assistant reply without the user having to refresh. Haiku runs in a
  // daemon thread on the backend, so the title may not be set the instant
  // the run finishes — poll a few times if it's still empty.
  useEffect(() => {
    if (!sessionId) return
    let cancelled = false
    let attempt = 0
    const maxAttempts = 8 // ~16s total at 2s spacing
    async function load() {
      try {
        const s = await getSession(sessionId!)
        if (cancelled) return
        setSession(s)
        if (!(s.title || '').trim() && attempt < maxAttempts) {
          attempt += 1
          setTimeout(load, 2000)
        }
      } catch (e) {
        console.error('load session metadata failed', e)
      }
    }
    load()
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, state?.running])

  // Hydrate transcript from the server once per session, then attach to any
  // in-flight run so the live token stream resumes from where we left off.
  useEffect(() => {
    if (!sessionId) return
    const cur = snapshot(sessionId)
    let cancelled = false
    if (!cur.hydrated) {
      getSessionMessages(sessionId)
        .then((msgs) => {
          if (cancelled) return
          setTranscript(sessionId, transcriptFromServerMessages(msgs))
        })
        .catch((e) => console.error('hydrate session messages failed', e))
    }
    void attachIfRunning(sessionId)
    return () => {
      cancelled = true
      // Intentionally do NOT abort the SSE — leaving this session must not
      // stop its run. The subscription stays alive in runs.ts.
    }
  }, [sessionId])

  useEffect(() => {
    if (!state?.activeStatus) return
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [state?.activeStatus])

  useEffect(() => {
    tailRef.current?.scrollIntoView({ block: 'end', behavior: 'auto' })
  }, [state?.transcript, state?.streamBuf, state?.activeStatus, state?.running])

  useEffect(() => {
    if (!state?.running) inputRef.current?.focus()
  }, [state?.running])

  if (!sessionId || !state) {
    return (
      <div className="flex h-full items-center justify-center text-[var(--color-fg-dim)]">
        no session selected
      </div>
    )
  }

  async function send() {
    if (!sessionId) return
    const msg = input.trim()
    if (!msg || state?.running) return
    setInput('')
    await startRun(sessionId, msg)
  }

  async function handleCancel() {
    if (!sessionId) return
    await cancelSession(sessionId)
  }

  const { transcript, streamBuf, activeStatus, running, lastTool } = state
  return (
    <div
      className="flex h-full flex-col bg-[var(--color-bg)]"
      onClick={(e) => {
        const target = e.target as HTMLElement
        if (target.tagName === 'A' || target.tagName === 'BUTTON') return
        const sel = window.getSelection()
        if (sel && sel.toString().length > 0) return
        inputRef.current?.focus()
      }}
    >
      <div className="flex shrink-0 items-center justify-between border-b border-[var(--color-border)] px-4 py-2 text-xs uppercase tracking-wider text-[var(--color-fg-dim)]">
        <div className="flex min-w-0 items-center gap-3">
          <span className="shrink-0">chat</span>
          <span
            className="truncate normal-case text-[12px] text-[var(--color-fg)]"
            title={sessionId}
          >
            {(session?.title || '').trim() || 'Untitled session'}
          </span>
        </div>
        <div className="flex items-center gap-3">
          {running && (
            <button
              onClick={handleCancel}
              title="cancel run"
              className="rounded border border-[var(--color-border)] px-2 py-0.5 text-[10px] uppercase tracking-wider text-[var(--color-bad)] hover:border-[var(--color-bad)]"
            >
              cancel
            </button>
          )}
          <span
            className={running ? 'text-[var(--color-warn)]' : 'text-[var(--color-good)]'}
            title={lastTool ? `running tool: ${lastTool}` : undefined}
          >
            {running ? `● running${lastTool ? ` · ${lastTool}` : ''}` : '○ idle'}
          </span>
        </div>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-3 text-[14px] leading-relaxed">
        {transcript.map((line, i) => (
          <div key={i} className="py-0.5">
            {line.kind === 'assistant' ? (
              <div className="flex items-start gap-1">
                <LinePrefix kind={line.kind} />
                <div className="min-w-0 flex-1">
                  <Markdown text={line.text} />
                </div>
              </div>
            ) : (
              <div className="whitespace-pre-wrap break-words">
                <LinePrefix kind={line.kind} />
                <span
                  className={
                    line.kind === 'status'
                      ? 'text-[var(--color-fg-dim)]'
                      : line.kind === 'system'
                        ? 'text-[var(--color-fg-dim)]'
                        : line.kind === 'error'
                          ? 'text-[var(--color-bad)]'
                          : 'text-[var(--color-fg)]'
                  }
                >
                  {line.text}
                </span>
              </div>
            )}
          </div>
        ))}

        {activeStatus && (
          <div className="py-0.5 whitespace-pre-wrap break-words">
            <span className="select-none text-[var(--color-fg-dim)]">·&nbsp;</span>
            <span className="text-[var(--color-fg-dim)]">{activeStatus.text}</span>
            <span className="ml-2 text-[var(--color-fg-dim)] opacity-70">
              ({fmtElapsed(now - activeStatus.startedAt)})
            </span>
          </div>
        )}

        {streamBuf && (
          <div className="py-0.5">
            <div className="flex items-start gap-1">
              <span className="select-none font-semibold text-[var(--color-good)]">Agent:&nbsp;</span>
              <div className="min-w-0 flex-1">
                <Markdown text={streamBuf} />
                <span className="cursor-blink ml-0.5 text-[var(--color-fg-dim)]">█</span>
              </div>
            </div>
          </div>
        )}

        {running && !streamBuf && !activeStatus && (
          <div className="py-0.5 text-[var(--color-fg-dim)]">
            <span className="select-none">·&nbsp;</span>
            <span className="cursor-blink">█</span>
          </div>
        )}

        {!running && (
          <div className="flex items-baseline py-0.5">
            <span className="select-none text-[var(--color-accent)]">{PROMPT}&nbsp;</span>
            <input
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  send()
                }
              }}
              autoFocus
              spellCheck={true}
              autoComplete="on"
              autoCorrect="on"
              autoCapitalize="sentences"
              className="flex-1 border-none bg-transparent p-0 font-[var(--font-mono)] text-[14px] text-[var(--color-fg)] placeholder:text-[var(--color-fg-dim)] focus:outline-none focus:ring-0"
            />
          </div>
        )}

        <div ref={tailRef} />
      </div>
    </div>
  )
}

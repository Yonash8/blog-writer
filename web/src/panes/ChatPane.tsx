import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { chatStream } from '../lib/api'
import { getChatId, rotateChatId } from '../lib/conversation'

type Line =
  | { kind: 'user'; text: string }
  | { kind: 'status'; text: string }
  | { kind: 'assistant'; text: string }
  | { kind: 'error'; text: string }
  | { kind: 'system'; text: string }

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

export default function ChatPane() {
  const [lines, setLines] = useState<Line[]>([])
  const [streamBuf, setStreamBuf] = useState<string>('')
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [chatId, setChatId] = useState<string>(() => getChatId())
  // Single in-place "active" status line — repeated status polls just update
  // .text and let the clock keep ticking.
  const [activeStatus, setActiveStatus] = useState<{ text: string; startedAt: number } | null>(null)
  const [now, setNow] = useState<number>(Date.now())

  function newConversation() {
    const id = rotateChatId()
    setChatId(id)
    setLines([])
    setStreamBuf('')
    setInput('')
    setActiveStatus(null)
  }
  const scrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const tailRef = useRef<HTMLDivElement>(null)

  // Tick the clock once a second while a status is active.
  useEffect(() => {
    if (!activeStatus) return
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [activeStatus])

  // Scroll the bottom of the transcript into view whenever content changes.
  useEffect(() => {
    tailRef.current?.scrollIntoView({ block: 'end', behavior: 'auto' })
  }, [lines, streamBuf, busy, activeStatus])

  // Keep the input focused unless the user is selecting text elsewhere.
  useEffect(() => {
    if (!busy) inputRef.current?.focus()
  }, [busy])

  function append(line: Line) {
    setLines((prev) => [...prev, line])
  }

  async function send() {
    const msg = input.trim()
    if (!msg || busy) return
    // Slash commands handled client-side
    if (msg === '/new' || msg === '/clear' || msg === '/reset') {
      newConversation()
      return
    }
    setInput('')
    setBusy(true)
    setStreamBuf('')
    setActiveStatus(null)
    append({ kind: 'user', text: msg })

    // Finalize the active status (if any) into a permanent transcript line,
    // freezing the elapsed time at the moment it was superseded.
    const finalizeActive = () => {
      setActiveStatus((cur) => {
        if (cur) {
          const elapsed = fmtElapsed(Date.now() - cur.startedAt)
          setLines((prev) => [...prev, { kind: 'status', text: `${cur.text} (${elapsed})` }])
        }
        return null
      })
    }

    try {
      for await (const evt of chatStream(msg, chatId)) {
        if (evt.type === 'status') {
          // Strip any trailing "(30s)" / "(1m 30s)" the backend may include —
          // we render our own ticking clock.
          const cleaned = evt.text.replace(/\s*\(\d+\s*[hms](?:\s+\d+\s*[hms])*\)\s*$/i, '').trim()
          setActiveStatus((cur) => ({
            text: cleaned,
            startedAt: cur?.startedAt ?? Date.now(),
          }))
        } else if (evt.type === 'token') {
          // First token after a status block → freeze the status line
          finalizeActive()
          // If a tool ran between assistant turns, commit any prior streaming
          // text so subsequent tokens start on a fresh line.
          setStreamBuf((cur) => cur + evt.text)
        } else if (evt.type === 'done') {
          finalizeActive()
          setStreamBuf('')
          if (evt.error) {
            append({ kind: 'error', text: evt.message })
          } else {
            append({ kind: 'assistant', text: evt.message })
          }
        }
      }
    } catch (err) {
      finalizeActive()
      setStreamBuf('')
      append({ kind: 'error', text: `request failed: ${err instanceof Error ? err.message : String(err)}` })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div
      className="flex h-full flex-col bg-[var(--color-bg)]"
      onClick={(e) => {
        // Click anywhere in the pane → refocus the prompt, unless the user
        // is interacting with a link/button or selecting text.
        const target = e.target as HTMLElement
        if (target.tagName === 'A' || target.tagName === 'BUTTON') return
        const sel = window.getSelection()
        if (sel && sel.toString().length > 0) return
        inputRef.current?.focus()
      }}
    >
      <div className="flex shrink-0 items-center justify-between border-b border-[var(--color-border)] px-4 py-2 text-xs uppercase tracking-wider text-[var(--color-fg-dim)]">
        <div className="flex items-center gap-3">
          <span>chat</span>
          <span className="normal-case text-[10px] text-[var(--color-fg-dim)]">
            {chatId.slice(0, 12)}…
          </span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={newConversation}
            disabled={busy}
            title="start a new conversation (/new)"
            className="rounded border border-[var(--color-border)] px-2 py-0.5 text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)] hover:border-[var(--color-accent-dim)] hover:text-[var(--color-fg)] disabled:opacity-50"
          >
            + new
          </button>
          <span className={busy ? 'text-[var(--color-warn)]' : 'text-[var(--color-good)]'}>
            {busy ? '● running' : '○ idle'}
          </span>
        </div>
      </div>

      {/* One scrollable terminal — transcript and the live prompt share the same flow. */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-3 text-[14px] leading-relaxed">
        {lines.map((line, i) => (
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

        {/* Live in-place status line — backend polls collapse here, clock ticks. */}
        {activeStatus && (
          <div className="py-0.5 whitespace-pre-wrap break-words">
            <span className="select-none text-[var(--color-fg-dim)]">·&nbsp;</span>
            <span className="text-[var(--color-fg-dim)]">{activeStatus.text}</span>
            <span className="ml-2 text-[var(--color-fg-dim)] opacity-70">
              ({fmtElapsed(now - activeStatus.startedAt)})
            </span>
          </div>
        )}

        {/* Streaming assistant tokens (live preview) */}
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

        {/* Busy with no streamed tokens yet — show a "thinking" indicator inline */}
        {busy && !streamBuf && (
          <div className="py-0.5 text-[var(--color-fg-dim)]">
            <span className="select-none">·&nbsp;</span>
            <span className="cursor-blink">█</span>
          </div>
        )}

        {/* Active prompt — the input lives inside the transcript flow */}
        {!busy && (
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

// Per-session run controller.
//
// Lives outside React so SSE subscriptions survive component unmounts —
// switching sessions in the sidebar must NOT kill the in-flight stream of
// the session you just left.
//
// Each session has one RunState (running flag, transcript, streaming buffer,
// last-status, last-tool) and at most one active AbortController for its SSE
// subscription. React components call subscribe(sessionId, callback) to
// re-render when state changes, and read snapshot(sessionId) for the current
// view.
//
// On page reload the in-memory state is gone but the *server's* run buffer
// is retained for ~5 min. attachIfRunning re-hooks the stream using the
// last_seq we previously stored in sessionStorage so token replay catches up.

import { authHeaders } from '../lib/auth'

export type Line =
  | { kind: 'user'; text: string }
  | { kind: 'status'; text: string }
  | { kind: 'assistant'; text: string }
  | { kind: 'error'; text: string }
  | { kind: 'system'; text: string }

export interface ActiveStatus {
  text: string
  startedAt: number
}

export interface RunState {
  sessionId: string
  transcript: Line[]
  streamBuf: string
  activeStatus: ActiveStatus | null
  running: boolean
  lastSeq: number
  lastTool: string | null
  lastStatus: string | null
  // True when we successfully hydrated transcript from the server.
  hydrated: boolean
}

type Listener = (state: RunState) => void

const _states = new Map<string, RunState>()
const _listeners = new Map<string, Set<Listener>>()
const _subscriptions = new Map<string, AbortController>()

function _seqKey(sessionId: string): string {
  return `blog-writer.last-seq.${sessionId}`
}

function _loadLastSeq(sessionId: string): number {
  const raw = sessionStorage.getItem(_seqKey(sessionId))
  return raw ? Number(raw) || 0 : 0
}

function _saveLastSeq(sessionId: string, seq: number): void {
  try {
    sessionStorage.setItem(_seqKey(sessionId), String(seq))
  } catch {
    // sessionStorage full / disabled — non-fatal.
  }
}

function _emptyState(sessionId: string): RunState {
  return {
    sessionId,
    transcript: [],
    streamBuf: '',
    activeStatus: null,
    running: false,
    lastSeq: _loadLastSeq(sessionId),
    lastTool: null,
    lastStatus: null,
    hydrated: false,
  }
}

function _getOrCreate(sessionId: string): RunState {
  let s = _states.get(sessionId)
  if (!s) {
    s = _emptyState(sessionId)
    _states.set(sessionId, s)
  }
  return s
}

function _notify(sessionId: string): void {
  const s = _states.get(sessionId)
  if (!s) return
  const subs = _listeners.get(sessionId)
  if (!subs) return
  for (const cb of subs) cb(s)
}

function _patch(sessionId: string, patch: Partial<RunState>): void {
  const s = _getOrCreate(sessionId)
  Object.assign(s, patch)
  _notify(sessionId)
}

export function snapshot(sessionId: string): RunState {
  return _getOrCreate(sessionId)
}

export function subscribe(sessionId: string, cb: Listener): () => void {
  let subs = _listeners.get(sessionId)
  if (!subs) {
    subs = new Set()
    _listeners.set(sessionId, subs)
  }
  subs.add(cb)
  return () => {
    subs!.delete(cb)
  }
}

export function setTranscript(sessionId: string, transcript: Line[]): void {
  _patch(sessionId, { transcript, hydrated: true })
}

export function clearSession(sessionId: string): void {
  const sub = _subscriptions.get(sessionId)
  if (sub) {
    sub.abort()
    _subscriptions.delete(sessionId)
  }
  _states.delete(sessionId)
  sessionStorage.removeItem(_seqKey(sessionId))
  _notify(sessionId)
}

// ---------------------------------------------------------------------------
// SSE event handler
// ---------------------------------------------------------------------------

type WireEvent =
  | { seq?: number; type: 'snapshot'; running: boolean; last_seq?: number; last_tool: string | null; last_status: string | null }
  | { seq: number; type: 'status'; text: string }
  | { seq: number; type: 'token'; text: string }
  | { seq: number; type: 'done'; message: string; article?: unknown }
  | { seq: number; type: 'error'; message?: string; code?: string }
  | { seq: number; type: 'idle'; running: boolean }
  | { seq: number; type: 'tool_start'; name: string }
  | { seq: number; type: 'tool_end'; name: string }

function _handleEvent(sessionId: string, ev: WireEvent): void {
  const s = _getOrCreate(sessionId)
  if (typeof ev.seq === 'number' && ev.seq > s.lastSeq) {
    s.lastSeq = ev.seq
    _saveLastSeq(sessionId, ev.seq)
  }
  switch (ev.type) {
    case 'snapshot': {
      s.running = ev.running
      s.lastTool = ev.last_tool
      s.lastStatus = ev.last_status
      break
    }
    case 'status': {
      // Strip trailing "(30s)" / "(1m 30s)" from old-format status lines —
      // we render our own ticking clock.
      const cleaned = ev.text.replace(/\s*\(\d+\s*[hms](?:\s+\d+\s*[hms])*\)\s*$/i, '').trim()
      s.lastStatus = cleaned
      s.activeStatus = { text: cleaned, startedAt: s.activeStatus?.startedAt ?? Date.now() }
      s.running = true
      break
    }
    case 'token': {
      // First token after a status block: freeze the status into the transcript.
      if (s.activeStatus) {
        s.transcript = [...s.transcript, { kind: 'status', text: s.activeStatus.text }]
        s.activeStatus = null
      }
      s.streamBuf += ev.text
      s.running = true
      break
    }
    case 'tool_start': {
      s.lastTool = ev.name
      break
    }
    case 'tool_end': {
      s.lastTool = null
      break
    }
    case 'done': {
      if (s.activeStatus) {
        s.transcript = [...s.transcript, { kind: 'status', text: s.activeStatus.text }]
        s.activeStatus = null
      }
      s.streamBuf = ''
      s.transcript = [...s.transcript, { kind: 'assistant', text: ev.message }]
      s.running = false
      s.lastTool = null
      break
    }
    case 'error': {
      if (s.activeStatus) {
        s.transcript = [...s.transcript, { kind: 'status', text: s.activeStatus.text }]
        s.activeStatus = null
      }
      s.streamBuf = ''
      s.transcript = [...s.transcript, { kind: 'error', text: ev.message ?? 'error' }]
      s.running = false
      s.lastTool = null
      break
    }
    case 'idle': {
      // Server timed out our long-poll; reattach from where we left off if
      // still running, otherwise stop.
      if (ev.running) {
        // Fire-and-forget reattach. Don't block the event loop here.
        void attachIfRunning(sessionId)
      } else {
        s.running = false
      }
      break
    }
  }
  _notify(sessionId)
}

// ---------------------------------------------------------------------------
// SSE consumer — shared by startRun and attachIfRunning.
// ---------------------------------------------------------------------------

async function _consumeStream(sessionId: string, response: Response): Promise<void> {
  if (!response.body) return
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  try {
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
              _handleEvent(sessionId, JSON.parse(line.slice(6)) as WireEvent)
            } catch {
              // skip malformed
            }
          }
        }
      }
    }
  } finally {
    _subscriptions.delete(sessionId)
  }
}

export async function startRun(sessionId: string, message: string): Promise<void> {
  const s = _getOrCreate(sessionId)
  // Optimistically append the user line so it shows up immediately.
  s.transcript = [...s.transcript, { kind: 'user', text: message }]
  s.streamBuf = ''
  s.activeStatus = null
  s.running = true
  _notify(sessionId)

  const ctl = new AbortController()
  // Cancel any prior subscription for this session before starting a new run.
  _subscriptions.get(sessionId)?.abort()
  _subscriptions.set(sessionId, ctl)

  try {
    const r = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/chat`, {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ message }),
      signal: ctl.signal,
    })
    if (!r.ok) {
      // 404 means the session was deleted server-side. Drop local state
      // and bounce to / so SessionLanding picks/creates a real one,
      // rather than leaving the user stuck sending into a dead URL.
      if (r.status === 404) {
        clearSession(sessionId)
        window.location.replace('/console/')
        return
      }
      _handleEvent(sessionId, { seq: s.lastSeq + 1, type: 'error', message: `request failed: ${r.status}` })
      return
    }
    await _consumeStream(sessionId, r)
  } catch (err) {
    if ((err as DOMException)?.name === 'AbortError') return
    _handleEvent(sessionId, {
      seq: snapshot(sessionId).lastSeq + 1,
      type: 'error',
      message: `request failed: ${err instanceof Error ? err.message : String(err)}`,
    })
  }
}

export async function attachIfRunning(sessionId: string): Promise<void> {
  // Don't double-attach.
  if (_subscriptions.has(sessionId)) return

  const ctl = new AbortController()
  _subscriptions.set(sessionId, ctl)
  const fromSeq = _loadLastSeq(sessionId)

  try {
    const url = `/api/sessions/${encodeURIComponent(sessionId)}/stream?from_seq=${fromSeq}`
    const r = await fetch(url, { headers: authHeaders(), signal: ctl.signal })
    if (!r.ok) {
      _subscriptions.delete(sessionId)
      return
    }
    await _consumeStream(sessionId, r)
  } catch (err) {
    if ((err as DOMException)?.name === 'AbortError') return
    // Network glitch — drop subscription so next snapshot can retry.
    _subscriptions.delete(sessionId)
  }
}

export function abortRun(sessionId: string): void {
  _subscriptions.get(sessionId)?.abort()
  _subscriptions.delete(sessionId)
}

// REST client for the /api/sessions/* endpoints.
//
// All calls go through authHeaders() so the bearer token is attached.
// The shape mirrors the backend Pydantic models in src/main.py.

import { authHeaders } from './auth'

export interface Session {
  id: string
  channel: string
  channel_user_id: string
  title: string | null
  created_at: string
  updated_at: string
  deleted_at?: string | null
  // Live run state — populated by the backend from the in-memory run buffer.
  running: boolean
  last_tool: string | null
  last_status: string | null
  run_id: string | null
}

export interface SessionMessage {
  id: string
  session_id: string
  channel: string
  channel_user_id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  created_at: string
}

export async function listSessions(): Promise<Session[]> {
  const r = await fetch('/api/sessions', { headers: authHeaders() })
  if (!r.ok) throw new Error(`list sessions: ${r.status}`)
  const data = (await r.json()) as { sessions: Session[] }
  return data.sessions
}

export async function getSession(sessionId: string): Promise<Session> {
  const r = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {
    headers: authHeaders(),
  })
  if (!r.ok) throw new Error(`get session: ${r.status}`)
  return r.json()
}

export async function createSession(title?: string): Promise<Session> {
  const r = await fetch('/api/sessions', {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ title: title ?? null }),
  })
  if (!r.ok) throw new Error(`create session: ${r.status}`)
  return r.json()
}

export async function renameSession(sessionId: string, title: string): Promise<Session> {
  const r = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'PATCH',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ title }),
  })
  if (!r.ok) throw new Error(`rename session: ${r.status}`)
  return r.json()
}

export async function deleteSession(sessionId: string): Promise<void> {
  const r = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
    headers: authHeaders(),
  })
  if (!r.ok) throw new Error(`delete session: ${r.status}`)
}

export async function getSessionMessages(sessionId: string): Promise<SessionMessage[]> {
  const r = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/messages`, {
    headers: authHeaders(),
  })
  if (!r.ok) throw new Error(`messages: ${r.status}`)
  const data = (await r.json()) as { messages: SessionMessage[] }
  return data.messages
}

export async function cancelSession(sessionId: string): Promise<void> {
  await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/cancel`, {
    method: 'POST',
    headers: authHeaders(),
  })
}

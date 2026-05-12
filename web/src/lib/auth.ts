// Auth token storage + helpers.
//
// Two credentials may live in localStorage:
//   - `blog-writer.auth-token`: the signed bearer token returned by POST
//     /api/auth/login. Preferred — has a 30-day TTL the server enforces.
//   - `blog-writer.api-key`: legacy raw-password fallback (kept so anyone with
//     a copy/pasted WEB_API_KEY can still bypass the login form).
//
// `authHeaders()` always sends the token if present, falling back to the raw
// key. Every authenticated request goes through this so the gate stays
// consistent.

const TOKEN_STORAGE = 'blog-writer.auth-token'
const LEGACY_KEY_STORAGE = 'blog-writer.api-key'

export function getAuthToken(): string | null {
  return localStorage.getItem(TOKEN_STORAGE)
}

export function setAuthToken(token: string): void {
  localStorage.setItem(TOKEN_STORAGE, token)
}

export function clearAuthToken(): void {
  localStorage.removeItem(TOKEN_STORAGE)
}

// Legacy aliases — keep so older code paths still compile during the refactor.
export function getApiKey(): string | null {
  return getAuthToken() || localStorage.getItem(LEGACY_KEY_STORAGE)
}

export function setApiKey(key: string): void {
  // Treat any value pasted into the legacy gate as a raw password.
  localStorage.setItem(LEGACY_KEY_STORAGE, key)
}

export function clearApiKey(): void {
  localStorage.removeItem(LEGACY_KEY_STORAGE)
  clearAuthToken()
}

export function authHeaders(extra: HeadersInit = {}): HeadersInit {
  const cred = getApiKey()
  return cred ? { ...extra, Authorization: `Bearer ${cred}` } : extra
}

export async function login(password: string): Promise<boolean> {
  const r = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ password }),
  })
  if (!r.ok) return false
  const data = (await r.json()) as { token?: string }
  if (!data.token) return false
  setAuthToken(data.token)
  // Drop any legacy raw-key entry now that we have a real signed token.
  localStorage.removeItem(LEGACY_KEY_STORAGE)
  return true
}

export async function checkAuth(): Promise<boolean> {
  if (!getApiKey()) return false
  try {
    const r = await fetch('/api/auth/me', { headers: authHeaders() })
    if (!r.ok) return false
    const data = (await r.json()) as { authenticated?: boolean }
    return !!data.authenticated
  } catch {
    return false
  }
}

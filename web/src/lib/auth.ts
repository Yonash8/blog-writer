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

/**
 * Install a global fetch interceptor so any 401 from `/api/*` (except the
 * auth endpoints themselves, which legitimately return 401 on bad
 * password) drops our credentials and bounces the user to the login
 * screen. Without this, a stale token sits in localStorage forever and
 * the SPA keeps making doomed requests.
 */
let _interceptorInstalled = false
export function installAuthInterceptor(): void {
  if (_interceptorInstalled) return
  _interceptorInstalled = true
  const orig = window.fetch.bind(window)
  window.fetch = async (input, init) => {
    const r = await orig(input, init)
    if (r.status === 401) {
      const url =
        typeof input === 'string'
          ? input
          : input instanceof URL
            ? input.toString()
            : input.url
      // Strip origin so the check works for both relative and absolute URLs.
      const path = url.replace(/^https?:\/\/[^/]+/, '')
      const isApi = path.startsWith('/api/')
      const isAuthEndpoint =
        path.startsWith('/api/auth/login') || path.startsWith('/api/auth/me')
      if (isApi && !isAuthEndpoint) {
        clearApiKey()
        // Hard reload so React state, runs.ts cached subscriptions, and any
        // background polls all reset cleanly. The next boot lands on
        // LoginGate because /api/auth/me reports unauthenticated.
        window.location.replace('/console/')
      }
    }
    return r
  }
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

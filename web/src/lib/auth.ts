const KEY_STORAGE = 'blog-writer.api-key'

export function getApiKey(): string | null {
  return localStorage.getItem(KEY_STORAGE)
}

export function setApiKey(key: string): void {
  localStorage.setItem(KEY_STORAGE, key)
}

export function clearApiKey(): void {
  localStorage.removeItem(KEY_STORAGE)
}

export function authHeaders(extra: HeadersInit = {}): HeadersInit {
  const key = getApiKey()
  return key ? { ...extra, 'X-API-Key': key } : extra
}

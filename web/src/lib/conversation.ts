const KEY = 'blog-writer.chat-id'

function newId(): string {
  // Short-ish, stable across reloads. crypto.randomUUID is available in
  // every browser that runs Vite-built JS.
  return 'web-' + crypto.randomUUID()
}

export function getChatId(): string {
  let id = localStorage.getItem(KEY)
  if (!id) {
    id = newId()
    localStorage.setItem(KEY, id)
  }
  return id
}

export function rotateChatId(): string {
  const id = newId()
  localStorage.setItem(KEY, id)
  return id
}

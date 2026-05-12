import { useEffect } from 'react'

export type KeyAction = 'toggle-logs' | 'toggle-traces' | 'toggle-config' | 'close'

export function useGlobalKeymap(onAction: (action: KeyAction) => void): void {
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const meta = e.metaKey || e.ctrlKey
      if (e.key === 'Escape') {
        onAction('close')
        return
      }
      if (!meta) return
      const k = e.key.toLowerCase()
      if (k === 'l') {
        e.preventDefault()
        onAction('toggle-logs')
      } else if (k === 't') {
        e.preventDefault()
        onAction('toggle-traces')
      } else if (k === ',') {
        e.preventDefault()
        onAction('toggle-config')
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onAction])
}

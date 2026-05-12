export function CloseButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      title="close (esc)"
      aria-label="close"
      className="flex h-6 w-6 items-center justify-center rounded text-[var(--color-fg-dim)] transition-colors hover:bg-[var(--color-border)] hover:text-[var(--color-bad)]"
    >
      <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
        <line x1="2" y1="2" x2="10" y2="10" />
        <line x1="10" y1="2" x2="2" y2="10" />
      </svg>
    </button>
  )
}

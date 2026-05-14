import { useEffect, useState } from 'react'
import { getAgents, patchAgentConfig, patchPrompts, type AgentDefinition } from '../lib/api'
import { CloseButton } from '../lib/ui'

export default function ConfigDrawer({ onClose }: { onClose: () => void }) {
  const [agents, setAgents] = useState<AgentDefinition[] | null>(null)
  const [activeId, setActiveId] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [pendingConfig, setPendingConfig] = useState<Record<string, string>>({})
  const [pendingPrompts, setPendingPrompts] = useState<Record<string, string>>({})

  useEffect(() => {
    getAgents()
      .then((r) => {
        setAgents(r.agents)
        if (r.agents.length > 0) setActiveId(r.agents[0].id)
      })
      .catch((e) => setErr(String(e)))
  }, [])

  const active = agents?.find((a) => a.id === activeId) ?? null

  function setConfigVal(key: string, value: string) {
    setPendingConfig((p) => ({ ...p, [key]: value }))
  }

  function setPromptVal(key: string, value: string) {
    setPendingPrompts((p) => ({ ...p, [key]: value }))
  }

  async function save() {
    setSaving(true)
    setErr(null)
    try {
      if (Object.keys(pendingConfig).length > 0) {
        await patchAgentConfig(pendingConfig)
      }
      if (Object.keys(pendingPrompts).length > 0) {
        await patchPrompts(pendingPrompts)
      }
      // Refresh
      const r = await getAgents()
      setAgents(r.agents)
      setPendingConfig({})
      setPendingPrompts({})
    } catch (e) {
      setErr(String(e))
    } finally {
      setSaving(false)
    }
  }

  const dirty = Object.keys(pendingConfig).length > 0 || Object.keys(pendingPrompts).length > 0

  return (
    <div className="drawer-bottom absolute inset-x-0 bottom-0 z-30 flex h-[75vh] flex-col border-t border-[var(--color-border)] bg-[var(--color-panel)] shadow-2xl">
      <div className="flex items-center justify-between border-b border-[var(--color-border)] px-4 py-2 text-xs uppercase tracking-wider">
        <span className="text-[var(--color-accent)]">config / agents</span>
        <div className="flex items-center gap-2">
          {dirty && (
            <button
              onClick={save}
              disabled={saving}
              className="rounded border border-[var(--color-good)] bg-[var(--color-good)]/10 px-3 py-0.5 text-[var(--color-good)] disabled:opacity-50"
            >
              {saving ? 'saving…' : 'save changes'}
            </button>
          )}
          <CloseButton onClick={onClose} />
        </div>
      </div>

      {err && <div className="border-b border-[var(--color-bad)] bg-[var(--color-bad)]/10 px-4 py-2 text-[var(--color-bad)]">{err}</div>}

      <div className="flex flex-1 overflow-hidden">
        <div className="w-56 overflow-y-auto border-r border-[var(--color-border)] py-2">
          {!agents && <div className="px-4 text-[var(--color-fg-dim)]">loading…</div>}
          {agents?.map((a) => (
            <button
              key={a.id}
              onClick={() => setActiveId(a.id)}
              className={
                'block w-full px-4 py-2 text-left ' +
                (activeId === a.id
                  ? 'bg-[var(--color-bg)] text-[var(--color-accent)]'
                  : 'text-[var(--color-fg)] hover:bg-[var(--color-bg)]')
              }
            >
              <div className="truncate">{a.name}</div>
              <div className="truncate text-[10px] text-[var(--color-fg-dim)]">{a.id}</div>
            </button>
          ))}
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-4">
          {active && (
            <>
              <div className="mb-1 text-lg text-[var(--color-fg)]">{active.name}</div>
              <div className="mb-4 text-xs text-[var(--color-fg-dim)]">{active.description}</div>

              <Section title="model / config">
                {active.config_keys.map((ck) => {
                  const cur = pendingConfig[ck.key] ?? active.config[ck.key] ?? ''
                  if (ck.type === 'provider_model') {
                    const flat = Object.values(active.providers_models).flat()
                    return (
                      <Row key={ck.key} label={ck.label}>
                        <select
                          value={cur}
                          onChange={(e) => setConfigVal(ck.key, e.target.value)}
                          className="w-full rounded border border-[var(--color-border)] bg-[var(--color-bg)] px-2 py-1 text-[var(--color-fg)] focus:border-[var(--color-accent)] focus:outline-none"
                        >
                          {!flat.includes(cur) && cur && <option value={cur}>{cur} (custom)</option>}
                          {Object.entries(active.providers_models).map(([prov, models]) => (
                            <optgroup key={prov} label={prov}>
                              {models.map((m) => (
                                <option key={m} value={m}>
                                  {m}
                                </option>
                              ))}
                            </optgroup>
                          ))}
                        </select>
                      </Row>
                    )
                  }
                  if (ck.type === 'number') {
                    return (
                      <Row key={ck.key} label={ck.label}>
                        <input
                          type="number"
                          min={ck.min}
                          max={ck.max}
                          value={cur}
                          onChange={(e) => setConfigVal(ck.key, e.target.value)}
                          className="w-32 rounded border border-[var(--color-border)] bg-[var(--color-bg)] px-2 py-1 text-[var(--color-fg)] focus:border-[var(--color-accent)] focus:outline-none"
                        />
                      </Row>
                    )
                  }
                  return (
                    <Row key={ck.key} label={ck.label}>
                      <input
                        value={cur}
                        onChange={(e) => setConfigVal(ck.key, e.target.value)}
                        className="w-full rounded border border-[var(--color-border)] bg-[var(--color-bg)] px-2 py-1 text-[var(--color-fg)] focus:border-[var(--color-accent)] focus:outline-none"
                      />
                    </Row>
                  )
                })}
              </Section>

              {active.prompts.length > 0 && (
                <Section title="prompts">
                  {active.prompts.map((pname) => {
                    const cur = pendingPrompts[pname] ?? active.prompt_contents[pname] ?? ''
                    return (
                      <div key={pname} className="mb-3">
                        <div className="mb-1 text-xs text-[var(--color-fg-dim)]">{pname}</div>
                        <textarea
                          value={cur}
                          onChange={(e) => setPromptVal(pname, e.target.value)}
                          rows={8}
                          className="w-full rounded border border-[var(--color-border)] bg-[var(--color-bg)] px-2 py-1 text-[var(--color-fg)] focus:border-[var(--color-accent)] focus:outline-none"
                        />
                      </div>
                    )
                  })}
                </Section>
              )}

              {(active.tools.length > 0 || active.sub_agents.length > 0) && (
                <Section title="capabilities">
                  {active.tools.length > 0 && (
                    <Row label="tools">
                      <div className="flex flex-wrap gap-1">
                        {active.tools.map((t) => (
                          <span key={t} className="rounded bg-[var(--color-border)] px-1.5 py-0.5 text-[10px]">
                            {t}
                          </span>
                        ))}
                      </div>
                    </Row>
                  )}
                  {active.sub_agents.length > 0 && (
                    <Row label="sub-agents">
                      <div className="flex flex-wrap gap-1">
                        {active.sub_agents.map((s) => (
                          <span key={s} className="rounded bg-[var(--color-accent-dim)] px-1.5 py-0.5 text-[10px]">
                            {s}
                          </span>
                        ))}
                      </div>
                    </Row>
                  )}
                </Section>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-6">
      <div className="mb-2 border-b border-[var(--color-border)] pb-1 text-xs uppercase tracking-wider text-[var(--color-fg-dim)]">
        {title}
      </div>
      {children}
    </div>
  )
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mb-2 flex items-start gap-4">
      <div className="w-32 shrink-0 pt-1 text-xs text-[var(--color-fg-dim)]">{label}</div>
      <div className="flex-1">{children}</div>
    </div>
  )
}

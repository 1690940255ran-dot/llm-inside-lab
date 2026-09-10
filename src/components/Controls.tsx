/**
 * 通用 UI 控件：滑块、分段选择器、开关
 * 全部为受控组件，样式统一走 global.css
 */
import { useRef, useState, type ReactNode } from 'react'
import { exportNodeAsPng } from '../core/exportImage'
import { useLang } from '../i18n'

export type ExportState = 'idle' | 'busy' | 'ok' | 'error'

export function Slider(props: {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (v: number) => void
  format?: (v: number) => string
  hint?: string
  disabled?: boolean
}) {
  const { label, value, min, max, step = 1, onChange, format, hint, disabled } = props
  return (
    <div className="control" style={{ minWidth: hint ? 200 : 150, opacity: disabled ? 0.45 : 1 }}>
      <label title={hint}>
        <span>{label}</span>
        <b>{format ? format(value) : value}</b>
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  )
}

export function Segmented<T extends string | number>(props: {
  options: { value: T; label: string }[]
  value: T
  onChange: (v: T) => void
}) {
  return (
    <div className="segmented">
      {props.options.map((o) => (
        <button
          key={String(o.value)}
          className={o.value === props.value ? 'active' : ''}
          onClick={() => props.onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Toggle(props: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label
      className="btn"
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 8,
        cursor: 'pointer',
        userSelect: 'none',
        background: props.checked ? 'var(--accent-soft)' : 'var(--surface)',
        borderColor: props.checked ? 'var(--accent)' : 'var(--border-strong)',
        color: props.checked ? 'var(--accent)' : 'var(--text-2)',
      }}
    >
      <input
        type="checkbox"
        checked={props.checked}
        onChange={(e) => props.onChange(e.target.checked)}
        style={{ accentColor: 'var(--accent)' }}
      />
      {props.label}
    </label>
  )
}

/**
 * 「导出图片」按钮
 *
 * 传 target（一个 ref）就导出那个节点；不传就导出最近的一张卡片（由 Card 自己接）。
 * 实现见 core/exportImage.ts：SVG foreignObject + 内联 CSS，零依赖。
 */
export function ExportButton(props: {
  target: React.RefObject<HTMLElement>
  filename: string
  label?: string
}) {
  const { t } = useLang()
  const [state, setState] = useState<ExportState>('idle')
  const [msg, setMsg] = useState('')

  const run = async () => {
    if (!props.target.current || state === 'busy') return
    setState('busy')
    setMsg('')
    try {
      const res = await exportNodeAsPng(props.target.current, { filename: props.filename })
      setState('ok')
      setMsg(`${res.width}×${res.height}`)
    } catch (e: any) {
      setState('error')
      setMsg(e?.message ?? String(e))
    } finally {
      setTimeout(() => setState('idle'), 2600)
    }
  }

  const text =
    state === 'busy'
      ? t('exporting')
      : state === 'ok'
        ? `✓ ${t('exportOk')} ${msg}`
        : state === 'error'
          ? `✕ ${t('exportFail')}`
          : props.label ?? t('exportImage')

  return (
    <button
      className={`btn-export${state === 'error' ? ' err' : ''}${state === 'ok' ? ' ok' : ''}`}
      onClick={run}
      disabled={state === 'busy'}
      title={state === 'error' ? msg : t('exportImageHint')}
    >
      {text}
    </button>
  )
}

export function Card(props: {
  title?: ReactNode
  hint?: ReactNode
  /** 传了这个就会在标题栏右侧出现「导出图片」按钮 */
  exportName?: string
  children: ReactNode
}) {
  const ref = useRef<HTMLDivElement>(null)
  return (
    <div className="card" ref={ref}>
      {(props.title || props.exportName) && (
        <div className="card-title">
          <span>{props.title}</span>
          {props.hint && <span className="hint">{props.hint}</span>}
          {props.exportName && (
            <>
              <span style={{ flex: 1, minWidth: 8 }} />
              <ExportButton target={ref} filename={props.exportName} />
            </>
          )}
        </div>
      )}
      {props.children}
    </div>
  )
}

export function Stats(props: { items: { k: string; v: string }[] }) {
  return (
    <div className="stats">
      {props.items.map((it) => (
        <div className="stat" key={it.k}>
          <div className="k">{it.k}</div>
          <div className="v">{it.v}</div>
        </div>
      ))}
    </div>
  )
}

/**
 * 通用 UI 控件：滑块、分段选择器、开关
 * 全部为受控组件，样式统一走 global.css
 */
import type { ReactNode } from 'react'

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

export function Card(props: { title?: ReactNode; hint?: ReactNode; children: ReactNode }) {
  return (
    <div className="card">
      {props.title && (
        <div className="card-title">
          <span>{props.title}</span>
          {props.hint && <span className="hint">{props.hint}</span>}
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

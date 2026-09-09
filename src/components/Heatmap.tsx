/**
 * 通用矩阵热力图
 * 用真正的 <table> 而不是 canvas：单元格能直接响应 hover / 点击，
 * 也天然支持横向滚动，长句子在手机上不会糊成一团。
 */
import { useState } from 'react'
import { heatColor } from '../core/color'

export interface HoverInfo {
  i: number
  j: number
  value: number
}

export function Heatmap(props: {
  matrix: number[][]
  rowLabels: string[]
  colLabels?: string[]
  /** 该格是否是被掩码掉的未来位置（画斜纹表示"看不到"） */
  isMasked?: (i: number, j: number) => boolean
  colorOf?: (v: number) => string
  onHover?: (info: HoverInfo | null) => void
  onSelectRow?: (i: number) => void
  selectedRow?: number
  maxValue?: number
}) {
  const { matrix, rowLabels, colLabels, isMasked, colorOf = heatColor } = props
  const [hover, setHover] = useState<HoverInfo | null>(null)
  const n = matrix.length

  const setInfo = (i: number, j: number, value: number) => {
    const info = { i, j, value }
    setHover(info)
    props.onHover?.(info)
  }

  const clear = () => {
    setHover(null)
    props.onHover?.(null)
  }

  return (
    <div className="heat-wrap">
      <table className="heat">
        <thead>
          <tr>
            {colLabels && <th />}
            {colLabels
              ? colLabels.map((c, j) => (
                  <th key={j} title={c}>
                    {n > 26 ? j : c.length > 3 ? c.slice(0, 3) : c}
                  </th>
                ))
              : null}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th className="row-label" title={rowLabels[i]}>
                {rowLabels[i]?.length > 5 ? rowLabels[i].slice(0, 5) : rowLabels[i]}
              </th>
              {row.map((v, j) => {
                const masked = isMasked?.(i, j) ?? false
                const isHL = hover ? hover.i === i || hover.j === j : false
                return (
                  <td
                    key={j}
                    className={[masked ? 'masked' : '', isHL ? (hover?.i === i ? 'hl-row' : 'hl-col') : '']
                      .filter(Boolean)
                      .join(' ')}
                    style={{
                      background: masked ? 'var(--surface-2)' : colorOf(v),
                    }}
                    onMouseEnter={() => !masked && setInfo(i, j, v)}
                    onMouseLeave={clear}
                    onClick={() => !masked && props.onSelectRow?.(i)}
                    title={masked ? '被因果掩码遮挡' : `${rowLabels[i]} → ${rowLabels[j] ?? j}: ${v.toFixed(3)}`}
                  />
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function HeatLegend(props: { min?: string; max?: string; colorOf?: (v: number) => string }) {
  const colorOf = props.colorOf ?? heatColor
  const stops = [0, 0.25, 0.5, 0.75, 1]
  const gradient = `linear-gradient(to right, ${stops.map((s) => colorOf(s)).join(', ')})`
  return (
    <div className="legend">
      <span>{props.min ?? '0'}</span>
      <div className="legend-bar" style={{ background: gradient }} />
      <span>{props.max ?? '1'}</span>
    </div>
  )
}

/** 横向条形图：展示"某个 token 在看谁" */
export function BarList(props: {
  items: { label: string; value: number }[]
  max?: number
  format?: (v: number) => string
}) {
  const max = props.max ?? Math.max(...props.items.map((i) => i.value), 1e-6)
  return (
    <div className="bars">
      {props.items.map((it, idx) => (
        <div className="bar-row" key={idx}>
          <div className="label" title={it.label}>
            {it.label}
          </div>
          <div className="bar-track">
            <div className="bar-fill" style={{ width: `${(it.value / max) * 100}%` }} />
          </div>
          <div className="val">{props.format ? props.format(it.value) : it.value.toFixed(3)}</div>
        </div>
      ))}
    </div>
  )
}

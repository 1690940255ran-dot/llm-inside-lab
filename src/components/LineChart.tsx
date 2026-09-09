/** 极简折线图：够用就好，不引图表库 */
export interface Line {
  name: string
  color: string
  points: { x: number; y: number }[]
  dashed?: boolean
}

export function LineChart(props: {
  lines: Line[]
  xLabel: string
  yLabel: string
  height?: number
  yFormat?: (v: number) => string
  xFormat?: (v: number) => string
}) {
  const W = 620
  const H = props.height ?? 260
  const padL = 74
  const padB = 34
  const padT = 14
  const padR = 12

  const allX = props.lines.flatMap((l) => l.points.map((p) => p.x))
  const allY = props.lines.flatMap((l) => l.points.map((p) => p.y))
  if (allX.length === 0) return null

  const xMin = Math.min(...allX)
  const xMax = Math.max(...allX)
  const yMin = 0
  const yMax = Math.max(...allY) || 1

  const sx = (x: number) => padL + ((x - xMin) / (xMax - xMin || 1)) * (W - padL - padR)
  const sy = (y: number) => H - padB - ((y - yMin) / (yMax - yMin || 1)) * (H - padB - padT)

  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((t) => yMin + t * (yMax - yMin))
  const xTicks = [0, 0.5, 1].map((t) => xMin + t * (xMax - xMin))

  return (
    <div>
      <div className="chip-row" style={{ marginBottom: 6 }}>
        {props.lines.map((l) => (
          <span
            key={l.name}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 6,
              fontSize: 12,
              color: 'var(--text-2)',
            }}
          >
            <span
              style={{
                width: 16,
                height: 3,
                background: l.color,
                display: 'inline-block',
                borderRadius: 2,
                opacity: l.dashed ? 0.6 : 1,
              }}
            />
            {l.name}
          </span>
        ))}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }}>
        {yTicks.map((t, i) => (
          <g key={i}>
            <line
              x1={padL}
              y1={sy(t)}
              x2={W - padR}
              y2={sy(t)}
              stroke="var(--border)"
              strokeWidth={0.6}
            />
            <text x={padL - 8} y={sy(t)} textAnchor="end" dominantBaseline="central" fontSize={10} fill="var(--text-3)">
              {props.yFormat ? props.yFormat(t) : t.toFixed(0)}
            </text>
          </g>
        ))}
        {xTicks.map((t, i) => (
          <text
            key={i}
            x={sx(t)}
            y={H - padB + 16}
            textAnchor="middle"
            fontSize={10}
            fill="var(--text-3)"
          >
            {props.xFormat ? props.xFormat(t) : t.toFixed(0)}
          </text>
        ))}
        <text x={W - padR} y={H - 4} textAnchor="end" fontSize={10} fill="var(--text-3)">
          {props.xLabel}
        </text>
        <text x={padL - 8} y={padT - 4} textAnchor="end" fontSize={10} fill="var(--text-3)">
          {props.yLabel}
        </text>
        {props.lines.map((l) => (
          <polyline
            key={l.name}
            points={l.points.map((p) => `${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ')}
            fill="none"
            stroke={l.color}
            strokeWidth={2}
            strokeDasharray={l.dashed ? '5 4' : undefined}
          />
        ))}
      </svg>
    </div>
  )
}

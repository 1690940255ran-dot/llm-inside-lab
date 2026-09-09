/** token 彩色卡片行：分词模块和注意力模块共用 */
import { tokenColor } from '../core/color'
import type { Token } from '../core/bpe'

export function TokenChips(props: {
  tokens: Token[]
  highlight?: number
  onHover?: (i: number | null) => void
  showId?: boolean
}) {
  const { tokens, highlight, onHover, showId = true } = props
  return (
    <div className="chip-row tight">
      {tokens.map((t, i) => {
        const c = tokenColor(i)
        const display = t.text === ' ' ? '␣' : t.text === '\n' ? '⏎' : t.text
        return (
          <span
            key={i}
            className={`token${highlight === i ? ' hover' : ''}${t.oov ? ' oov' : ''}${
              /\s/.test(t.text) ? ' ws' : ''
            }`}
            style={{ background: c.bg, color: c.fg }}
            onMouseEnter={() => onHover?.(i)}
            onMouseLeave={() => onHover?.(null)}
            title={t.oov ? `「${t.text}」不在词表中（训练语料没见过）` : `token ${t.id}`}
          >
            <span>{display === '' ? '·' : display}</span>
            {showId && <span className="tid">{t.oov ? 'OOV' : t.id}</span>}
          </span>
        )
      })}
    </div>
  )
}

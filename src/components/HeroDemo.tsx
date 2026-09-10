/**
 * 首页动图：一条流水线自动跑一遍
 *
 * 这不是装饰动画 —— 里面的每一步都由 core/ 里那套已被单测覆盖的函数算出来：
 *   分词      core/bpe.ts 的 encode（同一个 SHARED_MODEL）
 *   注意力    core/attention.ts 的 computeAttention（真的 Q/K/V + 缩放 + 因果掩码 + softmax）
 *   候选分布  tokenVector 的余弦相似度当 logits，再过 core/sampling.ts 的 sampleNext
 *
 * 唯一的"示意"是那 5 个候选词是写死的（真实词表有 5 万个），
 * 页面上也如实标了这一点。
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { useLang } from '../i18n'
import { encode } from '../core/bpe'
import { SHARED_MODEL } from '../core/sharedModel'
import { computeAttention } from '../core/attention'
import { tokenVector, cosine } from '../core/embedding'
import { sampleNext } from '../core/sampling'
import { heatColor } from '../core/color'

const TICK_MS = 380
/** 列宽有限，注意力矩阵最多画到 10×10 */
const MAX_TOKENS = 10
const MAX_STEPS = 4

const SEED_TEXT = {
  zh: '大语言模型通过预测下一个',
  en: 'the model predicts the next',
}

const CANDIDATES = {
  zh: ['词', '字', '模型', '概率', '句子'],
  en: [' token', ' word', ' model', ' prob', ' sentence'],
}

const STAGE_LABEL = {
  zh: ['① 输入文本', '② 切成 token', '③ 注意力：谁看谁', '④ 下一个 token 的分布'],
  en: ['① input text', '② tokens', '③ attention: who looks at whom', '④ next-token distribution'],
}

const NOTE = {
  zh: '分词与注意力走的是本站 core/ 里同一套函数（真的 BPE、真的 Q/K/V + 因果掩码 + softmax）；只有 5 个候选词是写死的示意——真实词表有 5 万个。',
  en: 'Tokenization and attention run the very same functions from core/ (real BPE, real Q/K/V + causal mask + softmax). Only the 5 candidates are hard-coded for illustration — a real vocabulary has 50k of them.',
}

const STAGE_CAPTION = {
  zh: ['一个数一个字往外蹦', '每个色块是一个 token', '每一格是一行归一化后的权重', '抽中的那个拼回输入，继续'],
  en: ['one piece at a time', 'each chip is one token', 'each cell is a normalised weight', 'the winner is appended and it repeats'],
}

export function HeroDemo() {
  const { lang, t } = useLang()
  const [playing, setPlaying] = useState(true)
  const [tick, setTick] = useState(0)
  const [extra, setExtra] = useState<string[]>([])
  const timer = useRef<number | null>(null)

  const base = SEED_TEXT[lang]
  const text = base + extra.join('')

  const tokens = useMemo(() => {
    const raw = encode(text, SHARED_MODEL).tokens.map((x) => x.text)
    return raw.slice(0, MAX_TOKENS)
  }, [text])

  const n = tokens.length
  const attn = useMemo(
    () =>
      computeAttention(tokens, {
        nLayers: 1,
        nHeads: 1,
        dModel: 32,
        temperature: 1,
        causal: true,
        distanceDecay: 0.12,
      }),
    [tokens],
  )
  const W = attn.weights[0]?.[0] ?? []

  const dist = useMemo(() => {
    const last = tokens[tokens.length - 1] ?? ''
    const e = tokenVector(last, 32)
    const cands = CANDIDATES[lang]
    const logits = cands.map((c) => cosine(e, tokenVector(c, 32)) * 6)
    const step = sampleNext(logits, { temperature: 0.9, topK: 0, topP: 0.9 }, 20260909)
    const rows = cands.map((c, i) => ({ label: c, p: step.final[i] ?? 0 })).sort((a, b) => b.p - a.p)
    return rows
  }, [tokens, lang])

  // 阶段划分：先亮 token → 再逐行填注意力 → 再长条形图 → 抽中 → 拼回去
  const revealTokens = Math.min(tick, n)
  const attnRows = Math.min(Math.max(tick - n, 0), n)
  const barsIn = Math.min(Math.max(tick - 2 * n, 0), 2)
  const picked = tick >= 2 * n + 3
  const done = tick >= 2 * n + 6

  useEffect(() => {
    if (!playing) return
    timer.current = window.setInterval(() => setTick((v) => v + 1), TICK_MS)
    return () => {
      if (timer.current) window.clearInterval(timer.current)
    }
  }, [playing])

  // 一轮结束：把抽中的候选拼回输入，重新开始。
  // 注意要保证「文本列」和「token 列」对得上：拼完之后 token 数一旦超过 MAX_TOKENS
  // （那样 chip 会被截断、和左边的文本不一致），就直接回到初始句重新循环。
  useEffect(() => {
    if (!done) return
    const winner = dist[0]?.label ?? ''
    const next = [...extra, winner]
    const rawCount = encode(base + next.join(''), SHARED_MODEL).tokens.length
    setExtra(rawCount > MAX_TOKENS || next.length > MAX_STEPS ? [] : next)
    setTick(0)
  }, [done])

  // 换语言 / 卸载时把动画复位
  useEffect(() => {
    setExtra([])
    setTick(0)
  }, [lang])

  const stage = revealTokens < n ? 0 : attnRows < n ? 1 : barsIn < 2 ? 2 : 3
  const labels = STAGE_LABEL[lang]

  return (
    <div className="card hero-demo-card">
      <div className="card-title">
        <span>{t('heroDemoTitle')}</span>
        <span className="hint">{t('heroDemoHint')}</span>
        <span style={{ flex: 1, minWidth: 8 }} />
        <button className="btn-export" onClick={() => setPlaying((p) => !p)}>
          {playing ? t('pause') : t('play')}
        </button>
      </div>

      <div className="hd-grid">
        {labels.map((label, i) => (
          <div key={i} className={`hd-col${i === stage ? ' on' : ''}`}>
            <div className="hd-label">
              {label}
              <span className="hd-caption">{STAGE_CAPTION[lang][i]}</span>
            </div>

            <div className="hd-body">
              {i === 0 && (
                <div className="hd-text">
                  {text.split('').map((ch, k) => (
                    <span key={k} className="hd-ch">
                      {ch === ' ' ? '␣' : ch}
                    </span>
                  ))}
                  <span className="hd-caret" />
                </div>
              )}

              {i === 1 && (
                <div className="hd-chips">
                  {tokens.map((tk, k) => (
                    <span
                      key={k}
                      className="hd-chip"
                      style={{
                        opacity: k < revealTokens ? 1 : 0.12,
                        transform: k < revealTokens ? 'translateY(0)' : 'translateY(4px)',
                      }}
                    >
                      {tk === ' ' ? '␣' : tk}
                    </span>
                  ))}
                </div>
              )}

              {i === 2 && (
                <div className="hd-heat" style={{ gridTemplateColumns: `repeat(${n}, 1fr)` }}>
                  {W.flatMap((row, r) =>
                    row.map((v, c) => {
                      const shown = r < attnRows
                      const masked = c > r
                      return (
                        <div
                          key={`${r}-${c}`}
                          className="hd-cell"
                          style={{
                            background: !shown
                              ? 'var(--surface-2)'
                              : masked
                                ? 'repeating-linear-gradient(45deg, var(--surface-2) 0 3px, transparent 3px 6px)'
                                : heatColor(v / (Math.max(...row.filter((_, j) => j <= r)) || 1)),
                            transitionDelay: `${(r * n + c) * 6}ms`,
                          }}
                        />
                      )
                    }),
                  )}
                </div>
              )}

              {i === 3 && (
                <div className="hd-bars">
                  {dist.map((r, k) => {
                    const isWin = k === 0
                    return (
                      <div className="hd-bar-row" key={r.label}>
                        <span className="hd-bar-label">{r.label === ' ' ? '␣' : r.label}</span>
                        <span className="hd-bar-track">
                          <span
                            className={`hd-bar-fill${isWin && picked ? ' win' : ''}`}
                            style={{ width: barsIn >= 1 ? `${Math.max(r.p * 100, 2)}%` : '0%' }}
                          />
                        </span>
                        <span className="hd-bar-val">
                          {barsIn >= 1 ? `${(r.p * 100).toFixed(0)}%` : ''}
                        </span>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="note" style={{ marginTop: 10 }}>
        {NOTE[lang]}
      </div>
    </div>
  )
}

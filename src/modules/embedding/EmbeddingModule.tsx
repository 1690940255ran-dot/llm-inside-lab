/**
 * 模块三：嵌入与位置编码
 *
 * 注意力之前的两步准备：
 *   1. 把 token id 查表变成 d 维向量（嵌入）
 *   2. 把"位置"信息加进去，否则 Transformer 根本分不清语序
 * 这里把两种主流位置编码都摆出来对比：正弦编码 vs RoPE。
 */
import { useMemo, useState } from 'react'
import { Card, Slider, Stats } from '../../components/Controls'
import { HeatLegend, Heatmap } from '../../components/Heatmap'
import { Principle } from '../../components/Principle'
import { TokenChips } from '../../components/Tokens'
import { cosine, pca2d, tokenVector } from '../../core/embedding'
import { positionalSimilarity, ropeAngles, sinusoidalPE } from '../../core/positional'
import { divergingColor, tokenColor } from '../../core/color'
import { SHARED_MODEL } from '../../core/sharedModel'
import { encode } from '../../core/bpe'
import { SAMPLE_TEXTS } from '../../core/corpus'

const D_MODEL = 48
const D_SHOW = 24 // 热图里只画前 24 维，画满太挤

export function EmbeddingModule() {
  const [text, setText] = useState('猫追着老鼠跑，因为它饿了。')
  const [posIndex, setPosIndex] = useState(3)
  const [pairIndex, setPairIndex] = useState(0)

  const tokenObjs = useMemo(() => encode(text, SHARED_MODEL).tokens.slice(0, 28), [text])
  const labels = tokenObjs.map((t) => (t.text === ' ' ? '␣' : t.text))
  const n = labels.length

  const emb = useMemo(() => labels.map((t) => tokenVector(t, D_MODEL)), [labels])
  const embShow = useMemo(() => emb.map((v) => v.slice(0, D_SHOW)), [emb])
  const pe = useMemo(() => sinusoidalPE(n, D_MODEL), [n])
  const peShow = useMemo(() => pe.map((v) => v.slice(0, D_SHOW)), [pe])
  const posSim = useMemo(() => positionalSimilarity(pe), [pe])
  const angles = useMemo(() => ropeAngles(n, D_MODEL), [n])
  const scatter = useMemo(() => pca2d(emb), [emb])

  // 归一化到 -1..1 方便用双向配色
  const normScale = (m: number[][]) => {
    let max = 0
    for (const row of m) for (const v of row) max = Math.max(max, Math.abs(v))
    return max === 0 ? m : m.map((row) => row.map((v) => v / max))
  }
  const embNorm = useMemo(() => normScale(embShow), [embShow])
  const peNorm = useMemo(() => normScale(peShow), [peShow])

  // RoPE 演示：取第 pairIndex 组的两维，看它随位置怎么转
  const rotPoints = labels.map((_, i) => {
    const theta = angles[i]?.[pairIndex] ?? 0
    const r = 0.82
    return { x: Math.cos(theta) * r, y: Math.sin(theta) * r, pos: i }
  })

  const simToFirst = useMemo(
    () => labels.map((_, i) => (i === 0 ? 1 : cosine(pe[0] ?? [], pe[i] ?? []))),
    [labels, pe],
  )

  return (
    <div>
      <div className="module-head">
        <h2>③ 嵌入与位置编码：给 token 一个坐标，再告诉它站在第几位</h2>
        <div className="lead">
          嵌入把离散的编号变成连续的向量，让"相似"这件事可以被计算；
          位置编码则把语序信息补回去 —— 因为自注意力本身是排列等变的，打乱顺序它根本察觉不到。
        </div>
      </div>

      <Principle
        title="为什么自注意力必须额外加位置信息？"
        formula={`嵌入：  X = Embedding[token_id]       形状 [n, d_model]

正弦位置编码（原始 Transformer）：
  PE[pos, 2i]   = sin(pos / 10000^(2i/d))
  PE[pos, 2i+1] = cos(pos / 10000^(2i/d))

RoPE（LLaMA / Qwen / Mistral 等在用）：
  把 d 维两两分组，第 j 组在位置 pos 处旋转角度 θ = pos / 10000^(2j/d)
  即：q'ᵀk' 只依赖于 (pos_q - pos_k)  ← 天然表达"相对距离"`}
        analogy="嵌入相当于给每个词发一张身份证（一串数字，越相似的词数字越接近）。但一堆身份证散在桌上是没有顺序的——所以还要给每张卡盖一个「座位号」印章。正弦编码是直接把座位号写死在卡上；RoPE 则是把卡片本身转一个角度，转多少由座位号决定，于是两张卡之间的夹角天然就编码了它们隔了多远。"
        detail={
          <>
            <p>
              两种方案的关键差别：<strong>正弦编码是"加"上去的绝对位置</strong>，模型得自己学会从绝对位置里推断相对关系；
              <strong>RoPE 是"旋转"，点积结果天然只跟相对距离有关</strong>，这也是它外推长上下文能力更好的原因之一。
            </p>
            <p>
              下面那张「位置相似度」图很值得看：它呈现明显的对角条带，说明
              <em>相似度只取决于两个位置的距离，而不是具体是第几位</em>。这正是相对位置感的来源。
            </p>
          </>
        }
        warn="嵌入向量同样是确定性模拟生成的（真实嵌入矩阵是训练出来的）。但为了让「语义相近的词在向量空间里相近」这件事可见，这里刻意让首字符相同、字符类别相同的 token 共享一部分向量成分，所以 PCA 图里能看到成簇现象。"
      />

      <Card title="输入文本">
        <textarea rows={2} value={text} onChange={(e) => setText(e.target.value)} />
        <div className="chip-row" style={{ marginTop: 10 }}>
          {SAMPLE_TEXTS.map((s) => (
            <button key={s.label} className="btn" onClick={() => setText(s.text)}>
              {s.label}
            </button>
          ))}
        </div>
      </Card>

      <Card title={`嵌入向量 X（前 ${D_SHOW} 维，已按最大值归一化）`} hint="每一行是一个 token 的身份证">
        <Heatmap matrix={embNorm} rowLabels={labels} colorOf={divergingColor} />
        <HeatLegend min="-1" max="1" colorOf={divergingColor} />
        <div style={{ marginTop: 12 }}>
          <TokenChips tokens={tokenObjs} showId={false} />
        </div>
      </Card>

      <Card title="正弦位置编码 PE" hint="注意左侧维度几乎不动（低频），右侧维度剧烈震荡（高频）">
        <Heatmap matrix={peNorm} rowLabels={labels.map((_, i) => `pos ${i}`)} colorOf={divergingColor} />
        <HeatLegend min="-1" max="1" colorOf={divergingColor} />
        <div style={{ marginTop: 12 }}>
          <Stats
            items={[
              { k: '序列长度 n', v: String(n) },
              { k: '维度 d_model', v: String(D_MODEL) },
              { k: '编码基数', v: '10000' },
              { k: `pos 0 与 pos ${Math.min(posIndex, n - 1)} 相似度`, v: simToFirst[Math.min(posIndex, n - 1)]?.toFixed(3) ?? '-' },
            ]}
          />
          <div style={{ marginTop: 10 }}>
            <Slider
              label="对比位置"
              value={posIndex}
              min={0}
              max={Math.max(0, n - 1)}
              onChange={setPosIndex}
              format={(v) => `pos ${v}`}
            />
          </div>
        </div>
      </Card>

      <Card title="位置向量之间的相似度" hint="对角条带 = 相似度只跟相对距离有关">
        <Heatmap
          matrix={posSim}
          rowLabels={labels.map((_, i) => `pos ${i}`)}
          colLabels={labels.map((_, i) => `${i}`)}
          colorOf={divergingColor}
        />
        <HeatLegend min="-1" max="1" colorOf={divergingColor} />
      </Card>

      <Card title="RoPE：把向量旋转一个跟位置有关的角度" hint="同一组维度在不同位置上转过的角度">
        <div className="controls" style={{ marginBottom: 12 }}>
          <Slider
            label="维度组 j"
            value={pairIndex}
            min={0}
            max={Math.max(0, Math.floor(D_MODEL / 2) - 1)}
            onChange={setPairIndex}
            format={(v) => `第 ${v} 组`}
            hint="组号越小 = 频率越低 = 转得越慢，越能表达长距离"
          />
        </div>
        <svg viewBox="-1.1 -1.1 2.2 2.2" width="100%" style={{ maxWidth: 320, display: 'block' }}>
          <circle cx={0} cy={0} r={1} fill="none" stroke="var(--border-strong)" strokeWidth={0.01} />
          <line x1={-1} y1={0} x2={1} y2={0} stroke="var(--border)" strokeWidth={0.008} />
          <line x1={0} y1={-1} x2={0} y2={1} stroke="var(--border)" strokeWidth={0.008} />
          {rotPoints.map((p, i) => {
            const c = tokenColor(i)
            return (
              <g key={i}>
                <line
                  x1={0}
                  y1={0}
                  x2={p.x}
                  y2={p.y}
                  stroke={c.fg}
                  strokeWidth={i === Math.min(posIndex, n - 1) ? 0.035 : 0.014}
                  opacity={i < n ? 0.85 : 0.2}
                />
                <circle cx={p.x} cy={p.y} r={0.045} fill={c.bg} stroke={c.fg} strokeWidth={0.012} />
              </g>
            )
          })}
        </svg>
        <div className="note" style={{ marginTop: 8 }}>
          每条彩色射线是「第 {pairIndex} 组维度」在某个位置上的方向。组号 j 越小频率越低，相邻位置转过的角度越小，
          越擅长捕捉长距离依赖；j 越大转得飞快，只能区分很近的位置。这就是 RoPE 能同时建模"远近"的原因。
        </div>
      </Card>

      <Card title="嵌入向量降维（PCA → 2D）" hint="语义/字形相近的 token 会靠在一起">
        {scatter.points.length > 1 ? (
          <ScatterPlot
            points={scatter.points}
            labels={labels}
            explained={scatter.explained}
          />
        ) : (
          <div className="note">至少需要 2 个 token。</div>
        )}
      </Card>
    </div>
  )
}

function ScatterPlot(props: {
  points: { x: number; y: number }[]
  labels: string[]
  explained: [number, number]
}) {
  const W = 520
  const H = 300
  const pad = 34
  const xs = props.points.map((p) => p.x)
  const ys = props.points.map((p) => p.y)
  const xMin = Math.min(...xs)
  const xMax = Math.max(...xs)
  const yMin = Math.min(...ys)
  const yMax = Math.max(...ys)
  const sx = (v: number) => pad + ((v - xMin) / (xMax - xMin || 1)) * (W - pad * 2)
  const sy = (v: number) => H - pad - ((v - yMin) / (yMax - yMin || 1)) * (H - pad * 2)

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" className="scatter">
        {props.points.map((p, i) => {
          const c = tokenColor(i)
          return (
            <g key={i}>
              <circle cx={sx(p.x)} cy={sy(p.y)} r={7} fill={c.bg} stroke={c.fg} strokeWidth={1.2} />
              <text x={sx(p.x)} y={sy(p.y) - 12} textAnchor="middle">
                {props.labels[i]}
              </text>
            </g>
          )
        })}
      </svg>
      <div className="note">
        主成分 1 解释了 {(props.explained[0] * 100).toFixed(1)}% 的方差，主成分 2 解释了{' '}
        {(props.explained[1] * 100).toFixed(1)}%。这只是 48 维压到 2 维的粗略投影，别过度解读。
      </div>
    </div>
  )
}

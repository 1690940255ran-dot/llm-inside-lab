/**
 * 模块二：嵌入与位置编码
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
import { useLang, type Lang } from '../../i18n'

const D_MODEL = 48
const D_SHOW = 24

const zh = {
  h2: '② 嵌入与位置编码：给 token 一个坐标，再告诉它站在第几位',
  lead: '嵌入把离散的编号变成连续的向量，让「相似」这件事可以被计算；位置编码则把语序信息补回去 —— 因为自注意力本身是排列等变的，打乱顺序它根本察觉不到。',
  principleTitle: '为什么自注意力必须额外加位置信息？',
  d1:
    '两种方案的关键差别：正弦编码是「加」上去的绝对位置，模型得自己学会从绝对位置里推断相对关系；RoPE 是「旋转」，点积结果天然只跟相对距离有关，这也是它外推长上下文能力更好的原因之一。',
  d2:
    '下面那张「位置相似度」图很值得看：它呈现明显的对角条带，说明相似度只取决于两个位置的距离，而不是具体是第几位。这正是相对位置感的来源。',
  warn: '嵌入向量同样是确定性模拟生成的（真实嵌入矩阵是训练出来的）。但为了让「语义相近的词在向量空间里相近」这件事可见，这里刻意让首字符相同、字符类别相同的 token 共享一部分向量成分，所以 PCA 图里能看到成簇现象。',
  embTitle: '嵌入向量 X（前',
  embTitle2: '维，已按最大值归一化）',
  embHint: '每一行是一个 token 的身份证',
  peTitle: '正弦位置编码 PE',
  peHint: '注意左侧维度几乎不动（低频），右侧维度剧烈震荡（高频）',
  len: '序列长度 n',
  dim: '维度 d_model',
  base: '编码基数',
  simTo: '相似度',
  comparePos: '对比位置',
  posSimTitle: '位置向量之间的相似度',
  posSimHint: '对角条带 = 相似度只跟相对距离有关',
  ropeTitle: 'RoPE：把向量旋转一个跟位置有关的角度',
  ropeHint: '同一组维度在不同位置上转过的角度',
  ropeGroup: '维度组 j',
  ropeGroupHint: '组号越小 = 频率越低 = 转得越慢，越能表达长距离',
  ropeNote1: '每条彩色射线是「第',
  ropeNote2:
    '组维度」在某个位置上的方向。组号 j 越小频率越低，相邻位置转过的角度越小，越擅长捕捉长距离依赖；j 越大转得飞快，只能区分很近的位置。这就是 RoPE 能同时建模「远近」的原因。',
  pcaTitle: '嵌入向量降维（PCA → 2D）',
  pcaHint: '语义/字形相近的 token 会靠在一起',
  pcaNote1: '主成分 1 解释了',
  pcaNote2: '的方差，主成分 2 解释了',
  pcaNote3: '。这只是 48 维压到 2 维的粗略投影，别过度解读。',
  need2: '至少需要 2 个 token。',
}

const en: typeof zh = {
  h2: '② Embeddings & Positional Encoding: a coordinate and a seat number',
  lead:
    'Embeddings turn discrete ids into continuous vectors so that "similar" becomes computable. Positional encoding puts word order back in — self-attention is permutation-equivariant, so shuffled input looks identical to it.',
  principleTitle: 'Why does self-attention need explicit position information?',
  d1:
    'The key difference: sinusoidal encoding adds an absolute position vector, so the model has to infer relative distance by itself. RoPE rotates, which makes the dot product depend only on the relative offset — and that is a big part of why it extrapolates to longer contexts better.',
  d2:
    'The "positional similarity" plot below is worth staring at: it shows clear diagonal banding, meaning similarity depends on distance between positions, not on which positions they are. That is exactly where the sense of relative position comes from.',
  warn: 'Embeddings here are deterministic simulations too (real ones are trained). But so that "similar words sit close together" is actually visible, tokens sharing a first character or a character class deliberately share part of their vector — which is why the PCA plot shows clusters.',
  embTitle: 'Embedding matrix X (first ',
  embTitle2: ' dims, normalised by max abs)',
  embHint: 'each row is one token’s identity card',
  peTitle: 'Sinusoidal positional encoding PE',
  peHint: 'left dims barely move (low frequency), right dims oscillate wildly (high frequency)',
  len: 'sequence length n',
  dim: 'd_model',
  base: 'base',
  simTo: 'similarity',
  comparePos: 'compare position',
  posSimTitle: 'Similarity between positional vectors',
  posSimHint: 'diagonal bands = similarity depends only on relative distance',
  ropeTitle: 'RoPE: rotate the vector by a position-dependent angle',
  ropeHint: 'the same dimension pair rotated differently at each position',
  ropeGroup: 'dimension group j',
  ropeGroupHint: 'smaller j = lower frequency = slower rotation = longer range',
  ropeNote1: 'Each ray is the direction of dimension group ',
  ropeNote2:
    ' at some position. Smaller j means lower frequency: neighbouring positions barely differ, which is what captures long-range dependency. Large j spins fast and can only tell nearby positions apart. That is how RoPE models both near and far.',
  pcaTitle: 'Embeddings projected to 2D (PCA)',
  pcaHint: 'tokens similar in meaning or shape end up close together',
  pcaNote1: 'Principal component 1 explains ',
  pcaNote2: ' of the variance, component 2 explains ',
  pcaNote3: '. This is a crude squash of 48 dims into 2 — do not over-read it.',
  need2: 'Need at least 2 tokens.',
}

const DICT: Record<Lang, typeof zh> = { zh, en }

export function EmbeddingModule() {
  const { lang, t } = useLang()
  const c = DICT[lang]

  const [text, setText] = useState(SAMPLE_TEXTS[2].text)
  const [posIndex, setPosIndex] = useState(3)
  const [pairIndex, setPairIndex] = useState(0)

  const tokenObjs = useMemo(() => encode(text, SHARED_MODEL).tokens.slice(0, 28), [text])
  const labels = tokenObjs.map((x) => (x.text === ' ' ? '␣' : x.text))
  const n = labels.length

  const emb = useMemo(() => labels.map((x) => tokenVector(x, D_MODEL)), [labels])
  const embShow = useMemo(() => emb.map((v) => v.slice(0, D_SHOW)), [emb])
  const pe = useMemo(() => sinusoidalPE(n, D_MODEL), [n])
  const peShow = useMemo(() => pe.map((v) => v.slice(0, D_SHOW)), [pe])
  const posSim = useMemo(() => positionalSimilarity(pe), [pe])
  const angles = useMemo(() => ropeAngles(n, D_MODEL), [n])
  const scatter = useMemo(() => pca2d(emb), [emb])

  const normScale = (m: number[][]) => {
    let max = 0
    for (const row of m) for (const v of row) max = Math.max(max, Math.abs(v))
    return max === 0 ? m : m.map((row) => row.map((v) => v / max))
  }
  const embNorm = useMemo(() => normScale(embShow), [embShow])
  const peNorm = useMemo(() => normScale(peShow), [peShow])

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
        <h2>{c.h2}</h2>
        <div className="lead">{c.lead}</div>
      </div>

      <Principle
        title={c.principleTitle}
        formula={`embedding:  X = Embedding[token_id]        shape [n, d_model]

sinusoidal (original Transformer):
  PE[pos, 2i]   = sin(pos / 10000^(2i/d))
  PE[pos, 2i+1] = cos(pos / 10000^(2i/d))

RoPE (LLaMA / Qwen / Mistral):
  group d dims into pairs, rotate pair j at position pos by θ = pos / 10000^(2j/d)
  ⇒ q'ᵀk' depends only on (pos_q − pos_k)   ← relative distance, for free`}
        analogy={
          lang === 'zh'
            ? '嵌入相当于给每个词发一张身份证（一串数字，越相似的词数字越接近）。但一堆身份证散在桌上是没有顺序的——所以还要给每张卡盖一个「座位号」印章。正弦编码是直接把座位号写死在卡上；RoPE 则是把卡片本身转一个角度，转多少由座位号决定，于是两张卡之间的夹角天然就编码了它们隔了多远。'
            : 'An embedding is an ID card for each word: a string of numbers, and similar words get similar numbers. But a pile of ID cards on a table has no order — so each card also gets a "seat number" stamped on it. Sinusoidal encoding writes the seat number directly onto the card; RoPE instead rotates the card by an angle determined by the seat number, so the angle between any two cards encodes how far apart they sit.'
        }
        detail={
          <>
            <p>{c.d1}</p>
            <p>{c.d2}</p>
          </>
        }
        warn={c.warn}
      />

      <Card title={t('inputText')}>
        <textarea rows={2} value={text} onChange={(e) => setText(e.target.value)} />
        <div className="chip-row" style={{ marginTop: 10 }}>
          {SAMPLE_TEXTS.map((s) => (
            <button key={s.label} className="btn" onClick={() => setText(s.text)}>
              {lang === 'zh' ? s.label : s.labelEn}
            </button>
          ))}
        </div>
      </Card>

      <Card
        title={`${c.embTitle}${D_SHOW} ${c.embTitle2}`}
        hint={c.embHint}
        exportName="02-embedding-heatmap"
      >
        <Heatmap matrix={embNorm} rowLabels={labels} colorOf={divergingColor} />
        <HeatLegend min="-1" max="1" colorOf={divergingColor} />
        <div style={{ marginTop: 12 }}>
          <TokenChips tokens={tokenObjs} showId={false} />
        </div>
      </Card>

      <Card title={c.peTitle} hint={c.peHint}>
        <Heatmap
          matrix={peNorm}
          rowLabels={labels.map((_, i) => `pos ${i}`)}
          colorOf={divergingColor}
        />
        <HeatLegend min="-1" max="1" colorOf={divergingColor} />
        <div style={{ marginTop: 12 }}>
          <Stats
            items={[
              { k: c.len, v: String(n) },
              { k: c.dim, v: String(D_MODEL) },
              { k: c.base, v: '10000' },
              {
                k: `pos 0 ↔ pos ${Math.min(posIndex, n - 1)} ${c.simTo}`,
                v: simToFirst[Math.min(posIndex, n - 1)]?.toFixed(3) ?? '-',
              },
            ]}
          />
          <div style={{ marginTop: 10 }}>
            <Slider
              label={c.comparePos}
              value={posIndex}
              min={0}
              max={Math.max(0, n - 1)}
              onChange={setPosIndex}
              format={(v) => `pos ${v}`}
            />
          </div>
        </div>
      </Card>

      <Card title={c.posSimTitle} hint={c.posSimHint}>
        <Heatmap
          matrix={posSim}
          rowLabels={labels.map((_, i) => `pos ${i}`)}
          colLabels={labels.map((_, i) => `${i}`)}
          colorOf={divergingColor}
        />
        <HeatLegend min="-1" max="1" colorOf={divergingColor} />
      </Card>

      <Card title={c.ropeTitle} hint={c.ropeHint} exportName="02-embedding-rope">
        <div className="controls" style={{ marginBottom: 12 }}>
          <Slider
            label={c.ropeGroup}
            value={pairIndex}
            min={0}
            max={Math.max(0, Math.floor(D_MODEL / 2) - 1)}
            onChange={setPairIndex}
            format={(v) => (lang === 'zh' ? `第 ${v} 组` : `group ${v}`)}
            hint={c.ropeGroupHint}
          />
        </div>
        <svg viewBox="-1.1 -1.1 2.2 2.2" width="100%" style={{ maxWidth: 320, display: 'block' }}>
          <circle cx={0} cy={0} r={1} fill="none" stroke="var(--border-strong)" strokeWidth={0.01} />
          <line x1={-1} y1={0} x2={1} y2={0} stroke="var(--border)" strokeWidth={0.008} />
          <line x1={0} y1={-1} x2={0} y2={1} stroke="var(--border)" strokeWidth={0.008} />
          {rotPoints.map((p, i) => {
            const col = tokenColor(i)
            return (
              <g key={i}>
                <line
                  x1={0}
                  y1={0}
                  x2={p.x}
                  y2={p.y}
                  stroke={col.fg}
                  strokeWidth={i === Math.min(posIndex, n - 1) ? 0.035 : 0.014}
                  opacity={i < n ? 0.85 : 0.2}
                />
                <circle cx={p.x} cy={p.y} r={0.045} fill={col.bg} stroke={col.fg} strokeWidth={0.012} />
              </g>
            )
          })}
        </svg>
        <div className="note" style={{ marginTop: 8 }}>
          {c.ropeNote1}
          {pairIndex}
          {c.ropeNote2}
        </div>
      </Card>

      <Card title={c.pcaTitle} hint={c.pcaHint}>
        {scatter.points.length > 1 ? (
          <ScatterPlot points={scatter.points} labels={labels} explained={scatter.explained} note={c} />
        ) : (
          <div className="note">{c.need2}</div>
        )}
      </Card>
    </div>
  )
}

function ScatterPlot(props: {
  points: { x: number; y: number }[]
  labels: string[]
  explained: [number, number]
  note: typeof zh
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
          const col = tokenColor(i)
          return (
            <g key={i}>
              <circle cx={sx(p.x)} cy={sy(p.y)} r={7} fill={col.bg} stroke={col.fg} strokeWidth={1.2} />
              <text x={sx(p.x)} y={sy(p.y) - 12} textAnchor="middle">
                {props.labels[i]}
              </text>
            </g>
          )
        })}
      </svg>
      <div className="note">
        {props.note.pcaNote1} {(props.explained[0] * 100).toFixed(1)}% {props.note.pcaNote2}{' '}
        {(props.explained[1] * 100).toFixed(1)}% {props.note.pcaNote3}
      </div>
    </div>
  )
}

/**
 * 模块三（主推）：多头自注意力
 *
 * 核心交互：
 *   - 输入任意文本，实时算出每一层每一个头的注意力权重
 *   - 切换层 / 头，悬停看具体数值，点击某一行看"这个 token 在看谁"
 *   - 提供「本层所有头一览」，一眼看出不同头在干完全不同的事
 *   - 可调温度、因果掩码、距离衰减，观察分布形态怎么变
 *   - 可选：加载真实模型权重（transformers.js，浏览器内推理），把模拟换成真值
 */
import { useEffect, useMemo, useState } from 'react'
import { Card, Slider, Stats, Toggle } from '../../components/Controls'
import { BarList, HeatLegend, Heatmap, type HoverInfo } from '../../components/Heatmap'
import { Principle } from '../../components/Principle'
import { TokenChips } from '../../components/Tokens'
import {
  computeAttention,
  DEFAULT_ATTENTION_CONFIG,
  PATTERN_INFO,
  headMetrics,
  rowTopK,
  type AttentionConfig,
} from '../../core/attention'
import { heatColor, tokenColor } from '../../core/color'
import { SHARED_MODEL } from '../../core/sharedModel'
import { encode } from '../../core/bpe'
import { SAMPLE_TEXTS } from '../../core/corpus'
import {
  REAL_MODELS,
  getStatus,
  load,
  runAttention,
  unload,
  type RealAttention,
  type RealStatus,
} from '../../core/realModel'
import { useLang, type Lang } from '../../i18n'

const MAX_TOKENS = 44

const zh = {
  h2: '③ 多头自注意力：模型到底在「看」哪里',
  lead1: '自注意力的本质只有一句话：',
  lead2:
    '每个位置带着自己的问题（Query）去和所有位置的名字牌（Key）比对，按匹配程度把大家的内容（Value）加权平均过来',
  lead3: '。多个头并行做这件事，每个头关注的角度不同。',
  principleTitle: '注意力是怎么算出来的？',
  whyScale: '为什么要除以 √d_k？',
  whyScaleBody:
    '维度越高，点积的绝对值越大，softmax 会被推到极陡的区域，梯度几乎为零。除以 √d_k 相当于把方差拉回 1，让训练稳定。你可以把「温度」滑块调小，直观感受一下分布变陡之后发生了什么。',
  whyHeads: '为什么一定要多个头？',
  whyHeadsBody:
    '一个头的注意力矩阵是「平均主义」的——它只能表达一种关系。分成多个头之后，有的头专门看前一个词，有的专门盯句首，有的负责抓语义相似，模型就能同时建模多种关系。下面「本层所有头一览」里，你能清楚看到它们的图案完全不一样。',
  warn: '⚠️ 诚实说明：默认显示的是确定性模拟结果，不是真实模型的前向输出。但计算流程、形状规律与真实 GPT 类模型高度一致：不同头会稳定呈现「前一个 token / 句首 sink / 标点 / 内容相似」等文献中反复观察到的典型模式。想看真值，用下面的「加载真实模型」。',
  inputHint: '换一段话，注意力图案会立刻变',
  truncated: '文本较长，已只取前',
  truncated2: '个 token 参与计算（再多热力图就糊了）。',
  struct: '模型结构参数',
  layers: '层数 L',
  heads: '注意力头数 H',
  headsHint: '每层',
  headsHint2: '个头，每个头',
  headsHint3: '维',
  temperature: 'softmax 温度 T',
  temperatureHint: '越大分布越平，越小越尖锐',
  decay: '距离衰减',
  decayHint: '越大越只关注附近',
  causal: '因果掩码（不看未来）',
  pickTitle: '选择层与头',
  pickLayer: '层',
  pickLayerHint: '浅层偏位置/局部，深层偏语义/全局',
  layerBtn: '第',
  layerBtn2: '层',
  allHeads: '本层所有头一览',
  allHeadsHint: '点选一个头看大图 —— 注意它们的图案差别有多大',
  headBtn: '头',
  weightTitle: '注意力权重',
  weightHint: '行 = 谁在看（Query），列 = 被看的是谁（Key）；每行之和为 1',
  personality: '这个头的性格',
  hoverHint: '把鼠标移到格子上看具体数值，点击某一行可以看它的完整注意力分布。',
  tokenAt: '第',
  tokenUnit: '个 token',
  looksAt: '看向第',
  weightIs: '的权重 =',
  strong: '（关注很强）',
  weak: '（几乎不看）',
  rowTitle: '在看谁',
  rowHint: '点击热力图任意一行可切换',
  metrics: '这个头的量化指标',
  entropy: '平均注意力熵',
  maxWeight: '平均最大权重',
  sink: '句首 sink 占比',
  dk: '每头维度 d_k',
  metricsNote1: '熵越低 = 越「专注」（只盯着少数几个位置）；熵接近 log(n)=',
  metricsNote2:
    '说明几乎在平均地看所有人。「句首 sink 占比」高说明这个头把大量注意力无脑丢给了第一个 token —— 这是真实大模型里非常普遍的现象，也是很多推理加速方案（比如 StreamingLLM）必须保留首 token 的原因。',
  realBadge: '真实权重',
  simBadge: '模拟',
  realTokens: '真实 token 数',
}

const en: typeof zh = {
  h2: '③ Multi-Head Attention: what is the model looking at?',
  lead1: 'Self-attention boils down to one sentence: ',
  lead2:
    'every position takes its own question (Query), matches it against everyone’s name tag (Key), and pulls back a weighted average of their content (Value)',
  lead3: '. Several heads do this in parallel, each with a different angle.',
  principleTitle: 'How is attention actually computed?',
  whyScale: 'Why divide by √d_k?',
  whyScaleBody:
    'The higher the dimension, the larger the dot products, which pushes softmax into a saturated region where gradients vanish. Dividing by √d_k pulls the variance back to 1 and keeps training stable. Drag the temperature slider down to feel what a saturated distribution looks like.',
  whyHeads: 'Why multiple heads at all?',
  whyHeadsBody:
    'A single head is "averaging-only" — it can express just one relation. With several heads, one looks at the previous token, one parks on the first token, one matches semantics, and the model models all of them at once. The all-heads strip below makes the difference obvious.',
  warn: '⚠️ Honest note: by default you are seeing a deterministic simulation, not a real forward pass. But the computation path and the shapes match real GPT-style models closely: heads reliably show the "previous token / first-token sink / delimiter / content matching" patterns reported over and over in the literature. For the real thing, use "Load real model" below.',
  inputHint: 'Change the sentence and the attention pattern changes instantly',
  truncated: 'Text is long; only the first ',
  truncated2: ' tokens are used (any more and the heatmap turns to mush).',
  struct: 'Architecture knobs',
  layers: 'Layers L',
  heads: 'Attention heads H',
  headsHint: 'per layer',
  headsHint2: ', each head ',
  headsHint3: ' dims',
  temperature: 'softmax temperature T',
  temperatureHint: 'higher = flatter, lower = sharper',
  decay: 'Distance decay',
  decayHint: 'higher = more local',
  causal: 'Causal mask (no peeking ahead)',
  pickTitle: 'Pick a layer and a head',
  pickLayer: 'Layer',
  pickLayerHint: 'shallow = positional/local, deep = semantic/global',
  layerBtn: 'L',
  layerBtn2: '',
  allHeads: 'All heads in this layer',
  allHeadsHint: 'click one to enlarge it — notice how different the patterns are',
  headBtn: 'H',
  weightTitle: 'Attention weights',
  weightHint: 'row = who is looking (Query), column = who is looked at (Key); each row sums to 1',
  personality: 'this head’s personality',
  hoverHint: 'Hover a cell for the exact value; click a row to see its full distribution.',
  tokenAt: 'token #',
  tokenUnit: '',
  looksAt: 'attends to token #',
  weightIs: 'with weight =',
  strong: '(strong focus)',
  weak: '(barely looks)',
  rowTitle: 'is looking at',
  rowHint: 'click any row in the heatmap to switch',
  metrics: 'Quantified for this head',
  entropy: 'mean attention entropy',
  maxWeight: 'mean max weight',
  sink: 'first-token sink share',
  dk: 'per-head dims d_k',
  metricsNote1: 'Lower entropy = more focused. Entropy near log(n)=',
  metricsNote2:
    ' means it is looking at everyone about equally. A high sink share means the head dumps attention on the first token — extremely common in real LLMs, and exactly why StreamingLLM has to keep that first token forever.',
  realBadge: 'real weights',
  simBadge: 'simulated',
  realTokens: 'real token count',
}

const DICT: Record<Lang, typeof zh> = { zh, en }

/** 一个头的小缩略图，用于"本层所有头一览" */
function MiniHeat(props: {
  matrix: number[][]
  causal: boolean
  active: boolean
  onClick: () => void
  title: string
}) {
  const n = props.matrix.length
  const cell = Math.max(2, Math.min(6, Math.floor(120 / Math.max(n, 1))))
  return (
    <button
      onClick={props.onClick}
      title={props.title}
      style={{
        border: props.active ? '2px solid var(--accent)' : '1px solid var(--border)',
        background: 'var(--surface)',
        padding: 3,
        borderRadius: 6,
        cursor: 'pointer',
        lineHeight: 0,
      }}
    >
      {props.matrix.map((row, i) => (
        <div key={i} style={{ display: 'flex' }}>
          {row.map((v, j) => {
            const masked = props.causal && j > i
            return (
              <div
                key={j}
                style={{ width: cell, height: cell, background: masked ? 'var(--surface-2)' : heatColor(v) }}
              />
            )
          })}
        </div>
      ))}
    </button>
  )
}

export function AttentionModule() {
  const { lang, t } = useLang()
  const c = DICT[lang]

  const [text, setText] = useState(SAMPLE_TEXTS[0].text)
  const [cfg, setCfg] = useState<AttentionConfig>(DEFAULT_ATTENTION_CONFIG)
  const [layer, setLayer] = useState(0)
  const [head, setHead] = useState(0)
  const [hover, setHover] = useState<HoverInfo | null>(null)
  const [selectedRow, setSelectedRow] = useState(0)

  // —— 真实模型相关状态 ——
  const [realStatus, setRealStatus] = useState<RealStatus>(getStatus())
  const [progress, setProgress] = useState(0)
  const [realData, setRealData] = useState<RealAttention | null>(null)
  const [realError, setRealError] = useState('')
  const [useReal, setUseReal] = useState(false)
  const [modelId, setModelId] = useState<string>(REAL_MODELS[0].id)
  const [mirror, setMirror] = useState('https://hf-mirror.com')

  const tokenObjs = useMemo(() => encode(text, SHARED_MODEL).tokens, [text])
  const truncated = tokenObjs.length > MAX_TOKENS
  const tokens = useMemo(
    () => tokenObjs.slice(0, MAX_TOKENS).map((x) => (x.text === ' ' ? '␣' : x.text)),
    [tokenObjs],
  )

  const sim = useMemo(() => computeAttention(tokens, cfg), [tokens, cfg])

  const handleLoad = async () => {
    setRealStatus('loading')
    setProgress(0)
    setRealError('')
    try {
      await load(modelId, mirror, (p) => setProgress(p))
      setRealStatus('ready')
      setUseReal(true)
    } catch (e: any) {
      setRealStatus('error')
      setRealError(e?.message ?? String(e))
    }
  }

  const handleUnload = () => {
    unload()
    setRealData(null)
    setUseReal(false)
    setRealStatus('idle')
  }

  // 真实模型就绪后，文本变化时重新跑一次前向
  useEffect(() => {
    if (!useReal || realStatus !== 'ready') return
    let cancelled = false
    ;(async () => {
      try {
        const r = await runAttention(text)
        if (!cancelled) setRealData(r)
      } catch (e: any) {
        if (!cancelled) {
          setRealError(e?.message ?? String(e))
          setRealStatus('error')
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [useReal, realStatus, text])

  const isReal = useReal && realStatus === 'ready' && realData !== null
  const activeTokens = isReal ? realData.tokens : tokens
  const activeWeights = isReal ? realData.weights : sim.weights
  const nL = isReal ? realData.nLayers : cfg.nLayers
  const nH = isReal ? realData.nHeads : cfg.nHeads

  const safeLayer = Math.min(layer, Math.max(nL - 1, 0))
  const safeHead = Math.min(head, Math.max(nH - 1, 0))
  const matrix = activeWeights[safeLayer]?.[safeHead] ?? []
  const simSummary = isReal ? null : sim.summaries[safeLayer]?.[safeHead]
  const metrics = isReal ? headMetrics(matrix) : simSummary
  const patternInfo = simSummary ? PATTERN_INFO[simSummary.pattern] : null

  const topK = useMemo(
    () => (matrix[selectedRow] ? rowTopK(matrix[selectedRow], 8) : []),
    [matrix, selectedRow],
  )

  const patch = (p: Partial<AttentionConfig>) => setCfg((prev) => ({ ...prev, ...p }))
  const dk = isReal ? Math.floor(cfg.dModel / Math.max(nH, 1)) : Math.floor(cfg.dModel / cfg.nHeads)

  return (
    <div>
      <div className="module-head">
        <h2>{c.h2}</h2>
        <div className="lead">
          {c.lead1}
          <strong>{c.lead2}</strong>
          {c.lead3}
        </div>
      </div>

      <Principle
        title={c.principleTitle}
        formula={`Q = X · W_Q    K = X · W_K    V = X · W_V

scores  = Q · Kᵀ / √d_k        ← every query scores every key
scores += mask (future → -∞)   ← a causal LM must not peek ahead
weights = softmax(scores / T)   ← how each query spends its attention budget
output  = weights · V           ← pull information back by weight

multi-head: split d_model into h slices, run each independently, concat + project`}
        analogy={
          lang === 'zh'
            ? '把每个 token 想成一个参会的人。每个人手里有一张「我想找什么」的寻人启事（Query），胸前别着一张「我是谁」的名牌（Key），兜里揣着自己的资料（Value）。开会时每个人都去跟全场比对启事和名牌，越匹配就越多地抄对方的资料。多头就是同时开好几场主题不同的小会。'
            : 'Picture every token as a person at a conference. Each holds a "what I am looking for" flyer (Query), wears a "who I am" name tag (Key), and carries their own notes (Value). Everyone matches flyers against name tags across the room and copies notes in proportion to the match. Multi-head is simply several such meetups running in parallel, each with a different theme.'
        }
        detail={
          <>
            <p>
              <strong>{c.whyScale}</strong> {c.whyScaleBody}
            </p>
            <p>
              <strong>{c.whyHeads}</strong> {c.whyHeadsBody}
            </p>
          </>
        }
        warn={c.warn}
      />

      <Card title={t('inputText')} hint={c.inputHint}>
        <textarea rows={3} value={text} onChange={(e) => setText(e.target.value)} />
        <div className="chip-row" style={{ marginTop: 10 }}>
          {SAMPLE_TEXTS.map((s) => (
            <button key={s.label} className="btn" onClick={() => setText(s.text)}>
              {lang === 'zh' ? s.label : s.labelEn}
            </button>
          ))}
        </div>
        {truncated && !isReal && (
          <div className="note" style={{ marginTop: 8 }}>
            {c.truncated} {MAX_TOKENS} {c.truncated2}
          </div>
        )}
      </Card>

      <Card
        title={t('realSection')}
        hint={isReal ? c.realBadge : c.simBadge}
      >
        <div className="note" style={{ marginBottom: 10 }}>
          {t('realIntro')}
        </div>
        <div className="controls">
          <div className="control" style={{ minWidth: 200 }}>
            <label>
              <span>{t('modelLabel')}</span>
            </label>
            <select
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              style={{
                font: 'inherit',
                fontSize: 12.5,
                padding: '6px 8px',
                border: '1px solid var(--border-strong)',
                borderRadius: 6,
                background: 'var(--surface)',
                color: 'var(--text)',
              }}
            >
              {REAL_MODELS.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label} · {m.size}
                </option>
              ))}
            </select>
          </div>
          <div className="control" style={{ minWidth: 220 }}>
            <label>
              <span>{t('mirrorLabel')}</span>
            </label>
            <input
              type="text"
              value={mirror}
              onChange={(e) => setMirror(e.target.value)}
              style={{ font: 'inherit', fontSize: 12.5, padding: '6px 8px' }}
            />
          </div>
          {realStatus !== 'ready' ? (
            <button className="btn primary" onClick={handleLoad} disabled={realStatus === 'loading'}>
              {realStatus === 'loading' ? `${t('loading')} ${progress}%` : t('loadModel')}
            </button>
          ) : (
            <button className="btn" onClick={handleUnload}>
              {t('unload')}
            </button>
          )}
        </div>
        <div className="note" style={{ marginTop: 10 }}>
          {t('realNote')}
          {'　'}
          {t('mirrorHint')}
        </div>
        {realStatus === 'ready' && (
          <div className="controls" style={{ marginTop: 10 }}>
            <Toggle label={c.realBadge} checked={useReal} onChange={setUseReal} />
            {realData && (
              <span className="note">
                {realData.nLayers} {lang === 'zh' ? '层' : 'layers'} × {realData.nHeads}{' '}
                {lang === 'zh' ? '头' : 'heads'} · {c.realTokens}: {realData.tokens.length}
              </span>
            )}
          </div>
        )}
        {realStatus === 'error' && (
          <div className="warn" style={{ marginTop: 10 }}>
            {t('realFailed')}：{realError}
          </div>
        )}
      </Card>

      <Card title={c.struct}>
        <div className="controls">
          <Slider
            label={c.layers}
            value={cfg.nLayers}
            min={1}
            max={6}
            disabled={isReal}
            onChange={(v) => {
              patch({ nLayers: v })
              setLayer((l) => Math.min(l, v - 1))
            }}
            format={(v) => (lang === 'zh' ? `${v} 层` : `L${v}`)}
          />
          <Slider
            label={c.heads}
            value={cfg.nHeads}
            min={1}
            max={8}
            disabled={isReal}
            onChange={(v) => {
              patch({ nHeads: v })
              setHead((h) => Math.min(h, v - 1))
            }}
            format={(v) => (lang === 'zh' ? `${v} 头` : `H${v}`)}
            hint={`${c.headsHint} ${cfg.nHeads} ${c.headsHint2} ${Math.floor(cfg.dModel / cfg.nHeads)} ${c.headsHint3}`}
          />
          <Slider
            label={c.temperature}
            value={cfg.temperature}
            min={0.3}
            max={2.5}
            step={0.1}
            disabled={isReal}
            onChange={(v) => patch({ temperature: v })}
            format={(v) => v.toFixed(1)}
            hint={c.temperatureHint}
          />
          <Slider
            label={c.decay}
            value={cfg.distanceDecay}
            min={0}
            max={0.5}
            step={0.02}
            disabled={isReal}
            onChange={(v) => patch({ distanceDecay: v })}
            format={(v) => v.toFixed(2)}
            hint={c.decayHint}
          />
          <Toggle label={c.causal} checked={cfg.causal} onChange={(v) => patch({ causal: v })} />
        </div>
        {isReal && (
          <div className="note" style={{ marginTop: 8 }}>
            {lang === 'zh'
              ? '真实模型模式下，层数 / 头数 / 温度 / 衰减由模型本身决定，不能再调；因果掩码仍然可以开关（只影响显示，不改变真实权重）。'
              : 'With real weights, layers / heads / temperature / decay come from the model itself and are locked. The causal toggle only affects rendering, not the real weights.'}
          </div>
        )}
      </Card>

      <Card title={c.pickTitle}>
        <div className="card-title" style={{ marginBottom: 6 }}>
          <span>{c.pickLayer}</span>
          <span className="hint">{c.pickLayerHint}</span>
        </div>
        <div className="chip-row" style={{ marginBottom: 14 }}>
          {Array.from({ length: nL }, (_, l) => (
            <button
              key={l}
              className={`btn${l === safeLayer ? ' primary' : ''}`}
              onClick={() => setLayer(l)}
            >
              {lang === 'zh' ? `第 ${l + 1} 层` : `L${l + 1}`}
            </button>
          ))}
        </div>
        <div className="card-title" style={{ marginBottom: 6 }}>
          <span>{c.allHeads}</span>
          <span className="hint">{c.allHeadsHint}</span>
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
          {Array.from({ length: nH }, (_, h) => (
            <div key={h} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3 }}>
              <MiniHeat
                matrix={activeWeights[safeLayer]?.[h] ?? []}
                causal={cfg.causal}
                active={h === safeHead}
                onClick={() => setHead(h)}
                title={sim.patterns[safeLayer]?.[h] ? PATTERN_INFO[sim.patterns[safeLayer][h]].name : `H${h + 1}`}
              />
              <span style={{ fontSize: 11, color: h === safeHead ? 'var(--accent)' : 'var(--text-3)' }}>
                {lang === 'zh' ? `头 ${h + 1}` : `H${h + 1}`}
              </span>
            </div>
          ))}
        </div>
      </Card>

      <Card
        title={`${c.weightTitle} · L${safeLayer + 1} / H${safeHead + 1}`}
        hint={c.weightHint}
      >
        {patternInfo && (
          <div className="note" style={{ marginBottom: 10 }}>
            <b>
              {c.personality}：{patternInfo.name}
            </b>{' '}
            —— {patternInfo.desc}
          </div>
        )}
        <Heatmap
          matrix={matrix}
          rowLabels={activeTokens}
          colLabels={activeTokens}
          isMasked={(i, j) => cfg.causal && j > i}
          onHover={setHover}
          onSelectRow={setSelectedRow}
        />
        <HeatLegend min="0" max="1" />
        <div className="tooltip">
          {hover ? (
            <>
              {c.tokenAt}
              <b>{hover.i + 1}</b>「<b>{activeTokens[hover.i]}</b>」{c.looksAt} <b>{hover.j + 1}</b>「
              <b>{activeTokens[hover.j]}</b>」{c.weightIs} <b>{hover.value.toFixed(4)}</b>
              {hover.value > 0.3 ? `　${c.strong}` : hover.value < 0.02 ? `　${c.weak}` : ''}
            </>
          ) : (
            c.hoverHint
          )}
        </div>
      </Card>

      <Card
        title={`#${selectedRow + 1}「${activeTokens[selectedRow] ?? ''}」${c.rowTitle}`}
        hint={c.rowHint}
      >
        <BarList
          items={topK.map((x) => ({
            label: `${x.index + 1}. ${activeTokens[x.index] ?? ''}`,
            value: x.value,
          }))}
        />
        <div style={{ marginTop: 12 }}>
          {isReal ? (
            <div className="chip-row tight">
              {activeTokens.map((tk, i) => (
                <span
                  key={i}
                  className="token"
                  style={{
                    background: i === selectedRow ? 'var(--accent-soft)' : 'var(--surface-2)',
                    color: i === selectedRow ? 'var(--accent)' : 'var(--text)',
                  }}
                >
                  <span>{tk === ' ' ? '␣' : tk}</span>
                </span>
              ))}
            </div>
          ) : (
            <TokenChips
              tokens={tokenObjs.slice(0, MAX_TOKENS)}
              highlight={selectedRow}
              showId={false}
            />
          )}
        </div>
      </Card>

      {metrics && (
        <Card title={c.metrics}>
          <Stats
            items={[
              { k: c.entropy, v: metrics.avgEntropy.toFixed(2) },
              { k: c.maxWeight, v: metrics.avgMax.toFixed(3) },
              { k: c.sink, v: (metrics.sinkRatio * 100).toFixed(1) + '%' },
              { k: c.dk, v: String(dk) },
            ]}
          />
          <div className="note" style={{ marginTop: 10 }}>
            {c.metricsNote1}
            {Math.log(Math.max(activeTokens.length, 2)).toFixed(2)}
            {c.metricsNote2}
          </div>
        </Card>
      )}
    </div>
  )
}

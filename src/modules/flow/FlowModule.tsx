/**
 * 模块六：Transformer 层间数据流
 *
 * 把"一个 Block 里到底发生了什么"和"堆很多层之后表示怎么演化"两件事合在一起看。
 * 核心是一个可逐子步推进的动画：LN → Attention → 残差相加 → LN → FFN → 残差相加，
 * 每推进一层，右边同步显示这一层的表示热图、残差贡献和层间相似度。
 */
import { useEffect, useMemo, useState } from 'react'
import { Card, Slider, Stats } from '../../components/Controls'
import { BarList, HeatLegend, Heatmap } from '../../components/Heatmap'
import { Principle } from '../../components/Principle'
import { TokenChips } from '../../components/Tokens'
import { DEFAULT_TRANSFORMER, simulateForward, stateMatrix, type TransformerConfig } from '../../core/transformer'
import { divergingColor } from '../../core/color'
import { SHARED_MODEL } from '../../core/sharedModel'
import { encode } from '../../core/bpe'
import { SAMPLE_TEXTS } from '../../core/corpus'
import { useLang, type Lang } from '../../i18n'

const zh = {
  h2: '⑥ 层间数据流：一个 Block 里发生了什么',
  lead1: 'Transformer 最反直觉的一点是：',
  lead2: '从头到尾张量形状几乎不变',
  lead3: '，变的只有内容。真正让网络变深的是那条贯穿始终的残差高速公路 —— 每一层只往上面「加」一点增量。',
  principleTitle: 'pre-norm 结构在做什么？',
  d1a: 'FFN 为什么要先升维 4 倍？',
  d1b: '注意力负责「token 之间交流」，但它是线性的加权平均，表达能力有限。FFN 负责「每个 token 自己想清楚」，升维再降维相当于在高维空间里做一次非线性查表，学界普遍认为 FFN 才是模型存知识的地方。',
  d2a: '注意力分支和 FFN 分支，谁贡献大？',
  d2b: '看下面的残差贡献柱状图。通常 FFN 的相对幅度明显大于注意力，说明每一层「往箱子里塞的东西」主要来自 FFN。',
  d3a: '为什么深层可以堆到上百层？',
  d3b: '因为残差让每一层只需要学一个「小增量」。看层间相似度图你会发现：越靠后的层，相邻层表示越相似 —— 深层在做的是微调，而不是推倒重来。这也是「层剪枝」能奏效的原因。',
  warn: '数值来自确定性模拟的随机权重矩阵，不是训练过的模型。但张量形状、残差结构、pre-norm 顺序、层间相似度逐渐升高的趋势，都是真实且可验证的。',
  ln: 'LayerNorm',
  attn: '注意力',
  add: '＋ 残差',
  ffn: 'FFN 前馈',
  descLn: '先把每一行拉回均值 0 方差 1。稳定数值，也让后面的注意力分数尺度可控。',
  descAttn: '每个位置按权重把别人的信息加权平均过来 —— 这是唯一一处 token 之间发生信息交换的地方。',
  descAdd: '把注意力的输出加到原表示上。注意是「加」，不是「替换」，所以原始信息不会丢。',
  descLn2: '进入 FFN 之前再归一化一次。',
  descFfn: '升维到 4 倍再降回来，中间过一层 GELU 非线性。每个位置独立计算，token 之间不交流。',
  descAdd2: '再一次残差相加，一个 Block 结束。输出形状和输入完全一样，可以无限堆叠。',
  struct: '模型结构',
  layers: '层数 L',
  dModel: 'd_model',
  dff: 'd_ff（FFN 中间维）',
  speed: '动画速度',
  blockTitle: '单个 Block 的数据流',
  current: '当前：第',
  layerOf: '层',
  inputEmb: '输入嵌入',
  outputEmb: '输出 h',
  inputNote: '输入嵌入',
  inputNote2: '：token 嵌入 + 位置编码，形状 [n=',
  inputNote3: ', d=',
  inputNote4: ']。这就是送进第一个 Block 的东西。',
  residualLabel: '残差高速公路：每一层只是往上「加」一点',
  stateTitle: '当前表示热图',
  stateSim: '（模拟输入嵌入）',
  stateLayer: '（第',
  stateLayer2: '层 · ',
  stateHint: '每一行是一个 token 的前 32 维，已按最大值归一化',
  contribTitle: '残差贡献：每层往高速公路上加了多少',
  contribHint: '相对幅度 = 分支输出的范数 / 残差流的范数',
  contribNote: '两者都远小于 1，说明每一层只做了小幅修正 —— 这正是几十层能稳定训练的原因。FFN 的幅度通常大于注意力，因为参数量和表达容量都集中在它身上。',
  simTitle: '层间表示相似度',
  simHint: '越接近 1 = 两层表示越像；注意右下角会明显发白',
  simNote: '沿着对角线往外，颜色迅速变浅；而右下角（深层之间）几乎全白 —— 说明深层之间的表示已经非常接近，模型在做的是微调而非重构。',
  cfgTitle: '当前配置',
  seqLen: '序列长度 n',
  ffnRatio: 'FFN 放大倍数',
  params: '每 Block 参数量',
  attnBranch: '注意力',
}

const en: typeof zh = {
  h2: '⑥ Block data flow: what happens inside one layer',
  lead1: 'The most counter-intuitive thing about a Transformer: ',
  lead2: 'the tensor shape barely changes from start to finish',
  lead3: '. Only the content changes. What makes the network deep is the residual highway running through it — every layer only adds a small increment on top.',
  principleTitle: 'What is the pre-norm layout doing?',
  d1a: 'Why does FFN expand by 4× first?',
  d1b: 'Attention handles communication between tokens, but it is a linear weighted average with limited expressiveness. FFN lets each token "think for itself": expanding and projecting back is like a non-linear lookup in a high-dimensional space. The prevailing view is that FFN is where knowledge is stored.',
  d2a: 'Which branch contributes more, attention or FFN?',
  d2b: 'Look at the residual contribution chart below. FFN’s relative magnitude is usually clearly larger, meaning most of what each layer adds comes from FFN.',
  d3a: 'Why can you stack a hundred layers?',
  d3b: 'Because residuals let each layer learn only a small increment. In the similarity matrix you will see that deeper layers resemble their neighbours more and more — deep layers fine-tune rather than rebuild. That is also why layer pruning works at all.',
  warn: 'Numbers come from deterministic random weight matrices, not a trained model. But the tensor shapes, residual layout, pre-norm ordering and the rising layer-to-layer similarity are all real and verifiable.',
  ln: 'LayerNorm',
  attn: 'Attention',
  add: '＋ residual',
  ffn: 'FFN',
  descLn: 'Pull every row back to mean 0 variance 1 — stabilises values and keeps attention scores on a sane scale.',
  descAttn: 'Each position pulls in a weighted average of the others — the only place where tokens actually exchange information.',
  descAdd: 'Add the attention output to the original representation. Add, not replace, so nothing is lost.',
  descLn2: 'One more normalisation before entering the FFN.',
  descFfn: 'Expand to 4×, apply GELU, project back. Computed independently per position — no cross-token mixing here.',
  descAdd2: 'One more residual add and the block is done. Output shape matches input exactly, so blocks stack forever.',
  struct: 'Architecture',
  layers: 'layers L',
  dModel: 'd_model',
  dff: 'd_ff (FFN width)',
  speed: 'animation speed',
  blockTitle: 'Data flow through one block',
  current: 'current: layer ',
  layerOf: '',
  inputEmb: 'input embedding',
  outputEmb: 'output h',
  inputNote: 'Input embedding',
  inputNote2: ': token embedding + positional encoding, shape [n=',
  inputNote3: ', d=',
  inputNote4: ']. This is what enters the first block.',
  residualLabel: 'residual highway: each layer only adds a little',
  stateTitle: 'Current state heatmap',
  stateSim: ' (simulated input embedding)',
  stateLayer: ' (layer ',
  stateLayer2: ' · ',
  stateHint: 'each row is the first 32 dims of one token, normalised by max abs',
  contribTitle: 'Residual contribution: how much each layer adds',
  contribHint: 'relative magnitude = branch output norm / residual stream norm',
  contribNote: 'Both are far below 1, i.e. each layer only makes a small correction — which is exactly why dozens of layers train stably. FFN is usually larger than attention because both the parameter count and the capacity sit there.',
  simTitle: 'Layer-to-layer representation similarity',
  simHint: 'closer to 1 = more alike; note how the bottom-right washes out',
  simNote: 'Moving away from the diagonal the colour fades fast, and the bottom-right (deep layers) is almost white — deep layers are fine-tuning, not rebuilding.',
  cfgTitle: 'Current config',
  seqLen: 'sequence length n',
  ffnRatio: 'FFN expansion',
  params: 'params per block',
  attnBranch: 'attention',
}

const DICT: Record<Lang, typeof zh> = { zh, en }

export function FlowModule() {
  const { lang, t } = useLang()
  const c = DICT[lang]

  const [text, setText] = useState(SAMPLE_TEXTS[0].text)
  const [cfg, setCfg] = useState<TransformerConfig>(DEFAULT_TRANSFORMER)
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(700)

  const SUB_STEPS = useMemo(
    () => [
      { key: 'ln1', name: c.ln, shape: () => `[n, ${cfg.dModel}]`, desc: c.descLn },
      { key: 'attn', name: c.attn, shape: () => `[n, ${cfg.dModel}]`, desc: c.descAttn },
      { key: 'add1', name: c.add, shape: () => `[n, ${cfg.dModel}]`, desc: c.descAdd },
      { key: 'ln2', name: c.ln, shape: () => `[n, ${cfg.dModel}]`, desc: c.descLn2 },
      { key: 'ffn', name: c.ffn, shape: () => `[n, ${cfg.dModel}] → [n, ${cfg.dFF}] → [n, ${cfg.dModel}]`, desc: c.descFfn },
      { key: 'add2', name: c.add, shape: () => `[n, ${cfg.dModel}]`, desc: c.descAdd2 },
    ],
    [c, cfg.dModel, cfg.dFF],
  )

  const tokenObjs = useMemo(() => encode(text, SHARED_MODEL).tokens.slice(0, 20), [text])
  const labels = tokenObjs.map((x) => (x.text === ' ' ? '␣' : x.text))
  const trace = useMemo(() => (labels.length ? simulateForward(labels, cfg) : null), [labels, cfg])

  const totalSteps = 1 + cfg.nLayers * SUB_STEPS.length

  useEffect(() => {
    setStep((s) => Math.min(s, totalSteps - 1))
  }, [totalSteps])

  useEffect(() => {
    if (!playing) return
    const timer = setInterval(() => {
      setStep((s) => {
        if (s >= totalSteps - 1) {
          setPlaying(false)
          return s
        }
        return s + 1
      })
    }, speed)
    return () => clearInterval(timer)
  }, [playing, speed, totalSteps])

  const curLayer = step === 0 ? -1 : Math.floor((step - 1) / SUB_STEPS.length)
  const curSub = step === 0 ? -1 : (step - 1) % SUB_STEPS.length
  const layerTrace = trace && curLayer >= 0 ? trace.layers[curLayer] : null

  const currentState = useMemo(() => {
    if (!trace) return []
    if (step === 0) return stateMatrix(trace.h0)
    const tr = trace.layers[Math.min(curLayer, trace.layers.length - 1)]
    if (!tr) return []
    switch (SUB_STEPS[curSub]?.key) {
      case 'ln1':
      case 'attn':
        return stateMatrix(tr.attnOut)
      case 'add1':
      case 'ln2':
        return stateMatrix(tr.afterAttn)
      case 'ffn':
        return stateMatrix(tr.ffnOut)
      default:
        return stateMatrix(tr.hOut)
    }
  }, [trace, step, curLayer, curSub, SUB_STEPS])

  const patch = (p: Partial<TransformerConfig>) => setCfg((prev) => ({ ...prev, ...p }))

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
        formula={`modern (LLaMA / Qwen / Mistral) uses pre-norm:

  h ← h + Attention( LayerNorm(h) )
  h ← h + FFN( LayerNorm(h) )

the original Transformer used post-norm: h ← LayerNorm(h + Attention(h))
Looks like a one-bracket difference, but pre-norm keeps the residual path "clean" —
a straight line from input to output that no normalisation ever touches —
which is why it trains to hundreds of layers while post-norm needs careful warmup.`}
        analogy={
          lang === 'zh'
            ? '把残差流想成一条传送带，token 的表示就是上面流动的箱子。每一层不是把箱子换掉，而是打开箱子往里面塞一点新东西（注意力塞进上下文信息，FFN 塞进非线性变换的结果），然后原样送回传送带。塞了几十次之后，箱子里装的东西已经完全不同了，但它始终是同一个箱子 —— 这条传送带就是梯度能顺畅回传的原因。'
            : 'Think of the residual stream as a conveyor belt carrying a box per token. Each layer does not replace the box — it opens it, drops something in (attention adds context, FFN adds the result of a non-linear transform) and sends it back. After dozens of stops the contents are completely different, but it is still the same box. That belt is the reason gradients flow back cleanly.'
        }
        detail={
          <>
            <p>
              <strong>{c.d1a}</strong> {c.d1b}
            </p>
            <p>
              <strong>{c.d2a}</strong> {c.d2b}
            </p>
            <p>
              <strong>{c.d3a}</strong> {c.d3b}
            </p>
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

      <Card title={c.struct}>
        <div className="controls">
          <Slider
            label={c.layers}
            value={cfg.nLayers}
            min={2}
            max={8}
            onChange={(v) => patch({ nLayers: v })}
            format={(v) => (lang === 'zh' ? `${v} 层` : `L${v}`)}
          />
          <Slider
            label={c.dModel}
            value={cfg.dModel}
            min={32}
            max={96}
            step={16}
            onChange={(v) => patch({ dModel: v })}
            format={(v) => String(v)}
          />
          <Slider
            label={c.dff}
            value={cfg.dFF}
            min={64}
            max={384}
            step={32}
            onChange={(v) => patch({ dFF: v })}
            format={(v) => `${v}（${(v / cfg.dModel).toFixed(1)}×）`}
          />
          <Slider
            label={c.speed}
            value={speed}
            min={200}
            max={1500}
            step={100}
            onChange={setSpeed}
            format={(v) => `${v} ms`}
          />
        </div>
      </Card>

      <Card
        title={c.blockTitle}
        hint={
          step === 0
            ? `${c.current}— ${c.inputEmb}`
            : `${c.current}${curLayer + 1} ${c.layerOf} · ${SUB_STEPS[curSub]?.name}`
        }
        exportName="06-flow-block"
      >
        <BlockDiagram activeSub={curSub} dModel={cfg.dModel} dFF={cfg.dFF} zh={c} />
        <div className="controls" style={{ marginTop: 12 }}>
          <button className="btn primary" onClick={() => setPlaying((p) => !p)}>
            {playing ? t('pause') : t('play')}
          </button>
          <button className="btn" onClick={() => setStep((s) => Math.max(0, s - 1))} disabled={step === 0}>
            {t('prevStep')}
          </button>
          <button
            className="btn"
            onClick={() => setStep((s) => Math.min(totalSteps - 1, s + 1))}
            disabled={step >= totalSteps - 1}
          >
            {t('nextStep')}
          </button>
          <button
            className="btn"
            onClick={() => {
              setStep(0)
              setPlaying(false)
            }}
          >
            {t('reset')}
          </button>
          <span className="note">
            {step + 1} / {totalSteps} {t('stepProgress')}
          </span>
        </div>
        {step === 0 ? (
          <div className="note" style={{ marginTop: 10 }}>
            <b>{c.inputNote}</b>
            {c.inputNote2}
            {labels.length}
            {c.inputNote3}
            {cfg.dModel}
            {c.inputNote4}
          </div>
        ) : (
          <div className="note" style={{ marginTop: 10 }}>
            <b>{SUB_STEPS[curSub]?.name}</b>（{lang === 'zh' ? '第' : 'layer '} {curLayer + 1}{' '}
            {lang === 'zh' ? '层' : ''}，{SUB_STEPS[curSub]?.shape()}）：{SUB_STEPS[curSub]?.desc}
          </div>
        )}
      </Card>

      <Card
        title={`${c.stateTitle}${step === 0 ? c.stateSim : `${c.stateLayer}${curLayer + 1}${c.stateLayer2}${SUB_STEPS[curSub]?.name}）`}`}
        hint={c.stateHint}
        exportName="06-flow-state"
      >
        {currentState.length > 0 ? (
          <>
            <Heatmap matrix={currentState} rowLabels={labels} colorOf={divergingColor} />
            <HeatLegend min="-1" max="1" colorOf={divergingColor} />
          </>
        ) : (
          <div className="note">{t('enterText')}</div>
        )}
      </Card>

      {trace && (
        <>
          <Card title={c.contribTitle} hint={c.contribHint}>
            <BarList
              items={trace.layers.flatMap((l) => [
                { label: `L${l.layer + 1} ${c.attnBranch}`, value: l.attnRatio },
                { label: `L${l.layer + 1} FFN`, value: l.ffnRatio },
              ])}
              format={(v) => v.toFixed(3)}
            />
            <div className="note" style={{ marginTop: 10 }}>
              {c.contribNote}
            </div>
          </Card>

          <Card title={c.simTitle} hint={c.simHint}>
            <Heatmap
              matrix={trace.simMatrix}
              rowLabels={[lang === 'zh' ? '嵌入' : 'emb', ...trace.layers.map((l) => `L${l.layer + 1}`)]}
              colLabels={['0', ...trace.layers.map((l) => `${l.layer + 1}`)]}
              colorOf={divergingColor}
            />
            <HeatLegend min="-1" max="1" colorOf={divergingColor} />
            <div className="note" style={{ marginTop: 10 }}>
              {c.simNote}
            </div>
          </Card>

          <Card title={c.cfgTitle}>
            <Stats
              items={[
                { k: c.seqLen, v: String(labels.length) },
                { k: c.layers, v: String(cfg.nLayers) },
                { k: c.dModel, v: String(cfg.dModel) },
                { k: c.ffnRatio, v: (cfg.dFF / cfg.dModel).toFixed(1) + '×' },
                {
                  k: c.params,
                  v: `${((2 * cfg.dModel * cfg.dModel + 2 * cfg.dModel * cfg.dFF) / 1000).toFixed(1)}K`,
                },
              ]}
            />
            <div style={{ marginTop: 12 }}>
              <TokenChips tokens={tokenObjs} showId={false} />
            </div>
          </Card>
        </>
      )}
    </div>
  )
}

function BlockDiagram(props: {
  activeSub: number
  dModel: number
  dFF: number
  zh: typeof zh
}) {
  const { activeSub, dModel, dFF } = props
  const boxes = [
    { x: 8, w: 70, label: props.zh.inputEmb, shape: `[n,${dModel}]`, group: -1 },
    { x: 96, w: 44, label: 'LN', shape: `[n,${dModel}]`, group: 0 },
    { x: 158, w: 82, label: props.zh.attn, shape: `[n,${dModel}]`, group: 1 },
    { x: 258, w: 38, label: '＋', shape: '', group: 2 },
    { x: 314, w: 44, label: 'LN', shape: `[n,${dModel}]`, group: 3 },
    { x: 376, w: 82, label: props.zh.ffn, shape: `[n,${dFF}]`, group: 4 },
    { x: 476, w: 38, label: '＋', shape: '', group: 5 },
    { x: 532, w: 70, label: props.zh.outputEmb, shape: `[n,${dModel}]`, group: 6 },
  ]
  const Y = 46
  const H = 46
  const isActive = (g: number) => activeSub === g

  return (
    <div style={{ overflowX: 'auto' }}>
      <svg viewBox="0 0 620 200" width="100%" style={{ minWidth: 560 }}>
        <defs>
          <marker id="flow-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M2 1L8 5L2 9" fill="none" stroke="var(--text-3)" strokeWidth="1.5" strokeLinecap="round" />
          </marker>
        </defs>
        {boxes.slice(0, -1).map((b, i) => (
          <line
            key={`a${i}`}
            x1={b.x + b.w}
            y1={Y + H / 2}
            x2={boxes[i + 1].x - 4}
            y2={Y + H / 2}
            stroke="var(--text-3)"
            strokeWidth={1.2}
            markerEnd="url(#flow-arrow)"
          />
        ))}

        <polyline
          points={`43,${Y + H} 43,152 495,152 495,${Y + H}`}
          fill="none"
          stroke="var(--accent)"
          strokeWidth={1.6}
          strokeDasharray="6 4"
        />
        <line x1={277} y1={152} x2={277} y2={Y + H} stroke="var(--accent)" strokeWidth={1.6} strokeDasharray="6 4" />
        <text x={269} y={172} textAnchor="middle" fontSize={11} fill="var(--accent)">
          {props.zh.residualLabel}
        </text>

        {boxes.map((b, i) => {
          const active = isActive(b.group)
          return (
            <g key={i}>
              <rect
                x={b.x}
                y={Y}
                width={b.w}
                height={H}
                rx={8}
                fill={active ? 'var(--accent-soft)' : 'var(--surface)'}
                stroke={active ? 'var(--accent)' : 'var(--border-strong)'}
                strokeWidth={active ? 1.6 : 0.8}
              />
              <text
                x={b.x + b.w / 2}
                y={Y + 16}
                textAnchor="middle"
                dominantBaseline="central"
                fontSize={12}
                fill={active ? 'var(--accent)' : 'var(--text)'}
              >
                {b.label}
              </text>
              {b.shape && (
                <text
                  x={b.x + b.w / 2}
                  y={Y + 32}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fontSize={9.5}
                  fill="var(--text-3)"
                  fontFamily="var(--mono)"
                >
                  {b.shape}
                </text>
              )}
            </g>
          )
        })}
      </svg>
    </div>
  )
}

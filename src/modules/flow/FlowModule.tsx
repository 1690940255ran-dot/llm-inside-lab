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

/** 每一层内部的子步骤，用于动画推进 */
const SUB_STEPS = [
  { key: 'ln1', name: 'LayerNorm', shape: (d: number) => `[n, ${d}]`, desc: '先把每一行拉回均值 0 方差 1。稳定数值，也让后面的注意力分数尺度可控。' },
  { key: 'attn', name: '注意力', shape: (d: number) => `[n, ${d}]`, desc: '每个位置按权重把别人的信息加权平均过来 —— 这是唯一一处 token 之间发生信息交换的地方。' },
  { key: 'add1', name: '＋ 残差', shape: (d: number) => `[n, ${d}]`, desc: '把注意力的输出加到原表示上。注意是"加"，不是"替换"，所以原始信息不会丢。' },
  { key: 'ln2', name: 'LayerNorm', shape: (d: number) => `[n, ${d}]`, desc: '进入 FFN 之前再归一化一次。' },
  { key: 'ffn', name: 'FFN 前馈', shape: (d: number, dFF: number) => `[n, ${d}] → [n, ${dFF}] → [n, ${d}]`, desc: '升维到 4 倍再降回来，中间过一层 GELU 非线性。每个位置独立计算，token 之间不交流。' },
  { key: 'add2', name: '＋ 残差', shape: (d: number) => `[n, ${d}]`, desc: '再一次残差相加，一个 Block 结束。输出形状和输入完全一样，可以无限堆叠。' },
]

export function FlowModule() {
  const [text, setText] = useState('注意力机制让模型关注输入中不同位置的信息。')
  const [cfg, setCfg] = useState<TransformerConfig>(DEFAULT_TRANSFORMER)
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(700)

  const tokenObjs = useMemo(() => encode(text, SHARED_MODEL).tokens.slice(0, 20), [text])
  const labels = tokenObjs.map((t) => (t.text === ' ' ? '␣' : t.text))
  const trace = useMemo(
    () => (labels.length ? simulateForward(labels, cfg) : null),
    [labels, cfg],
  )

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
    const t = trace.layers[Math.min(curLayer, trace.layers.length - 1)]
    if (!t) return []
    switch (SUB_STEPS[curSub]?.key) {
      case 'ln1':
      case 'attn':
        return stateMatrix(t.attnOut)
      case 'add1':
      case 'ln2':
        return stateMatrix(t.afterAttn)
      case 'ffn':
        return stateMatrix(t.ffnOut)
      default:
        return stateMatrix(t.hOut)
    }
  }, [trace, step, curLayer, curSub])

  const patch = (p: Partial<TransformerConfig>) => setCfg((c) => ({ ...c, ...p }))

  return (
    <div>
      <div className="module-head">
        <h2>⑥ 层间数据流：一个 Block 里发生了什么</h2>
        <div className="lead">
          Transformer 最反直觉的一点是：<strong>从头到尾张量形状几乎不变</strong>，
          变的只有内容。真正让网络变深的是那条贯穿始终的残差高速公路 —— 每一层只往上面"加"一点增量。
        </div>
      </div>

      <Principle
        title="pre-norm 结构在做什么？"
        formula={`现代主流（LLaMA / Qwen / Mistral）用的 pre-norm：

  h ← h + Attention( LayerNorm(h) )
  h ← h + FFN( LayerNorm(h) )

原始 Transformer 用的是 post-norm：h ← LayerNorm(h + Attention(h))
差别看着小，但 pre-norm 的残差通路是"干净"的（从输入到输出有一条不经过任何归一化的直连），
所以深层更好训练，这也是现在几乎没人再用 post-norm 的原因。`}
        analogy="把残差流想成一条传送带，token 的表示就是上面流动的箱子。每一层不是把箱子换掉，而是打开箱子往里面塞一点新东西（注意力塞进上下文信息，FFN 塞进非线性变换的结果），然后原样送回传送带。塞了几十次之后，箱子里装的东西已经完全不同了，但它始终是同一个箱子 —— 这条传送带就是梯度能顺畅回传的原因。"
        detail={
          <>
            <p>
              <strong>FFN 为什么要先升维 4 倍？</strong> 注意力负责"token 之间交流"，但它是线性的加权平均，
              表达能力有限。FFN 负责"每个 token 自己想清楚"，升维再降维相当于在高维空间里做一次非线性查表，
              学界普遍认为 FFN 才是模型存知识的地方。
            </p>
            <p>
              <strong>注意力分支和 FFN 分支，谁贡献大？</strong> 看下面的残差贡献柱状图。
              通常 FFN 的相对幅度明显大于注意力，说明每一层"往箱子里塞的东西"主要来自 FFN。
            </p>
            <p>
              <strong>为什么深层可以堆到上百层？</strong> 因为残差让每一层只需要学一个"小增量"。
              看层间相似度图你会发现：越靠后的层，相邻层表示越相似 ——
              深层在做的是微调，而不是推倒重来。这也是"层剪枝"能奏效的原因。
            </p>
          </>
        }
        warn="数值来自确定性模拟的随机权重矩阵，不是训练过的模型。但张量形状、残差结构、pre-norm 顺序、层间相似度逐渐升高的趋势，都是真实且可验证的。"
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

      <Card title="模型结构">
        <div className="controls">
          <Slider
            label="层数 L"
            value={cfg.nLayers}
            min={2}
            max={8}
            onChange={(v) => patch({ nLayers: v })}
            format={(v) => `${v} 层`}
          />
          <Slider
            label="d_model"
            value={cfg.dModel}
            min={32}
            max={96}
            step={16}
            onChange={(v) => patch({ dModel: v })}
            format={(v) => String(v)}
          />
          <Slider
            label="d_ff（FFN 中间维）"
            value={cfg.dFF}
            min={64}
            max={384}
            step={32}
            onChange={(v) => patch({ dFF: v })}
            format={(v) => `${v}（${(v / cfg.dModel).toFixed(1)}×）`}
          />
          <Slider
            label="动画速度"
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
        title="单个 Block 的数据流"
        hint={`当前：第 ${curLayer + 1} 层 · ${step === 0 ? '输入嵌入' : SUB_STEPS[curSub]?.name}`}
      >
        <BlockDiagram activeSub={curSub} dModel={cfg.dModel} dFF={cfg.dFF} />
        <div className="controls" style={{ marginTop: 12 }}>
          <button className="btn primary" onClick={() => setPlaying((p) => !p)}>
            {playing ? '暂停' : '播放'}
          </button>
          <button className="btn" onClick={() => setStep((s) => Math.max(0, s - 1))} disabled={step === 0}>
            上一步
          </button>
          <button
            className="btn"
            onClick={() => setStep((s) => Math.min(totalSteps - 1, s + 1))}
            disabled={step >= totalSteps - 1}
          >
            下一步
          </button>
          <button
            className="btn"
            onClick={() => {
              setStep(0)
              setPlaying(false)
            }}
          >
            重置
          </button>
          <span className="note">
            第 {step + 1} / {totalSteps} 步
          </span>
        </div>
        {step === 0 ? (
          <div className="note" style={{ marginTop: 10 }}>
            <b>输入嵌入</b>：token 嵌入 + 位置编码，形状 [n={labels.length}, d={cfg.dModel}]。这就是送进第一个 Block 的东西。
          </div>
        ) : (
          <div className="note" style={{ marginTop: 10 }}>
            <b>{SUB_STEPS[curSub]?.name}</b>（第 {curLayer + 1} 层，形状 {SUB_STEPS[curSub]?.shape(cfg.dModel, cfg.dFF)}）：
            {SUB_STEPS[curSub]?.desc}
          </div>
        )}
      </Card>

      <Card
        title={`当前表示热图${step === 0 ? '（输入嵌入）' : `（第 ${curLayer + 1} 层 · ${SUB_STEPS[curSub]?.name}）`}`}
        hint="每一行是一个 token 的前 32 维，已按最大值归一化"
      >
        {currentState.length > 0 ? (
          <>
            <Heatmap matrix={currentState} rowLabels={labels} colorOf={divergingColor} />
            <HeatLegend min="-1" max="1" colorOf={divergingColor} />
          </>
        ) : (
          <div className="note">请输入文本。</div>
        )}
      </Card>

      {trace && (
        <>
          <Card title="残差贡献：每层往高速公路上加了多少" hint="相对幅度 = 分支输出的范数 / 残差流的范数">
            <BarList
              items={trace.layers.flatMap((l) => [
                { label: `L${l.layer + 1} 注意力`, value: l.attnRatio },
                { label: `L${l.layer + 1} FFN`, value: l.ffnRatio },
              ])}
              format={(v) => v.toFixed(3)}
            />
            <div className="note" style={{ marginTop: 10 }}>
              两者都远小于 1，说明每一层只做了小幅修正 —— 这正是几十层能稳定训练的原因。
              FFN 的幅度通常大于注意力，因为参数量和表达容量都集中在它身上。
            </div>
          </Card>

          <Card title="层间表示相似度" hint="越接近 1 = 两层表示越像；注意右下角会明显发白">
            <Heatmap
              matrix={trace.simMatrix}
              rowLabels={['嵌入', ...trace.layers.map((l) => `L${l.layer + 1}`)]}
              colLabels={['0', ...trace.layers.map((l) => `${l.layer + 1}`)]}
              colorOf={divergingColor}
            />
            <HeatLegend min="-1" max="1" colorOf={divergingColor} />
            <div className="note" style={{ marginTop: 10 }}>
              沿着对角线往外，颜色迅速变浅；而右下角（深层之间）几乎全白 ——
              说明深层之间的表示已经非常接近，模型在做的是微调而非重构。
            </div>
          </Card>

          <Card title="当前配置">
            <Stats
              items={[
                { k: '序列长度 n', v: String(labels.length) },
                { k: '层数 L', v: String(cfg.nLayers) },
                { k: 'd_model', v: String(cfg.dModel) },
                { k: 'FFN 放大倍数', v: (cfg.dFF / cfg.dModel).toFixed(1) + '×' },
                { k: '每 Block 参数量', v: `${((2 * cfg.dModel * cfg.dModel + 2 * cfg.dModel * cfg.dFF) / 1000).toFixed(1)}K` },
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

/** 单个 Block 的结构示意图，高亮当前子步骤 */
function BlockDiagram(props: { activeSub: number; dModel: number; dFF: number }) {
  const { activeSub, dModel, dFF } = props
  const boxes: { x: number; w: number; label: string; shape: string; group: number }[] = [
    { x: 8, w: 70, label: '输入 h', shape: `[n,${dModel}]`, group: -1 },
    { x: 96, w: 44, label: 'LN', shape: `[n,${dModel}]`, group: 0 },
    { x: 158, w: 82, label: 'Attention', shape: `[n,${dModel}]`, group: 1 },
    { x: 258, w: 38, label: '＋', shape: '', group: 2 },
    { x: 314, w: 44, label: 'LN', shape: `[n,${dModel}]`, group: 3 },
    { x: 376, w: 82, label: 'FFN', shape: `[n,${dFF}]`, group: 4 },
    { x: 476, w: 38, label: '＋', shape: '', group: 5 },
    { x: 532, w: 70, label: '输出 h', shape: `[n,${dModel}]`, group: 6 },
  ]
  const Y = 46
  const H = 46
  const isActive = (g: number) => activeSub === g

  return (
    <div style={{ overflowX: 'auto' }}>
      <svg viewBox="0 0 620 200" width="100%" style={{ minWidth: 560 }}>
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
        <defs>
          <marker id="flow-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M2 1L8 5L2 9" fill="none" stroke="var(--text-3)" strokeWidth="1.5" strokeLinecap="round" />
          </marker>
        </defs>

        {/* 残差高速公路 */}
        <polyline
          points={`43,${Y + H} 43,152 495,152 495,${Y + H}`}
          fill="none"
          stroke="var(--accent)"
          strokeWidth={1.6}
          strokeDasharray="6 4"
        />
        <line x1={277} y1={152} x2={277} y2={Y + H} stroke="var(--accent)" strokeWidth={1.6} strokeDasharray="6 4" />
        <text x={269} y={172} textAnchor="middle" fontSize={11} fill="var(--accent)">
          残差高速公路：每一层只是往上「加」一点
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

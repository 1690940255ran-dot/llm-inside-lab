/**
 * 模块二（主推）：多头自注意力
 *
 * 这是整个站点最核心的一块。核心交互：
 *   - 输入任意文本，实时算出每一层每一个头的注意力权重
 *   - 切换层 / 头，悬停看具体数值，点击某一行看"这个 token 在看谁"
 *   - 提供「本层所有头一览」，一眼看出不同头在干完全不同的事
 *   - 可调温度、因果掩码、距离衰减，观察分布形态怎么变
 */
import { useMemo, useState } from 'react'
import { Card, Slider, Stats, Toggle } from '../../components/Controls'
import { BarList, HeatLegend, Heatmap, type HoverInfo } from '../../components/Heatmap'
import { Principle } from '../../components/Principle'
import { TokenChips } from '../../components/Tokens'
import {
  computeAttention,
  DEFAULT_ATTENTION_CONFIG,
  PATTERN_INFO,
  rowTopK,
  type AttentionConfig,
} from '../../core/attention'
import { heatColor, tokenColor } from '../../core/color'
import { SHARED_MODEL } from '../../core/sharedModel'
import { encode } from '../../core/bpe'
import { SAMPLE_TEXTS } from '../../core/corpus'

const MAX_TOKENS = 44

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
                style={{
                  width: cell,
                  height: cell,
                  background: masked ? 'var(--surface-2)' : heatColor(v),
                }}
              />
            )
          })}
        </div>
      ))}
    </button>
  )
}

export function AttentionModule() {
  const [text, setText] = useState('注意力机制让模型关注输入中不同位置的信息，因为它是 Transformer 的心脏。')
  const [cfg, setCfg] = useState<AttentionConfig>(DEFAULT_ATTENTION_CONFIG)
  const [layer, setLayer] = useState(0)
  const [head, setHead] = useState(0)
  const [hover, setHover] = useState<HoverInfo | null>(null)
  const [selectedRow, setSelectedRow] = useState(0)

  const tokenObjs = useMemo(() => encode(text, SHARED_MODEL).tokens, [text])
  const truncated = tokenObjs.length > MAX_TOKENS
  const tokens = useMemo(
    () => tokenObjs.slice(0, MAX_TOKENS).map((t) => (t.text === ' ' ? '␣' : t.text)),
    [tokenObjs],
  )

  const result = useMemo(() => computeAttention(tokens, cfg), [tokens, cfg])

  const safeLayer = Math.min(layer, cfg.nLayers - 1)
  const safeHead = Math.min(head, cfg.nHeads - 1)
  const matrix = result.weights[safeLayer]?.[safeHead] ?? []
  const summary = result.summaries[safeLayer]?.[safeHead]
  const patternInfo = summary ? PATTERN_INFO[summary.pattern] : null

  const topK = useMemo(
    () => (matrix[selectedRow] ? rowTopK(matrix[selectedRow], 8) : []),
    [matrix, selectedRow],
  )

  const patch = (p: Partial<AttentionConfig>) => setCfg((c) => ({ ...c, ...p }))

  return (
    <div>
      <div className="module-head">
        <h2>② 多头自注意力：模型到底在"看"哪里</h2>
        <div className="lead">
          自注意力的本质只有一句话：<strong>每个位置带着自己的问题（Query）去和所有位置的名字牌（Key）比对，
          按匹配程度把大家的内容（Value）加权平均过来</strong>。多个头并行做这件事，每个头关注的角度不同。
        </div>
      </div>

      <Principle
        title="注意力是怎么算出来的？"
        formula={`Q = X · W_Q    K = X · W_K    V = X · W_V

scores = Q · Kᵀ / √d_k        ← 每个 query 对每个 key 打一个原始分
scores = scores + 掩码(未来位置 → -∞)   ← 因果语言模型不能偷看后面
weights = softmax(scores / T)  ← 归一化成"每个 query 的注意力预算怎么分"
output  = weights · V          ← 按权重把信息取回来

多头：把 d_model 维切成 h 份，每份独立算一遍，最后拼接 + 线性变换`}
        analogy="把每个 token 想成一个参会的人。每个人手里有一张「我想找什么」的寻人启事（Query），胸前别着一张「我是谁」的名牌（Key），兜里揣着自己的资料（Value）。开会时每个人都去跟全场比对启事和名牌，越匹配就越多地抄对方的资料。多头就是同时开好几场主题不同的小会。"
        detail={
          <>
            <p>
              <strong>为什么要除以 √d_k？</strong> 维度越高，点积的绝对值越大，softmax 会被推到极陡的区域，
              梯度几乎为零。除以 √d_k 相当于把方差拉回 1，让训练稳定。你可以把「温度」滑块调小，直观感受一下分布变陡之后发生了什么。
            </p>
            <p>
              <strong>为什么一定要多个头？</strong> 一个头的注意力矩阵是"平均主义"的——它只能表达一种关系。
              分成多个头之后，有的头专门看前一个词，有的专门盯句首，有的负责抓语义相似，模型就能同时建模多种关系。
              下面「本层所有头一览」里，你能清楚看到它们的图案完全不一样。
            </p>
          </>
        }
        warn="⚠️ 诚实说明：本页的权重是用确定性模拟算出来的教学示意，不是真实模型的前向结果（真实权重需要下载几百 MB 参数）。但它的计算流程、形状规律与真实 GPT 类模型高度一致：不同头会稳定呈现「前一个 token / 句首 sink / 标点 / 内容相似」等文献中反复观察到的典型模式。后续版本会支持接入 transformers.js 跑真实权重。"
      />

      <Card title="输入文本" hint="换一段话，注意力图案会立刻变">
        <textarea rows={3} value={text} onChange={(e) => setText(e.target.value)} />
        <div className="chip-row" style={{ marginTop: 10 }}>
          {SAMPLE_TEXTS.map((s) => (
            <button key={s.label} className="btn" onClick={() => setText(s.text)}>
              {s.label}
            </button>
          ))}
        </div>
        {truncated && (
          <div className="note" style={{ marginTop: 8 }}>
            文本较长，已只取前 {MAX_TOKENS} 个 token 参与计算（再多热力图就糊了）。
          </div>
        )}
      </Card>

      <Card title="模型结构参数">
        <div className="controls">
          <Slider
            label="层数 L"
            value={cfg.nLayers}
            min={1}
            max={6}
            onChange={(v) => {
              patch({ nLayers: v })
              setLayer((l) => Math.min(l, v - 1))
            }}
            format={(v) => `${v} 层`}
          />
          <Slider
            label="注意力头数 H"
            value={cfg.nHeads}
            min={1}
            max={8}
            onChange={(v) => {
              patch({ nHeads: v })
              setHead((h) => Math.min(h, v - 1))
            }}
            format={(v) => `${v} 头`}
            hint={`每层 ${cfg.nHeads} 个头，每个头 ${Math.floor(cfg.dModel / cfg.nHeads)} 维`}
          />
          <Slider
            label="softmax 温度 T"
            value={cfg.temperature}
            min={0.3}
            max={2.5}
            step={0.1}
            onChange={(v) => patch({ temperature: v })}
            format={(v) => v.toFixed(1)}
            hint="越大分布越平，越小越尖锐"
          />
          <Slider
            label="距离衰减"
            value={cfg.distanceDecay}
            min={0}
            max={0.5}
            step={0.02}
            onChange={(v) => patch({ distanceDecay: v })}
            format={(v) => v.toFixed(2)}
            hint="越大越只关注附近"
          />
          <Toggle
            label="因果掩码（不看未来）"
            checked={cfg.causal}
            onChange={(v) => patch({ causal: v })}
          />
        </div>
      </Card>

      <Card title="选择层与头">
        <div className="card-title" style={{ marginBottom: 6 }}>
          <span>层</span>
          <span className="hint">浅层偏位置/局部，深层偏语义/全局</span>
        </div>
        <div className="chip-row" style={{ marginBottom: 14 }}>
          {Array.from({ length: cfg.nLayers }, (_, l) => (
            <button
              key={l}
              className={`btn${l === safeLayer ? ' primary' : ''}`}
              onClick={() => setLayer(l)}
            >
              第 {l + 1} 层
            </button>
          ))}
        </div>
        <div className="card-title" style={{ marginBottom: 6 }}>
          <span>本层所有头一览</span>
          <span className="hint">点选一个头看大图 —— 注意它们的图案差别有多大</span>
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
          {Array.from({ length: cfg.nHeads }, (_, h) => (
            <div key={h} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3 }}>
              <MiniHeat
                matrix={result.weights[safeLayer]?.[h] ?? []}
                causal={cfg.causal}
                active={h === safeHead}
                onClick={() => setHead(h)}
                title={PATTERN_INFO[result.patterns[safeLayer]?.[h] ?? 'broad'].name}
              />
              <span style={{ fontSize: 11, color: h === safeHead ? 'var(--accent)' : 'var(--text-3)' }}>
                头 {h + 1}
              </span>
            </div>
          ))}
        </div>
      </Card>

      <Card
        title={`注意力权重 · 第 ${safeLayer + 1} 层 / 第 ${safeHead + 1} 头`}
        hint="行 = 谁在看（Query），列 = 被看的是谁（Key）；每行之和为 1"
      >
        {patternInfo && (
          <div className="note" style={{ marginBottom: 10 }}>
            <b>这个头的性格：{patternInfo.name}</b> —— {patternInfo.desc}
          </div>
        )}
        <Heatmap
          matrix={matrix}
          rowLabels={tokens}
          colLabels={tokens}
          isMasked={(i, j) => cfg.causal && j > i}
          onHover={setHover}
          onSelectRow={setSelectedRow}
        />
        <HeatLegend min="0" max="1" />
        <div className="tooltip">
          {hover ? (
            <>
              第 <b>{hover.i + 1}</b> 个 token「<b>{tokens[hover.i]}</b>」看向第 <b>{hover.j + 1}</b> 个 token「
              <b>{tokens[hover.j]}</b>」的权重 = <b>{hover.value.toFixed(4)}</b>
              {hover.value > 0.3 ? '　（关注很强）' : hover.value < 0.02 ? '　（几乎不看）' : ''}
            </>
          ) : (
            '把鼠标移到格子上看具体数值，点击某一行可以看它的完整注意力分布。'
          )}
        </div>
      </Card>

      <Card
        title={`第 ${selectedRow + 1} 个 token「${tokens[selectedRow] ?? ''}」在看谁`}
        hint="点击热力图任意一行可切换"
      >
        <BarList
          items={topK.map((t) => ({
            label: `${t.index + 1}. ${tokens[t.index] ?? ''}`,
            value: t.value,
          }))}
        />
        <div style={{ marginTop: 12 }}>
          <TokenChips tokens={tokenObjs.slice(0, MAX_TOKENS)} highlight={selectedRow} showId={false} />
        </div>
      </Card>

      {summary && (
        <Card title="这个头的量化指标">
          <Stats
            items={[
              { k: '平均注意力熵', v: summary.avgEntropy.toFixed(2) },
              { k: '平均最大权重', v: summary.avgMax.toFixed(3) },
              { k: '句首 sink 占比', v: (summary.sinkRatio * 100).toFixed(1) + '%' },
              { k: '每头维度 d_k', v: String(Math.floor(cfg.dModel / cfg.nHeads)) },
            ]}
          />
          <div className="note" style={{ marginTop: 10 }}>
            熵越低 = 越"专注"（只盯着少数几个位置）；熵接近 log(n)={Math.log(Math.max(tokens.length, 2)).toFixed(2)} 说明几乎在平均地看所有人。
            「句首 sink 占比」高说明这个头把大量注意力无脑丢给了第一个 token —— 这是真实大模型里非常普遍的现象，
            也是很多推理加速方案（比如 StreamingLLM）必须保留首 token 的原因。
          </div>
        </Card>
      )}
    </div>
  )
}

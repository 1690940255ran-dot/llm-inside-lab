/**
 * 模块四：逐 token 生成与采样
 *
 * 大模型"一个字一个字往外蹦"这件事，本质是：每一步都算出一个词表上的概率分布，然后从中抽一个。
 * 这个模块把抽签前的每一步都摊开：原始概率 → 温度 → top-k → top-p → 最终抽中谁。
 *
 * 概率分布来自一个真正的统计语言模型（从内置语料数出来的 bigram），
 * 它比 GPT 简单一万倍，但采样机制作用在它上面的效果完全一致。
 */
import { useEffect, useMemo, useState } from 'react'
import { Card, Slider, Stats } from '../../components/Controls'
import { Principle } from '../../components/Principle'
import { TokenChips } from '../../components/Tokens'
import { DEFAULT_CORPUS } from '../../core/corpus'
import { encode } from '../../core/bpe'
import { SHARED_MODEL } from '../../core/sharedModel'
import { nextLogits, trainNGram } from '../../core/ngram'
import {
  DEFAULT_SAMPLING,
  argmax,
  candidateCount,
  entropyOf,
  sampleNext,
  type SamplingConfig,
} from '../../core/sampling'

/** 语料上训练一次，全局复用 */
const NGRAM = trainNGram(DEFAULT_CORPUS, SHARED_MODEL)
const VOCAB_SIZE = NGRAM.vocab.length
const TOP_ROWS = 12

interface GenToken {
  text: string
  prob: number
  source: 'bigram' | 'unigram'
  candidates: number
  entropy: number
}

/** 从给定历史连续生成 steps 个 token，返回字符串 */
function generateChain(history: string[], cfg: SamplingConfig, seed: number, steps: number): string {
  const h = [...history]
  const out: string[] = []
  for (let i = 0; i < steps; i++) {
    const { logits } = nextLogits(NGRAM, h)
    const step = sampleNext(logits, cfg, seed + i * 7919)
    const t = NGRAM.vocab[step.chosen] ?? ''
    h.push(t)
    out.push(t)
  }
  return out.join('')
}

export function GenerationModule() {
  const [prompt, setPrompt] = useState('大语言模型通过预测下一个')
  const [cfg, setCfg] = useState<SamplingConfig>(DEFAULT_SAMPLING)
  const [generated, setGenerated] = useState<GenToken[]>([])
  const [seed, setSeed] = useState(42)
  const [auto, setAuto] = useState(false)
  const [compareSeed, setCompareSeed] = useState(1)

  const promptTokens = useMemo(
    () => encode(prompt, SHARED_MODEL).tokens.map((t) => t.text),
    [prompt],
  )
  const history = useMemo(
    () => [...promptTokens, ...generated.map((g) => g.text)],
    [promptTokens, generated],
  )

  const { logits, source } = useMemo(() => nextLogits(NGRAM, history), [history])

  // 每一步都用全新的参数重算一次分布（不真正落子），只为展示"如果现在抽，会怎么抽"
  const preview = useMemo(
    () => sampleNext(logits, cfg, seed + generated.length * 7919),
    [logits, cfg, seed, generated.length],
  )

  const greedyIndex = useMemo(() => argmax(preview.tempered), [preview])

  const rows = useMemo(() => {
    const idx = preview.tempered
      .map((p, i) => ({ p, i }))
      .sort((a, b) => b.p - a.p)
      .slice(0, TOP_ROWS)
      .map((o) => o.i)
    if (!idx.includes(preview.chosen)) idx.push(preview.chosen)
    if (!idx.includes(greedyIndex)) idx.push(greedyIndex)
    return idx
  }, [preview, greedyIndex])

  const stepOnce = () => {
    const g: GenToken = {
      text: NGRAM.vocab[preview.chosen] ?? '',
      prob: preview.final[preview.chosen] ?? 0,
      source,
      candidates: candidateCount(preview),
      entropy: entropyOf(preview.final),
    }
    setGenerated((prev) => [...prev, g])
  }

  useEffect(() => {
    if (!auto) return
    const timer = setInterval(stepOnce, 420)
    return () => clearInterval(timer)
  }, [auto, preview])

  useEffect(() => {
    if (generated.length >= 60) setAuto(false)
  }, [generated.length])

  const patch = (p: Partial<SamplingConfig>) => setCfg((c) => ({ ...c, ...p }))

  const comparisons = useMemo(
    () =>
      Array.from({ length: 5 }, (_, i) =>
        generateChain(promptTokens, cfg, compareSeed * 1000 + i * 131, 14),
      ),
    [promptTokens, cfg, compareSeed],
  )

  return (
    <div>
      <div className="module-head">
        <h2>④ 逐 token 生成：模型是怎么"抽签"的</h2>
        <div className="lead">
          模型每一步只做两件事：<strong>算出一个词表上的概率分布，然后从中抽一个 token</strong>，
          再把这个 token 拼回输入，继续下一步。温度、top-k、top-p 三个旋钮，改的都是"抽签之前那一步"。
        </div>
      </div>

      <Principle
        title="三个参数各自在改什么？"
        formula={`logits  ──(÷ 温度 T)──▶ softmax ──(top-k 截断)──▶ ──(top-p 核采样)──▶ 重新归一化 ──▶ 抽签

温度 T：  p_i ∝ exp(logit_i / T)
          T→0  分布变成 one-hot，等价于贪心，输出最"正经"也最啰嗦重复
          T=1  模型原本的分布
          T>1  长尾被抬起来，更容易出现意外但也可能胡说

top-k：   只保留概率最大的 k 个，其余置零后重新归一化
          k=1 就是贪心；k 小 → 稳但单调；k 大 → 多样但容易跑偏

top-p：   从大到小累加，保留累计概率刚超过 p 的最少候选
          好处是候选数自适应：模型很确定时只留 2 个，很纠结时留几十个`}
        analogy="温度像调酒师的手：手越稳（T 小），每次都倒同一款招牌酒；手越抖（T 大），越可能给你端出一杯没见过的东西。top-k 是「只在菜单前 k 道菜里选」，top-p 是「按人气从高往低拿，拿到累计人气占八成就停」——后者知道今天该拿几道菜，因为它会看分布本身有多集中。"
        detail={
          <>
            <p>
              实际生产里这三样一般<strong>不叠加猛调</strong>：常见做法是固定一个温和的 top-p（0.9 左右）
              配一个 0.7~1.0 的温度。同时开很强的 top-k 和很低的温度，等价于贪心，会迅速陷入复读。
            </p>
            <p>
              还有一个反直觉的点：<strong>温度改变的是 logits，而 top-k / top-p 改变的是概率分布的支撑集</strong>。
              所以温度会把整个分布的形状连续地拉伸，而 top-p 会直接把长尾砍断——下面的表格里，
              你能清楚看到哪些候选是被"砍掉"的（灰掉的行），哪些只是被"压低"了。
            </p>
          </>
        }
        warn="⚠️ 这里的概率分布来自一个从内置语料统计出来的 bigram 语言模型，不是 GPT。它生成的文本会很不通顺——这恰恰是好事：你能清楚看到采样参数如何独立地影响选择，而不被流畅的文本分散注意力。采样机制本身与真实大模型完全一致。"
      />

      <Card title="输入上下文">
        <textarea rows={2} value={prompt} onChange={(e) => setPrompt(e.target.value)} />
        <div className="chip-row" style={{ marginTop: 10 }}>
          <button className="btn" onClick={() => setPrompt('大语言模型通过预测下一个')}>
            中文示例
          </button>
          <button className="btn" onClick={() => setPrompt('the model predicts the next')}>
            英文示例
          </button>
          <button className="btn" onClick={() => setGenerated([])}>
            清空已生成
          </button>
        </div>
      </Card>

      <Card title="采样参数">
        <div className="controls">
          <Slider
            label="温度 T"
            value={cfg.temperature}
            min={0.1}
            max={2.5}
            step={0.05}
            onChange={(v) => patch({ temperature: v })}
            format={(v) => v.toFixed(2)}
            hint="T 越小越确定，越大越发散"
          />
          <Slider
            label="top-k"
            value={cfg.topK}
            min={0}
            max={40}
            onChange={(v) => patch({ topK: v })}
            format={(v) => (v === 0 ? '关闭' : `k=${v}`)}
            hint="0 = 不截断"
          />
          <Slider
            label="top-p"
            value={cfg.topP}
            min={0.05}
            max={1}
            step={0.05}
            onChange={(v) => patch({ topP: v })}
            format={(v) => (v >= 1 ? '关闭' : `p=${v.toFixed(2)}`)}
            hint="核采样阈值"
          />
          <Slider
            label="随机种子"
            value={seed}
            min={1}
            max={999}
            onChange={setSeed}
            format={(v) => `#${v}`}
            hint="同一种子结果可复现"
          />
        </div>
      </Card>

      <Card
        title="下一步的候选分布"
        hint={`条件来自最后 1 个 token（${source === 'bigram' ? 'bigram 命中' : '回退到 unigram'}）· 绿色行是本次抽中的`}
      >
        <table className="tbl">
          <thead>
            <tr>
              <th>候选 token</th>
              <th>原始 p</th>
              <th>温度后 p</th>
              <th>采样后 p</th>
              <th>状态</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((i) => {
              const kept = preview.kept[i]
              const chosen = i === preview.chosen
              return (
                <tr key={i} className={`${kept ? '' : 'dim'}${chosen ? ' chosen' : ''}`}>
                  <td title={NGRAM.vocab[i]}>
                    {NGRAM.vocab[i] === ' ' ? '␣' : NGRAM.vocab[i]}
                    {i === greedyIndex && <span style={{ color: 'var(--text-3)' }}> ▲贪心</span>}
                  </td>
                  <td>{(preview.base[i] * 100).toFixed(2)}%</td>
                  <td>{(preview.tempered[i] * 100).toFixed(2)}%</td>
                  <td>{kept ? (preview.final[i] * 100).toFixed(2) + '%' : '—'}</td>
                  <td style={{ fontFamily: 'inherit' }}>
                    {chosen ? '✓ 抽中' : kept ? '保留' : '被截断'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
        <div style={{ marginTop: 12 }}>
          <Stats
            items={[
              { k: '词表大小', v: String(VOCAB_SIZE) },
              { k: '保留候选数', v: String(candidateCount(preview)) },
              { k: '抽中概率', v: (preview.final[preview.chosen] * 100).toFixed(1) + '%' },
              { k: '分布熵', v: entropyOf(preview.final).toFixed(2) },
            ]}
          />
        </div>
        <div className="controls" style={{ marginTop: 12 }}>
          <button className="btn primary" onClick={stepOnce}>
            生成下一个 token
          </button>
          <button className="btn" onClick={() => setAuto((a) => !a)}>
            {auto ? '停止自动' : '自动连续生成'}
          </button>
        </div>
      </Card>

      <Card title="生成的文本" hint="每个色块是一步抽签的结果，鼠标悬停看它当时的概率">
        <div style={{ marginBottom: 10 }}>
          <div className="note" style={{ marginBottom: 6 }}>上下文</div>
          <TokenChips tokens={promptTokens.map((t, i) => ({ text: t, id: i, oov: false }))} showId={false} />
        </div>
        <div>
          <div className="note" style={{ marginBottom: 6 }}>已生成 {generated.length} 个 token</div>
          {generated.length > 0 ? (
            <div className="chip-row tight">
              {generated.map((g, i) => (
                <span
                  key={i}
                  className="token"
                  style={{ background: 'var(--accent-soft)', color: 'var(--accent)' }}
                  title={`第 ${i + 1} 步：从 ${g.candidates} 个候选中抽中，概率 ${(g.prob * 100).toFixed(1)}%（${g.source}）`}
                >
                  <span>{g.text === ' ' ? '␣' : g.text === '\n' ? '⏎' : g.text}</span>
                  <span className="tid">{(g.prob * 100).toFixed(0)}%</span>
                </span>
              ))}
            </div>
          ) : (
            <div className="note">还没有生成内容，点上面的按钮走一步。</div>
          )}
        </div>
      </Card>

      <Card title="同参数、不同种子的 5 次采样" hint="想看参数对多样性的影响，看这一栏最直观">
        <div className="controls" style={{ marginBottom: 10 }}>
          <button className="btn" onClick={() => setCompareSeed((s) => s + 1)}>
            重新采样 5 次
          </button>
          <span className="note">每次固定生成 14 个 token</span>
        </div>
        {comparisons.map((c, i) => (
          <div
            key={i}
            style={{
              fontFamily: 'var(--mono)',
              fontSize: 12.5,
              padding: '6px 10px',
              background: 'var(--surface-2)',
              borderRadius: 6,
              marginBottom: 6,
              wordBreak: 'break-all',
            }}
          >
            <span style={{ color: 'var(--text-3)', marginRight: 8 }}>#{i + 1}</span>
            {c || '（空）'}
          </div>
        ))}
        <div className="note" style={{ marginTop: 8 }}>
          把温度调到 0.1 再采样：5 条会几乎一模一样（复读机模式）。调到 2.0：5 条各不相同，但也开始胡言乱语。
          这就是真实调 API 时 temperature 的手感。
        </div>
      </Card>
    </div>
  )
}

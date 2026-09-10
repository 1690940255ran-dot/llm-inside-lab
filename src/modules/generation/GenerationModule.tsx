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
import { RealModelPanel } from '../../components/RealModelPanel'
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
import { useLang, type Lang } from '../../i18n'

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

function generateChain(history: string[], cfg: SamplingConfig, seed: number, steps: number): string {
  const h = [...history]
  const out: string[] = []
  for (let i = 0; i < steps; i++) {
    const { logits } = nextLogits(NGRAM, h)
    const step = sampleNext(logits, cfg, seed + i * 7919)
    const tk = NGRAM.vocab[step.chosen] ?? ''
    h.push(tk)
    out.push(tk)
  }
  return out.join('')
}

const zh = {
  h2: '④ 逐 token 生成：模型是怎么「抽签」的',
  lead1: '模型每一步只做两件事：',
  lead2: '算出一个词表上的概率分布，然后从中抽一个 token',
  lead3: '，再把这个 token 拼回输入，继续下一步。温度、top-k、top-p 三个旋钮，改的都是「抽签之前那一步」。',
  principleTitle: '三个参数各自在改什么？',
  d1: '实际生产里这三样一般不叠加猛调：常见做法是固定一个温和的 top-p（0.9 左右）配一个 0.7~1.0 的温度。同时开很强的 top-k 和很低的温度，等价于贪心，会迅速陷入复读。',
  d2a: '还有一个反直觉的点：',
  d2b: '温度改变的是 logits，而 top-k / top-p 改变的是概率分布的支撑集',
  d2c: '。所以温度会把整个分布的形状连续地拉伸，而 top-p 会直接把长尾砍断——下面的表格里，你能清楚看到哪些候选是被「砍掉」的（灰掉的行），哪些只是被「压低」了。',
  warn: '⚠️ 这里的概率分布来自一个从内置语料统计出来的 bigram 语言模型，不是 GPT。它生成的文本会很不通顺——这恰恰是好事：你能清楚看到采样参数如何独立地影响选择，而不被流畅的文本分散注意力。采样机制本身与真实大模型完全一致。',
  zhSample: '中文示例',
  enSample: '英文示例',
  clearGen: '清空已生成',
  params: '采样参数',
  temperature: '温度 T',
  temperatureHint: 'T 越小越确定，越大越发散',
  topk: 'top-k',
  topkHint: '0 = 不截断',
  topkOff: '关闭',
  topp: 'top-p',
  toppOff: '关闭',
  seed: '随机种子',
  seedHint: '同一种子结果可复现',
  tableTitle: '下一步的候选分布',
  bigramHit: 'bigram 命中',
  fallback: '回退到 unigram',
  cond: '条件来自最后 1 个 token（',
  cond2: '）· 绿色行是本次抽中的',
  colToken: '候选 token',
  colBase: '原始 p',
  colTemp: '温度后 p',
  colFinal: '采样后 p',
  colState: '状态',
  chosen: '✓ 抽中',
  kept: '保留',
  cut: '被截断',
  greedy: '▲贪心',
  vocabSize: '词表大小',
  keptCount: '保留候选数',
  chosenProb: '抽中概率',
  entropy: '分布熵',
  genNext: '生成下一个 token',
  autoGen: '自动连续生成',
  stopAuto: '停止自动',
  output: '生成的文本',
  outputHint: '每个色块是一步抽签的结果，鼠标悬停看它当时的概率',
  context: '上下文',
  generated: '已生成',
  generated2: '个 token',
  empty: '还没有生成内容，点上面的按钮走一步。',
  compareTitle: '同参数、不同种子的 5 次采样',
  compareHint: '想看参数对多样性的影响，看这一栏最直观',
  resample: '重新采样 5 次',
  fixed: '每次固定生成 14 个 token',
  compareNote: '把温度调到 0.1 再采样：5 条会几乎一模一样（复读机模式）。调到 2.0：5 条各不相同，但也开始胡言乱语。这就是真实调 API 时 temperature 的手感。',
  step: '步',
}

const en: typeof zh = {
  h2: '④ Token-by-token generation: how the model draws lots',
  lead1: 'Each step does exactly two things: ',
  lead2: 'compute a distribution over the vocabulary, then draw one token from it',
  lead3: ', append it to the input, and repeat. Temperature, top-k and top-p all modify the step right before the draw.',
  principleTitle: 'What does each knob change?',
  d1: 'In production you rarely crank all three: a mild top-p (~0.9) with a 0.7–1.0 temperature is the usual combo. A strong top-k plus a very low temperature is just greedy decoding in disguise, and it degenerates into repetition fast.',
  d2a: 'One counter-intuitive point: ',
  d2b: 'temperature changes the logits, while top-k / top-p change the support of the distribution',
  d2c: '. Temperature stretches the whole shape continuously; top-p chops the tail off. In the table below you can see exactly which candidates got chopped (greyed rows) and which merely got pushed down.',
  warn: '⚠️ The distribution here comes from a bigram model counted from the built-in corpus, not from GPT. The output will be far from fluent — which is the point: you can watch each knob act independently without fluent text stealing your attention. The sampling mechanics are identical to a real LLM.',
  zhSample: 'Chinese',
  enSample: 'English',
  clearGen: 'clear generated',
  params: 'Sampling parameters',
  temperature: 'temperature T',
  temperatureHint: 'lower = more deterministic, higher = more diverse',
  topk: 'top-k',
  topkHint: '0 = disabled',
  topkOff: 'off',
  topp: 'top-p',
  toppOff: 'off',
  seed: 'random seed',
  seedHint: 'same seed = reproducible output',
  tableTitle: 'Candidate distribution for the next token',
  bigramHit: 'bigram hit',
  fallback: 'fallback to unigram',
  cond: 'conditioned on the last token (',
  cond2: ') · the green row is the one that got drawn',
  colToken: 'candidate',
  colBase: 'raw p',
  colTemp: 'after T',
  colFinal: 'after filters',
  colState: 'state',
  chosen: '✓ drawn',
  kept: 'kept',
  cut: 'cut',
  greedy: '▲greedy',
  vocabSize: 'vocabulary',
  keptCount: 'candidates kept',
  chosenProb: 'drawn prob',
  entropy: 'entropy',
  genNext: 'generate next token',
  autoGen: 'generate continuously',
  stopAuto: 'stop',
  output: 'Generated text',
  outputHint: 'each chip is one draw — hover to see its probability at the time',
  context: 'context',
  generated: 'generated',
  generated2: 'tokens',
  empty: 'Nothing generated yet — press the button to take one step.',
  compareTitle: 'Five samples, same parameters, different seeds',
  compareHint: 'the clearest way to see how a knob affects diversity',
  resample: 'resample 5 times',
  fixed: '14 tokens each',
  compareNote: 'Set temperature to 0.1 and resample: all five come out nearly identical (parrot mode). Set it to 2.0: all five differ, and they start to babble. That is exactly how temperature feels when you call a real API.',
  step: 'step',
}

const DICT: Record<Lang, typeof zh> = { zh, en }

export function GenerationModule() {
  const { lang, t } = useLang()
  const c = DICT[lang]

  const [prompt, setPrompt] = useState('大语言模型通过预测下一个')
  const [cfg, setCfg] = useState<SamplingConfig>(DEFAULT_SAMPLING)
  const [generated, setGenerated] = useState<GenToken[]>([])
  const [seed, setSeed] = useState(42)
  const [auto, setAuto] = useState(false)
  const [compareSeed, setCompareSeed] = useState(1)

  const promptTokens = useMemo(() => encode(prompt, SHARED_MODEL).tokens.map((x) => x.text), [prompt])
  const history = useMemo(() => [...promptTokens, ...generated.map((g) => g.text)], [promptTokens, generated])

  const { logits, source } = useMemo(() => nextLogits(NGRAM, history), [history])

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

  const patch = (p: Partial<SamplingConfig>) => setCfg((prev) => ({ ...prev, ...p }))

  const comparisons = useMemo(
    () => Array.from({ length: 5 }, (_, i) => generateChain(promptTokens, cfg, compareSeed * 1000 + i * 131, 14)),
    [promptTokens, cfg, compareSeed],
  )

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
        formula={`logits ──(÷ T)──▶ softmax ──(top-k cut)──▶ ──(top-p nucleus)──▶ renormalise ──▶ draw

temperature T:  p_i ∝ exp(logit_i / T)
    T→0  → one-hot, same as greedy: most "proper" and most repetitive
    T=1  → the model's original distribution
    T>1  → the tail is lifted: surprising, but also more likely to be nonsense

top-k:   keep only the k largest, zero the rest, renormalise
    k=1 is greedy; small k = safe but dull; large k = diverse but drifty

top-p:   accumulate from the top and keep the fewest candidates exceeding p
    the candidate count adapts: 2 when the model is sure, dozens when it is not`}
        analogy={
          lang === 'zh'
            ? '温度像调酒师的手：手越稳（T 小），每次都倒同一款招牌酒；手越抖（T 大），越可能给你端出一杯没见过的东西。top-k 是「只在菜单前 k 道菜里选」，top-p 是「按人气从高往低拿，拿到累计人气占八成就停」——后者知道今天该拿几道菜，因为它会看分布本身有多集中。'
            : 'Temperature is the bartender’s hand: a steady hand (low T) pours the same house special every time; a shaky one (high T) may hand you something you have never seen. top-k is "only order from the first k dishes on the menu"; top-p is "take dishes by popularity until you have covered 80% of the votes" — the latter knows how many dishes to take today, because it looks at how concentrated the distribution is.'
        }
        detail={
          <>
            <p>{c.d1}</p>
            <p>
              {c.d2a}
              <strong>{c.d2b}</strong>
              {c.d2c}
            </p>
          </>
        }
        warn={c.warn}
      />

      <Card title={t('inputText')}>
        <textarea rows={2} value={prompt} onChange={(e) => setPrompt(e.target.value)} />
        <div className="chip-row" style={{ marginTop: 10 }}>
          <button className="btn" onClick={() => setPrompt('大语言模型通过预测下一个')}>
            {c.zhSample}
          </button>
          <button className="btn" onClick={() => setPrompt('the model predicts the next')}>
            {c.enSample}
          </button>
          <button className="btn" onClick={() => setGenerated([])}>
            {c.clearGen}
          </button>
        </div>
      </Card>

      <Card title={c.params}>
        <div className="controls">
          <Slider
            label={c.temperature}
            value={cfg.temperature}
            min={0.1}
            max={2.5}
            step={0.05}
            onChange={(v) => patch({ temperature: v })}
            format={(v) => v.toFixed(2)}
            hint={c.temperatureHint}
          />
          <Slider
            label={c.topk}
            value={cfg.topK}
            min={0}
            max={40}
            onChange={(v) => patch({ topK: v })}
            format={(v) => (v === 0 ? c.topkOff : `k=${v}`)}
            hint={c.topkHint}
          />
          <Slider
            label={c.topp}
            value={cfg.topP}
            min={0.05}
            max={1}
            step={0.05}
            onChange={(v) => patch({ topP: v })}
            format={(v) => (v >= 1 ? c.toppOff : `p=${v.toFixed(2)}`)}
          />
          <Slider
            label={c.seed}
            value={seed}
            min={1}
            max={999}
            onChange={setSeed}
            format={(v) => `#${v}`}
            hint={c.seedHint}
          />
        </div>
      </Card>

      <Card
        title={c.tableTitle}
        hint={`${c.cond}${source === 'bigram' ? c.bigramHit : c.fallback}${c.cond2}`}
        exportName="04-sampling-distribution"
      >
        <table className="tbl">
          <thead>
            <tr>
              <th>{c.colToken}</th>
              <th>{c.colBase}</th>
              <th>{c.colTemp}</th>
              <th>{c.colFinal}</th>
              <th>{c.colState}</th>
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
                    {i === greedyIndex && <span style={{ color: 'var(--text-3)' }}> {c.greedy}</span>}
                  </td>
                  <td>{(preview.base[i] * 100).toFixed(2)}%</td>
                  <td>{(preview.tempered[i] * 100).toFixed(2)}%</td>
                  <td>{kept ? (preview.final[i] * 100).toFixed(2) + '%' : '—'}</td>
                  <td style={{ fontFamily: 'inherit' }}>
                    {chosen ? c.chosen : kept ? c.kept : c.cut}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
        <div style={{ marginTop: 12 }}>
          <Stats
            items={[
              { k: c.vocabSize, v: String(VOCAB_SIZE) },
              { k: c.keptCount, v: String(candidateCount(preview)) },
              { k: c.chosenProb, v: (preview.final[preview.chosen] * 100).toFixed(1) + '%' },
              { k: c.entropy, v: entropyOf(preview.final).toFixed(2) },
            ]}
          />
        </div>
        <div className="controls" style={{ marginTop: 12 }}>
          <button className="btn primary" onClick={stepOnce}>
            {c.genNext}
          </button>
          <button className="btn" onClick={() => setAuto((a) => !a)}>
            {auto ? c.stopAuto : c.autoGen}
          </button>
        </div>
      </Card>

      {/*
        真实模型面板：上面那张表用的是 bigram，这里用真实权重算的 logits，
        但两边共用同一套采样参数 —— 于是可以直接对比"同样的旋钮在真假分布上的效果"。
      */}
      <RealModelPanel text={prompt + generated.map((g) => g.text).join('')} cfg={cfg} seed={seed} />

      <Card title={c.output} hint={c.outputHint}>
        <div style={{ marginBottom: 10 }}>
          <div className="note" style={{ marginBottom: 6 }}>
            {c.context}
          </div>
          <TokenChips tokens={promptTokens.map((x, i) => ({ text: x, id: i, oov: false }))} showId={false} />
        </div>
        <div>
          <div className="note" style={{ marginBottom: 6 }}>
            {c.generated} {generated.length} {c.generated2}
          </div>
          {generated.length > 0 ? (
            <div className="chip-row tight">
              {generated.map((g, i) => (
                <span
                  key={i}
                  className="token"
                  style={{ background: 'var(--accent-soft)', color: 'var(--accent)' }}
                  title={`${c.step} ${i + 1}：${g.candidates} · ${(g.prob * 100).toFixed(1)}%（${g.source}）`}
                >
                  <span>{g.text === ' ' ? '␣' : g.text === '\n' ? '⏎' : g.text}</span>
                  <span className="tid">{(g.prob * 100).toFixed(0)}%</span>
                </span>
              ))}
            </div>
          ) : (
            <div className="note">{c.empty}</div>
          )}
        </div>
      </Card>

      <Card title={c.compareTitle} hint={c.compareHint}>
        <div className="controls" style={{ marginBottom: 10 }}>
          <button className="btn" onClick={() => setCompareSeed((s) => s + 1)}>
            {c.resample}
          </button>
          <span className="note">{c.fixed}</span>
        </div>
        {comparisons.map((x, i) => (
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
            {x || '（空）'}
          </div>
        ))}
        <div className="note" style={{ marginTop: 8 }}>
          {c.compareNote}
        </div>
      </Card>
    </div>
  )
}

/**
 * 模块十：从零训一个迷你 GPT
 *
 * 前九个模块都是「看一个已经训好的模型」。这一页反过来：从随机权重开始，
 * 在浏览器里真的跑反向传播、真的用 AdamW 更新、真的把 loss 降下来。
 *
 * 页面要回答三个问题：
 *   1. 训练到底在做什么？—— 一条真的 loss 曲线 + 一段真的从乱码变像话的采样文本
 *   2. 代价是多少？—— 参数量、每 token FLOPs、花了多少秒
 *   3. 为什么这些小到可笑的模型也能讲出大模型的道理？—— 超参、初始化、warmup 的坑一个不少
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { Card, Segmented, Slider, Stats } from '../../components/Controls'
import { Heatmap, HeatLegend } from '../../components/Heatmap'
import { LineChart } from '../../components/LineChart'
import { Principle } from '../../components/Principle'
import {
  DEFAULT_MODEL,
  DEFAULT_TRAIN,
  PRESET_CORPORA,
  type ModelConfig,
  type ParamGroupKey,
  type TrainConfig,
} from '../../core/minigpt'
import type { MainToWorker, WorkerToMain } from '../../core/trainProtocol'
import { useLang } from '../../i18n'

const zh = {
  h2: '⑩ 从零训一个迷你 GPT',
  lead1:
    '前面九页都在看一个「已经训好的」模型内部长什么样。这一页换个方向：从随机权重开始，用浏览器里真的跑一遍反向传播和 AdamW，',
  lead2: '把 loss 真的降下来',
  lead3: '—— 你会看到一段真实的乱码慢慢变成像话的句子，也会看到这条路上所有的坑。',

  principleTitle: '这一页到底在算什么？',
  d1a: '全部代码就是一次前向、一次反向、一次参数更新，循环上千遍。',
  d1b: '模型是标准的 GPT 结构：词嵌入 + 位置嵌入，两层 pre-norm Transformer，输出投影和词嵌入共享同一份权重（weight tying，GPT-2 起就是这么做的）。反向传播是手写的，没有用任何自动微分框架 —— 所以页面上每一个数字都是真的算出来的。',
  d2a: '为什么学习率是 3e-2，而不是大模型常用的 3e-4？',
  d2b: '因为 Adam 对每个参数几乎都用同样的步长。大模型的权重量级约 0.02，3e-4 的步长相对权重是 1.5%，几千步才推得动；这个小模型同样只有一万八千多参数，用 3e-4 你盯着屏幕三十秒会以为它坏了。实测 3e-2 才能让「几十秒学会」成立 —— **大模型的超参配方不能直接搬给小模型。**',
  d3a: '为什么 loss 曲线是「先陡后平」？',
  d3b: '早期模型只需要学会「哪些字符常见」就能大幅降 loss —— 这一步信息量最大、最容易拿。越往后越要靠位置和上下文去区分细节，收益越来越小，曲线于是翘起来变平。默认那份中文语料重复度极高（一句话循环 60 次），曲线几乎直接俯冲到底；切到「长周期（更难）」那份，就能看到标准的「先陡后平」形状：前 200 步从 2.95 掉到 1.71，之后一千步都停在 1.4 附近。',
  warn: '这一页的模型只有一万八千多个参数（准确说 18176 + 32 × 词表大小），它的「会说话」本质上是把这份几十到几千字符的语料背了下来，不等于理解语言。另外要诚实说明：小模型训练对随机初始化相当敏感，实测 8 个种子里有 2 个会卡在坏区域（loss 停在 2.4 下不去，因为早期一次梯度爆炸把它推进了坏盆地）；页面里提供的种子都是实测能收敛的。这正是真实训练里研究初始化、warmup、梯度裁剪的原因。',

  corpusTitle: '第 1 步：给它一份语料',
  corpusHint: '字符级建模，重复度越高学得越快',
  preset: '内置语料',
  custom: '自定义',
  customPh: '在这里粘贴任何文本（中文、英文、代码都行），或者上传 .txt 文件',
  upload: '上传 .txt',
  apply: '用这份语料训练',
  corpusStats: '语料情况',
  nChars: '字符数',
  nKinds: '不同字符',
  nVocab: '词表大小',
  nDropped: '折成 UNK',
  period: '重复周期',
  periodNote:
    '「重复周期」是影响学习速度的头号因素，而且这个影响可以被量出来。固定这份中文语料（周期 12）、只改上下文窗口，实测 1200 步后的 batch loss：窗口 8（0.67 个周期）→ 2.22，窗口 12（1.00）→ 2.49，窗口 16（1.33）→ 1.29，窗口 24（2.00）→ 0.06，窗口 32（2.67）→ 0.011，窗口 48（4.00）→ 0.009。窗口装不下一个完整周期时，模型连「往前看一整个周期」都做不到，loss 几乎贴着随机基线 ln(12)=2.48 不动；要装下约 2 个周期，loss 才掉到 0.1 以下。这也正好和「⑨ 长上下文外推」接上 —— 上下文窗口不是「越大越好」，它首先是一道「能不能看到完整规律」的门槛。',
  tooShort: '语料太短了，至少要有上下文窗口的 3 倍长。',

  cfgTitle: '第 2 步：模型和数据',
  cfgHint: '改完点「重置」才会生效',
  layers: '层数',
  dModel: '隐藏维度',
  heads: '注意力头数',
  blockSize: '上下文长度',
  batch: '批大小',
  steps: '训练步数',
  lr: '学习率',
  warmup: 'warmup 步数',
  wd: '权重衰减',
  seedLabel: '初始权重',
  newSeed: '换一组初始权重',
  params: '参数量',
  flopsTok: '每 token FLOPs',
  tokensSeen: '训练见过 token 数',

  runTitle: '第 3 步：开训',
  start: '开始训练',
  pause: '暂停',
  resume: '继续',
  reset: '重置',
  stepLabel: '当前步',
  elapsed: '已用',
  eta: '预计剩余',
  msPerStep: '每步耗时',
  computing: '正在训练…',

  lossTitle: '验证 loss 曲线（这就是训练在发生）',
  lossHint: '每 20 步在固定验证集上评估一次',
  baseline: '随机猜测基线 log(词表大小)',
  lossNow: '当前 loss',
  lossInit: '初始 loss',
  lossBest: '最优 loss',
  ppl: '困惑度（= e^loss）',
  lossNote:
    '虚线是「什么都不学、均匀乱猜」的基线，就是 ln(词表大小)。曲线必须掉到这根线下面，才说明模型真的学到了东西。换个角度看更直观：默认语料下词表只有 13 个字符，乱猜时正确答案的概率是 1/13 ≈ 7.7%；曲线降到 0.014 时，这个概率是 e^−0.014 ≈ 98.6% —— 它几乎每次都猜对了。另外注意这条曲线**不是单调下降**的（默认语料在 300 步附近会反弹到 0.57），这是真实训练的常态：验证集只有 12 个固定 batch，warmup 刚结束时学习率又最大。',

  sampleTitle: '它到底学会了什么（最直观的证据）',
  sampleHint: '每一步都从同一个开头续写，温度为 0.4',
  sampleStep: '第 {n} 步',
  sampleEmpty: '还没有采样结果',
  sampleNote:
    '这是同一个模型在同一个开头下、不同训练阶段的续写。最开始是纯乱码 —— 它连「哪些字符存在」都不知道；几十步之后开始出现空格和常见字母；到最后能把它背下来的句子原样吐出来。中间没有任何一行代码在装样子。',

  attnTitle: '注意力从噪声长出结构',
  attnHint: '同一串输入、同一个头，训练前后对比',
  attnBefore: '训练前（第 0 步）',
  attnAfter: '当前',
  attnHead: '头',
  attnNote:
    '左边是随机初始化时的注意力：每一行几乎完全均匀（归一化熵 1.0000，最大权重 0.048 ≈ 1/21），也就是「谁也不特别看」。右边是训练后：每个头都塌成近似 one-hot（归一化熵 0.0000，最大权重 1.0000），一行只盯住一个位置。我们只给了它「预测下一个字符」这一个任务，没有任何一行代码告诉它该看哪里。要诚实补一句：落点并不是「永远回看一个周期」这种通用算法，而是「第 t 行盯第 p(t) 个位置」的一张查找表（p(t) 本身以语料周期重复）—— 它把答案背下来了，而不是学会了复制。',
  attnNone: '开始训练后这里会显示注意力快照',
  prevTok: '各头「盯住前一个位置」的平均权重',

  paramTitle: '这笔账有多大',
  paramHint: '参数量明细 + 和真模型对比',
  paramGroups: {
    tokEmb: '词嵌入（= 输出投影，共享）',
    posEmb: '位置嵌入',
    attn: '注意力（QKVO）',
    ffn: '前馈（W1/W2）',
    ln: 'LayerNorm',
  } satisfies Record<ParamGroupKey, string>,
  paramNote:
    '词嵌入和输出投影共享同一份权重，所以它只出现在明细里一次 —— 这是 GPT-2 以来省参数的标准做法。下面把我们的一万八千多参数和真模型放在同一个尺子上：GPT-2 124M 是它的 6 670 倍，GPT-3 175B 是它的 940 万倍。而这还只是**参数量**；预训练总算力约等于 `6 × 参数量 × 训练 token 数`，真模型的 token 数本身就是千亿量级，所以算力差距比参数差距还要再大好几个数量级。',

  orderNote: '数量级对比',
  ordOurs: '本页模型',
  ordGpt2: 'GPT-2 124M',
  ordGpt3: 'GPT-3 175B',
  times: '倍',
}

const en: typeof zh = {
  h2: '⑩ Train a mini GPT from scratch',
  lead1:
    'The previous nine pages all look inside a model that is already trained. This one goes the other way: start from random weights and actually run backpropagation and AdamW in your browser to ',
  lead2: 'really bring the loss down',
  lead3: ' — you will watch gibberish turn into real sentences, and hit every pitfall along the way.',

  principleTitle: 'What is this page actually computing?',
  d1a: 'The whole thing is one forward pass, one backward pass and one parameter update, repeated a thousand times.',
  d1b: 'The model is a standard GPT: token plus positional embeddings, two pre-norm Transformer blocks, and an output projection that shares weights with the token embedding (weight tying, as in GPT-2 onwards). The backward pass is hand-written with no autodiff framework — so every number on this page is genuinely computed.',
  d2a: 'Why is the learning rate 3e-2 and not the 3e-4 used for large models?',
  d2b: 'Because Adam takes almost the same step size for every parameter. Large model weights are around 0.02, so a 3e-4 step is 1.5% of a weight and takes thousands of steps to move anything; this model has only ~18.6k parameters, and at 3e-4 you would stare at the screen for thirty seconds convinced it was broken. 3e-2 is what makes "learns in tens of seconds" true — **recipes from large models do not transfer to small ones.**',
  d3a: 'Why does the loss curve drop steeply then flatten?',
  d3b: 'Early on, simply learning which characters are common buys a large drop — that is the cheapest, biggest chunk of information. After that, gains must come from position and context, so each step buys less and the curve bends over and flattens. The default Chinese corpus is so repetitive (one sentence repeated 60 times) that the curve dives almost straight down; switch to "longer period (harder)" to see the classic shape: 2.95 → 1.71 in the first 200 steps, then stuck near 1.4 for the next thousand.',
  warn: 'The model here has only ~18.6k parameters (exactly 18176 + 32 × vocab size); its "talking" is really memorising this corpus of a few dozen to a few thousand characters, which is not the same as understanding language. One more honest caveat: small models are quite sensitive to random initialisation — measured across 8 seeds, 2 got stuck in a bad region (loss frozen near 2.4 because an early gradient explosion pushed it into a bad basin). The seeds offered here are ones verified to converge. This is exactly why real training work studies initialisation, warmup and gradient clipping.',

  corpusTitle: 'Step 1: give it a corpus',
  corpusHint: 'character-level modelling — the more repetitive, the faster it learns',
  preset: 'Built-in corpora',
  custom: 'Custom',
  customPh: 'Paste any text here (Chinese, English, code — anything), or upload a .txt file',
  upload: 'Upload .txt',
  apply: 'Train on this corpus',
  corpusStats: 'Corpus stats',
  nChars: 'characters',
  nKinds: 'distinct chars',
  nVocab: 'vocab size',
  nDropped: 'mapped to UNK',
  period: 'repeat period',
  periodNote:
    'The repeat period is the single biggest factor in how fast it learns, and the effect can be measured. Fixing this Chinese corpus (period 12) and changing only the context window, the batch loss after 1200 steps is: window 8 (0.67 periods) → 2.22, 12 (1.00) → 2.49, 16 (1.33) → 1.29, 24 (2.00) → 0.06, 32 (2.67) → 0.011, 48 (4.00) → 0.009. When the window cannot hold one full period the model cannot even "look back exactly one period", and loss sits at the random-guess baseline ln(12)=2.48; it needs roughly 2 periods before loss falls below 0.1. This connects directly to module ⑨: a longer context window is not simply "better" — first of all it is the threshold for whether the model can see a complete pattern.',
  tooShort: 'Corpus too short — it needs at least 3× the context window.',

  cfgTitle: 'Step 2: model and data',
  cfgHint: 'changes take effect after you hit reset',
  layers: 'layers',
  dModel: 'hidden size',
  heads: 'heads',
  blockSize: 'context length',
  batch: 'batch size',
  steps: 'training steps',
  lr: 'learning rate',
  warmup: 'warmup steps',
  wd: 'weight decay',
  seedLabel: 'initial weights',
  newSeed: 'resample initial weights',
  params: 'parameters',
  flopsTok: 'FLOPs / token',
  tokensSeen: 'tokens seen',

  runTitle: 'Step 3: run it',
  start: 'Start training',
  pause: 'Pause',
  resume: 'Resume',
  reset: 'Reset',
  stepLabel: 'step',
  elapsed: 'elapsed',
  eta: 'ETA',
  msPerStep: 'ms / step',
  computing: 'training…',

  lossTitle: 'Validation loss curve (this is training happening)',
  lossHint: 'evaluated on a fixed validation set every 20 steps',
  baseline: 'random-guess baseline log(vocab size)',
  lossNow: 'current loss',
  lossInit: 'initial loss',
  lossBest: 'best loss',
  ppl: 'perplexity (= e^loss)',
  lossNote:
    'The dashed line is the "learn nothing, guess uniformly" baseline: ln(vocab size). The curve has to fall below it before the model has learned anything. Another way to read it: on the default corpus the vocabulary is just 13 characters, so uniform guessing puts 1/13 ≈ 7.7% on the right answer; at a loss of 0.014 that probability is e^−0.014 ≈ 98.6% — it now gets it right almost every time. Note also that the curve is **not monotonic** (the default corpus bounces back to 0.57 around step 300). That is normal for real training: the validation set is only 12 fixed batches, and the learning rate is at its largest just after warmup.',

  sampleTitle: 'What it actually learned (the most direct evidence)',
  sampleHint: 'same prompt at every checkpoint, temperature 0.4',
  sampleStep: 'step {n}',
  sampleEmpty: 'no samples yet',
  sampleNote:
    'This is the same model continuing from the same prompt at different points in training. At first it is pure noise — it does not even know which characters exist. After a few dozen steps spaces and common letters appear; by the end it reproduces the memorised sentence verbatim. Not one line of code is faking it.',

  attnTitle: 'Attention grows structure out of noise',
  attnHint: 'same input, same head, before vs after',
  attnBefore: 'before training (step 0)',
  attnAfter: 'current',
  attnHead: 'head',
  attnNote:
    'On the left is attention at random initialisation: every row is almost perfectly uniform (normalised entropy 1.0000, max weight 0.048 ≈ 1/21), meaning "nothing in particular is being looked at". On the right, after training, every head collapses to near one-hot (normalised entropy 0.0000, max weight 1.0000) — each row stares at exactly one position. We only ever gave it next-character prediction, and no line of code told it where to look. One honest addition: the target it settles on is not a general "always look back one period" algorithm but a lookup table "row t stares at position p(t)", with p(t) itself repeating at the corpus period — it memorised the answer rather than learning to copy.',
  attnNone: 'attention snapshots appear here once training starts',
  prevTok: 'average weight each head puts on the previous position',

  paramTitle: 'How big is this bill',
  paramHint: 'parameter breakdown + comparison with real models',
  paramGroups: {
    tokEmb: 'Token embedding (= output proj, tied)',
    posEmb: 'Positional embedding',
    attn: 'Attention (QKVO)',
    ffn: 'Feed-forward (W1/W2)',
    ln: 'LayerNorm',
  } satisfies Record<ParamGroupKey, string>,
  paramNote:
    'The token embedding and the output projection share one weight matrix, so it appears only once in the breakdown — the standard way to save parameters since GPT-2. Below, our ~18.6k parameters sit on the same ruler as real models: GPT-2 124M is 6,670× this mini model, and GPT-3 175B is 9.4 million×. And that is only **parameter count**; pretraining compute is roughly `6 × parameters × training tokens`, and real models see hundreds of billions of tokens — so the compute gap is orders of magnitude wider still.',

  orderNote: 'orders of magnitude',
  ordOurs: 'this page',
  ordGpt2: 'GPT-2 124M',
  ordGpt3: 'GPT-3 175B',
  times: '×',
}

/** 实测能收敛的种子。小模型对初始化敏感，这里只放验证过的那几个 */
const SEEDS = [123, 1, 7, 42, 777, 99991]

const HEAD_COLORS = ['var(--accent)', 'var(--ok, #3aa675)', 'var(--warn, #c98a1b)', 'var(--text-2)']

function fmtInt(n: number): string {
  return n.toLocaleString('en-US')
}
function fmtBig(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`
  return String(n)
}

interface ReadyInfo {
  vocabSize: number
  vocabChars: string[]
  droppedKinds: number
  corpusChars: number
  paramCount: number
  flopsPerToken: number
  paramBreakdown: { key: ParamGroupKey; size: number }[]
  initialLoss: number
}

interface Progress {
  step: number
  totalSteps: number
  batchLoss: number
  gradNorm: number
  lr: number
  elapsedMs: number
  trace: { step: number; loss: number }[]
  samples: { step: number; text: string }[]
  attnStep: number
  attnVersion: number
  attn?: number[][][]
  prevToken: number[]
  done: boolean
}

export function TrainModule() {
  const { lang } = useLang()
  const d = lang === 'zh' ? zh : en

  const [corpusId, setCorpusId] = useState(PRESET_CORPORA[0].id)
  const [corpus, setCorpus] = useState(PRESET_CORPORA[0].text)
  const [model, setModel] = useState<ModelConfig>({ ...DEFAULT_MODEL })
  const [train, setTrain] = useState<TrainConfig>({ ...DEFAULT_TRAIN })

  const [ready, setReady] = useState<ReadyInfo | null>(null)
  const [prog, setProg] = useState<Progress | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')
  const [head, setHead] = useState(0)

  const workerRef = useRef<Worker | null>(null)
  const initialAttnRef = useRef<number[][][] | null>(null)
  /** 最近一次收到的配置，重置时用 */
  const cfgRef = useRef({ model, train, corpus })
  cfgRef.current = { model, train, corpus }

  useEffect(() => {
    const w = new Worker(new URL('../../workers/trainWorker.ts', import.meta.url), { type: 'module' })
    w.onmessage = (ev: MessageEvent<WorkerToMain>) => {
      const m = ev.data
      if (m.type === 'ready') {
        setReady(m)
        setError('')
        initialAttnRef.current = null
      } else if (m.type === 'progress') {
        setProg((prev) => {
          const next: Progress = { ...m, attn: m.attn ?? prev?.attn }
          return next
        })
        if (m.attn && m.attnStep === 0 && !initialAttnRef.current) initialAttnRef.current = m.attn
        setRunning(!m.done && progRunningRef.current)
        if (m.done) setRunning(false)
      } else if (m.type === 'error') {
        setError(m.message)
        setRunning(false)
      }
    }
    workerRef.current = w
    const send = (msg: MainToWorker) => w.postMessage(msg)
    send({ type: 'init', corpus, model, train, maxVocab: 96 })
    return () => w.terminate()
    // 只在挂载时初始化一次；之后的重新初始化都由按钮显式触发
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  /** 用当前的语料 + 配置重新初始化（重建词表与模型） */
  const reinit = (nextModel = model, nextTrain = train, nextCorpus = corpus) => {
    setRunning(false)
    setProg(null)
    setReady(null)
    initialAttnRef.current = null
    progRunningRef.current = false
    workerRef.current?.postMessage({
      type: 'init',
      corpus: nextCorpus,
      model: nextModel,
      train: nextTrain,
      maxVocab: 96,
    } satisfies MainToWorker)
  }

  const progRunningRef = useRef(false)

  const start = () => {
    progRunningRef.current = true
    setRunning(true)
    workerRef.current?.postMessage({ type: 'start' } satisfies MainToWorker)
  }
  const stop = () => {
    progRunningRef.current = false
    setRunning(false)
    workerRef.current?.postMessage({ type: 'stop' } satisfies MainToWorker)
  }
  const reset = () => {
    progRunningRef.current = false
    setRunning(false)
    workerRef.current?.postMessage({ type: 'reset' } satisfies MainToWorker)
  }

  const applyCorpus = () => reinit(model, train, corpus)
  const applyConfig = () => reinit(model, train, corpus)
  const newSeed = () => {
    const i = SEEDS.indexOf(train.seed)
    const next = { ...train, seed: SEEDS[(i + 1 + SEEDS.length) % SEEDS.length] }
    setTrain(next)
    reinit(model, next, corpus)
  }

  const upload = (file: File) => {
    const reader = new FileReader()
    reader.onload = () => {
      const text = String(reader.result ?? '')
      setCorpus(text)
      setCorpusId('custom')
      reinit(model, train, text)
    }
    reader.readAsText(file, 'utf-8')
  }

  /* ---------------- 派生数据 ---------------- */

  const corpusStats = useMemo(() => {
    const set = new Set(corpus)
    return { chars: corpus.length, kinds: set.size, period: PRESET_CORPORA.find((p) => p.id === corpusId)?.period ?? 0 }
  }, [corpus, corpusId])

  const baseline = ready ? Math.log(ready.vocabSize) : 0
  const latest = prog?.trace[prog.trace.length - 1]?.loss ?? 0
  const bestLoss = prog ? Math.min(...prog.trace.map((t) => t.loss)) : 0
  const tokensSeen = prog ? prog.step * train.batchSize * model.blockSize : 0
  const msPerStep = prog && prog.step > 0 ? prog.elapsedMs / prog.step : 0
  const eta = prog && msPerStep > 0 ? Math.max(0, (prog.totalSteps - prog.step) * msPerStep) : 0

  const lossLines = useMemo(() => {
    const lines = [
      {
        name: d.baseline,
        color: 'var(--text-3)',
        dashed: true,
        points: [
          { x: 0, y: baseline },
          { x: train.steps, y: baseline },
        ],
      },
    ]
    if (prog && prog.trace.length > 1) {
      lines.push({
        name: 'validation loss',
        color: 'var(--accent)',
        dashed: false,
        points: prog.trace.map((t) => ({ x: t.step, y: t.loss })),
      })
    }
    return lines
  }, [prog, baseline, train.steps, d.baseline])

  const attn = prog?.attn ?? null
  const attnLabels = ready ? ready.vocabChars.slice(0, model.blockSize) : []

  return (
    <div>
      <div className="module-head">
        <h2>{d.h2}</h2>
        <div className="lead">
          {d.lead1}
          <strong>{d.lead2}</strong>
          {d.lead3}
        </div>
      </div>

      <Principle
        title={d.principleTitle}
        formula={undefined}
        analogy={d.d1a}
        detail={
          <>
            <p>{d.d1b}</p>
            <p style={{ marginTop: 8 }}>
              <b>{d.d2a}</b> {d.d2b}
            </p>
            <p style={{ marginTop: 8 }}>
              <b>{d.d3a}</b> {d.d3b}
            </p>
          </>
        }
        warn={d.warn}
      />

      {error && <div className="warn" style={{ marginBottom: 12 }}>{error === 'corpus-too-short' ? d.tooShort : error}</div>}

      {/* ---------------- 语料 ---------------- */}
      <Card title={d.corpusTitle} hint={d.corpusHint}>
        <div className="chip-row" style={{ marginBottom: 8 }}>
          {PRESET_CORPORA.map((p) => (
            <button
              key={p.id}
              className={`chip${corpusId === p.id ? ' active' : ''}`}
              onClick={() => {
                setCorpusId(p.id)
                setCorpus(p.text)
                reinit(model, train, p.text)
              }}
            >
              {lang === 'zh' ? p.label : p.labelEn}
            </button>
          ))}
          <button
            className={`chip${corpusId === 'custom' ? ' active' : ''}`}
            onClick={() => setCorpusId('custom')}
          >
            {d.custom}
          </button>
          <label className="btn" style={{ cursor: 'pointer' }}>
            {d.upload}
            <input
              type="file"
              accept=".txt,text/plain"
              style={{ display: 'none' }}
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) upload(f)
                e.target.value = ''
              }}
            />
          </label>
        </div>
        <textarea
          rows={5}
          value={corpus}
          placeholder={d.customPh}
          onChange={(e) => {
            setCorpus(e.target.value)
            setCorpusId('custom')
          }}
        />
        <div style={{ display: 'flex', gap: 8, marginTop: 8, flexWrap: 'wrap', alignItems: 'center' }}>
          <button className="btn primary" onClick={applyCorpus}>
            {d.apply}
          </button>
          {corpusStats.period > 0 && (
            <span className="hint">
              {d.period}: {corpusStats.period}
            </span>
          )}
        </div>
        <Stats
          items={[
            { k: d.nChars, v: fmtInt(corpusStats.chars) },
            { k: d.nKinds, v: fmtInt(corpusStats.kinds) },
            { k: d.nVocab, v: ready ? String(ready.vocabSize) : '–' },
            { k: d.nDropped, v: ready ? String(ready.droppedKinds) : '–' },
          ]}
        />
        {corpusStats.period > 0 && <div className="hint" style={{ marginTop: 6 }}>{d.periodNote}</div>}
      </Card>

      {/* ---------------- 配置 ---------------- */}
      <Card title={d.cfgTitle} hint={d.cfgHint}>
        <div className="controls">
          <Slider label={d.layers} value={model.nLayers} min={1} max={4} onChange={(v) => setModel({ ...model, nLayers: v })} />
          <Segmented
            value={model.dModel}
            options={[
              { value: 16, label: '16' },
              { value: 24, label: '24' },
              { value: 32, label: '32' },
              { value: 48, label: '48' },
            ]}
            onChange={(v) => setModel({ ...model, dModel: v })}
          />
          <Segmented
            value={model.nHeads}
            options={[
              { value: 1, label: '1' },
              { value: 2, label: '2' },
              { value: 4, label: '4' },
              { value: 8, label: '8' },
            ]}
            onChange={(v) => setModel({ ...model, nHeads: v })}
          />
          <Slider label={d.blockSize} value={model.blockSize} min={8} max={64} step={4} onChange={(v) => setModel({ ...model, blockSize: v })} />
          <Slider label={d.batch} value={train.batchSize} min={2} max={32} step={2} onChange={(v) => setTrain({ ...train, batchSize: v })} />
          <Slider label={d.steps} value={train.steps} min={200} max={3000} step={100} onChange={(v) => setTrain({ ...train, steps: v, warmup: Math.round(v * 0.1) })} />
          <Slider
            label={d.lr}
            value={train.lr}
            min={0.001}
            max={0.1}
            step={0.001}
            onChange={(v) => setTrain({ ...train, lr: v })}
            format={(v) => v.toFixed(3)}
          />
          <Slider label={d.warmup} value={train.warmup} min={0} max={Math.max(50, Math.round(train.steps / 2))} step={10} onChange={(v) => setTrain({ ...train, warmup: v })} />
          <Slider label={d.wd} value={train.weightDecay} min={0} max={0.1} step={0.005} onChange={(v) => setTrain({ ...train, weightDecay: v })} format={(v) => v.toFixed(3)} />
        </div>
        <div style={{ display: 'flex', gap: 8, marginTop: 10, flexWrap: 'wrap', alignItems: 'center' }}>
          <button className="btn primary" onClick={applyConfig}>
            {d.reset}
          </button>
          <button className="btn" onClick={newSeed}>
            {d.newSeed}
          </button>
          <span className="hint">
            {d.seedLabel}: {train.seed}
          </span>
        </div>
        <Stats
          items={[
            { k: d.params, v: ready ? fmtInt(ready.paramCount) : '–' },
            { k: d.flopsTok, v: ready ? fmtBig(ready.flopsPerToken) : '–' },
            { k: d.tokensSeen, v: fmtBig(tokensSeen) },
          ]}
        />
      </Card>

      {/* ---------------- 运行 ---------------- */}
      <Card title={d.runTitle}>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
          {running ? (
            <button className="btn primary" onClick={stop}>
              {d.pause}
            </button>
          ) : (
            <button className="btn primary" onClick={start} disabled={!ready}>
              {prog && prog.step > 0 ? d.resume : d.start}
            </button>
          )}
          <button className="btn" onClick={reset} disabled={!ready}>
            {d.reset}
          </button>
          {running && <span className="hint">{d.computing}</span>}
        </div>
        <Stats
          items={[
            { k: d.stepLabel, v: prog ? `${prog.step} / ${prog.totalSteps}` : '–' },
            { k: d.elapsed, v: prog ? `${(prog.elapsedMs / 1000).toFixed(1)}s` : '–' },
            { k: d.eta, v: prog && running ? `${(eta / 1000).toFixed(0)}s` : '–' },
            { k: d.msPerStep, v: msPerStep ? msPerStep.toFixed(1) : '–' },
          ]}
        />
      </Card>

      {/* ---------------- loss 曲线 ---------------- */}
      <Card title={d.lossTitle} hint={d.lossHint} exportName="10-training-loss">
        <Stats
          items={[
            { k: d.lossNow, v: prog ? latest.toFixed(4) : '–' },
            { k: d.lossInit, v: prog ? prog.trace[0].loss.toFixed(4) : '–' },
            { k: d.lossBest, v: prog ? bestLoss.toFixed(4) : '–' },
            { k: d.ppl, v: prog ? Math.exp(latest).toFixed(2) : '–' },
          ]}
        />
        <LineChart
          lines={lossLines}
          xLabel={d.stepLabel}
          yLabel="loss"
          height={280}
          xFormat={(v) => v.toFixed(0)}
          yFormat={(v) => v.toFixed(1)}
        />
        <p className="note">{d.lossNote}</p>
      </Card>

      {/* ---------------- 采样 ---------------- */}
      <Card title={d.sampleTitle} hint={d.sampleHint} exportName="10-training-samples">
        {prog && prog.samples.length > 0 ? (
          <div className="sample-list">
            {prog.samples.map((s) => (
              <div className="sample-row" key={s.step}>
                <div className="k">{d.sampleStep.replace('{n}', String(s.step))}</div>
                <div className="v">{s.text}</div>
              </div>
            ))}
          </div>
        ) : (
          <div className="hint">{d.sampleEmpty}</div>
        )}
        <p className="note">{d.sampleNote}</p>
      </Card>

      {/* ---------------- 注意力 ---------------- */}
      <Card title={d.attnTitle} hint={d.attnHint} exportName="10-training-attention">
        {attn && initialAttnRef.current ? (
          <>
            <div className="chip-row" style={{ marginBottom: 8 }}>
              {Array.from({ length: model.nHeads }, (_, h) => (
                <button key={h} className={`chip${head === h ? ' active' : ''}`} onClick={() => setHead(h)}>
                  {d.attnHead} {h}
                </button>
              ))}
            </div>
            <div className="two-col">
              <div>
                <div className="hint" style={{ marginBottom: 4 }}>
                  {d.attnBefore}
                </div>
                <Heatmap
                  matrix={initialAttnRef.current[head] ?? []}
                  rowLabels={attnLabels}
                  isMasked={(i, j) => j > i}
                />
              </div>
              <div>
                <div className="hint" style={{ marginBottom: 4 }}>
                  {d.attnAfter}（step {prog?.attnStep ?? 0}）
                </div>
                <Heatmap matrix={attn[head] ?? []} rowLabels={attnLabels} isMasked={(i, j) => j > i} />
              </div>
            </div>
            <HeatLegend />
            <div style={{ marginTop: 12 }}>
              <div className="hint" style={{ marginBottom: 6 }}>
                {d.prevTok}
              </div>
              <div className="bars">
                {prog?.prevToken.map((v, h) => (
                  <div className="bar-row" key={h}>
                    <div className="label">
                      {d.attnHead} {h}
                    </div>
                    <div className="bar-track">
                      <div className="bar-fill" style={{ width: `${Math.min(100, v * 100)}%`, background: HEAD_COLORS[h % HEAD_COLORS.length] }} />
                    </div>
                    <div className="val">{v.toFixed(3)}</div>
                  </div>
                ))}
              </div>
            </div>
          </>
        ) : (
          <div className="hint">{d.attnNone}</div>
        )}
        <p className="note">{d.attnNote}</p>
      </Card>

      {/* ---------------- 参数账 ---------------- */}
      <Card title={d.paramTitle} hint={d.paramHint} exportName="10-training-params">
        {ready && (
          <>
            <div className="bars">
              {ready.paramBreakdown.map((g) => (
                <div className="bar-row" key={g.key}>
                  <div className="label" title={d.paramGroups[g.key] ?? g.key}>
                    {d.paramGroups[g.key] ?? g.key}
                  </div>
                  <div className="bar-track">
                    <div className="bar-fill" style={{ width: `${(g.size / ready.paramBreakdown[0].size) * 100}%`, opacity: 0.35 + 0.65 * (g.size / ready.paramBreakdown[0].size) }} />
                  </div>
                  <div className="val">{fmtInt(g.size)}</div>
                </div>
              ))}
            </div>
            <div style={{ marginTop: 14 }}>
              <div className="hint" style={{ marginBottom: 6 }}>
                {d.orderNote}
              </div>
              <div className="bars">
                {[
                  { label: d.ordOurs, value: ready.paramCount },
                  { label: d.ordGpt2, value: 124e6 },
                  { label: d.ordGpt3, value: 175e9 },
                ].map((r) => (
                  <div className="bar-row" key={r.label}>
                    <div className="label">{r.label}</div>
                    <div className="bar-track">
                      <div className="bar-fill" style={{ width: `${Math.max(0.15, (Math.log10(r.value) / Math.log10(175e9)) * 100)}%` }} />
                    </div>
                    <div className="val">
                      {fmtBig(r.value)}
                      {r.value > ready.paramCount && ` (${fmtBig(r.value / ready.paramCount)}${d.times})`}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
        <p className="note">{d.paramNote}</p>
      </Card>
    </div>
  )
}

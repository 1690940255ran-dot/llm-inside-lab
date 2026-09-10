/**
 * 训练 Worker：真正跑训练循环的地方
 *
 * 为什么必须开线程？一次默认训练是 1200 步、十几秒的纯计算。
 * 放主线程会让页面完全失去响应。放到 Worker 里之后，主线程只需要收消息、画曲线。
 *
 * 为什么还要按时间分片？Worker 的事件队列要被处理，必须「让出执行权」。
 * 如果一口气跑完 1200 步，期间收到的 stop / reset 只能等训练结束才生效 ——
 * 表现出来就是「点了停止没反应」。所以每跑满 40ms 就交还一次控制权。
 */
/// <reference lib="webworker" />
import {
  MiniGPT,
  buildVocab,
  encodeCorpus,
  lrAt,
  sampleBatch,
  vocabChar,
  type CharVocab,
  type ModelConfig,
  type TrainConfig,
} from '../core/minigpt'
import { mulberry32 } from '../core/random'
import type { MainToWorker, WorkerToMain } from '../core/trainProtocol'

/** 每跑满这么多毫秒就交还一次控制权 */
const SLICE_MS = 40
/** 验证集 batch 数：曲线靠它，比单个训练 batch 的 loss 稳定得多 */
const VAL_BATCHES = 12
/** 曲线与注意力快照的评估间隔（步） */
const EVAL_EVERY = 20
/** 采样生成的字符数 */
const SAMPLE_LEN = 48
/** 语料字符数上限：再长也背不完，曲线会平得没有信息量 */
const MAX_CORPUS = 20000

let model: MiniGPT | null = null
let data: Int32Array | null = null
let vocab: CharVocab | null = null
let mcfg: ModelConfig | null = null
let tcfg: TrainConfig | null = null
let valBatches: { inputs: Int32Array; targets: Int32Array }[] = []
/** 数据采样用的随机流。持有成一个模块级状态，"暂停再继续"才不会重复同一批数据 */
let dataRnd: () => number = () => 0

let step = 0
let stopped = false
let running = false
let startedAt = 0
let elapsedMs = 0
let lastBatchLoss = 0
let lastGradNorm = 0

let trace: { step: number; loss: number }[] = []
let samples: { step: number; text: string }[] = []
let attnStep = -1
let attnVersion = 0
let attnData: number[][][] | null = null
let prevToken: number[] = []
let attnFresh = false
let attnEvery = 1

const post = (msg: WorkerToMain) => (self as unknown as Worker).postMessage(msg)

function decode(ids: number[]): string {
  if (!vocab) return ''
  return ids.map((id) => vocabChar(vocab as CharVocab, id)).join('')
}

/** 在固定验证集上评估：曲线上的点，比随手抓一个 batch 的 loss 稳定得多 */
function evaluate(): number {
  if (!model) return 0
  let s = 0
  for (const b of valBatches) s += model.loss(b.inputs, b.targets)
  return s / valBatches.length
}

function captureSample(): void {
  if (!model || !data) return
  const ids = model.sample([data[0]], SAMPLE_LEN, 0.4, 1234, 0)
  samples.push({ step, text: decode(ids) })
}

function captureAttn(): void {
  if (!model || !data || !mcfg) return
  const ids = Array.from(data.slice(0, mcfg.blockSize))
  attnData = model.attnSnapshot(ids, 0)
  prevToken = model.prevTokenScore(0)
  attnStep = step
  attnVersion++
  attnFresh = true
}

function buildModel(seed: number): void {
  if (!vocab || !mcfg || !tcfg) return
  model = new MiniGPT({
    vocabSize: vocab.itos.length,
    model: mcfg,
    batchSize: tcfg.batchSize,
    seed,
  })
}

function resetState(): void {
  if (!tcfg) return
  step = 0
  stopped = false
  running = false
  elapsedMs = 0
  lastBatchLoss = 0
  lastGradNorm = 0
  trace = []
  samples = []
  attnStep = -1
  attnVersion = 0
  attnData = null
  prevToken = []
  attnFresh = false
  attnEvery = Math.max(1, Math.floor(tcfg.steps / 12))
  dataRnd = mulberry32(tcfg.seed)
}

function init(msg: Extract<MainToWorker, { type: 'init' }>): void {
  try {
    mcfg = msg.model
    tcfg = msg.train

    const corpus = msg.corpus.slice(0, MAX_CORPUS)
    if (corpus.length < mcfg.blockSize * 3) {
      post({ type: 'error', message: 'corpus-too-short' })
      return
    }
    vocab = buildVocab(corpus, msg.maxVocab)
    data = encodeCorpus(corpus, vocab)

    resetState()
    buildModel(tcfg.seed)

    // 固定的验证集，全程不变 —— 否则曲线会被「换了评估数据」污染
    const vrnd = mulberry32(999)
    valBatches = Array.from({ length: VAL_BATCHES }, () =>
      sampleBatch(data as Int32Array, tcfg!.batchSize, mcfg!.blockSize, vrnd),
    )

    const initialLoss = evaluate()
    trace.push({ step: 0, loss: initialLoss })
    captureSample()
    captureAttn()

    post({
      type: 'ready',
      vocabSize: vocab.itos.length,
      vocabChars: vocab.itos.map((_, i) => vocabChar(vocab as CharVocab, i)),
      droppedKinds: vocab.droppedKinds,
      corpusChars: corpus.length,
      paramCount: (model as MiniGPT).paramCount(),
      flopsPerToken: (model as MiniGPT).flopsPerToken(),
      paramBreakdown: (model as MiniGPT).paramBreakdown(),
      initialLoss,
      textPerStepMs: 0,
    })
    postProgress(false)
  } catch (e) {
    post({ type: 'error', message: e instanceof Error ? e.message : String(e) })
  }
}

function postProgress(done: boolean): void {
  if (!tcfg) return
  post({
    type: 'progress',
    step,
    totalSteps: tcfg.steps,
    batchLoss: lastBatchLoss,
    gradNorm: lastGradNorm,
    lr: lrAt(Math.max(0, step - 1), tcfg),
    elapsedMs,
    trace: trace.slice(),
    samples: samples.slice(),
    attnStep,
    attnVersion,
    attn: attnFresh ? attnData ?? undefined : undefined,
    prevToken,
    done,
  })
  attnFresh = false
}

/** 时间分片的主循环：跑满 SLICE_MS 就交还控制权 */
function runSlice(): void {
  if (!model || !data || !tcfg || !mcfg || !running) return
  const t0 = Date.now()

  while (Date.now() - t0 < SLICE_MS) {
    if (step >= tcfg.steps) break
    const { inputs, targets } = sampleBatch(data, tcfg.batchSize, mcfg.blockSize, dataRnd)
    const r = model.trainStep(inputs, targets, lrAt(step, tcfg), tcfg.weightDecay, tcfg.gradClip)
    lastBatchLoss = r.loss
    lastGradNorm = r.gradNorm
    step++
    if (step % EVAL_EVERY === 0) trace.push({ step, loss: evaluate() })
    if (step % attnEvery === 0) captureAttn()
  }
  elapsedMs = Date.now() - startedAt

  const done = step >= tcfg.steps
  if (done) captureSample()
  postProgress(done)

  if (done || stopped) {
    // 停下来时把注意力快照再抓一次，让用户看到"停在这一步"的样子
    if (!done) captureAttn()
    running = false
    stopped = false
    return
  }
  setTimeout(runSlice, 0)
}

self.onmessage = (ev: MessageEvent<MainToWorker>) => {
  const msg = ev.data
  switch (msg.type) {
    case 'init':
      init(msg)
      break
    case 'start':
      if (running || !model || !tcfg) break
      running = true
      stopped = false
      startedAt = Date.now() - elapsedMs
      setTimeout(runSlice, 0)
      break
    case 'stop':
      stopped = true
      break
    case 'reset':
      // 用同样的种子重建，回到完全一样的初始权重
      if (!mcfg || !tcfg || !data) break
      resetState()
      buildModel(tcfg.seed)
      trace.push({ step: 0, loss: evaluate() })
      captureSample()
      captureAttn()
      postProgress(false)
      break
  }
}

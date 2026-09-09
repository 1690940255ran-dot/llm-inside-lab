/**
 * 采样：温度 / top-k / top-p
 *
 * 模型最后一层输出的是 logits（未归一化的分数），要先变成概率才能"抽签"。
 * 这三个参数就是在抽签之前对分布做手脚：
 *   - 温度 T：缩放 logits。T<1 把差距拉大（更确定），T>1 抹平差距（更发散）
 *   - top-k：只留概率最大的 k 个，其余直接丢掉再归一化
 *   - top-p：从大到小累加，只保留累计概率刚超过 p 的最少候选（核采样）
 */
import { mulberry32 } from './random'

export interface SamplingConfig {
  temperature: number
  /** 0 或 >= 词表大小 表示不启用 */
  topK: number
  /** 1 表示不启用 */
  topP: number
}

export const DEFAULT_SAMPLING: SamplingConfig = {
  temperature: 1,
  topK: 0,
  topP: 1,
}

export interface SampleStep {
  /** 原始概率（softmax(logits)） */
  base: number[]
  /** 温度之后、过滤之前的概率 */
  tempered: number[]
  /** 最终用于抽签的概率（被过滤掉的位置为 0） */
  final: number[]
  /** 是否进入了最终候选集 */
  kept: boolean[]
  chosen: number
}

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits)
  const exps = logits.map((x) => Math.exp(x - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map((x) => x / sum)
}

/** 按概率抽签，rnd 是 [0,1) 随机源 */
function pick(probs: number[], rnd: number): number {
  let acc = 0
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i]
    if (rnd < acc) return i
  }
  return probs.length - 1
}

/** 完整走一遍：原始 → 温度 → top-k → top-p → 抽签 */
export function sampleNext(logits: number[], cfg: SamplingConfig, seed: number): SampleStep {
  const base = softmax(logits)
  const tempered = softmax(logits.map((l) => l / cfg.temperature))

  let work = [...tempered]
  const V = work.length

  // top-k
  if (cfg.topK > 0 && cfg.topK < V) {
    const order = work.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p)
    const keep = new Set(order.slice(0, cfg.topK).map((o) => o.i))
    work = work.map((p, i) => (keep.has(i) ? p : 0))
  }

  // top-p（核采样）
  if (cfg.topP < 1) {
    const order = work.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p)
    let acc = 0
    const keep = new Set<number>()
    for (const o of order) {
      keep.add(o.i)
      acc += o.p
      if (acc >= cfg.topP) break // 保留"跨过阈值"的那个，这是标准做法
    }
    work = work.map((p, i) => (keep.has(i) ? p : 0))
  }

  const sum = work.reduce((a, b) => a + b, 0)
  const final = sum > 0 ? work.map((p) => p / sum) : new Array(V).fill(1 / V)

  return {
    base,
    tempered,
    final,
    kept: work.map((p) => p > 0),
    chosen: pick(final, mulberry32(seed)()),
  }
}

/** 贪心：永远取概率最大的 */
export function argmax(probs: number[]): number {
  let best = 0
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i
  return best
}

/** 分布的熵，用来衡量"有多纠结" */
export function entropyOf(probs: number[]): number {
  let h = 0
  for (const p of probs) if (p > 1e-12) h -= p * Math.log(p)
  return h
}

/** 候选集大小（最终概率非零的个数） */
export function candidateCount(step: SampleStep): number {
  return step.kept.filter(Boolean).length
}

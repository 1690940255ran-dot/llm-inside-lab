/**
 * 真实的 BPE（Byte Pair Encoding）分词算法
 *
 * 这是 GPT-2 / LLaMA / Qwen 等模型使用的分词方法的核心思想：
 *   1. 先把文本切成「预分词单元」（这里：连续的汉字算一个单元、连续英文字母算一个单元、标点各自独立）
 *   2. 每个单元拆成单字符，统计语料里出现频率最高的一对相邻字符
 *   3. 把这一对合并成一个新符号，记进合并表，重复第 2 步 N 次
 *   4. 编码新文本时，按合并表的优先级从高到低依次把字符粘回去
 *
 * 这里没有硬编码任何结果，合并表真的是从内置语料里学出来的，
 * 所以调整「合并次数」或换一份语料，词表会真的跟着变。
 */
import { DEFAULT_CORPUS } from './corpus'

const SPLIT = '\u0001' // 合并表 key 的分隔符，避免和真实字符冲突
const CJK_RE = /[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff]/

/** 预分词：把一段文本切成若干独立单元，BPE 只在单元内部合并，不会跨单元 */
export function pretokenize(text: string): string[] {
  const out: string[] = []
  let buf = ''
  let bufType: 'none' | 'cjk' | 'word' = 'none'

  const flush = () => {
    if (buf) {
      out.push(buf)
      buf = ''
    }
    bufType = 'none'
  }

  for (const ch of text) {
    if (/\s/.test(ch)) {
      flush()
      out.push(ch) // 空白自己成一个单元，不去污染词表
      continue
    }
    let type: 'cjk' | 'word' | 'punct' = 'punct'
    if (CJK_RE.test(ch)) type = 'cjk'
    else if (/[A-Za-z0-9]/.test(ch)) type = 'word'

    if (type === 'punct') {
      flush()
      out.push(ch)
      continue
    }
    if (bufType !== type) flush()
    buf += ch
    bufType = type
  }
  flush()
  return out
}

export interface Merge {
  a: string
  b: string
  rank: number // 越小优先级越高
  count: number // 合并时这对相邻符号在语料中出现的次数
}

export interface BPEModel {
  numMerges: number
  merges: Merge[]
  mergeRank: Map<string, number>
  idToToken: string[]
  tokenToId: Map<string, number>
  corpusSize: number
}

/** 在语料上训练 BPE，学习 numMerges 次合并 */
export function trainBPE(corpus: string = DEFAULT_CORPUS, numMerges = 160): BPEModel {
  // 1. 统计每个预分词单元的出现次数
  const freq = new Map<string, number>()
  for (const pt of pretokenize(corpus)) freq.set(pt, (freq.get(pt) ?? 0) + 1)

  // 2. 每个单元初始化为「字符数组」
  const symbols = new Map<string, string[]>()
  const chars = new Set<string>()
  for (const [pt] of freq) {
    symbols.set(pt, [...pt])
    for (const c of pt) chars.add(c)
  }

  const merges: Merge[] = []
  const mergeRank = new Map<string, number>()

  // 3. 反复找出现次数最多的相邻对并合并
  for (let step = 0; step < numMerges; step++) {
    const pairCount = new Map<string, number>()
    for (const [pt, syms] of symbols) {
      const f = freq.get(pt) ?? 0
      for (let i = 0; i + 1 < syms.length; i++) {
        const key = syms[i] + SPLIT + syms[i + 1]
        pairCount.set(key, (pairCount.get(key) ?? 0) + f)
      }
    }
    let bestKey: string | null = null
    let bestCount = 0
    for (const [k, c] of pairCount) {
      // 次数相同时按字典序取，保证结果稳定可复现
      if (c > bestCount || (c === bestCount && bestKey !== null && k < bestKey)) {
        bestKey = k
        bestCount = c
      }
    }
    if (!bestKey || bestCount < 2) break // 没有值得再合并的对了

    const [a, b] = bestKey.split(SPLIT)
    merges.push({ a, b, rank: step, count: bestCount })
    mergeRank.set(a + SPLIT + b, step)

    for (const [pt, syms] of symbols) {
      const next: string[] = []
      for (let i = 0; i < syms.length; i++) {
        if (i + 1 < syms.length && syms[i] === a && syms[i + 1] === b) {
          next.push(a + b)
          i++
        } else next.push(syms[i])
      }
      symbols.set(pt, next)
    }
  }

  // 4. 构造词表：基础字符在前，学到的合并 token 按优先级排在后面
  const idToToken = [...chars].sort()
  const tokenToId = new Map<string, number>()
  idToToken.forEach((t, i) => tokenToId.set(t, i))
  for (const m of merges) {
    const t = m.a + m.b
    if (!tokenToId.has(t)) {
      tokenToId.set(t, idToToken.length)
      idToToken.push(t)
    }
  }

  return {
    numMerges: merges.length,
    merges,
    mergeRank,
    idToToken,
    tokenToId,
    corpusSize: corpus.length,
  }
}

export interface Token {
  text: string
  id: number
  oov: boolean // 训练语料里没见过的字符
}

export interface EncodeStep {
  symbols: string[]
  pair?: [string, string]
  rank?: number
}

export interface EncodeResult {
  tokens: Token[]
  /** 每个预分词单元的合并过程，用于播放动画 */
  stepsByUnit: EncodeStep[][]
}

/** 用训练好的模型给一段文本分词，并记录每一步合并过程 */
export function encode(text: string, model: BPEModel): EncodeResult {
  const tokens: Token[] = []
  const stepsByUnit: EncodeStep[][] = []

  for (const pt of pretokenize(text)) {
    let syms = [...pt]
    const steps: EncodeStep[] = [{ symbols: [...syms] }]

    while (syms.length > 1) {
      let bestRank = Infinity
      let bestIdx = -1
      for (let i = 0; i + 1 < syms.length; i++) {
        const r = model.mergeRank.get(syms[i] + SPLIT + syms[i + 1])
        if (r !== undefined && r < bestRank) {
          bestRank = r
          bestIdx = i
        }
      }
      if (bestIdx < 0) break
      const pair: [string, string] = [syms[bestIdx], syms[bestIdx + 1]]
      syms.splice(bestIdx, 2, pair[0] + pair[1])
      steps.push({ symbols: [...syms], pair, rank: bestRank })
    }
    stepsByUnit.push(steps)

    for (const s of syms) {
      const id = model.tokenToId.get(s)
      tokens.push({ text: s, id: id ?? -1, oov: id === undefined })
    }
  }
  return { tokens, stepsByUnit }
}

/** 字符级分词，作为对照基线 */
export function encodeByChar(text: string): Token[] {
  return [...text].map((c, i) => ({ text: c, id: i, oov: false }))
}

export interface TokenStats {
  tokenCount: number
  charCount: number
  avgTokenLength: number
  /** 相对于字符级的压缩倍数，越高说明 token 越"大" */
  compression: number
}

export function computeStats(tokens: Token[], text: string): TokenStats {
  const charCount = [...text].length
  const tokenCount = tokens.length || 1
  return {
    tokenCount: tokens.length,
    charCount,
    avgTokenLength: charCount / tokenCount,
    compression: charCount / tokenCount,
  }
}

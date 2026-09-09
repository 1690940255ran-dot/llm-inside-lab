/**
 * 从内置语料统计出来的 bigram 语言模型
 *
 * 为什么不用真模型？还是那个约束：不下载权重、离线可跑。
 * 但"采样机制"这件事本身不需要大模型——任何能给出 next-token 概率分布的东西都行。
 *
 * 所以这里用一个真正的统计语言模型（带加平滑的 bigram），
 * 它是从语料里数出来的真实分布，不是拍脑袋造的。
 * 温度 / top-k / top-p 作用在它上面的效果，和作用在 GPT 的 logits 上完全一样。
 */
import { encode, pretokenize, type BPEModel } from './bpe'

export interface NGramModel {
  vocab: string[]
  tokenToIndex: Map<string, number>
  /** bigram[a] = Map<下一个 token 的 index, 出现次数> */
  bigram: Map<number, Map<number, number>>
  bigramTotal: Map<number, number>
  unigram: number[]
  unigramTotal: number
}

const ALPHA = 0.35 // 加平滑系数，避免零概率

export function trainNGram(corpus: string, bpe: BPEModel): NGramModel {
  const tokens = encode(corpus, bpe).tokens.map((t) => t.text)
  const tokenToIndex = new Map<string, number>()
  const vocab: string[] = []
  const unigram: number[] = []

  const indexOf = (t: string) => {
    let i = tokenToIndex.get(t)
    if (i === undefined) {
      i = vocab.length
      vocab.push(t)
      tokenToIndex.set(t, i)
      unigram.push(0)
    }
    return i
  }

  const bigram = new Map<number, Map<number, number>>()
  const bigramTotal = new Map<number, number>()
  let unigramTotal = 0

  for (let i = 0; i < tokens.length; i++) {
    const a = indexOf(tokens[i])
    unigram[a] += 1
    unigramTotal += 1
    if (i + 1 < tokens.length) {
      const b = indexOf(tokens[i + 1])
      let row = bigram.get(a)
      if (!row) {
        row = new Map()
        bigram.set(a, row)
      }
      row.set(b, (row.get(b) ?? 0) + 1)
      bigramTotal.set(a, (bigramTotal.get(a) ?? 0) + 1)
    }
  }

  return { vocab, tokenToIndex, bigram, bigramTotal, unigram, unigramTotal }
}

/**
 * 给定历史 token，给出下一个 token 的 logits（对数概率）
 * 优先用 bigram，历史没见过就退回 unigram。
 */
export function nextLogits(
  ngram: NGramModel,
  history: string[],
): { logits: number[]; source: 'bigram' | 'unigram' } {
  const V = ngram.vocab.length
  const last = history[history.length - 1]
  const a = last !== undefined ? ngram.tokenToIndex.get(last) : undefined
  const row = a !== undefined ? ngram.bigram.get(a) : undefined
  const total = a !== undefined ? (ngram.bigramTotal.get(a) ?? 0) : 0

  const logits = new Array(V).fill(0)
  if (row && total > 0) {
    for (let i = 0; i < V; i++) {
      const c = row.get(i) ?? 0
      logits[i] = Math.log((c + ALPHA) / (total + ALPHA * V))
    }
    return { logits, source: 'bigram' }
  }
  for (let i = 0; i < V; i++) {
    const c = ngram.unigram[i] ?? 0
    logits[i] = Math.log((c + ALPHA) / (ngram.unigramTotal + ALPHA * V))
  }
  return { logits, source: 'unigram' }
}

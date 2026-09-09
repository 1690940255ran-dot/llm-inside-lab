/**
 * 多头自注意力的模拟计算
 *
 * ⚠️ 诚实声明：这里算出来的不是任何真实模型的注意力权重。
 * 真实权重需要跑一次完整前向传播（几十 MB 到几 GB 的权重文件），
 * 与本站点「纯前端、零依赖、离线可跑」的目标冲突。
 *
 * 所以这里的做法是：
 *   1. 用确定性伪随机矩阵算出 Q、K（真实模型里这也是矩阵乘法，只是权重是学出来的）
 *   2. 叠加一层「头行为偏置」，让不同的头稳定地呈现出文献里观察到的典型模式
 *      （前一个词、句首 attention sink、标点、内容相似、稀疏激等）
 *   3. 再加上距离衰减、因果掩码、温度缩放，最后走标准的 softmax
 *
 * 这样得到的图，形状、规律和真实 GPT-2 的注意力图高度相似，
 * 足以让学习者建立起「不同头在干不同的事」的直觉，而且是确定性的、可复现的。
 */
import { cosine, tokenVector } from './embedding'
import { hashString, mulberry32, randomMatrix, softmax } from './random'

export type HeadPattern =
  | 'previous'
  | 'next'
  | 'self'
  | 'first'
  | 'delimiter'
  | 'content'
  | 'broad'
  | 'sparse'

export const PATTERN_INFO: Record<HeadPattern, { name: string; desc: string }> = {
  previous: {
    name: '前一个 token',
    desc: '几乎只看紧邻的上一个词。这是语言模型最基础的模式，相当于在学二元语法（bigram）。',
  },
  next: {
    name: '后一个 token',
    desc: '反向关注下一个位置，常见于双向编码器（BERT 类）或者需要处理逆序依赖的场景。',
  },
  self: {
    name: '关注自身',
    desc: '主要看自己，相当于「原样把当前信息传下去」，常和残差连接配合保留原始语义。',
  },
  first: {
    name: '句首（attention sink）',
    desc: '大量权重压在第一个 token 上。真实模型里非常常见，句首 token 像一个「泄压阀」，多余的概率质量往这里丢。',
  },
  delimiter: {
    name: '标点 / 分隔符',
    desc: '聚焦逗号、句号等分隔符。这些位置往往聚合了整句的概括信息，模型拿它当「存档点」。',
  },
  content: {
    name: '内容相似',
    desc: '关注语义或拼写相近的 token。深层常见，用来做指代消解、同义聚合这类真正需要理解的事。',
  },
  broad: {
    name: '广谱平均',
    desc: '权重摊得比较平，把整段上下文做一个加权平均，相当于提取全局主题。',
  },
  sparse: {
    name: '稀疏激发',
    desc: '只被少数特定 token 强烈激活，看起来像"关键词检索"，常和某个具体特征绑定。',
  },
}

export interface AttentionConfig {
  nLayers: number
  nHeads: number
  dModel: number
  /** softmax 温度：越大分布越平，越小越尖锐 */
  temperature: number
  /** 是否使用因果掩码（解码器必须开，看不到未来） */
  causal: boolean
  /** 距离衰减系数：越大越倾向于关注近处 */
  distanceDecay: number
}

export const DEFAULT_ATTENTION_CONFIG: AttentionConfig = {
  nLayers: 4,
  nHeads: 6,
  dModel: 48,
  temperature: 1,
  causal: true,
  distanceDecay: 0.12,
}

/** 给每一层每一个头分配一个"性格"：浅层偏位置，深层偏内容 */
function assignPattern(layer: number, head: number, nLayers: number): HeadPattern {
  const shallow: HeadPattern[] = ['previous', 'self', 'first', 'delimiter', 'next']
  const middle: HeadPattern[] = ['content', 'previous', 'delimiter', 'broad']
  const deep: HeadPattern[] = ['content', 'broad', 'sparse', 'first']
  const ratio = nLayers <= 1 ? 0 : layer / (nLayers - 1)
  const pool = ratio < 0.34 ? shallow : ratio < 0.7 ? middle : deep
  return pool[(head + layer * 3) % pool.length]
}

const DELIMITER_RE = /[，。！？；：、,.!?;:\s"'（）《》]/

export interface HeadSummary {
  pattern: HeadPattern
  avgEntropy: number
  avgMax: number
  /** 第一个 token 平均拿到的权重，用于观察 attention sink */
  sinkRatio: number
}

export interface AttentionResult {
  /** weights[layer][head][i][j] = 第 i 个 token 对第 j 个 token 的注意力 */
  weights: number[][][][]
  patterns: HeadPattern[][]
  summaries: HeadSummary[][]
  /** 每个 token 的模拟嵌入向量 */
  embeddings: number[][]
}

function entropyOf(p: number[]): number {
  let h = 0
  for (const x of p) if (x > 1e-12) h -= x * Math.log(x)
  return h
}

export function computeAttention(tokens: string[], cfg: AttentionConfig): AttentionResult {
  const n = tokens.length
  const emb = tokens.map((t) => tokenVector(t, cfg.dModel))
  const dHead = Math.max(1, Math.floor(cfg.dModel / cfg.nHeads))

  const weights: number[][][][] = []
  const patterns: HeadPattern[][] = []
  const summaries: HeadSummary[][] = []

  for (let l = 0; l < cfg.nLayers; l++) {
    const layerW: number[][][] = []
    const layerP: HeadPattern[] = []
    const layerS: HeadSummary[] = []

    for (let h = 0; h < cfg.nHeads; h++) {
      const pattern = assignPattern(l, h, cfg.nLayers)
      const seed = hashString(`L${l}-H${h}-d${dHead}`)
      const Wq = randomMatrix(cfg.dModel, dHead, seed, 1 / Math.sqrt(cfg.dModel))
      const Wk = randomMatrix(cfg.dModel, dHead, seed + 1, 1 / Math.sqrt(cfg.dModel))

      // Q = E·Wq, K = E·Wk，形状都是 [n][dHead]
      const Q: number[][] = emb.map((e) => Wq[0].map((_, j) => e.reduce((a, v, k) => a + v * Wq[k][j], 0)))
      const K: number[][] = emb.map((e) => Wk[0].map((_, j) => e.reduce((a, v, k) => a + v * Wk[k][j], 0)))

      const rnd = mulberry32(seed + 99)
      const matrix: number[][] = []
      let entropySum = 0
      let maxSum = 0
      let sinkSum = 0

      for (let i = 0; i < n; i++) {
        const logits = new Array(n).fill(0)
        for (let j = 0; j < n; j++) {
          // (1) 内容项：Q·K^T / sqrt(dHead)
          let content = 0
          for (let k = 0; k < dHead; k++) content += Q[i][k] * K[j][k]
          content = content / Math.sqrt(dHead)

          // (2) 头行为偏置
          let bias = 0
          switch (pattern) {
            case 'previous':
              if (j === i - 1) bias += 5
              break
            case 'next':
              if (j === i + 1) bias += 5
              break
            case 'self':
              if (j === i) bias += 4.5
              break
            case 'first':
              if (j === 0) bias += 5
              break
            case 'delimiter':
              if (DELIMITER_RE.test(tokens[j] ?? '')) bias += 3.8
              break
            case 'content':
              bias += 4 * cosine(emb[i], emb[j])
              break
            case 'sparse':
              if (rnd() < 0.14) bias += 4
              break
            case 'broad':
            default:
              break
          }

          // (3) 距离衰减：越远越不关注（真实模型里由 RoPE / 相对位置偏置产生类似效果）
          const decay = -cfg.distanceDecay * Math.abs(i - j)
          logits[j] = (content + bias + decay) / cfg.temperature
        }

        // (4) 因果掩码：解码器不能偷看未来
        if (cfg.causal) {
          for (let j = i + 1; j < n; j++) logits[j] = -1e9
        }

        const row = softmax(logits)
        matrix.push(row)
        entropySum += entropyOf(row)
        maxSum += Math.max(...row)
        sinkSum += row[0] ?? 0
      }

      layerW.push(matrix)
      layerP.push(pattern)
      layerS.push({
        pattern,
        avgEntropy: n ? entropySum / n : 0,
        avgMax: n ? maxSum / n : 0,
        sinkRatio: n ? sinkSum / n : 0,
      })
    }
    weights.push(layerW)
    patterns.push(layerP)
    summaries.push(layerS)
  }

  return { weights, patterns, summaries, embeddings: emb }
}

/** 把注意力矩阵的一行按权重从大到小排出来，用于画"这个 token 在看谁" */
export function rowTopK(row: number[], k: number): { index: number; value: number }[] {
  return row
    .map((value, index) => ({ index, value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, k)
}

export interface HeadMetrics {
  avgEntropy: number
  avgMax: number
  sinkRatio: number
}

/**
 * 对任意注意力矩阵算同一组指标。
 * 真实模型加载后没有"头性格"可言（那是模拟时注入的），但熵、最大权重、sink 占比照样能算。
 */
export function headMetrics(weights: number[][]): HeadMetrics {
  const n = weights.length
  if (n === 0) return { avgEntropy: 0, avgMax: 0, sinkRatio: 0 }
  let e = 0
  let m = 0
  let s = 0
  for (const row of weights) {
    e += entropyOf(row)
    m += Math.max(...row)
    s += row[0] ?? 0
  }
  return { avgEntropy: e / n, avgMax: m / n, sinkRatio: s / n }
}

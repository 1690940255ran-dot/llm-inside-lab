/**
 * 模拟一个完整 Transformer Block 的前向传播
 *
 * 采用现代主流（LLaMA / Qwen / GPT-2 之后）的 **pre-norm** 结构：
 *
 *   h ← h + Attn(LN(h))
 *   h ← h + FFN(LN(h))
 *
 * 关注点不是数值有多准，而是把三件事讲清楚：
 *   1. 张量形状从头到尾基本不变（[n, d_model]），变的只有内容
 *   2. 残差连接是一条"高速公路"，每一层只往上加一点点增量
 *   3. 越深的层，表示变化越小 —— 层间相似度图会显现出明显的块状结构
 */
import { tokenVector } from './embedding'
import { computeAttention } from './attention'
import { mulberry32, randomMatrix, hashString } from './random'
import { sinusoidalPE } from './positional'

export interface TransformerConfig {
  nLayers: number
  dModel: number
  /** FFN 中间层维度，默认为 4 × d_model */
  dFF: number
}

export const DEFAULT_TRANSFORMER: TransformerConfig = {
  nLayers: 6,
  dModel: 48,
  dFF: 192,
}

export interface LayerTrace {
  layer: number
  hIn: number[][]
  attnOut: number[][]
  afterAttn: number[][]
  ffnOut: number[][]
  hOut: number[][]
  /** 注意力分支输出的相对幅度 ||Attn|| / ||h|| */
  attnRatio: number
  /** FFN 分支输出的相对幅度 ||FFN|| / ||h|| */
  ffnRatio: number
}

export interface ForwardTrace {
  h0: number[][]
  layers: LayerTrace[]
  /** (L+1) × (L+1) 的层间表示相似度，index 0 是嵌入层输出 */
  simMatrix: number[][]
}

/** 行级 Layer Normalization */
function layerNorm(x: number[][]): number[][] {
  const d = x[0]?.length ?? 0
  const eps = 1e-5
  return x.map((row) => {
    const mean = row.reduce((a, b) => a + b, 0) / (d || 1)
    const var_ = row.reduce((a, b) => a + (b - mean) * (b - mean), 0) / (d || 1)
    const s = Math.sqrt(var_ + eps)
    return row.map((v) => (v - mean) / s)
  })
}

function matmul(a: number[][], b: number[][]): number[][] {
  const n = a.length
  const m = b[0]?.length ?? 0
  const k = b.length
  const out: number[][] = Array.from({ length: n }, () => new Array(m).fill(0))
  for (let i = 0; i < n; i++) {
    for (let p = 0; p < k; p++) {
      const av = a[i][p]
      if (av === 0) continue
      for (let j = 0; j < m; j++) out[i][j] += av * b[p][j]
    }
  }
  return out
}

function gelu(v: number): number {
  // tanh 近似，够用且便宜
  return 0.5 * v * (1 + Math.tanh(0.7978845608 * (v + 0.044715 * v * v * v)))
}

function add(a: number[][], b: number[][]): number[][] {
  return a.map((row, i) => row.map((v, j) => v + (b[i]?.[j] ?? 0)))
}

function meanRowNorm(x: number[][]): number {
  if (x.length === 0) return 0
  const sum = x.reduce((acc, row) => acc + Math.sqrt(row.reduce((a, v) => a + v * v, 0)), 0)
  return sum / x.length
}

/** 两层表示之间的平均余弦相似度（逐 token 求余弦再平均） */
function reprSim(a: number[][], b: number[][]): number {
  if (a.length === 0) return 1
  let s = 0
  for (let t = 0; t < a.length; t++) {
    let dot = 0
    let na = 0
    let nb = 0
    for (let i = 0; i < a[t].length; i++) {
      dot += a[t][i] * b[t][i]
      na += a[t][i] * a[t][i]
      nb += b[t][i] * b[t][i]
    }
    const den = Math.sqrt(na) * Math.sqrt(nb)
    s += den === 0 ? 0 : dot / den
  }
  return s / a.length
}

export function simulateForward(tokens: string[], cfg: TransformerConfig): ForwardTrace {
  const n = tokens.length
  const d = cfg.dModel

  // 输入 = 词嵌入 + 位置编码（广播相加）
  const pe = sinusoidalPE(n, d)
  const h0 = tokens.map((t, i) => {
    const e = tokenVector(t, d)
    return e.map((v, k) => v + (pe[i]?.[k] ?? 0) * 0.6)
  })

  const layers: LayerTrace[] = []
  let h = h0

  for (let l = 0; l < cfg.nLayers; l++) {
    const seed = hashString(`block-${l}-${d}-${cfg.dFF}`)

    // —— 注意力分支 ——
    const ln1 = layerNorm(h)
    const att = computeAttention(tokens, {
      nLayers: 1,
      nHeads: 1,
      dModel: d,
      temperature: 1,
      causal: true,
      distanceDecay: 0.1,
    })
    const W = att.weights[0]?.[0] ?? []
    const Wv = randomMatrix(d, d, seed, 1 / Math.sqrt(d))
    const V = matmul(ln1, Wv) // [n, d]
    // attnOut[i] = Σ_j w_ij · V_j
    const attnOut: number[][] = Array.from({ length: n }, () => new Array(d).fill(0))
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const w = W[i]?.[j] ?? 0
        if (w === 0) continue
        for (let k = 0; k < d; k++) attnOut[i][k] += w * V[j][k]
      }
    }
    const afterAttn = add(h, attnOut)

    // —— FFN 分支 ——
    const ln2 = layerNorm(afterAttn)
    const W1 = randomMatrix(d, cfg.dFF, seed + 11, 1 / Math.sqrt(d))
    const W2 = randomMatrix(cfg.dFF, d, seed + 22, 1 / Math.sqrt(cfg.dFF))
    const hidden = matmul(ln2, W1).map((row) => row.map(gelu)) // [n, dFF]
    const ffnOut = matmul(hidden, W2) // [n, d]
    const hOut = add(afterAttn, ffnOut)

    layers.push({
      layer: l,
      hIn: h,
      attnOut,
      afterAttn,
      ffnOut,
      hOut,
      attnRatio: meanRowNorm(attnOut) / (meanRowNorm(h) || 1),
      ffnRatio: meanRowNorm(ffnOut) / (meanRowNorm(afterAttn) || 1),
    })
    h = hOut
  }

  // 层间相似度：把每一层的输出（含 h0）两两比较
  const all = [h0, ...layers.map((l) => l.hOut)]
  const simMatrix = all.map((a) => all.map((b) => reprSim(a, b)))

  return { h0, layers, simMatrix }
}

/** 把某一层某个 token 的表示取出来，用于画热图 */
export function stateMatrix(state: number[][], maxDim = 32): number[][] {
  const max = Math.max(...state.flat().map(Math.abs), 1e-6)
  return state.map((row) => row.slice(0, maxDim).map((v) => v / max))
}

/**
 * 位置编码相关计算
 *
 * 自注意力本身是「无序」的：把一句话的词打乱，注意力输出也会跟着打乱。
 * 所以必须显式地把位置信息加进嵌入向量里，这就是位置编码。
 * 本文件提供两种经典方案的数值：
 *   - 正弦位置编码（原始 Transformer）：PE[pos][2i] = sin(pos / 10000^(2i/d))
 *   - RoPE 旋转位置编码（LLaMA / Qwen 等在用）：把向量按二维一组旋转 pos * θ_i 的角度
 */
export const BASE = 10000

/** 正弦位置编码矩阵，形状 [n][d] */
export function sinusoidalPE(n: number, d: number): number[][] {
  const pe: number[][] = []
  for (let pos = 0; pos < n; pos++) {
    const row = new Array(d).fill(0)
    for (let i = 0; i < d; i += 2) {
      const freq = 1 / Math.pow(BASE, i / d)
      row[i] = Math.sin(pos * freq)
      if (i + 1 < d) row[i + 1] = Math.cos(pos * freq)
    }
    pe.push(row)
  }
  return pe
}

/** RoPE 的旋转角：第 j 组（两个维度一组）在位置 pos 上要转多少弧度 */
export function ropeAngles(n: number, d: number): number[][] {
  const out: number[][] = []
  for (let pos = 0; pos < n; pos++) {
    const row: number[] = []
    for (let j = 0; j < Math.floor(d / 2); j++) {
      row.push(pos / Math.pow(BASE, (2 * j) / d))
    }
    out.push(row)
  }
  return out
}

/** 对一个 d 维向量的第 pairIndex 组（2 维）施加 RoPE 旋转，返回旋转后的两维坐标 */
export function ropeRotate(
  vec: number[],
  pairIndex: number,
  pos: number,
  d: number,
): [number, number] {
  const theta = pos / Math.pow(BASE, (2 * pairIndex) / d)
  const x = vec[pairIndex * 2] ?? 0
  const y = vec[pairIndex * 2 + 1] ?? 0
  return [x * Math.cos(theta) - y * Math.sin(theta), x * Math.sin(theta) + y * Math.cos(theta)]
}

function cosine(a: number[], b: number[]): number {
  let dot = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  const den = Math.sqrt(na) * Math.sqrt(nb)
  return den === 0 ? 0 : dot / den
}

/**
 * 位置向量之间的余弦相似度矩阵 [n][n]
 * 这张图很值得看：正弦编码的相似度只和「相对距离」有关，
 * 所以图上会呈现出沿对角线方向的条带 —— 这正是它能表达相对位置的原因。
 */
export function positionalSimilarity(pe: number[][]): number[][] {
  const n = pe.length
  const out: number[][] = []
  for (let i = 0; i < n; i++) {
    const row: number[] = []
    for (let j = 0; j < n; j++) row.push(cosine(pe[i], pe[j]))
    out.push(row)
  }
  return out
}

/**
 * 模拟的 token 嵌入向量
 *
 * 真实模型的嵌入矩阵是训练出来的，浏览器里没法凭空拿到。
 * 这里的做法是「确定性伪造」：同一个 token 永远得到同一个向量，
 * 并且让「首字符相同」和「字符类别相同」的 token 分享一部分成分，
 * 于是降维之后能看到成簇的现象 —— 这正好可以用来讲「语义相近的词在向量空间里相近」。
 *
 * 注意：这是教学示意用的模拟向量，不是任何真实模型的权重。
 */
import { hashString, mulberry32 } from './random'

function charClass(token: string): string {
  const c = token[0] ?? ''
  if (/[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/.test(c)) return 'cjk'
  if (/[A-Za-z]/.test(c)) return 'latin'
  if (/[0-9]/.test(c)) return 'digit'
  if (/\s/.test(c)) return 'space'
  return 'punct'
}

function rawVector(seedKey: string, dim: number): number[] {
  const rnd = mulberry32(hashString(seedKey))
  const v: number[] = []
  for (let i = 0; i < dim; i++) v.push(rnd() * 2 - 1)
  return v
}

const cache = new Map<string, number[]>()

/** 取一个 token 的模拟嵌入向量（带缓存，同一 token 结果恒定） */
export function tokenVector(token: string, dim = 48): number[] {
  const key = `${dim}::${token}`
  const hit = cache.get(key)
  if (hit) return hit

  const own = rawVector('tok:' + token, dim)
  const byPrefix = rawVector('pre:' + (token[0] ?? ''), dim)
  const byClass = rawVector('cls:' + charClass(token), dim)

  const v = new Array(dim)
  for (let i = 0; i < dim; i++) {
    v[i] = 0.55 * own[i] + 0.25 * byPrefix[i] + 0.2 * byClass[i]
  }
  const norm = Math.sqrt(v.reduce((a, b) => a + b * b, 0)) || 1
  const out = v.map((x) => x / norm)
  cache.set(key, out)
  return out
}

export function cosine(a: number[], b: number[]): number {
  let dot = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  const d = Math.sqrt(na) * Math.sqrt(nb)
  return d === 0 ? 0 : dot / d
}

/** 把一批高维向量降到二维（PCA，幂迭代 + 收缩法），用于画散点图 */
export function pca2d(vectors: number[][]): {
  points: { x: number; y: number }[]
  explained: [number, number]
} {
  const n = vectors.length
  const d = vectors[0]?.length ?? 0
  if (n === 0 || d === 0) return { points: [], explained: [0, 0] }
  if (n === 1) return { points: [{ x: 0, y: 0 }], explained: [1, 0] }

  // 中心化
  const mean = new Array(d).fill(0)
  for (const v of vectors) for (let i = 0; i < d; i++) mean[i] += v[i] / n
  const X = vectors.map((v) => v.map((x, i) => x - mean[i]))

  // 协方差矩阵 C = X^T X / n
  const C: number[][] = Array.from({ length: d }, () => new Array(d).fill(0))
  for (const row of X) {
    for (let i = 0; i < d; i++) {
      for (let j = i; j < d; j++) {
        const val = (row[i] * row[j]) / n
        C[i][j] += val
        if (i !== j) C[j][i] += val
      }
    }
  }

  let trace = 0
  for (let i = 0; i < d; i++) trace += C[i][i]

  const eigen: { vec: number[]; val: number }[] = []
  let cur = C
  for (let k = 0; k < 2; k++) {
    // 幂迭代求最大特征向量
    const rnd = mulberry32(1234 + k * 77)
    let v = Array.from({ length: d }, () => rnd() * 2 - 1)
    let val = 0
    for (let it = 0; it < 150; it++) {
      const u = new Array(d).fill(0)
      for (let i = 0; i < d; i++) {
        let s = 0
        for (let j = 0; j < d; j++) s += cur[i][j] * v[j]
        u[i] = s
      }
      const norm = Math.sqrt(u.reduce((a, b) => a + b * b, 0))
      if (norm < 1e-12) break
      v = u.map((x) => x / norm)
      let vcv = 0
      for (let i = 0; i < d; i++) {
        let s = 0
        for (let j = 0; j < d; j++) s += cur[i][j] * v[j]
        vcv += v[i] * s
      }
      val = vcv
    }
    eigen.push({ vec: v, val })
    // 收缩：C ← C - λ v v^T
    const next: number[][] = Array.from({ length: d }, () => new Array(d).fill(0))
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) next[i][j] = cur[i][j] - val * v[i] * v[j]
    }
    cur = next
  }

  const points = X.map((row) => ({
    x: row.reduce((a, b, i) => a + b * eigen[0].vec[i], 0),
    y: row.reduce((a, b, i) => a + b * eigen[1].vec[i], 0),
  }))
  const explained: [number, number] =
    trace === 0
      ? [0, 0]
      : [Math.max(0, eigen[0].val) / trace, Math.max(0, eigen[1].val) / trace]
  return { points, explained }
}

/**
 * 确定性伪随机工具
 *
 * 为什么不用 Math.random()？
 * 站点里的"模拟权重"必须满足：同样的 token + 同样的层/头 → 每次得到完全一样的结果。
 * 否则用户每次输入都会看到乱跳的数字，无法建立"这个头稳定地关注某处"的直觉。
 * 所以这里统一用「字符串哈希 → 种子 → 伪随机序列」的方式生成，保证可复现。
 */

/** FNV-1a 字符串哈希，把任意字符串映射成 32 位无符号整数种子 */
export function hashString(s: string): number {
  let h = 2166136261 >>> 0
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

/** mulberry32：小巧、质量足够的伪随机数生成器，返回 [0,1) */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return function () {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** 生成一个确定性的标准正态分布随机数（Box-Muller 变换） */
export function gaussian(rnd: () => number): number {
  const u = Math.max(rnd(), 1e-9)
  const v = rnd()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

/** 生成确定性的随机矩阵，元素服从 N(0, scale^2) */
export function randomMatrix(
  rows: number,
  cols: number,
  seed: number,
  scale = 1,
): number[][] {
  const rnd = mulberry32(seed)
  const m: number[][] = []
  for (let i = 0; i < rows; i++) {
    const row: number[] = []
    for (let j = 0; j < cols; j++) row.push(gaussian(rnd) * scale)
    m.push(row)
  }
  return m
}

export function softmax(logits: number[]): number[] {
  const max = Math.max(...logits)
  const exps = logits.map((x) => Math.exp(x - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map((x) => (sum === 0 ? 1 / logits.length : x / sum))
}

export function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v))
}

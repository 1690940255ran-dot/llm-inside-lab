/**
 * KV Cache 的收益模型
 *
 * 解码阶段为什么慢？因为每生成一个 token 都要把整个序列重算一遍注意力。
 * KV Cache 的思路很简单：前面 token 的 K 和 V 已经算过了，存起来，下一步只算新 token 的 Q。
 *
 * 计算量模型（只统计注意力部分，4 是 QK^T 与 AV 两个矩阵乘各 2 倍 FLOPs）：
 *   无缓存：第 t 步要跑一整次前向，序列长 s = n + t → 4·s²·d·L
 *   有缓存：prefill 一次 4·n²·d·L；之后每步只算 1 个 query 对 s 个 key → 4·s·d·L
 * 于是复杂度从 O(m·n²) 降到 O(n² + m·n)，这就是加速的来源，也是显存的代价。
 */
export interface KVCacheConfig {
  nPrompt: number
  nGen: number
  nLayers: number
  /** Query 头数，决定注意力计算量 */
  nHeads: number
  /** KV 头数，决定缓存大小。GQA / MQA 模型里它远小于 nHeads */
  nKVHeads: number
  dHead: number
  batch: number
  bytesPerParam: number
}

export const DEFAULT_KV: KVCacheConfig = {
  nPrompt: 512,
  nGen: 128,
  nLayers: 28,
  nHeads: 28,
  nKVHeads: 4,
  dHead: 128,
  batch: 1,
  bytesPerParam: 2, // fp16
}

export interface KVReport {
  /** prefill 阶段注意力 FLOPs */
  prefillFlops: number
  /** 解码阶段（无缓存）注意力 FLOPs */
  decodeNoCache: number
  /** 解码阶段（有缓存）注意力 FLOPs */
  decodeWithCache: number
  totalNoCache: number
  totalWithCache: number
  speedup: number
  /** 单条序列单个 token 的 KV 显存（字节） */
  bytesPerToken: number
  /** 全部 batch、整个序列的 KV 显存（字节） */
  cacheBytes: number
  /** 逐 token 累计计算量曲线 */
  series: { step: number; noCache: number; withCache: number }[]
}

const F = 4 // 两个矩阵乘 × 2 FLOPs/MAC

export function analyzeKV(cfg: KVCacheConfig): KVReport {
  const d = cfg.dHead * cfg.nHeads
  const L = cfg.nLayers
  const n = cfg.nPrompt
  const m = cfg.nGen

  const prefillFlops = F * n * n * d * L
  let decodeNoCache = 0
  let decodeWithCache = 0
  const series: { step: number; noCache: number; withCache: number }[] = []
  let cumNo = prefillFlops
  let cumWith = prefillFlops

  for (let t = 1; t <= m; t++) {
    const s = n + t
    const no = F * s * s * d * L // 整段重算
    const with_ = F * s * d * L // 只算新 token 的 query
    decodeNoCache += no
    decodeWithCache += with_
    cumNo += no
    cumWith += with_
    series.push({ step: t, noCache: cumNo, withCache: cumWith })
  }

  // 注意只算 KV 头数：这就是 GQA / MQA 能大幅省显存的原因
  const bytesPerToken = 2 * L * cfg.nKVHeads * cfg.dHead * cfg.bytesPerParam // K 和 V 各一份
  const cacheBytes = bytesPerToken * (n + m) * cfg.batch

  const totalNoCache = prefillFlops + decodeNoCache
  const totalWithCache = prefillFlops + decodeWithCache

  return {
    prefillFlops,
    decodeNoCache,
    decodeWithCache,
    totalNoCache,
    totalWithCache,
    speedup: totalWithCache > 0 ? totalNoCache / totalWithCache : 1,
    bytesPerToken,
    cacheBytes,
    series,
  }
}

export function fmtFlops(v: number): string {
  if (v >= 1e15) return (v / 1e15).toFixed(2) + ' PFLOPs'
  if (v >= 1e12) return (v / 1e12).toFixed(2) + ' TFLOPs'
  if (v >= 1e9) return (v / 1e9).toFixed(2) + ' GFLOPs'
  if (v >= 1e6) return (v / 1e6).toFixed(2) + ' MFLOPs'
  return v.toFixed(0) + ' FLOPs'
}

export function fmtBytes(v: number): string {
  if (v >= 1024 ** 4) return (v / 1024 ** 4).toFixed(2) + ' TB'
  if (v >= 1024 ** 3) return (v / 1024 ** 3).toFixed(2) + ' GB'
  if (v >= 1024 ** 2) return (v / 1024 ** 2).toFixed(2) + ' MB'
  if (v >= 1024) return (v / 1024).toFixed(2) + ' KB'
  return v.toFixed(0) + ' B'
}

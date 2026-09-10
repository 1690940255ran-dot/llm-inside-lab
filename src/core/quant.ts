/**
 * 量化（Quantization）
 *
 * 这个文件**真的**把一组浮点权重压成整数再还原，误差是算出来的，不是画出来的。
 *
 * 它想回答三个问题：
 *   1. 4 bit 到底损失了多少？—— 看 SQNR 和量化台阶
 *   2. 为什么加一个离群值就崩？—— 对称量化的 scale 由 max|x| 决定，
 *      一个 40σ 的离群值会把 scale 撑大 40 倍，剩下 99% 的权重挤在 1~2 个台阶上
 *   3. per-tensor / per-channel / group-wise / NF4 各自好在哪？—— 同一份数据上比出来
 *
 * 权重分布按照真实 LLM 的样子造：主体是高斯分布，外加极少数极大的离群值
 * （真实模型里的 outlier features 就是集中在少数通道上的这种尖峰）。
 */
import { gaussian, mulberry32 } from './random'

export type QuantScheme = 'sym' | 'asym' | 'nf4'
export type QuantMode = 'tensor' | 'channel' | 'group'

/**
 * NF4（NormalFloat4）码本，来自 QLoRA 论文 / bitsandbytes。
 * 它不是均匀的：数值按标准正态的分位数摆放，密集处（0 附近）台阶多，稀疏处台阶少。
 * 所以同样 4 bit，对高斯权重它比均匀量化准得多 —— 下面 metrics 里能直接看到。
 */
export const NF4_CODEBOOK = [
  -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
  -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
  0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
  0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
]

export interface QuantConfig {
  bits: number
  scheme: QuantScheme
  mode: QuantMode
  /** group-wise 每组的元素数 */
  groupSize: number
  /** per-channel 的通道数 */
  nChannels: number
  /** 离群值幅度，单位是权重的标准差。1 = 就是普通权重，40 = 40σ 的尖峰 */
  outlierScale: number
  /** 有几个离群值 */
  outlierCount: number
  /** 权重个数 */
  n: number
  seed: number
}

export const DEFAULT_QUANT: QuantConfig = {
  bits: 4,
  scheme: 'sym',
  mode: 'tensor',
  groupSize: 32,
  nChannels: 8,
  outlierScale: 40,
  outlierCount: 2,
  n: 256,
  seed: 7,
}

/** 造一组像真实 LLM 权重的数据：高斯主体 + 少数离群尖峰 */
export function makeWeights(cfg: QuantConfig): number[] {
  const rnd = mulberry32(cfg.seed)
  const v = Array.from({ length: cfg.n }, () => gaussian(rnd))
  const k = Math.min(cfg.outlierCount, cfg.n)
  for (let i = 0; i < k; i++) {
    // 均匀铺开位置，避免离群值全挤在一起
    const idx = Math.floor(((i + 0.5) / k) * cfg.n) % cfg.n
    v[idx] = cfg.outlierScale * (i % 2 === 0 ? 1 : -1)
  }
  return v
}

export interface QuantResult {
  cfg: QuantConfig
  values: number[]
  codes: number[]
  dequant: number[]
  scales: number[]
  zeros: number[]
  /** 用到的不同码字个数 —— 离群值毁掉精度的最直观指标 */
  usedLevels: number
  /** 理论台阶数（2^bits 或 16） */
  totalLevels: number
  maxAbsErr: number
  mse: number
  /** 信噪比（dB），越高越好。0 dB 表示误差和信号一样大 */
  snrDb: number
  /** 按数值排序后的下标，画台阶图用 */
  order: number[]
}

function groupBounds(cfg: QuantConfig): number[] {
  const n = cfg.n
  let count: number
  if (cfg.mode === 'tensor') count = 1
  else if (cfg.mode === 'channel') count = Math.max(1, cfg.nChannels)
  else count = Math.max(1, Math.ceil(n / Math.max(1, cfg.groupSize)))

  const edges: number[] = [0]
  for (let g = 1; g <= count; g++) edges.push(Math.round((g / count) * n))
  return edges
}

function nearestNf4(x: number): number {
  let best = 0
  let bestD = Infinity
  for (let i = 0; i < NF4_CODEBOOK.length; i++) {
    const d = Math.abs(NF4_CODEBOOK[i] - x)
    if (d < bestD) {
      bestD = d
      best = i
    }
  }
  return best
}

export function quantize(cfg: QuantConfig): QuantResult {
  const values = makeWeights(cfg)
  const n = values.length
  const edges = groupBounds(cfg)
  const gCount = edges.length - 1

  const codes = new Array<number>(n).fill(0)
  const dequant = new Array<number>(n).fill(0)
  const scales = new Array<number>(gCount).fill(0)
  const zeros = new Array<number>(gCount).fill(0)

  const qmax = Math.pow(2, cfg.bits - 1) - 1
  const levels = Math.pow(2, cfg.bits)

  for (let g = 0; g < gCount; g++) {
    const lo = edges[g]
    const hi = edges[g + 1]
    if (hi <= lo) continue
    const slice = values.slice(lo, hi)

    if (cfg.scheme === 'nf4') {
      let maxAbs = 0
      for (const x of slice) maxAbs = Math.max(maxAbs, Math.abs(x))
      const scale = maxAbs > 0 ? maxAbs : 1
      scales[g] = scale
      zeros[g] = 0
      for (let i = lo; i < hi; i++) {
        const idx = nearestNf4(values[i] / scale)
        codes[i] = idx
        dequant[i] = NF4_CODEBOOK[idx] * scale
      }
      continue
    }

    if (cfg.scheme === 'sym') {
      let maxAbs = 0
      for (const x of slice) maxAbs = Math.max(maxAbs, Math.abs(x))
      const scale = maxAbs > 0 ? maxAbs / qmax : 1
      scales[g] = scale
      zeros[g] = 0
      for (let i = lo; i < hi; i++) {
        const c = Math.max(-qmax - 1, Math.min(qmax, Math.round(values[i] / scale)))
        codes[i] = c
        dequant[i] = c * scale
      }
      continue
    }

    // 非对称：把 [min, max] 线性映射到 [0, levels-1]，额外存一个 zero point
    let mn = Infinity
    let mx = -Infinity
    for (const x of slice) {
      mn = Math.min(mn, x)
      mx = Math.max(mx, x)
    }
    const scale = mx > mn ? (mx - mn) / (levels - 1) : 1
    const zero = Math.round(-mn / scale)
    scales[g] = scale
    zeros[g] = zero
    for (let i = lo; i < hi; i++) {
      const c = Math.max(0, Math.min(levels - 1, Math.round(values[i] / scale) + zero))
      codes[i] = c
      dequant[i] = (c - zero) * scale
    }
  }

  let se = 0
  let maxAbsErr = 0
  let mean = 0
  for (let i = 0; i < n; i++) {
    const e = values[i] - dequant[i]
    se += e * e
    maxAbsErr = Math.max(maxAbsErr, Math.abs(e))
    mean += values[i]
  }
  mean /= n
  let variance = 0
  for (let i = 0; i < n; i++) variance += (values[i] - mean) ** 2
  variance /= n

  const mse = se / n
  const snrDb = mse > 0 ? 10 * Math.log10(variance / mse) : 120

  const totalLevels = cfg.scheme === 'nf4' ? NF4_CODEBOOK.length : levels
  const usedLevels = new Set(codes).size

  const order = values
    .map((v, i) => ({ v, i }))
    .sort((a, b) => a.v - b.v)
    .map((x) => x.i)

  return {
    cfg,
    values,
    codes,
    dequant,
    scales,
    zeros,
    usedLevels,
    totalLevels,
    maxAbsErr,
    mse,
    snrDb,
    order,
  }
}

/* ------------------------------------------------------------------ */
/* 方案对比：同一份权重，不同量化策略各损失多少                         */
/* ------------------------------------------------------------------ */

export interface QuantComparison {
  id: string
  label: string
  labelEn: string
  snrDb: number
  mse: number
  usedLevels: number
  /** 每个权重占的字节数（含量化元数据） */
  bytesPerWeight: number
}

export function compareSchemes(base: QuantConfig): QuantComparison[] {
  const cases: { id: string; label: string; labelEn: string; patch: Partial<QuantConfig> }[] = [
    { id: 'fp16', label: '基准（不量化）', labelEn: 'baseline (no quant)', patch: { bits: 16, scheme: 'sym', mode: 'tensor' } },
    { id: 't8', label: 'INT8 per-tensor', labelEn: 'INT8 per-tensor', patch: { bits: 8, scheme: 'sym', mode: 'tensor' } },
    { id: 't4', label: 'INT4 per-tensor', labelEn: 'INT4 per-tensor', patch: { bits: 4, scheme: 'sym', mode: 'tensor' } },
    { id: 'c4', label: 'INT4 per-channel', labelEn: 'INT4 per-channel', patch: { bits: 4, scheme: 'sym', mode: 'channel' } },
    { id: 'g4', label: 'INT4 group=32', labelEn: 'INT4 group=32', patch: { bits: 4, scheme: 'sym', mode: 'group', groupSize: 32 } },
    { id: 'nf4', label: 'NF4 group=32', labelEn: 'NF4 group=32', patch: { bits: 4, scheme: 'nf4', mode: 'group', groupSize: 32 } },
    { id: 'a4', label: 'INT4 非对称 group=32', labelEn: 'INT4 asym group=32', patch: { bits: 4, scheme: 'asym', mode: 'group', groupSize: 32 } },
  ]

  return cases.map((c) => {
    const cfg = { ...base, ...c.patch }
    if (c.id === 'fp16') {
      // 不量化：fp16 的相对误差小到可以当基准，误差记 0
      return {
        id: c.id,
        label: c.label,
        labelEn: c.labelEn,
        snrDb: 120,
        mse: 0,
        usedLevels: 0,
        bytesPerWeight: 2,
      }
    }
    const r = quantize(cfg)
    return {
      id: c.id,
      label: c.label,
      labelEn: c.labelEn,
      snrDb: r.snrDb,
      mse: r.mse,
      usedLevels: r.usedLevels,
      bytesPerWeight: bytesPerWeight(cfg.bits, cfg.mode, cfg.groupSize),
    }
  })
}

/**
 * 每个权重占多少字节 —— 真正决定显存的是这个数，不是裸的 bits/8。
 *
 * group-wise 量化要额外存每组的 scale 和 zero point（各 fp16，共 4 字节）。
 * 所以 groupSize 越小精度越好，但元数据开销越大 —— 这是真实存在的取舍：
 *   group=128 → 4/8 + 4/128 = 0.531 B/权重
 *   group=32  → 4/8 + 4/32  = 0.625 B/权重（约多 18%）
 */
export function bytesPerWeight(bits: number, mode: QuantMode, groupSize: number): number {
  const raw = bits / 8
  if (mode === 'group' && groupSize > 0) return raw + 4 / groupSize
  if (mode === 'channel') return raw + 4 / 128 // 通道数通常远多于 128，近似忽略
  return raw
}

export function fmtGB(bytes: number): string {
  if (bytes >= 1024 ** 3) return (bytes / 1024 ** 3).toFixed(2) + ' GB'
  if (bytes >= 1024 ** 2) return (bytes / 1024 ** 2).toFixed(1) + ' MB'
  return (bytes / 1024).toFixed(1) + ' KB'
}

/** 常用精度下，某个规模的模型权重占多少显存 */
export function memoryTable(nParams: number, groupSize = 128) {
  return [
    { label: 'FP16', bytesPerWeight: 2, bits: 16, mode: 'tensor' as QuantMode },
    { label: 'INT8', bytesPerWeight: 1, bits: 8, mode: 'tensor' as QuantMode },
    { label: 'INT4 group=128', bits: 4, mode: 'group' as QuantMode, bytesPerWeight: bytesPerWeight(4, 'group', groupSize) },
    { label: 'NF4 group=64', bits: 4, mode: 'group' as QuantMode, bytesPerWeight: bytesPerWeight(4, 'group', 64) },
    { label: 'INT3 group=128', bits: 3, mode: 'group' as QuantMode, bytesPerWeight: bytesPerWeight(3, 'group', groupSize) },
    { label: 'INT2 group=64', bits: 2, mode: 'group' as QuantMode, bytesPerWeight: bytesPerWeight(2, 'group', 64) },
  ].map((f) => ({ ...f, bytes: f.bytesPerWeight * nParams }))
}

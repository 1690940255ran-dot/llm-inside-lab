/**
 * 长上下文外推（Context Extension）
 *
 * 一个只在 4K 上训过的模型，为什么直接喂 32K 就崩？
 * 这个文件把几何层面的原因算清楚，并且**真的**实现三种主流改写方案：
 *
 *   linear (PI)  位置统一除以 s —— 把所有波长的相位都压扁
 *   ntk          改 RoPE 的 base —— 高频维度几乎不动（保住局部精度），
 *                低频维度被拉伸 s 倍（换来长距离覆盖）
 *   yarn         在 ntk 的基础上分频段：波长 < L/β_fast 的维度完全不动，
 *                波长 > L/β_slow 的维度完整拉伸，中间平滑过渡
 *
 * 所有曲线都是真算的，核心量有三个：
 *   1. 每个维度的波长 λ_j = 2π / θ_j
 *   2. 距离 Δ 处的相位差 Δ·θ_j —— 超过 π 就发生别名（绕圈后分不清了）
 *   3. RoPE 的期望内积（注意力分数）(1/n)·Σ_j cos(Δ·θ_j)，
 *      这是随机单位 q/k 下 RoPE 点积的解析期望，不是编的
 *
 * 诚实声明：这里只做**几何**分析。真实模型外推失败还牵涉训练时学到的注意力模式，
 * 几何上"能区分"不等于模型"会用"，所以下面的几何上界会偏乐观 —— 页面里会写明。
 */
import { clamp } from './random'

export type RopeMethod = 'none' | 'linear' | 'ntk' | 'yarn'

export interface ContextConfig {
  /** 单个头的维度 d */
  dHead: number
  /** 训练时的上下文长度 */
  trainLen: number
  /** 想外推到多长 */
  targetLen: number
  /** RoPE base（θ 的基数），LLaMA 系默认 10000 */
  base: number
  method: RopeMethod
  /** YaRN 高频边界系数 */
  betaFast: number
  /** YaRN 低频边界系数 */
  betaSlow: number
  /** 是否施加 YaRN 的注意力温度补偿 1/(0.1·ln s + 1) */
  yarnTemp: boolean
}

export const DEFAULT_CONTEXT: ContextConfig = {
  dHead: 128,
  trainLen: 4096,
  targetLen: 32768,
  base: 10000,
  method: 'none',
  betaFast: 32,
  betaSlow: 1,
  yarnTemp: false,
}

export interface DimInfo {
  j: number
  /** 原始角频率 */
  theta0: number
  /** 改写后的角频率 */
  theta: number
  /** 原始波长 */
  lambda0: number
  /** 改写后的波长 */
  lambda: number
  /** 拉伸倍数 */
  stretch: number
  /** 在训练长度内转了几圈 —— 圈数多的维度负责精细位置 */
  turnsInTrain: number
  /** 相位差首次达到 π（Nyquist）的距离，超过它这个维度就分不清了 */
  aliasOnset: number
  /** yarn 的过渡系数：0 = 不缩放，1 = 完全缩放 */
  ramp: number
}

/** 按方法把 RoPE 每个维度对的角频率算出来 */
export function ropeDims(cfg: ContextConfig): DimInfo[] {
  const d = cfg.dHead
  const half = Math.floor(d / 2)
  const s = cfg.targetLen / cfg.trainLen
  const out: DimInfo[] = []

  // ntk：把 base 换成 base · s^(d/(d-2))
  const baseNtk = cfg.base * Math.pow(s, d / (d - 2))

  for (let j = 0; j < half; j++) {
    const exp = (2 * j) / d
    const theta0 = Math.pow(cfg.base, -exp)
    let theta = theta0
    let ramp = 0

    if (cfg.method === 'linear') {
      theta = theta0 / s
      ramp = 1
    } else if (cfg.method === 'ntk') {
      theta = Math.pow(baseNtk, -exp)
      // 实际拉伸倍数
      ramp = clamp(Math.log(theta0 / theta) / Math.log(s), 0, 1)
    } else if (cfg.method === 'yarn') {
      const lambda0 = (2 * Math.PI) / theta0
      const lo = cfg.trainLen / cfg.betaFast
      const hi = cfg.trainLen / cfg.betaSlow
      if (lambda0 <= lo) ramp = 0
      else if (lambda0 >= hi) ramp = 1
      else ramp = (lambda0 - lo) / (hi - lo)
      theta = theta0 / Math.pow(s, ramp)
    }

    const lambda0 = (2 * Math.PI) / theta0
    const lambda = (2 * Math.PI) / theta
    out.push({
      j,
      theta0,
      theta,
      lambda0,
      lambda,
      stretch: lambda / lambda0,
      turnsInTrain: (cfg.trainLen * theta) / (2 * Math.PI),
      aliasOnset: Math.PI / theta,
      ramp,
    })
  }
  return out
}

/**
 * RoPE 的期望注意力分数（相对距离 Δ）
 *
 * 对随机单位向量 q、k，加上 RoPE 之后点积的解析期望是 (1/n)·Σ_j cos(Δ·θ_j)。
 * 这就是「RoPE 远程衰减」那条性质，也是判断外推时注意力会不会失焦的直接依据：
 * 分数掉到 0 附近，意味着远距离位置的注意力 logit 和大家没差别（变成均匀分布）。
 */
export function ropeScore(dims: DimInfo[], delta: number): number {
  let s = 0
  for (const d of dims) s += Math.cos(delta * d.theta)
  return s / dims.length
}

/** 对数等距的距离网格：Δ 从 1 到 targetLen，取 n 个点 */
export function deltaGrid(targetLen: number, n = 160): number[] {
  const lo = Math.log(1)
  const hi = Math.log(Math.max(targetLen, 2))
  const out: number[] = []
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1)
    out.push(Math.exp(lo + t * (hi - lo)))
  }
  return out
}

export interface ContextReport {
  cfg: ContextConfig
  scale: number
  dims: DimInfo[]
  /** 距离 Δ 处的期望注意力分数 */
  score: { delta: number; value: number }[]
  /** 距离 Δ 处还没发生别名的维度占比 */
  resolvable: { delta: number; value: number }[]
  /** 几何上位置仍然唯一的最大距离 = 最长波长的一半 */
  geometricHorizon: number
  /** 训练长度时的分数（构成"分布内"基线） */
  scoreAtTrain: number
  /** 目标长度时的分数 */
  scoreAtTarget: number
  /** 有多少比例的维度在训练长度内转了不到一整圈（这些维度负责长距离） */
  lowFreqShare: number
  /**
   * 最小可分辨间隔 = π / θ_max（最短波长的半波长，Nyquist）。
   * 它衡量**局部精度**：间隔比它更近的两个位置，没有任何维度能区分。
   * 线性插值把所有 θ 都除以 s，这个值就跟着放大 s 倍 —— 这是 PI 牺牲的代价，
   * 也是 NTK / YaRN 坚持不动高频维度的原因。
   */
  minGap: number
}

export function analyzeContext(cfg: ContextConfig, gridN = 160): ContextReport {
  const dims = ropeDims(cfg)
  const s = cfg.targetLen / cfg.trainLen
  const grid = deltaGrid(Math.max(cfg.targetLen, cfg.trainLen), gridN)

  const score = grid.map((delta) => {
    let v = ropeScore(dims, delta)
    // YaRN 的注意力温度补偿：把 logits 乘 1/t，等价于把分数整体缩放
    if (cfg.yarnTemp && cfg.method === 'yarn') {
      const t = 0.1 * Math.log(s) + 1
      v = v / t
    }
    return { delta, value: v }
  })

  const resolvable = grid.map((delta) => {
    let alive = 0
    for (const d of dims) if (delta * d.theta < Math.PI) alive++
    return { delta, value: alive / dims.length }
  })

  const maxLambda = Math.max(...dims.map((d) => d.lambda))
  const geometricHorizon = maxLambda / 2
  const lowFreq = dims.filter((d) => d.turnsInTrain < 1).length
  const maxTheta = Math.max(...dims.map((d) => d.theta))

  return {
    cfg,
    scale: s,
    dims,
    score,
    resolvable,
    geometricHorizon,
    scoreAtTrain: ropeScore(dims, cfg.trainLen),
    scoreAtTarget: ropeScore(dims, cfg.targetLen),
    lowFreqShare: lowFreq / dims.length,
    minGap: Math.PI / maxTheta,
  }
}

/** 四种方法的分数曲线，画在同一张图上对比 */
export function compareMethods(
  cfg: ContextConfig,
  methods: RopeMethod[] = ['none', 'linear', 'ntk', 'yarn'],
): Record<string, { delta: number; value: number }[]> {
  const out: Record<string, { delta: number; value: number }[]> = {}
  for (const m of methods) {
    out[m] = analyzeContext({ ...cfg, method: m }).score
  }
  return out
}

/**
 * 相位缠绕轨迹：几个代表维度上，位置 t 的相位折到 [-π, π] 之后是多少。
 *
 * 这张图最容易看懂外推为什么崩：位置走远之后，相位开始来回折返，
 * 不同位置的相位变得一样 → 模型再也分不清它们在哪儿。
 */
export function phaseTrack(
  cfg: ContextConfig,
  jList: number[],
  positions: number[],
): { j: number; phase: number[]; theta: number }[] {
  const dims = ropeDims(cfg)
  const wrap = (x: number) => {
    const m = ((x + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI)
    return m - Math.PI
  }
  return jList
    .filter((j) => j >= 0 && j < dims.length)
    .map((j) => ({
      j,
      theta: dims[j].theta,
      phase: positions.map((t) => wrap(t * dims[j].theta)),
    }))
}

/** 每个维度改写前后的波长，画频率谱用 */
export function wavelengthSpectrum(cfg: ContextConfig): {
  dims: { j: number; base: number; scaled: number; ramp: number }[]
} {
  const dims = ropeDims(cfg)
  const half = Math.floor(cfg.dHead / 2)
  const ref = ropeDims({ ...cfg, method: 'none' })
  void half
  return {
    dims: dims.map((d, i) => ({
      j: d.j,
      base: ref[i].lambda0,
      scaled: d.lambda,
      ramp: d.ramp,
    })),
  }
}

export const METHOD_LABEL: Record<RopeMethod, { zh: string; en: string }> = {
  none: { zh: '不改（原始 RoPE）', en: 'none (vanilla RoPE)' },
  linear: { zh: '线性位置插值 PI', en: 'linear position interpolation' },
  ntk: { zh: 'NTK-aware（改 base）', en: 'NTK-aware (scale the base)' },
  yarn: { zh: 'YaRN（分频段）', en: 'YaRN (piecewise by wavelength)' },
}

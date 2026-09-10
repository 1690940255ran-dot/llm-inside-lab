/**
 * MoE（Mixture of Experts，稀疏专家）路由
 *
 * 这个文件不做"画图式模拟"：token 向量、专家路由向量、gate 概率、top-k 选择、
 * 容量丢 token、辅助损失（aux loss）、均衡偏置、专家专精更新，全部按真实公式算。
 *
 * ── 它复现的真实现象 ────────────────────────────────────────────────
 * 1. 路由会自己长出分工：专精力是 k-means 式的质心更新，专家被拉向
 *    "选中自己的那些 token"的质心，于是 token 按主题排好序后，热力图呈块对角。
 * 2. 但分工天然不均：语料话题频率本身就是 Zipf 分布，热门话题的 token 挤在
 *    少数几个专家上，剩下的专家分不到 token —— 这就是"负载塌缩"与"死专家"。
 * 3. 治它有两条流派，本文件把两条都实现了，而且**作用在同一组参数上**
 *    （每个专家一个 logit 偏置 b_e），所以两者的差别只来自信号本身：
 *
 *      · aux   —— Switch Transformer 的负载均衡辅助损失，用它的解析梯度
 *                 ∂L/∂b_e = n_E · f_e · avg_i[ p_e(1-p_e) ]
 *                 注意 f_e 是"分到的 token 占比"，死专家 f_e = 0
 *                 ⇒ **梯度恒等于 0，结构上不可能救回死专家**。
 *      · bias  —— DeepSeek-V3 的 aux-loss-free 均衡，不用梯度，用实测负载偏差
 *                 当反馈信号（一个 P 控制器）。死专家负载为 0，偏差最负，
 *                 偏置被顶到最高 ⇒ **能把死专家重新点活**。
 *
 * 这个对比不是我们编的：Switch（2021）用 aux loss，DeepSeek-V3（2024）在技术
 * 报告里明确说 aux loss 会损害模型质量、且均衡效果受限于梯度信号，于是改成了
 * 直接对偏置做负载反馈。两边的强度和代价在这个模块里都能亲手扫出来。
 *
 * 与站点其它模块一样，所有随机数都来自确定性种子，同样的配置永远得到同样的结果。
 */
import { gaussian, mulberry32, softmax } from './random'

/** 均衡力流派：none = 不治；aux = 辅助损失梯度；bias = 负载反馈偏置 */
export type BalanceMode = 'none' | 'aux' | 'bias'

export interface MoEConfig {
  /** 路由专家总数 */
  nExperts: number
  /** 每个 token 激活几个专家 */
  topK: number
  /** 训练用的 token 数 */
  nTokens: number
  /** 潜在主题数（真实语料的话题本来就不均匀，这是负载倾斜的根源） */
  nTopics: number
  /** 路由训练步数 */
  steps: number
  /** 路由学习率 */
  lr: number
  /** 均衡力强度。含义随 balanceMode 变：aux 是损失权重，bias 是反馈增益 */
  balanceWeight: number
  balanceMode: BalanceMode
  /** 容量因子，0 = 不限容量。1.0 = 恰好平均分配 */
  capacityFactor: number
  /** 路由温度，越小越"赢者通吃" */
  temperature: number
  /** token / 专家向量维度 */
  dim: number
  /** 主题内散度：0 = 每个主题的 token 完全一样，越大簇越糊。对应真实语料里话题的"纯度" */
  noise: number
  seed: number
}

export const DEFAULT_MOE: MoEConfig = {
  nExperts: 8,
  // 默认 top-1：Switch Transformer 用的就是 top-1，而且它是最早引入 aux loss 的模型。
  // top-1 的硬分配让负载倾斜最直观，切成 top-2 就能看到倾斜明显缓和。
  topK: 1,
  nTokens: 256,
  // 48 个长尾话题：话题数远多于专家数，8 个专家都分得到活干，
  // 倾斜完全来自"话题频率本身是 Zipf"这一层，而不是"专家比簇还多"。
  nTopics: 48,
  steps: 80,
  lr: 0.3,
  balanceWeight: 0,
  balanceMode: 'none',
  capacityFactor: 0,
  temperature: 0.35,
  dim: 32,
  noise: 0.6,
  seed: 20260910,
}

/**
 * 教学场景，一键切换就能把这套机制的正反面看一遍。
 *
 * 下面注释里的数字都是本文件在**默认配置下实测**的，由 tests/moe.test.ts 的断言守着：
 *   ① none       负载 26:24:34:65:22:34:28:23，倾斜 2.03 倍，偏置全 0
 *   ② aux w=1    倾斜 1.16，但偏置被推到 |b|=2.21 才勉强压住
 *   ③ bias w=2   倾斜 1.06，偏置 |b| 只要 0.30（比 aux 小一个数量级）
 *   ④ aux w=4    偏置推到 8.85，均衡力自己把一个专家打死（dead=1）、纯度 0.545→0.470
 *   ⑤ cap 1.25   倾斜 1.38，代价是 24 个 token 被完全丢弃
 *   ⑥ topics=3   话题数少于专家数 ⇒ 结构性死专家 3 个、倾斜 3.56，均衡力也只能在专家间倒腾
 */
export const MOE_SCENARIOS: { id: string; label: string; labelEn: string; patch: Partial<MoEConfig> }[] = [
  { id: 'collapse', label: '① 不治：负载倾斜', labelEn: '① untreated: imbalanced', patch: { balanceMode: 'none', balanceWeight: 0 } },
  { id: 'aux', label: '② aux loss 梯度', labelEn: '② aux-loss gradient', patch: { balanceMode: 'aux', balanceWeight: 1 } },
  { id: 'bias', label: '③ 负载反馈偏置', labelEn: '③ load-feedback bias', patch: { balanceMode: 'bias', balanceWeight: 2 } },
  { id: 'overkill', label: '④ aux 开过头', labelEn: '④ aux over-driven', patch: { balanceMode: 'aux', balanceWeight: 4 } },
  { id: 'capacity', label: '⑤ 容量因子 1.25', labelEn: '⑤ capacity factor 1.25', patch: { balanceMode: 'none', balanceWeight: 0, capacityFactor: 1.25 } },
  { id: 'starved', label: '⑥ 专家多于簇', labelEn: '⑥ more experts than clusters', patch: { balanceMode: 'none', balanceWeight: 0, nTopics: 3 } },
]

export interface BalancePoint {
  w: number
  imbalance: number
  entropy: number
  deadExperts: number
  /** 活跃专家（load > 0）的平均专精纯度 */
  livePurity: number
  /** 最终偏置的绝对值均值，用来看均衡力到底把参数推了多远 */
  biasNorm: number
}

/**
 * 扫一遍均衡力强度，看两个指标怎么此消彼长。
 *
 * 这是这个模块最值得看的一张图：两条流派的曲线形状完全不同 ——
 * aux 的熵很快撞顶、死专家却降不下去（梯度消失）；bias 能把死专家压到 0，
 * 但增益开太大时路由结构被偏置冲掉，专精纯度反而掉下去。
 */
export function sweepBalance(cfg: MoEConfig, weights: number[], mode?: BalanceMode): BalancePoint[] {
  return weights.map((w) => {
    const r = trainRouter({ ...cfg, balanceWeight: w, balanceMode: mode ?? (cfg.balanceMode === 'none' ? 'bias' : cfg.balanceMode) })
    const live = r.expertPurity.filter((_, e) => r.final.load[e] > 0)
    return {
      w,
      imbalance: r.final.imbalance,
      entropy: r.final.entropy,
      deadExperts: r.final.deadExperts,
      livePurity: live.length > 0 ? live.reduce((a, b) => a + b, 0) / live.length : 0,
      biasNorm: r.final.bias.reduce((a, b) => a + Math.abs(b), 0) / Math.max(1, r.final.bias.length),
    }
  })
}

export interface TokenSet {
  tokens: number[][]
  topicOf: number[]
  /** Zipf 分布下的主题频率（百分比） */
  topicShare: number[]
  dim: number
}

function normalize(v: number[]): number[] {
  let n = 0
  for (const x of v) n += x * x
  n = Math.sqrt(n)
  return n === 0 ? v.slice() : v.map((x) => x / n)
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
  const den = Math.sqrt(na) * Math.sqrt(nb)
  return den === 0 ? 0 : dot / den
}

/**
 * 造一组有潜在结构的 token。
 *
 * 关键设计：主题频率服从 Zipf 分布，而不是均匀分布。
 * 真实语料里话题热度本来就高度不均（新闻里"体育"比"考古"多得多），
 * 这正是 MoE 负载倾斜的第一层原因 —— 不是路由器的锅，是数据本来就偏。
 */
export function makeTokenSet(
  n: number,
  dim: number,
  nTopics: number,
  seed: number,
  /** 主题内噪声，相对主题中心的模长（中心是单位向量，所以 0.3 就是 30% 的散度） */
  noise = 0.3,
): TokenSet {
  const rnd = mulberry32(seed)

  const centers: number[][] = []
  for (let t = 0; t < nTopics; t++) {
    centers.push(normalize(Array.from({ length: dim }, () => gaussian(rnd))))
  }

  // Zipf：w_t ∝ 1/(t+1)
  const w = Array.from({ length: nTopics }, (_, i) => 1 / (i + 1))
  const total = w.reduce((a, b) => a + b, 0)
  const cum: number[] = []
  let acc = 0
  for (const x of w) {
    acc += x / total
    cum.push(acc)
  }

  const tokens: number[][] = []
  const topicOf: number[] = []
  for (let i = 0; i < n; i++) {
    const u = rnd()
    let t = cum.findIndex((c) => u <= c)
    if (t < 0) t = nTopics - 1
    topicOf.push(t)
    // 分量级噪声要乘 1/√dim，否则模长会变成 noise·√dim，把单位长度的主题中心整个淹掉
    const s = noise / Math.sqrt(dim)
    tokens.push(normalize(centers[t].map((x) => x + gaussian(rnd) * s)))
  }

  const counts = new Array(nTopics).fill(0)
  for (const t of topicOf) counts[t]++
  const topicShare = counts.map((c) => (c / n) * 100)

  return { tokens, topicOf, topicShare, dim }
}

/**
 * 路由器的全部可训练参数。
 *
 * 刻意把「路由」和「专家计算」分成两组参数：真实 MoE 里 router 是一个独立的
 * 线性层，expert 权重是另一套 FFN 参数，两者互不干扰。
 *
 * vectors 负责「谁擅长什么」（方向，决定 token 落到谁头上）；
 * bias 负责「该多分多少」（每专家一个标量，是均衡力的唯一作用点）。
 * 两者参数空间分离，均衡力才不会去破坏学好的分工。
 */
export interface RouterState {
  /** [nExperts][dim] 路由向量，真实 MoE 里是 router 线性层的第 e 行 */
  vectors: number[][]
  /** [nExperts] logit 偏置，DeepSeek-V3 的 aux-loss-free 均衡就加在这里 */
  bias: number[]
}

export function makeRouterState(nExperts: number, dim: number, seed: number): RouterState {
  const rnd = mulberry32(seed ^ 0x9e3779b9)
  return {
    vectors: Array.from({ length: nExperts }, () =>
      normalize(Array.from({ length: dim }, () => gaussian(rnd))),
    ),
    bias: new Array(nExperts).fill(0),
  }
}

export interface RouteStep {
  step: number
  /** [nTokens][nExperts] gate 概率（softmax 之后，top-k 之前） */
  prob: number[][]
  /** [nTokens] 实际落到的专家（容量满时可能少于 topK） */
  assign: number[][]
  /** [nTokens] 在落到的专家上重新归一化后的权重（gate 概率带权，用来加权求和专家输出） */
  weight: number[][]
  /** [nTokens] 是否被完全丢弃（所有候选专家都满了） */
  dropped: boolean[]
  /** 分到的专家数少于 topK 的 token 数 —— 这才是容量不足的真实代价 */
  partialDropped: number
  /** [nExperts] 分到的 token 数 */
  load: number[]
  /** [nExperts] 平均 gate 概率 P_e */
  probMass: number[]
  /**
   * [nExperts] 辅助损失对偏置的梯度 ∂L/∂b_e = n_E · f_e · avg_i[ p_e(1-p_e) ]
   *
   * 注意 f_e 用的是"分到的 token 占比"，所以**死专家的这一项精确等于 0**：
   * aux loss 对死专家没有任何直接作用力。这条曲线画出来就是 aux 流派
   * 曲线又抖又非单调、开大了还会自己造出新死专家的根本原因。
   */
  auxGrad: number[]
  /** [nExperts] 当前偏置快照 */
  bias: number[]
  /** 负载均衡辅助损失 N·Σ f_e·P_e，完美均衡时下界为 1 */
  auxLoss: number
  /** max(load) / mean(load)，完美均衡 = 1 */
  imbalance: number
  /** 负载为 0 的专家数 */
  deadExperts: number
  droppedCount: number
  /** 归一化到 [0,1] 的负载熵，1 = 完全均衡 */
  entropy: number
}

function routingStats(load: number[], tokens: number, nExperts: number, probMass: number[]) {
  const placed = load.reduce((a, b) => a + b, 0)
  const mean = placed / nExperts
  const imbalance = mean > 0 ? Math.max(...load) / mean : 0
  const deadExperts = load.filter((c) => c === 0).length

  let H = 0
  if (placed > 0) {
    for (const c of load) {
      if (c > 0) {
        const p = c / placed
        H -= p * Math.log(p)
      }
    }
  }
  const entropy = nExperts > 1 ? H / Math.log(nExperts) : 1

  const f = load.map((c) => c / tokens)
  const auxLoss = nExperts * f.reduce((a, fe, e) => a + fe * probMass[e], 0)

  return { imbalance, deadExperts, entropy, auxLoss }
}

/**
 * 辅助损失对偏置的解析梯度。
 *
 * L = n_E · Σ_e f_e · P_e ，其中 P_e = (1/n)Σ_i p_i,e ，∂p_i,e/∂b_e = p_i,e(1-p_i,e)。
 * f_e 按 Switch 的做法当常数（stop-gradient），于是
 *
 *     ∂L/∂b_e = n_E · f_e · (1/n) · Σ_i p_i,e(1-p_i,e)
 *
 * f_e = 0 时整项为 0 —— 死专家对 aux loss 完全"免疫"。
 */
export function auxGradient(load: number[], prob: number[][], nExperts: number): number[] {
  const n = prob.length
  const out = new Array(nExperts).fill(0)
  if (n === 0) return out
  for (let e = 0; e < nExperts; e++) {
    let acc = 0
    for (let i = 0; i < n; i++) {
      const p = prob[i][e]
      acc += p * (1 - p)
    }
    out[e] = (nExperts * (load[e] / n) * acc) / n
  }
  return out
}

/**
 * 单步路由：logit = cos(token, router_e)/T + b_e → softmax → top-k → 容量限制
 *
 * 超容量的 token 直接丢弃，与 GShard/Switch 的 capacity dropping 一致。
 */
export function routeStep(
  tokens: number[][],
  state: RouterState,
  cfg: MoEConfig,
  step: number,
): RouteStep {
  const nE = cfg.nExperts
  const n = tokens.length

  const prob: number[][] = []
  for (let i = 0; i < n; i++) {
    const logits: number[] = []
    for (let e = 0; e < nE; e++) {
      logits.push(cosine(tokens[i], state.vectors[e]) / cfg.temperature + state.bias[e])
    }
    prob.push(softmax(logits))
  }

  const capacity =
    cfg.capacityFactor > 0
      ? Math.max(1, Math.ceil(((n * cfg.topK) / nE) * cfg.capacityFactor))
      : Number.POSITIVE_INFINITY

  const load = new Array(nE).fill(0)
  const assign: number[][] = []
  const weight: number[][] = []
  const dropped: boolean[] = []

  for (let i = 0; i < n; i++) {
    const order = Array.from({ length: nE }, (_, e) => e).sort(
      (a, b) => prob[i][b] - prob[i][a] || a - b,
    )
    const picked: number[] = []
    for (let k = 0; k < Math.min(cfg.topK, nE); k++) {
      const e = order[k]
      if (load[e] < capacity) {
        picked.push(e)
        load[e]++
      }
    }
    // 落在哪些专家（索引）和它们的 gate 概率要分开：
    // 权重来自"概率"这一组，之前直接拿索引去归一化会得到完全错误的加权。
    const raw = picked.map((e) => prob[i][e])
    const s = raw.reduce((a, b) => a + b, 0)
    assign.push(picked)
    weight.push(raw.map((x) => (s > 0 ? x / s : 0)))
    dropped.push(picked.length === 0)
  }

  const probMass = new Array(nE).fill(0)
  for (let i = 0; i < n; i++) for (let e = 0; e < nE; e++) probMass[e] += prob[i][e]
  for (let e = 0; e < nE; e++) probMass[e] /= n

  const stats = routingStats(load, n, nE, probMass)

  return {
    step,
    prob,
    assign,
    weight,
    dropped,
    load,
    probMass,
    auxGrad: auxGradient(load, prob, nE),
    bias: state.bias.slice(),
    partialDropped: assign.filter((a) => a.length < Math.min(cfg.topK, nE)).length,
    droppedCount: dropped.filter(Boolean).length,
    ...stats,
  }
}

/**
 * 一步参数更新，两股力作用在**不同的参数**上：
 *
 * ① 专精力（作用在路由向量 vectors 上）
 *    k-means 式的质心更新：router_e 朝"选中自己的那些 token"的质心靠。
 *    梯度 ∝ (token - cos·router_e)，也就是切向分量。这一步会让专家越来越像
 *    自己负责的 token，于是更常被选中 —— 路由的自我强化就来自这里。
 *
 * ② 均衡力（作用在偏置 bias 上，两种流派二选一）
 *    · aux：对辅助损失 L = n_E·Σ f_e·P_e 求偏置的解析梯度。
 *           f_e 当常数（stop-gradient，Switch 的做法），于是
 *           ∂L/∂b_e = n_E · f_e · avg_i[ p_e(1-p_e) ]
 *           死专家 f_e = 0 ⇒ 梯度恰好为 0，永远推不动它。
 *    · bias：不看梯度，直接看实测负载偏差 dev_e = load_e/mean - 1，
 *           b_e -= gain · dev_e。死专家 dev_e = -1（最负），偏置被顶到最高，
 *           于是它重新被抽中 ⇒ 这一步就是 DeepSeek-V3 的思路（它用 sign，我们按比例）。
 */
export function updateRouter(
  tokens: number[][],
  state: RouterState,
  cfg: MoEConfig,
  route: RouteStep,
): RouterState {
  const nE = cfg.nExperts
  const n = tokens.length
  const vectors = state.vectors.map((e) => e.slice())
  const bias = state.bias.slice()

  // ① 专精力
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < route.assign[i].length; k++) {
      const e = route.assign[i][k]
      const g = route.weight[i][k]
      const c = cosine(tokens[i], vectors[e])
      for (let d = 0; d < cfg.dim; d++) {
        vectors[e][d] += cfg.lr * g * (tokens[i][d] - c * vectors[e][d])
      }
    }
  }

  // ② 均衡力
  if (cfg.balanceWeight > 0 && cfg.balanceMode !== 'none') {
    if (cfg.balanceMode === 'aux') {
      for (let e = 0; e < nE; e++) {
        bias[e] -= cfg.lr * cfg.balanceWeight * route.auxGrad[e]
      }
    } else {
      const placed = route.load.reduce((a, b) => a + b, 0)
      const mean = placed / nE
      for (let e = 0; e < nE; e++) {
        const dev = mean > 0 ? route.load[e] / mean - 1 : 0
        bias[e] -= (cfg.lr * cfg.balanceWeight * dev) / nE
      }
    }
  }

  return { vectors: vectors.map(normalize), bias }
}

export interface MoEReport {
  cfg: MoEConfig
  tokenSet: TokenSet
  /** 逐步的统计，用来画训练过程曲线 */
  history: RouteStep[]
  /** 训练结束时的路由快照 */
  final: RouteStep
  /** 按主题排好序的 token 下标 —— 热力图按这个顺序显示才能看出块对角结构 */
  orderByTopic: number[]
  /** token i 的最终 gate 概率（已按 orderByTopic 重排） */
  sortedProb: number[][]
  sortedTopic: number[]
  /** 专家 → 主题 的归属矩阵 [nExperts][nTopics]，元素是该专家收到的该主题 token 数 */
  expertTopic: number[][]
  /** 每个专家最主要的主题（-1 = 没有明确偏好） */
  expertTopTopic: number[]
  /** 每个专家的"专精纯度"：主主题占比，1/nTopics = 完全没专精 */
  expertPurity: number[]
}

export function trainRouter(cfg: MoEConfig): MoEReport {
  const tokenSet = makeTokenSet(cfg.nTokens, cfg.dim, cfg.nTopics, cfg.seed, cfg.noise)
  let state = makeRouterState(cfg.nExperts, cfg.dim, cfg.seed)

  const history: RouteStep[] = []
  for (let s = 0; s < cfg.steps; s++) {
    const route = routeStep(tokenSet.tokens, state, cfg, s)
    history.push(route)
    state = updateRouter(tokenSet.tokens, state, cfg, route)
  }
  const final = routeStep(tokenSet.tokens, state, cfg, cfg.steps)

  const orderByTopic = tokenSet.topicOf
    .map((t, i) => ({ t, i }))
    .sort((a, b) => a.t - b.t || a.i - b.i)
    .map((x) => x.i)

  const expertTopic: number[][] = Array.from({ length: cfg.nExperts }, () =>
    new Array(cfg.nTopics).fill(0),
  )
  for (let i = 0; i < cfg.nTokens; i++) {
    for (const e of final.assign[i]) expertTopic[e][tokenSet.topicOf[i]]++
  }

  const expertTopTopic: number[] = []
  const expertPurity: number[] = []
  for (let e = 0; e < cfg.nExperts; e++) {
    const row = expertTopic[e]
    const sum = row.reduce((a, b) => a + b, 0)
    if (sum === 0) {
      expertTopTopic.push(-1)
      expertPurity.push(0)
      continue
    }
    let best = 0
    for (let t = 1; t < row.length; t++) if (row[t] > row[best]) best = t
    expertTopTopic.push(best)
    expertPurity.push(row[best] / sum)
  }

  return {
    cfg,
    tokenSet,
    history,
    final,
    orderByTopic,
    sortedProb: orderByTopic.map((i) => final.prob[i]),
    sortedTopic: orderByTopic.map((i) => tokenSet.topicOf[i]),
    expertTopic,
    expertTopTopic,
    expertPurity,
  }
}

/* ------------------------------------------------------------------ */
/* 参数量预算：MoE 最反直觉的地方 —— 总参数和激活参数差一个数量级      */
/* ------------------------------------------------------------------ */

export interface MoEAnatomy {
  /** 词嵌入 + 每层注意力 + LayerNorm，这部分每个 token 都要走 */
  denseParams: number
  /** 单个专家的 FFN 参数（gate / up / down 三个矩阵） */
  expertParams: number
  totalParams: number
  activeParams: number
  /** 激活占比，越小越"稀疏" */
  activeRatio: number
  /** 专家参数占总参数的比例 */
  expertShare: number
  /** 同样激活参数量的稠密模型，总参数就是 activeParams —— 用它当参照 */
  denseEquivalent: number
}

export function moeAnatomy(cfg: {
  dModel: number
  nLayers: number
  ffnDim: number
  nExperts: number
  nShared: number
  topK: number
  vocab: number
}): MoEAnatomy {
  const attn = 4 * cfg.dModel * cfg.dModel
  const norms = 2 * cfg.dModel
  const densePerLayer = attn + norms
  const denseParams = cfg.vocab * cfg.dModel + cfg.nLayers * densePerLayer
  const expertParams = 3 * cfg.dModel * cfg.ffnDim

  const expertTotal = (cfg.nExperts + cfg.nShared) * expertParams
  const expertActive = (cfg.topK + cfg.nShared) * expertParams

  const totalExperts = cfg.nLayers * expertTotal
  const activeExperts = cfg.nLayers * expertActive

  const totalParams = denseParams + totalExperts
  const activeParams = denseParams + activeExperts

  return {
    denseParams,
    expertParams,
    totalParams,
    activeParams,
    activeRatio: totalParams > 0 ? activeParams / totalParams : 0,
    expertShare: totalParams > 0 ? totalExperts / totalParams : 0,
    denseEquivalent: activeParams,
  }
}

export interface MoEArchPreset {
  id: string
  name: string
  nameEn: string
  note: string
  noteEn: string
  /** 官方公布的参数量（十亿），用来核对我们的公式算得对不对 */
  published: { totalB: number; activeB: number } | null
  arch: {
    dModel: number
    nLayers: number
    ffnDim: number
    nExperts: number
    nShared: number
    topK: number
    vocab: number
  }
}

export const MOE_ARCH_PRESETS: MoEArchPreset[] = [
  {
    id: 'mixtral',
    name: 'Mixtral 8x7B',
    nameEn: 'Mixtral 8x7B',
    note: '开源 MoE 的普及者，8 选 2',
    noteEn: 'The model that popularized open MoE, top-2 of 8',
    published: { totalB: 46.7, activeB: 12.9 },
    arch: {
      dModel: 4096,
      nLayers: 32,
      ffnDim: 14336,
      nExperts: 8,
      nShared: 0,
      topK: 2,
      vocab: 32000,
    },
  },
  {
    id: 'qwen3',
    name: 'Qwen3-30B-A3B',
    nameEn: 'Qwen3-30B-A3B',
    note: '128 选 8，细粒度专家',
    noteEn: 'top-8 of 128 — fine-grained experts',
    published: { totalB: 30.5, activeB: 3.3 },
    arch: {
      dModel: 2048,
      nLayers: 48,
      ffnDim: 768,
      nExperts: 128,
      nShared: 0,
      topK: 8,
      vocab: 151936,
    },
  },
  {
    id: 'tiny',
    name: '教学用小型 MoE',
    nameEn: 'Tiny teaching MoE',
    note: '小到能一眼看完的比例',
    noteEn: 'Small enough to read the ratios at a glance',
    published: null,
    arch: {
      dModel: 512,
      nLayers: 8,
      ffnDim: 1408,
      nExperts: 8,
      nShared: 0,
      topK: 2,
      vocab: 8000,
    },
  },
]

export function fmtParams(n: number): string {
  if (n >= 1e12) return (n / 1e12).toFixed(2) + ' T'
  if (n >= 1e9) return (n / 1e9).toFixed(2) + ' B'
  if (n >= 1e6) return (n / 1e6).toFixed(1) + ' M'
  if (n >= 1e3) return (n / 1e3).toFixed(1) + ' K'
  return String(Math.round(n))
}

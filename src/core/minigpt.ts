/**
 * 迷你 GPT：一个「真的能被训练起来」的字符级 Transformer
 *
 * 前面八个模块都是「拿一个已经训好的模型，看它内部长什么样」。
 * 这个模块反过来：**从随机权重开始，在浏览器里真的跑反向传播**，
 * 让 loss 真的降下来，让注意力图真的从噪声变成有结构。
 *
 * 为什么不用现成的自动微分库？
 *   1. 站点全程零依赖（transformers.js 只用来加载真模型），不想为了教学引一个框架；
 *   2. 更重要的是「算法是真算的」这条原则 —— 前向和反向都手写，才敢说每个数字都是算出来的；
 *   3. 手写反向还有个教学好处：梯度从哪条路径流回、哪一项对应哪一步，代码里一目了然。
 *
 * 为了能在 JS 单线程里跑得动，全部用 Float32Array 扁平存储 + 行优先，
 * 并且所有缓冲区按容量预分配、循环中复用，不制造 GC 压力。
 *
 * 结构就是标准 GPT（pre-norm，权重共享）：
 *
 *   x = tokEmb[ids] + posEmb[pos]
 *   每层：
 *     a  = LN1(x);  o = MHA(a);  xMid = x + Wo·o
 *     f  = LN2(xMid); h = GELU(f·W1);  x = xMid + W2·h
 *   xf = LNf(x)
 *   logits = xf · tokEmbᵀ        ← weight tying，GPT-2 起就这么做
 *
 * 反向传播的关键约定：残差流是「多路汇入」的，所以 dx 一律用**累加**而不是赋值 ——
 * 这是手写反向最容易出错的地方，也是这里所有 `+=` 存在的原因。
 */
import { gaussian, mulberry32 } from './random'

/* ------------------------------------------------------------------ *
 * 配置
 * ------------------------------------------------------------------ */

export interface ModelConfig {
  nLayers: number
  dModel: number
  nHeads: number
  /** FFN 中间层维度，通常 4 × dModel */
  dFF: number
  /** 训练时一次看多少个字符（上下文长度） */
  blockSize: number
}

export interface TrainConfig {
  steps: number
  batchSize: number
  lr: number
  weightDecay: number
  /** 梯度范数裁剪阈值，0 表示不裁 */
  gradClip: number
  seed: number
  /** 前多少步做线性 warmup（从 ~0 升到 lr） */
  warmup: number
  /** 余弦衰减的终点倍率：结束时降到 lr × 这个值 */
  minLrRatio: number
}

/**
 * paramBreakdown() 返回的分组键。
 *
 * 键 → 显示文案的映射属于界面，由消费模块的 i18n 字典负责；
 * core 层只产出键，不产出文字 —— 否则切到英文界面时这一块会露出中文。
 */
export type ParamGroupKey = 'tokEmb' | 'posEmb' | 'attn' | 'ffn' | 'ln'

/**
 * 默认模型：2 层 d=32、约 1.86 万参数（`18176 + 32 × 词表大小`，随语料浮动）。
 *
 * 尺寸是实测出来的：单步纯训练约 14 ms，而站点里每 20 步还要在 12 个验证 batch 上
 * 评估一次，所以界面上「每步耗时」实测约 20~22 ms —— 1200 步约 25~30 秒。
 * 而这一步数足够把一份周期较短的语料从验证 loss 2.5834 压到 0.0137（困惑度 1.014）——
 * 「几十秒里亲眼看到它学会说话」是这一页的全部意义，所以尺寸必须服务于这个目标。
 */
export const DEFAULT_MODEL: ModelConfig = {
  nLayers: 2,
  dModel: 32,
  nHeads: 4,
  dFF: 64,
  blockSize: 32,
}

/**
 * 默认训练配置。几个数字都不是惯例，而是这份小模型实测出来的：
 *
 * · lr = 3e-2 比大模型的 3e-4 / 6e-4 大两个数量级。原因是 Adam 对每个参数都用
 *   几乎相同的步长，而这个模型的权重量级只有 0.02 —— 3e-4 的步长相对权重只有 1.5%，
 *   几百步内根本推不动；3e-2 才让「几十秒学会」成立。这本身就是一条值得讲的结论：
 *   **大模型的超参配方不能直接搬到小模型上。**
 * · warmup 占 10%、余弦衰减到 2%：不加这两项，实测 loss 会先降到 0.5 再反弹到 1.6，
 *   典型的「后期步长相对权重尺度太大，把刚学会的东西砸掉」。
 */
export const DEFAULT_TRAIN: TrainConfig = {
  steps: 1200,
  batchSize: 8,
  lr: 0.03,
  weightDecay: 0.01,
  gradClip: 1,
  seed: 123,
  warmup: 120,
  minLrRatio: 0.02,
}

/**
 * 内置语料。
 *
 * 三条语料是有意按「一个周期有多长」排序的 —— 这是实测下来决定学习速度的头号因素：
 * 如果重复周期能整个塞进 blockSize（上下文窗口），模型只要学会「往前看一个周期」
 * 就能把 loss 打到很低；周期一旦超出窗口，模型就只能去硬学局部 n-gram，明显更吃力。
 * 这正好和「⑨ 长上下文外推」那一页接上。
 *
 * `period` 的定义是**一个完整重复单元的长度（含它末尾的分隔符）**。
 * 这三个数字都被 tests/minigpt.test.ts 里的最小周期断言盯着 ——
 * 早期版本这里手写成了 13 / 23 / 49，全都和真实周期差 1~3，
 * 而「周期」在本模块里是要拿去做结论的量，错一个数就会把结论带偏，所以现在用测试锁死。
 */
export const PRESET_CORPORA: {
  id: string
  label: string
  labelEn: string
  /** 一个重复单元的字符数，用来解释「周期 vs 窗口」 */
  period: number
  text: string
}[] = [
  {
    id: 'zh',
    label: '中文短句（推荐）',
    labelEn: 'Chinese phrase (recommended)',
    // '注意力就是让模型看哪里。' = 12 字
    period: 12,
    text: Array.from({ length: 60 }, () => '注意力就是让模型看哪里。').join(''),
  },
  {
    id: 'en',
    label: '英文短句',
    labelEn: 'English phrase',
    // 'the cat sat on the mat. ' = 24 字符（含句末空格）
    period: 24,
    text: Array.from({ length: 60 }, () => 'the cat sat on the mat. ').join(''),
  },
  {
    id: 'hard',
    label: '长周期（更难）',
    labelEn: 'Longer period (harder)',
    // 'the cat sat on the mat, and the dog ran to the log.' = 51 字符，另有 1 个换行
    period: 52,
    text: Array.from({ length: 40 }, () => 'the cat sat on the mat, and the dog ran to the log.').join('\n'),
  },
]

/**
 * 学习率调度：线性 warmup + 余弦衰减。
 *
 * 为什么要 warmup？刚初始化时梯度方向和尺度都很不可靠，一上来就用大 lr
 * 容易把模型推进一个坏区域（loss 先降后翘就是典型症状）。warmup 让前几十步
 * 先用小步探一探，再加速；余弦衰减则在后期收小步长，让 loss 稳定地压下去。
 */
export function lrAt(step: number, cfg: TrainConfig): number {
  const warm = Math.max(1, cfg.warmup)
  if (step < warm) return cfg.lr * ((step + 1) / warm)
  const t = Math.min(1, (step - warm) / Math.max(1, cfg.steps - warm))
  const minLr = cfg.lr * cfg.minLrRatio
  return minLr + (cfg.lr - minLr) * 0.5 * (1 + Math.cos(Math.PI * t))
}

/* ------------------------------------------------------------------ *
 * 字符词表
 * ------------------------------------------------------------------ */

export interface CharVocab {
  /** id → 字符；最后一个始终是 UNK */
  itos: string[]
  stoi: Map<string, number>
  unk: number
  /** 被截断掉（映射到 UNK）的字符种类数 */
  droppedKinds: number
}

/**
 * 从语料里建字符词表。
 *
 * 字符级是最适合「小到能在浏览器里训」的粒度：词表只有几十到一两百，
 * 而权重共享的输出投影和词嵌入是同一块矩阵，所以词表大一点也不会吃掉全部参数。
 * 这正好和「①分词」模块形成对照：那边讲子词怎么把词表压小，这边干脆不分子词。
 */
export function buildVocab(text: string, maxSize = 96): CharVocab {
  const freq = new Map<string, number>()
  for (const ch of text) freq.set(ch, (freq.get(ch) ?? 0) + 1)

  const sorted = [...freq.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
  const keep = maxSize > 0 ? sorted.slice(0, Math.max(1, maxSize - 1)) : sorted

  const itos = keep.map((e) => e[0])
  const stoi = new Map<string, number>()
  itos.forEach((c, i) => stoi.set(c, i))
  itos.push('\u0000UNK')
  return {
    itos,
    stoi,
    unk: itos.length - 1,
    droppedKinds: Math.max(0, sorted.length - keep.length),
  }
}

/** 展示用：把 UNK 的占位符换成人能看懂的符号 */
export function vocabChar(v: CharVocab, id: number): string {
  if (id === v.unk) return '␀'
  return v.itos[id]
}

/** 把整份文本编码成 id 序列 */
export function encodeCorpus(text: string, v: CharVocab): Int32Array {
  const out = new Int32Array(text.length)
  for (let i = 0; i < text.length; i++) {
    const id = v.stoi.get(text[i])
    out[i] = id === undefined ? v.unk : id
  }
  return out
}

/** 从编码后的语料里随机抽一个 batch：每条样本是长度为 blockSize 的连续片段 */
export function sampleBatch(
  data: Int32Array,
  batchSize: number,
  blockSize: number,
  rnd: () => number,
): { inputs: Int32Array; targets: Int32Array } {
  const inputs = new Int32Array(batchSize * blockSize)
  const targets = new Int32Array(batchSize * blockSize)
  const span = data.length - blockSize - 1
  for (let b = 0; b < batchSize; b++) {
    const start = span > 0 ? Math.floor(rnd() * span) : 0
    for (let t = 0; t < blockSize; t++) {
      inputs[b * blockSize + t] = data[start + t]
      targets[b * blockSize + t] = data[start + t + 1] ?? data[start + t]
    }
  }
  return { inputs, targets }
}

/* ------------------------------------------------------------------ *
 * 参数张量
 * ------------------------------------------------------------------ */

interface Tensor {
  name: string
  size: number
  data: Float32Array
  grad: Float32Array
  /** AdamW 的一阶 / 二阶动量 */
  m: Float32Array
  v: Float32Array
  /** 是否参与权重衰减（1 维的 bias 和 LayerNorm 的 gain 不参与，这是标准做法） */
  decay: boolean
}

/** 一层的激活缓存：反向要用，所以必须逐层分开存，不能共用缓冲区 */
interface LayerCache {
  xIn: Float32Array
  a1: Float32Array
  xh1: Float32Array
  inv1: Float32Array
  q: Float32Array
  k: Float32Array
  v: Float32Array
  o: Float32Array
  attnOut: Float32Array
  xMid: Float32Array
  a2: Float32Array
  xh2: Float32Array
  inv2: Float32Array
  hpre: Float32Array
  hact: Float32Array
  ffnOut: Float32Array
  /** [B, H, T, T] 注意力概率 */
  prob: Float32Array
}

const LN_EPS = 1e-5
const GELU_C = 0.7978845608
const GELU_A = 0.044715

/** tanh 近似的 GELU，及与之严格配套的导数（导数不配套，梯度校验就一定过不了） */
export function gelu(v: number): number {
  return 0.5 * v * (1 + Math.tanh(GELU_C * (v + GELU_A * v * v * v)))
}
export function dGelu(v: number): number {
  const u = GELU_C * (v + GELU_A * v * v * v)
  const t = Math.tanh(u)
  const du = GELU_C * (1 + 3 * GELU_A * v * v)
  return 0.5 * (1 + t) + 0.5 * v * (1 - t * t) * du
}

export interface StepResult {
  loss: number
  /** 裁剪前的全局梯度范数 */
  gradNorm: number
  /** 裁剪后实际用到的梯度范数 */
  clippedNorm: number
  lr: number
}

export class MiniGPT {
  readonly vocabSize: number
  readonly cfg: ModelConfig
  readonly batchSize: number
  readonly d: number
  readonly hd: number

  readonly tensors: Tensor[] = []
  private byName = new Map<string, Tensor>()

  private tokEmb!: Tensor
  private posEmb!: Tensor
  private lnfg!: Tensor
  private lnfb!: Tensor
  private p: {
    ln1g: Tensor; ln1b: Tensor
    Wq: Tensor; bq: Tensor
    Wk: Tensor; bk: Tensor
    Wv: Tensor; bv: Tensor
    Wo: Tensor; bo: Tensor
    ln2g: Tensor; ln2b: Tensor
    W1: Tensor; b1: Tensor
    W2: Tensor; b2: Tensor
  }[] = []

  private caches: LayerCache[] = []
  private capBT = 0

  /** 残差流与全局临时量 */
  private g = new Map<string, Float32Array>()
  /** 训练步计数（Adam 偏差修正 + warmup 用） */
  step = 0

  constructor(init: { vocabSize: number; model: ModelConfig; batchSize: number; seed: number }) {
    const { model, seed } = init
    if (model.dModel % model.nHeads !== 0) {
      throw new Error(`dModel(${model.dModel}) 必须能被 nHeads(${model.nHeads}) 整除`)
    }
    if (model.blockSize < 2) throw new Error('blockSize 至少要 2')
    this.vocabSize = init.vocabSize
    this.cfg = { ...model }
    this.batchSize = Math.max(1, init.batchSize)
    this.d = model.dModel
    this.hd = model.dModel / model.nHeads

    const rnd = mulberry32((seed ^ 0x9e3779b9) >>> 0)
    const { d } = this
    const V = this.vocabSize
    const ff = model.dFF

    // GPT-2 的初始化尺度：~N(0, 0.02)；残差分支的输出投影再按 1/sqrt(2L) 缩小，
    // 这一步不是仪式，它直接决定 2 层以上还能不能训得动。
    const std = 0.02
    const projScale = 1 / Math.sqrt(2 * model.nLayers)

    /**
     * 初始权重按 ±3σ 截断。
     *
     * 为什么要多这一步？Box-Muller 偶尔会吐出 6σ 级别的值（u 很接近 0 时
     * sqrt(-2·ln u) 能到 6 以上）。在只有一万八千参数的模型里，一个大到 6σ 的初始权重
     * 就足以把训练推进一个坏盆地 —— 实测有种子因此完全学不动（loss 卡在 2.2，
     * 而其他种子都能降到 0.4 以下）。截断到 ±3σ 把这种运气成分压掉。
     */
    const gauss = () => {
      const g = gaussian(rnd)
      return g > 3 ? 3 : g < -3 ? -3 : g
    }

    const mk = (name: string, size: number, scale: number, decay = true) => {
      const t: Tensor = {
        name,
        size,
        data: new Float32Array(size),
        grad: new Float32Array(size),
        m: new Float32Array(size),
        v: new Float32Array(size),
        decay,
      }
      for (let i = 0; i < size; i++) t.data[i] = gauss() * scale
      this.tensors.push(t)
      this.byName.set(name, t)
      return t
    }

    this.tokEmb = mk('tokEmb', V * d, std)
    this.posEmb = mk('posEmb', model.blockSize * d, 0.01)

    /** LayerNorm 的 gain 必须初始化为 1（不是 N(0,1)）：
     *  随机 gain 会让一部分通道开天窗、另一部分放大，小模型上很容易训不稳。 */
    const mkGain = (name: string, size: number) => {
      const t = mk(name, size, 0, false)
      t.data.fill(1)
      return t
    }

    for (let l = 0; l < model.nLayers; l++) {
      const lp = `L${l}.`
      this.p.push({
        ln1g: mkGain(lp + 'ln1.g', d), ln1b: mk(lp + 'ln1.b', d, 0, false),
        Wq: mk(lp + 'Wq', d * d, std), bq: mk(lp + 'bq', d, 0, false),
        Wk: mk(lp + 'Wk', d * d, std), bk: mk(lp + 'bk', d, 0, false),
        Wv: mk(lp + 'Wv', d * d, std), bv: mk(lp + 'bv', d, 0, false),
        Wo: mk(lp + 'Wo', d * d, std * projScale), bo: mk(lp + 'bo', d, 0, false),
        ln2g: mkGain(lp + 'ln2.g', d), ln2b: mk(lp + 'ln2.b', d, 0, false),
        W1: mk(lp + 'W1', d * ff, std), b1: mk(lp + 'b1', ff, 0, false),
        W2: mk(lp + 'W2', ff * d, std * projScale), b2: mk(lp + 'b2', d, 0, false),
      })
    }
    this.lnfg = mkGain('lnf.g', d)
    this.lnfb = mk('lnf.b', d, 0, false)

    this.alloc(this.batchSize * model.blockSize)
  }

  /** 参数量。词嵌入与输出投影共享同一份权重，所以这里没有重复计数。 */
  paramCount(): number {
    return this.tensors.reduce((a, t) => a + t.size, 0)
  }

  /**
   * 每层参数量明细，用来画「参数都花在哪了」。
   *
   * 返回的是**分组键**而不是显示文案：core 层不应该产出界面文字，
   * 否则切到英文界面时这一块会露出中文（这个 bug 真的发生过）。
   * 键 → 文案的映射由模块的 i18n 字典负责。
   */
  paramBreakdown(): { key: ParamGroupKey; size: number }[] {
    const groups: { key: ParamGroupKey; size: number }[] = []
    const push = (key: ParamGroupKey, test: (n: string) => boolean) => {
      const size = this.tensors.filter((t) => test(t.name)).reduce((a, t) => a + t.size, 0)
      if (size > 0) groups.push({ key, size })
    }
    push('tokEmb', (n) => n === 'tokEmb')
    push('posEmb', (n) => n === 'posEmb')
    push('attn', (n) => /\.(Wq|Wk|Wv|Wo|bq|bk|bv|bo)$/.test(n))
    push('ffn', (n) => /\.(W1|W2|b1|b2)$/.test(n))
    push('ln', (n) => /ln/.test(n))
    return groups.sort((a, b) => b.size - a.size)
  }

  /**
   * 每个 token 前向大约多少 FLOPs（一次乘法 + 一次加法算 2）。
   * 这是业界估算训练算力的常用式子，模块里会拿它去乘 token 数、对比真模型的量级。
   */
  flopsPerToken(): number {
    const { nLayers, dModel, dFF } = this.cfg
    const perLayer = 4 * dModel * dModel + 2 * dModel * dFF
    const output = 2 * dModel * this.vocabSize
    return 2 * (nLayers * perLayer + output)
  }

  weightStats(): { name: string; std: number; absMax: number; size: number }[] {
    return this.tensors.map((t) => {
      let sum = 0
      let mx = 0
      for (let i = 0; i < t.size; i++) {
        sum += t.data[i] * t.data[i]
        const a = Math.abs(t.data[i])
        if (a > mx) mx = a
      }
      return { name: t.name, std: Math.sqrt(sum / Math.max(1, t.size)), absMax: mx, size: t.size }
    })
  }

  /* ---------------- 缓冲区 ---------------- */

  private alloc(BT: number) {
    if (BT <= this.capBT) return
    this.capBT = BT
    const d = this.d
    const ff = this.cfg.dFF
    const H = this.cfg.nHeads
    const T = this.cfg.blockSize

    for (const n of ['x', 'xf', 'xhF', 'dx', 'dx2', 'dA', 'dB', 'dC']) {
      this.g.set(n, new Float32Array(BT * d))
    }
    this.g.set('invF', new Float32Array(BT))
    for (const n of ['dFF', 'dFF2']) this.g.set(n, new Float32Array(BT * ff))
    for (const n of ['logits', 'probs', 'dlogits']) this.g.set(n, new Float32Array(BT * this.vocabSize))

    this.caches = []
    for (let l = 0; l < this.cfg.nLayers; l++) {
      const F = (n: number) => new Float32Array(n)
      this.caches.push({
        xIn: F(BT * d), a1: F(BT * d), xh1: F(BT * d), inv1: F(BT),
        q: F(BT * d), k: F(BT * d), v: F(BT * d),
        o: F(BT * d), attnOut: F(BT * d), xMid: F(BT * d),
        a2: F(BT * d), xh2: F(BT * d), inv2: F(BT),
        hpre: F(BT * ff), hact: F(BT * ff), ffnOut: F(BT * d),
        prob: F(BT * H * T),
      })
    }
  }

  private gb(n: string, len: number): Float32Array {
    let a = this.g.get(n)
    if (!a || a.length < len) {
      a = new Float32Array(len)
      this.g.set(n, a)
    }
    return a
  }

  /* ---------------- 前向 ---------------- */

  /** 只跑前向，把 logits 留在缓冲区里（loss / 采样 / 注意力取数都走它） */
  forward(inputs: Int32Array): void {
    const { d, hd } = this
    const { nHeads: H, nLayers: L, blockSize: T, dFF: ff } = this.cfg
    const V = this.vocabSize
    const BT = inputs.length
    const B = Math.ceil(BT / T)
    if (BT > this.capBT) this.alloc(BT)

    const tokEmb = this.tokEmb.data
    const posEmb = this.posEmb.data
    const x = this.gb('x', BT * d)

    for (let i = 0; i < BT; i++) {
      const t = i % T
      const ti = inputs[i] * d
      const pi = t * d
      const xi = i * d
      for (let k = 0; k < d; k++) x[xi + k] = tokEmb[ti + k] + posEmb[pi + k]
    }

    for (let l = 0; l < L; l++) {
      const P = this.p[l]
      const C = this.caches[l]
      C.xIn.set(x.subarray(0, BT * d))

      lnForward(x, P.ln1g.data, P.ln1b.data, BT, d, C.xh1, C.a1, C.inv1)
      linear(C.a1, P.Wq.data, P.bq.data, C.q, BT, d, d)
      linear(C.a1, P.Wk.data, P.bk.data, C.k, BT, d, d)
      linear(C.a1, P.Wv.data, P.bv.data, C.v, BT, d, d)
      attentionForward(C.q, C.k, C.v, C.o, B, T, H, hd, C.prob)
      linear(C.o, P.Wo.data, P.bo.data, C.attnOut, BT, d, d)

      for (let i = 0; i < BT * d; i++) x[i] += C.attnOut[i]
      C.xMid.set(x.subarray(0, BT * d))

      lnForward(x, P.ln2g.data, P.ln2b.data, BT, d, C.xh2, C.a2, C.inv2)
      linear(C.a2, P.W1.data, P.b1.data, C.hpre, BT, d, ff)
      for (let i = 0; i < BT * ff; i++) C.hact[i] = gelu(C.hpre[i])
      linear(C.hact, P.W2.data, P.b2.data, C.ffnOut, BT, d, d)
      for (let i = 0; i < BT * d; i++) x[i] += C.ffnOut[i]
    }

    // 最终 LN + 输出投影（权重共享）
    const xf = this.gb('xf', BT * d)
    lnForward(x, this.lnfg.data, this.lnfb.data, BT, d, this.gb('xhF', BT * d), xf, this.gb('invF', BT))

    const logits = this.gb('logits', BT * V)
    for (let i = 0; i < BT; i++) {
      const xi = i * d
      const li = i * V
      for (let j = 0; j < V; j++) {
        const tj = j * d
        let s = 0
        for (let k = 0; k < d; k++) s += xf[xi + k] * tokEmb[tj + k]
        logits[li + j] = s
      }
    }
  }

  /** 前向 + 交叉熵 loss */
  loss(inputs: Int32Array, targets: Int32Array): number {
    this.forward(inputs)
    const V = this.vocabSize
    const BT = inputs.length
    const logits = this.gb('logits', BT * V)
    const probs = this.gb('probs', BT * V)
    let loss = 0
    for (let i = 0; i < BT; i++) {
      const li = i * V
      let mx = -Infinity
      for (let j = 0; j < V; j++) if (logits[li + j] > mx) mx = logits[li + j]
      let sum = 0
      for (let j = 0; j < V; j++) {
        const e = Math.exp(logits[li + j] - mx)
        probs[li + j] = e
        sum += e
      }
      const inv = 1 / (sum || 1)
      for (let j = 0; j < V; j++) probs[li + j] *= inv
      loss += -Math.log(Math.max(probs[li + targets[i]], 1e-12))
    }
    return loss / BT
  }

  /* ---------------- 反向 ---------------- */

  backward(inputs: Int32Array, targets: Int32Array): void {
    const { d, hd } = this
    const { nHeads: H, nLayers: L, blockSize: T, dFF: ff } = this.cfg
    const V = this.vocabSize
    const BT = inputs.length
    const B = Math.ceil(BT / T)
    const probs = this.gb('probs', BT * V)
    const xf = this.gb('xf', BT * d)

    // —— dlogits = (p − onehot) / BT ——
    const dlogits = this.gb('dlogits', BT * V)
    const invBT = 1 / BT
    for (let i = 0; i < BT; i++) {
      const li = i * V
      for (let j = 0; j < V; j++) {
        dlogits[li + j] = (probs[li + j] - (j === targets[i] ? 1 : 0)) * invBT
      }
    }

    // —— 输出投影（与词嵌入共享）：dTokEmb += dlogitsᵀ·xf，dxf = dlogits·tokEmb ——
    const te = this.tokEmb
    for (let i = 0; i < BT; i++) {
      const li = i * V
      const xi = i * d
      for (let j = 0; j < V; j++) {
        const g = dlogits[li + j]
        if (g === 0) continue
        const tj = j * d
        for (let k = 0; k < d; k++) te.grad[tj + k] += g * xf[xi + k]
      }
    }
    const dxf = this.gb('dx2', BT * d)
    for (let i = 0; i < BT * d; i++) dxf[i] = 0
    for (let i = 0; i < BT; i++) {
      const li = i * V
      const xi = i * d
      for (let j = 0; j < V; j++) {
        const g = dlogits[li + j]
        if (g === 0) continue
        const tj = j * d
        for (let k = 0; k < d; k++) dxf[xi + k] += g * te.data[tj + k]
      }
    }

    // —— 最终 LN 反向 → dx（残差流梯度起点） ——
    const dx = this.gb('dx', BT * d)
    for (let i = 0; i < BT * d; i++) dx[i] = 0
    lnBackward(dxf, this.gb('xhF', BT * d), this.gb('invF', BT), this.lnfg.data, this.lnfg.grad, this.lnfb.grad, BT, d, this.gb('dA', BT * d), dx)

    const dMid = this.gb('dB', BT * d)
    const dAttn = this.gb('dC', BT * d)

    for (let l = L - 1; l >= 0; l--) {
      const P = this.p[l]
      const C = this.caches[l]

      // ===== FFN 反向 =====
      // x = xMid + ffnOut  → 两条路都要给 dffnOut
      zeroGrad(P.W2); zeroGrad(P.b2)
      const dHact = this.gb('dFF', BT * ff)
      for (let i = 0; i < BT * ff; i++) dHact[i] = 0
      linearBackward(dx, C.hact, P.W2.data, P.W2.grad, P.b2.grad, dHact, BT, ff, d)

      const dHpre = this.gb('dFF2', BT * ff)
      for (let i = 0; i < BT * ff; i++) dHpre[i] = dHact[i] * dGelu(C.hpre[i])

      zeroGrad(P.W1); zeroGrad(P.b1)
      const dA2 = this.gb('dA', BT * d)
      for (let i = 0; i < BT * d; i++) dA2[i] = 0
      linearBackward(dHpre, C.a2, P.W1.data, P.W1.grad, P.b1.grad, dA2, BT, d, ff)

      // LN2 反向往 xMid 累加（这里是残差汇入点，必须 +=）
      for (let i = 0; i < BT * d; i++) dMid[i] = dx[i]
      lnBackward(dA2, C.xh2, C.inv2, P.ln2g.data, P.ln2g.grad, P.ln2b.grad, BT, d, this.gb('dFF3', BT * d), dMid)

      // ===== 注意力反向 =====
      // xMid = xIn + attnOut，所以 dattnOut 与 dxIn 都拿到 dMid 的那一份
      zeroGrad(P.Wo); zeroGrad(P.bo)
      for (let i = 0; i < BT * d; i++) dAttn[i] = 0
      linearBackward(dMid, C.o, P.Wo.data, P.Wo.grad, P.bo.grad, dAttn, BT, d, d)

      const dQ = this.gb('dq', BT * d)
      const dK = this.gb('dk', BT * d)
      const dV = this.gb('dv', BT * d)
      for (let i = 0; i < BT * d; i++) {
        dQ[i] = 0; dK[i] = 0; dV[i] = 0
      }
      attentionBackward(dAttn, C.q, C.k, C.v, C.prob, dQ, dK, dV, B, T, H, hd)

      zeroGrad(P.Wq); zeroGrad(P.bq); zeroGrad(P.Wk); zeroGrad(P.bk)
      zeroGrad(P.Wv); zeroGrad(P.bv)
      const dA1 = this.gb('dA', BT * d)
      for (let i = 0; i < BT * d; i++) dA1[i] = 0
      linearBackward(dQ, C.a1, P.Wq.data, P.Wq.grad, P.bq.grad, dA1, BT, d, d)
      linearBackward(dK, C.a1, P.Wk.data, P.Wk.grad, P.bk.grad, dA1, BT, d, d)
      linearBackward(dV, C.a1, P.Wv.data, P.Wv.grad, P.bv.grad, dA1, BT, d, d)

      const dxIn = this.gb('dx2', BT * d)
      for (let i = 0; i < BT * d; i++) dxIn[i] = dMid[i]
      lnBackward(dA1, C.xh1, C.inv1, P.ln1g.data, P.ln1g.grad, P.ln1b.grad, BT, d, this.gb('dFF4', BT * d), dxIn)

      // 交给上一层的梯度 = 本层输入的梯度
      for (let i = 0; i < BT * d; i++) dx[i] = dxIn[i]
    }

    // —— 嵌入反向：x = tokEmb[id] + posEmb[t] ——
    const dPos = this.posEmb.grad
    const dTok = this.tokEmb.grad
    for (let i = 0; i < BT; i++) {
      const t = i % T
      const xi = i * d
      const pi = t * d
      const ti = inputs[i] * d
      for (let k = 0; k < d; k++) {
        dTok[ti + k] += dx[xi + k]
        dPos[pi + k] += dx[xi + k]
      }
    }
  }

  /** 一次完整的训练步：前向 → 反向 → AdamW。返回本步指标。 */
  trainStep(inputs: Int32Array, targets: Int32Array, lr: number, weightDecay: number, gradClip: number): StepResult {
    const loss = this.loss(inputs, targets)
    this.zeroGrads()
    this.backward(inputs, targets)
    const r = this.adamwStep(lr, weightDecay, gradClip)
    return { loss, gradNorm: r.gradNorm, clippedNorm: r.clippedNorm, lr }
  }

  /* ---------------- 优化器 ---------------- */

  zeroGrads(): void {
    for (const t of this.tensors) t.grad.fill(0)
  }

  gradNorm(): number {
    let s = 0
    for (const t of this.tensors) {
      for (let i = 0; i < t.size; i++) s += t.grad[i] * t.grad[i]
    }
    return Math.sqrt(s)
  }

  /**
   * AdamW：一阶动量 + 二阶动量 + 偏差修正 + 解耦权重衰减。
   *
   * 为什么非得用它？裸 SGD 在几百步内几乎不动，而 Adam 的自适应步长让
   * 「几十秒里就能看出 loss 在降」成为可能 —— 这是本模块能塞进浏览器的前提。
   */
  adamwStep(lr: number, weightDecay: number, gradClip: number): { gradNorm: number; clippedNorm: number } {
    const raw = this.gradNorm()
    const scale = gradClip > 0 && raw > gradClip ? gradClip / raw : 1
    const b1 = 0.9
    const b2 = 0.95
    const eps = 1e-8
    this.step += 1
    const bc1 = 1 - Math.pow(b1, this.step)
    const bc2 = 1 - Math.pow(b2, this.step)

    for (const T of this.tensors) {
      const { data, grad, m, v } = T
      const wd = T.decay ? weightDecay : 0
      for (let i = 0; i < T.size; i++) {
        const g = grad[i] * scale
        m[i] = b1 * m[i] + (1 - b1) * g
        v[i] = b2 * v[i] + (1 - b2) * g * g
        const mh = m[i] / bc1
        const vh = v[i] / bc2
        data[i] -= lr * (mh / (Math.sqrt(vh) + eps) + wd * data[i])
      }
    }
    return { gradNorm: raw, clippedNorm: raw * scale }
  }

  /* ---------------- 采样 ---------------- */

  /**
   * 自回归采样 —— 唯一能让用户「亲眼看到训练进度」的地方：
   * 第 0 步吐出来的是乱码，第 600 步吐出来的是像话的字符序列。
   */
  sample(prompt: number[], n: number, temperature: number, seed: number, topK = 0): number[] {
    const { blockSize: T } = this.cfg
    const V = this.vocabSize
    const ids = prompt.length ? prompt.slice() : [0]
    const rnd = mulberry32(seed >>> 0)
    const out: number[] = []
    const seq = new Int32Array(T)
    const pr = new Float64Array(V)

    for (let s = 0; s < n; s++) {
      const start = Math.max(0, ids.length - T)
      const len = ids.length - start
      for (let i = 0; i < T; i++) {
        // 左侧用首个 id 补齐，这样「当前位置」永远是下标 T-1
        seq[i] = i < T - len ? ids[start] : ids[start + (i - (T - len))]
      }
      this.forward(seq)
      const logits = this.gb('logits', T * V)
      const li = (T - 1) * V

      const temp = Math.max(1e-4, temperature)
      let mx = -Infinity
      for (let j = 0; j < V; j++) if (logits[li + j] > mx) mx = logits[li + j]
      let sum = 0
      for (let j = 0; j < V; j++) {
        const e = Math.exp((logits[li + j] - mx) / temp)
        pr[j] = e
        sum += e
      }
      for (let j = 0; j < V; j++) pr[j] /= sum

      if (topK > 0 && topK < V) {
        // top-k：只留概率最高的 k 个，其余置零后重新归一化
        const arr = Array.from({ length: V }, (_, j) => j).sort((a, b) => pr[b] - pr[a])
        const keep = new Set(arr.slice(0, topK))
        let total = 0
        for (let j = 0; j < V; j++) {
          if (keep.has(j)) total += pr[j]
          else pr[j] = 0
        }
        for (let j = 0; j < V; j++) pr[j] = total > 0 ? pr[j] / total : keep.has(j) ? 1 / topK : 0
      }

      let r = rnd()
      let pick = V - 1
      for (let j = 0; j < V; j++) {
        r -= pr[j]
        if (r <= 0) {
          pick = j
          break
        }
      }
      ids.push(pick)
      out.push(pick)
    }
    return out
  }

  /* ---------------- 可视化取数 ---------------- */

  /**
   * 对给定 id 序列跑一次前向，返回第 layer 层每个头的 T×T 注意力矩阵。
   * 训练前后各调一次，就能看出「注意力真的从噪声变成了有结构的东西」。
   */
  attnSnapshot(ids: number[], layer = 0): number[][][] {
    const { blockSize: T, nHeads: H } = this.cfg
    const seq = new Int32Array(T)
    for (let t = 0; t < T; t++) seq[t] = ids[Math.min(t, ids.length - 1)] ?? 0
    this.forward(seq)
    const C = this.caches[Math.max(0, Math.min(this.cfg.nLayers - 1, layer))]
    const maps: number[][][] = []
    for (let h = 0; h < H; h++) {
      const m: number[][] = []
      for (let t = 0; t < T; t++) {
        const row: number[] = []
        for (let j = 0; j < T; j++) row.push(C.prob[(h * T + t) * T + j])
        m.push(row)
      }
      maps.push(m)
    }
    return maps
  }

  /** 每个头在「关注前一个位置」上的平均权重 —— 训练中最直观的一个涌现指标 */
  prevTokenScore(layer: number): number[] {
    const C = this.caches[Math.max(0, Math.min(this.cfg.nLayers - 1, layer))]
    const { blockSize: T, nHeads: H } = this.cfg
    const out: number[] = []
    for (let h = 0; h < H; h++) {
      let s = 0
      let n = 0
      for (let t = 1; t < T; t++) {
        s += C.prob[(h * T + t) * T + (t - 1)]
        n++
      }
      out.push(n ? s / n : 0)
    }
    return out
  }
}

/* ------------------------------------------------------------------ *
 * 无状态的张量运算
 * 全部行优先扁平：A[M,K] · B[K,N] → C[M,N]
 * ------------------------------------------------------------------ */

function zeroGrad(t: Tensor) {
  t.grad.fill(0)
}

/** y = x·W + b，W 形状 [K,N] */
export function linear(
  x: Float32Array,
  W: Float32Array,
  bias: Float32Array | null,
  y: Float32Array,
  M: number,
  K: number,
  N: number,
) {
  for (let i = 0; i < M; i++) {
    const xi = i * K
    const yi = i * N
    for (let j = 0; j < N; j++) y[yi + j] = bias ? bias[j] : 0
    for (let p = 0; p < K; p++) {
      const a = x[xi + p]
      if (a === 0) continue
      const wp = p * N
      for (let j = 0; j < N; j++) y[yi + j] += a * W[wp + j]
    }
  }
}

/**
 * 线性层反向：
 *   dW += xᵀ·dy     db += Σ dy     dx += dy·Wᵀ
 * dx 用累加而非赋值 —— 同一个 x 可能同时喂给 Wq/Wk/Wv 三个投影。
 */
export function linearBackward(
  dy: Float32Array,
  x: Float32Array,
  W: Float32Array,
  dW: Float32Array,
  db: Float32Array | null,
  dx: Float32Array,
  M: number,
  K: number,
  N: number,
) {
  for (let i = 0; i < M; i++) {
    const xi = i * K
    const yi = i * N
    if (db) {
      for (let j = 0; j < N; j++) db[j] += dy[yi + j]
    }
    for (let p = 0; p < K; p++) {
      const a = x[xi + p]
      const wp = p * N
      let acc = 0
      for (let j = 0; j < N; j++) {
        const g = dy[yi + j]
        if (g === 0) continue
        dW[wp + j] += a * g
        acc += g * W[wp + j]
      }
      dx[xi + p] += acc
    }
  }
}

/** LayerNorm 前向。xhOut 是归一化中间量（反向要用），yOut 是最终输出，invStdOut 是每行的 1/σ。 */
export function lnForward(
  x: Float32Array,
  g: Float32Array,
  b: Float32Array,
  M: number,
  D: number,
  xhOut: Float32Array,
  yOut: Float32Array,
  invStdOut?: Float32Array,
): Float32Array {
  for (let i = 0; i < M; i++) {
    const off = i * D
    let mean = 0
    for (let k = 0; k < D; k++) mean += x[off + k]
    mean /= D
    let varr = 0
    for (let k = 0; k < D; k++) {
      const dv = x[off + k] - mean
      varr += dv * dv
    }
    varr /= D
    const inv = 1 / Math.sqrt(varr + LN_EPS)
    if (invStdOut) invStdOut[i] = inv
    for (let k = 0; k < D; k++) {
      const xh = (x[off + k] - mean) * inv
      xhOut[off + k] = xh
      yOut[off + k] = xh * g[k] + b[k]
    }
  }
  return yOut
}

/**
 * LayerNorm 反向：dg/db 累加，dx 累加（残差分支要叠上去）。
 *
 * 这里必须乘回 1/σ，这是手写 LN 反向最经典的漏项：
 *   y = (x−μ)/σ · g + b   ⇒   ∂y_i/∂x_j = (1/σ)(δ_ij − 1/D − u_i·u_j/D)
 * 少了 1/σ，梯度就整体错一个尺度 —— 训练还能跑，但学到的方向是错的。
 */
export function lnBackward(
  dy: Float32Array,
  xh: Float32Array,
  invStd: Float32Array,
  g: Float32Array,
  dg: Float32Array,
  db: Float32Array,
  M: number,
  D: number,
  dxh: Float32Array,
  dx: Float32Array,
) {
  for (let i = 0; i < M; i++) {
    const off = i * D
    const inv = invStd[i] ?? 1
    let sumDxh = 0
    let sumDxhXh = 0
    for (let k = 0; k < D; k++) {
      const d = dy[off + k]
      const xhv = xh[off + k]
      const dxhv = d * g[k]
      dxh[off + k] = dxhv
      sumDxh += dxhv
      sumDxhXh += dxhv * xhv
    }
    const invD = 1 / D
    const meanDxh = sumDxh * invD
    const meanDxhXh = sumDxhXh * invD
    for (let k = 0; k < D; k++) {
      dx[off + k] += inv * (dxh[off + k] - meanDxh - xh[off + k] * meanDxhXh)
      dg[k] += dy[off + k] * xh[off + k]
      db[k] += dy[off + k]
    }
  }
}

/** 因果多头注意力前向，概率写进 prob [B,H,T,T] */
export function attentionForward(
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  out: Float32Array,
  B: number,
  T: number,
  H: number,
  hd: number,
  prob: Float32Array,
) {
  const d = H * hd
  const scale = 1 / Math.sqrt(hd)
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      const off = h * hd
      const pBase = (b * H + h) * T * T
      for (let t = 0; t < T; t++) {
        const qi = (b * T + t) * d + off
        const rowOff = pBase + t * T
        let mx = -Infinity
        for (let j = 0; j <= t; j++) {
          const ki = (b * T + j) * d + off
          let s = 0
          for (let c = 0; c < hd; c++) s += q[qi + c] * k[ki + c]
          s *= scale
          prob[rowOff + j] = s
          if (s > mx) mx = s
        }
        for (let j = t + 1; j < T; j++) prob[rowOff + j] = 0
        let sum = 0
        for (let j = 0; j <= t; j++) {
          const e = Math.exp(prob[rowOff + j] - mx)
          prob[rowOff + j] = e
          sum += e
        }
        const inv = sum > 0 ? 1 / sum : 1
        for (let j = 0; j <= t; j++) prob[rowOff + j] *= inv

        const oi = (b * T + t) * d + off
        for (let c = 0; c < hd; c++) out[oi + c] = 0
        for (let j = 0; j <= t; j++) {
          const p = prob[rowOff + j]
          if (p === 0) continue
          const vi = (b * T + j) * d + off
          for (let c = 0; c < hd; c++) out[oi + c] += p * v[vi + c]
        }
      }
    }
  }
}

/** 因果多头注意力反向：dQ/dK/dV 全部累加 */
export function attentionBackward(
  dOut: Float32Array,
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  prob: Float32Array,
  dQ: Float32Array,
  dK: Float32Array,
  dV: Float32Array,
  B: number,
  T: number,
  H: number,
  hd: number,
) {
  const d = H * hd
  const scale = 1 / Math.sqrt(hd)
  const dS = new Float32Array(T)
  const dP = new Float32Array(T)
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      const off = h * hd
      const pBase = (b * H + h) * T * T
      for (let t = 0; t < T; t++) {
        const rowOff = pBase + t * T
        const oi = (b * T + t) * d + off
        for (let j = 0; j <= t; j++) {
          const vi = (b * T + j) * d + off
          let s = 0
          for (let c = 0; c < hd; c++) s += dOut[oi + c] * v[vi + c]
          dP[j] = s
        }
        for (let j = 0; j <= t; j++) {
          const p = prob[rowOff + j]
          if (p === 0) continue
          const vi = (b * T + j) * d + off
          for (let c = 0; c < hd; c++) dV[vi + c] += p * dOut[oi + c]
        }
        // softmax 反向
        let dot = 0
        for (let j = 0; j <= t; j++) dot += prob[rowOff + j] * dP[j]
        for (let j = 0; j <= t; j++) dS[j] = prob[rowOff + j] * (dP[j] - dot)

        const qi = (b * T + t) * d + off
        for (let j = 0; j <= t; j++) {
          const ds = dS[j] * scale
          if (ds === 0) continue
          const ki = (b * T + j) * d + off
          for (let c = 0; c < hd; c++) {
            dQ[qi + c] += ds * k[ki + c]
            dK[ki + c] += ds * q[qi + c]
          }
        }
      }
    }
  }
}

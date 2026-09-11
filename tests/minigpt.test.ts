/**
 * 迷你 GPT 的单元测试
 *
 * 这个文件里最重要的不是「训练能降 loss」那几条，而是**梯度正确性检验**。
 * 手写反向传播如果错了，训练照样"能跑"、loss 也常常会下降 —— 只是下降的原因不对，
 * 而且这种错误在页面上肉眼完全看不出来。
 *
 * 但检验方法本身有个坑：float32 前向的网络里，逐参数有限差分几乎不可用 ——
 * 小梯度的信号（Δloss ~ 1e-7）会被 float32 的舍入噪声（~1e-6）直接淹没，
 * 于是会报出大量"解析梯度=0 而数值梯度≠0"的假阳性。我们就在这上面栽过一次。
 *
 * 所以主判据换成**下降方向判据**：
 *   若 g 真是 ∇L，那么对足够小的 η 有  L(θ − η·g) ≈ L(θ) − η·|g|²
 * 右边是 |g|² 量级的信号，比噪声高好几个数量级，任何符号或系数错误都会立刻暴露。
 */
import { describe, expect, it } from 'vitest'
import {
  DEFAULT_MODEL,
  DEFAULT_TRAIN,
  MiniGPT,
  PRESET_CORPORA,
  attentionBackward,
  attentionForward,
  buildVocab,
  encodeCorpus,
  lnBackward,
  lnForward,
  lrAt,
  sampleBatch,
  vocabChar,
  type ModelConfig,
  type ParamGroupKey,
  type TrainConfig,
} from '../src/core/minigpt'
import { mulberry32 } from '../src/core/random'

function fixedData(V: number, BT: number, seed: number) {
  const inputs = new Int32Array(BT)
  const targets = new Int32Array(BT)
  const rnd = mulberry32(seed)
  for (let i = 0; i < BT; i++) {
    inputs[i] = Math.floor(rnd() * V)
    targets[i] = Math.floor(rnd() * V)
  }
  return { inputs, targets }
}

/** 用「沿 −g 走一步的实际下降量」检验梯度，返回 实际/预测 的比值 */
function descentRatio(nLayers: number): number {
  const V = 11
  const model: ModelConfig = { nLayers, dModel: 16, nHeads: 2, dFF: 32, blockSize: 8 }
  const m = new MiniGPT({ vocabSize: V, model, batchSize: 2, seed: 4242 })
  const { inputs, targets } = fixedData(V, 2 * model.blockSize, 99)

  const l0 = m.loss(inputs, targets)
  m.zeroGrads()
  m.backward(inputs, targets)
  let g2 = 0
  for (const t of m.tensors) for (let i = 0; i < t.size; i++) g2 += t.grad[i] * t.grad[i]

  const eta = 1e-6
  for (const t of m.tensors) for (let i = 0; i < t.size; i++) t.data[i] -= eta * t.grad[i]
  const l1 = m.loss(inputs, targets)
  return (l1 - l0) / (-eta * g2)
}

describe('字符词表', () => {
  it('按词频取前 N 个，其余折成 UNK', () => {
    const v = buildVocab('aaabbbccde', 4) // 上限 4 → 留 3 个字符 + UNK
    expect(v.itos.length).toBe(4)
    expect(v.itos.slice(0, 3)).toEqual(['a', 'b', 'c'])
    expect(v.unk).toBe(3)
    expect(v.droppedKinds).toBe(2) // d、e 被折掉
    expect(v.stoi.get('a')).toBe(0)
  })

  it('词表够大时不产生 UNK', () => {
    const v = buildVocab('abcabc', 96)
    expect(v.itos.length).toBe(4) // a b c + UNK 占位
    expect(v.droppedKinds).toBe(0)
  })

  it('vocabChar 把 UNK 渲染成可见符号', () => {
    const v = buildVocab('aaabbbccde', 4)
    expect(vocabChar(v, v.unk)).toBe('␀')
    expect(vocabChar(v, 0)).toBe('a')
  })

  it('encodeCorpus 把词表外的字符映射到 unk', () => {
    const v = buildVocab('aaabbbccde', 4)
    const ids = encodeCorpus('aXd', v)
    expect(Array.from(ids)).toEqual([0, v.unk, v.unk])
  })
})

describe('批采样', () => {
  it('形状正确，targets 是 inputs 右移一位', () => {
    const data = Int32Array.from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    const { inputs, targets } = sampleBatch(data, 3, 4, mulberry32(1))
    expect(inputs.length).toBe(12)
    expect(targets.length).toBe(12)
    for (let b = 0; b < 3; b++) {
      for (let t = 0; t < 4; t++) {
        expect(targets[b * 4 + t]).toBe(inputs[b * 4 + t] + 1)
      }
    }
    // 所有取值都在语料范围内
    for (const v of inputs) expect(v).toBeGreaterThanOrEqual(0)
    for (const v of targets) expect(v).toBeLessThan(data.length)
  })

  it('同样的种子给出同样的批', () => {
    const data = Int32Array.from(Array.from({ length: 500 }, (_, i) => i % 17))
    const a = sampleBatch(data, 4, 8, mulberry32(7))
    const b = sampleBatch(data, 4, 8, mulberry32(7))
    expect(Array.from(a.inputs)).toEqual(Array.from(b.inputs))
  })
})

describe('学习率调度', () => {
  const cfg: TrainConfig = { ...DEFAULT_TRAIN, steps: 1000, lr: 0.02, warmup: 100, minLrRatio: 0.05 }

  it('warmup 阶段线性上升', () => {
    expect(lrAt(0, cfg)).toBeCloseTo(0.02 / 100)
    expect(lrAt(49, cfg)).toBeCloseTo(0.02 * 0.5, 5)
    expect(lrAt(99, cfg)).toBeCloseTo(0.02, 6)
  })

  it('warmup 之后单调下降到 minLrRatio', () => {
    let prev = lrAt(100, cfg)
    for (let s = 120; s <= 1000; s += 20) {
      const cur = lrAt(s, cfg)
      expect(cur).toBeLessThanOrEqual(prev + 1e-9)
      prev = cur
    }
    expect(lrAt(999, cfg)).toBeCloseTo(0.02 * 0.05, 4)
  })
})

describe('梯度正确性（下降方向判据）', () => {
  it('1 层：L(θ−ηg) 的实际下降量应等于 η|g|²', () => {
    const ratio = descentRatio(1)
    expect(ratio).toBeGreaterThan(0.9)
    expect(ratio).toBeLessThan(1.1)
  })

  it('2 层：同样成立（残差汇入处必须累加而非覆盖）', () => {
    const ratio = descentRatio(2)
    expect(ratio).toBeGreaterThan(0.9)
    expect(ratio).toBeLessThan(1.1)
  })
})

describe('基础算子的反向', () => {
  it('因果注意力：dQ / dK / dV 与数值梯度一致（只看量级够大的分量）', () => {
    const B = 2
    const T = 5
    const H = 2
    const hd = 3
    const D = H * hd
    const r1 = mulberry32(21)
    const r2 = mulberry32(22)
    const r3 = mulberry32(23)
    const r4 = mulberry32(24)
    const q = Float32Array.from({ length: B * T * D }, () => r1() * 2 - 1)
    const k = Float32Array.from({ length: B * T * D }, () => r2() * 2 - 1)
    const v = Float32Array.from({ length: B * T * D }, () => r3() * 2 - 1)
    const dOut = Float32Array.from({ length: B * T * D }, () => r4() * 2 - 1)

    const prob = new Float32Array(B * H * T * T)
    const out = new Float32Array(B * T * D)
    attentionForward(q, k, v, out, B, T, H, hd, prob)
    const dQ = new Float32Array(B * T * D)
    const dK = new Float32Array(B * T * D)
    const dV = new Float32Array(B * T * D)
    attentionBackward(dOut, q, k, v, prob, dQ, dK, dV, B, T, H, hd)

    // 注意力概率每一行（含掩码外）必须归一化
    for (let b = 0; b < B; b++) {
      for (let h = 0; h < H; h++) {
        for (let t = 0; t < T; t++) {
          let s = 0
          for (let j = 0; j < T; j++) s += prob[(b * H + h) * T * T + t * T + j]
          expect(s).toBeCloseTo(1, 5)
          // 因果掩码：未来位置必须是 0
          for (let j = t + 1; j < T; j++) expect(prob[(b * H + h) * T * T + t * T + j]).toBe(0)
        }
      }
    }

    const f = (which: 'q' | 'k' | 'v') => {
      const p2 = new Float32Array(B * H * T * T)
      const o2 = new Float32Array(B * T * D)
      attentionForward(q, k, v, o2, B, T, H, hd, p2)
      let s = 0
      for (let i = 0; i < B * T * D; i++) s += dOut[i] * o2[i]
      return s
    }
    const eps = 1e-3
    for (const [arr, darr, which] of [
      [q, dQ, 'q'],
      [k, dK, 'k'],
      [v, dV, 'v'],
    ] as const) {
      for (let i = 0; i < arr.length; i++) {
        const orig = arr[i]
        arr[i] = orig + eps
        const p = f(which)
        arr[i] = orig - eps
        const m = f(which)
        arr[i] = orig
        const num = (p - m) / (2 * eps)
        const ana = darr[i]
        // 只检验量级足够大的分量：小分量的有限差分被 float32 噪声淹没（见文件头注释）
        if (Math.max(Math.abs(num), Math.abs(ana)) > 1e-2) {
          expect(Math.abs(num - ana) / (Math.abs(num) + Math.abs(ana))).toBeLessThan(0.02)
        }
      }
    }
  })

  it('LayerNorm 反向：gain/bias/x 的梯度都对', () => {
    const M = 5
    const D = 9
    const r1 = mulberry32(11)
    const r2 = mulberry32(12)
    const r3 = mulberry32(13)
    const r4 = mulberry32(14)
    const x = Float32Array.from({ length: M * D }, () => (r1() * 2 - 1) * 2)
    const g = Float32Array.from({ length: D }, () => r2() * 2 - 1)
    const b = Float32Array.from({ length: D }, () => r3() * 2 - 1)
    const dy = Float32Array.from({ length: M * D }, () => r4() * 2 - 1)

    const xh = new Float32Array(M * D)
    const y = new Float32Array(M * D)
    const inv = new Float32Array(M)
    lnForward(x, g, b, M, D, xh, y, inv)
    const dg = new Float32Array(D)
    const db = new Float32Array(D)
    const dx = new Float32Array(M * D)
    lnBackward(dy, xh, inv, g, dg, db, M, D, new Float32Array(M * D), dx)

    // 归一化中间量 xh 必须是零均值单位方差（y = xh·g + b，b 是随机的，所以要看 xh）
    for (let i = 0; i < M; i++) {
      let mean = 0
      for (let k = 0; k < D; k++) mean += xh[i * D + k]
      mean /= D
      expect(mean).toBeCloseTo(0, 5)
      let vv = 0
      for (let k = 0; k < D; k++) vv += (xh[i * D + k] - mean) ** 2
      expect(vv / D).toBeCloseTo(1, 3)
    }

    const loss = () => {
      const xh2 = new Float32Array(M * D)
      const y2 = new Float32Array(M * D)
      lnForward(x, g, b, M, D, xh2, y2, new Float32Array(M))
      let s = 0
      for (let i = 0; i < M * D; i++) s += dy[i] * y2[i]
      return s
    }
    const eps = 1e-3
    const check = (arr: Float32Array, darr: Float32Array) => {
      for (let i = 0; i < arr.length; i++) {
        const orig = arr[i]
        arr[i] = orig + eps
        const p = loss()
        arr[i] = orig - eps
        const m = loss()
        arr[i] = orig
        const num = (p - m) / (2 * eps)
        const ana = darr[i]
        if (Math.max(Math.abs(num), Math.abs(ana)) > 1e-2) {
          expect(Math.abs(num - ana) / (Math.abs(num) + Math.abs(ana))).toBeLessThan(0.02)
        }
      }
    }
    check(g, dg)
    check(b, db)
    check(x, dx)
  })
})

describe('模型自检', () => {
  it('参数量等于各张量之和，且共享的词嵌入只算一次', () => {
    const m = new MiniGPT({ vocabSize: 20, model: DEFAULT_MODEL, batchSize: 4, seed: 1 })
    const sum = m.tensors.reduce((a, t) => a + t.size, 0)
    expect(m.paramCount()).toBe(sum)
    // 输出投影与词嵌入共享，所以不存在第二份 [V, d] 的投影矩阵
    expect(m.tensors.filter((t) => t.name === 'tokEmb').length).toBe(1)
  })

  it('参数明细各项之和等于总参数量', () => {
    const m = new MiniGPT({ vocabSize: 20, model: DEFAULT_MODEL, batchSize: 4, seed: 1 })
    const sum = m.paramBreakdown().reduce((a, g) => a + g.size, 0)
    expect(sum).toBe(m.paramCount())
  })

  /**
   * 回归测试：曾经这里返回的是中文显示名，切到英文界面时「参数都花在哪了」
   * 那张条形图会直接露出中文。core 层只允许给键，文案由模块的 i18n 字典负责。
   */
  it('参数明细只返回分组键，绝不返回界面文案', () => {
    const m = new MiniGPT({ vocabSize: 20, model: DEFAULT_MODEL, batchSize: 4, seed: 1 })
    const groups = m.paramBreakdown()
    const allowed: ParamGroupKey[] = ['tokEmb', 'posEmb', 'attn', 'ffn', 'ln']
    for (const g of groups) {
      expect(allowed).toContain(g.key)
      // 键必须是纯 ASCII 标识符，不能混进任何 CJK 字符
      expect(/^[a-zA-Z]+$/.test(g.key)).toBe(true)
    }
    // 默认配置下五组都应该非空（词嵌入/位置嵌入/注意力/前馈/LayerNorm）
    expect(groups.map((g) => g.key).sort()).toEqual([...allowed].sort())
  })

  it('每 token FLOPs 随层数线性增长', () => {
    const one = new MiniGPT({ vocabSize: 20, model: { ...DEFAULT_MODEL, nLayers: 1 }, batchSize: 4, seed: 1 })
    const two = new MiniGPT({ vocabSize: 20, model: { ...DEFAULT_MODEL, nLayers: 2 }, batchSize: 4, seed: 1 })
    expect(two.flopsPerToken()).toBeGreaterThan(one.flopsPerToken())
    expect(two.flopsPerToken()).toBeLessThan(one.flopsPerToken() * 2.2)
  })

  it('dModel 不能被头数整除时直接报错，而不是悄悄算错', () => {
    expect(
      () => new MiniGPT({ vocabSize: 10, model: { ...DEFAULT_MODEL, dModel: 30, nHeads: 4 }, batchSize: 2, seed: 1 }),
    ).toThrow()
  })

  it('同样的种子给出完全一样的初始 loss（确定性）', () => {
    const V = 12
    const model: ModelConfig = { nLayers: 2, dModel: 16, nHeads: 2, dFF: 32, blockSize: 8 }
    const { inputs, targets } = fixedData(V, 16, 5)
    const a = new MiniGPT({ vocabSize: V, model, batchSize: 2, seed: 777 })
    const b = new MiniGPT({ vocabSize: V, model, batchSize: 2, seed: 777 })
    expect(a.loss(inputs, targets)).toBe(b.loss(inputs, targets))
    const c = new MiniGPT({ vocabSize: V, model, batchSize: 2, seed: 778 })
    expect(c.loss(inputs, targets)).not.toBe(a.loss(inputs, targets))
  })

  it('初始权重被截断在 ±3σ 内（防止离群初始值破坏训练）', () => {
    const m = new MiniGPT({ vocabSize: 30, model: DEFAULT_MODEL, batchSize: 4, seed: 3 })
    for (const t of m.tensors) {
      // 只检查高斯初始化的权重；LayerNorm 的 gain 固定为 1、bias 固定为 0
      if (!t.decay) continue
      let mx = 0
      for (let i = 0; i < t.size; i++) mx = Math.max(mx, Math.abs(t.data[i]))
      // 初始化 std 最大 0.02（残差输出投影更小），3σ = 0.06
      expect(mx).toBeLessThanOrEqual(0.06 + 1e-6)
      expect(mx).toBeGreaterThan(0)
    }
  })
})

describe('训练确实在学', () => {
  it('周期等于 3 的语料能被背下来', () => {
    const text = 'abc'.repeat(120)
    const vocab = buildVocab(text)
    const data = encodeCorpus(text, vocab)
    const model: ModelConfig = { nLayers: 1, dModel: 32, nHeads: 4, dFF: 64, blockSize: 16 }
    const cfg: TrainConfig = {
      steps: 300, batchSize: 8, lr: 0.03, weightDecay: 0.01, gradClip: 1, seed: 5, warmup: 30, minLrRatio: 0.05,
    }
    const m = new MiniGPT({ vocabSize: vocab.itos.length, model, batchSize: cfg.batchSize, seed: cfg.seed })
    const vrnd = mulberry32(999)
    const val = sampleBatch(data, cfg.batchSize, model.blockSize, vrnd)
    const before = m.loss(val.inputs, val.targets)
    const rnd = mulberry32(cfg.seed)
    for (let i = 0; i < cfg.steps; i++) {
      const b = sampleBatch(data, cfg.batchSize, model.blockSize, rnd)
      m.trainStep(b.inputs, b.targets, lrAt(i, cfg), cfg.weightDecay, cfg.gradClip)
    }
    const after = m.loss(val.inputs, val.targets)
    expect(before).toBeGreaterThan(1.0) // 随机时接近 ln(4)=1.386
    expect(after).toBeLessThan(0.3)
    expect(after).toBeLessThan(before)
  })

  it('采样出来的 id 全部合法', () => {
    const vocab = buildVocab(PRESET_CORPORA[0].text)
    const m = new MiniGPT({ vocabSize: vocab.itos.length, model: DEFAULT_MODEL, batchSize: 4, seed: 1 })
    const ids = m.sample([0], 40, 0.8, 42)
    expect(ids.length).toBe(40)
    for (const id of ids) {
      expect(id).toBeGreaterThanOrEqual(0)
      expect(id).toBeLessThan(vocab.itos.length)
    }
  })

  it('训练前注意力接近均匀（这就是"什么都没学"的样子）', () => {
    const vocab = buildVocab(PRESET_CORPORA[0].text)
    const data = encodeCorpus(PRESET_CORPORA[0].text, vocab)
    const m = new MiniGPT({ vocabSize: vocab.itos.length, model: DEFAULT_MODEL, batchSize: 4, seed: 1 })
    const maps = m.attnSnapshot(Array.from(data.slice(0, DEFAULT_MODEL.blockSize)), 0)
    for (const head of maps) {
      for (let t = 4; t < head.length; t++) {
        let e = 0
        for (let j = 0; j <= t; j++) {
          const p = head[t][j]
          if (p > 0) e -= p * Math.log(p)
        }
        // 归一化熵：1 = 完全均匀，0 = 只看一个位置
        expect(e / Math.log(t + 1)).toBeGreaterThan(0.95)
      }
    }
  })

  /**
   * 这条和上一条是配对的：站点上「注意力从噪声长出结构」那两张热力图，
   * 左边对应上一条（归一化熵 ≈ 1.00），右边对应这一条（归一化熵 ≈ 0）。
   * 没有这条断言，右边那张图到底是不是真的算出来的就没有证据。
   */
  it('训练后注意力塌成近似 one-hot（这就是"学出结构"的样子）', () => {
    const vocab = buildVocab(PRESET_CORPORA[0].text)
    const data = encodeCorpus(PRESET_CORPORA[0].text, vocab)
    const m = new MiniGPT({
      vocabSize: vocab.itos.length,
      model: DEFAULT_MODEL,
      batchSize: DEFAULT_TRAIN.batchSize,
      seed: DEFAULT_TRAIN.seed,
    })
    const rnd = mulberry32(DEFAULT_TRAIN.seed)
    for (let i = 0; i < DEFAULT_TRAIN.steps; i++) {
      const b = sampleBatch(data, DEFAULT_TRAIN.batchSize, DEFAULT_MODEL.blockSize, rnd)
      m.trainStep(b.inputs, b.targets, lrAt(i, DEFAULT_TRAIN), DEFAULT_TRAIN.weightDecay, DEFAULT_TRAIN.gradClip)
    }
    const maps = m.attnSnapshot(Array.from(data.slice(0, DEFAULT_MODEL.blockSize)), 0)

    let worstEntropy = 1
    for (const head of maps) {
      for (let t = 8; t < head.length; t++) {
        let e = 0
        for (let j = 0; j <= t; j++) {
          const p = head[t][j]
          if (p > 0) e -= p * Math.log(p)
        }
        worstEntropy = Math.min(worstEntropy, e / Math.log(t + 1))
      }
    }
    // 实测 1200 步后归一化熵全部压到 1e-3 以下（打印出来是 0.0000）
    expect(worstEntropy).toBeLessThan(1e-2)

    // 而且每个头在末行都出现了权重接近 1.0 的"只盯一个位置"
    for (const head of maps) {
      const mx = Math.max(...head[DEFAULT_MODEL.blockSize - 1])
      expect(mx).toBeGreaterThan(0.99)
    }
  })
})

describe('内置语料', () => {
  /**
   * 最小重复周期：最小的 p 使得 s[i] === s[i+p] 对全部 i 成立。
   *
   * 为什么值得单独立个测试？因为「周期」在本模块里不是装饰性字段 ——
   * 「周期能不能塞进上下文窗口」是模块⑩ 的核心教学结论，要拿 period 去和 blockSize 比。
   * 早期版本这里手写成 13 / 23 / 49，而真实值是 12 / 24 / 52，三个全错，
   * 结论的方向虽然没错但数字全被带偏。这个断言就是为了让这种错误不可能再溜过去。
   */
  function minimalPeriod(s: string): number {
    for (let p = 1; p <= s.length; p++) {
      let ok = true
      for (let i = 0; i + p < s.length; i++) {
        if (s[i] !== s[i + p]) {
          ok = false
          break
        }
      }
      if (ok) return p
    }
    return s.length
  }

  it('三个内置语料都有足够长度且周期可辨识', () => {
    for (const p of PRESET_CORPORA) {
      expect(p.text.length).toBeGreaterThan(500)
      expect(p.period).toBeGreaterThan(0)
      expect(p.period).toBeLessThan(p.text.length / 10)
      // label 中英都要有
      expect(p.label.length).toBeGreaterThan(0)
      expect(p.labelEn.length).toBeGreaterThan(0)
    }
  })

  it('声明的 period 必须等于实测最小周期', () => {
    for (const p of PRESET_CORPORA) {
      expect({ id: p.id, period: p.period }).toEqual({ id: p.id, period: minimalPeriod(p.text) })
    }
  })

  it('三条语料的周期单调递增（按难度排序）', () => {
    const periods = PRESET_CORPORA.map((p) => p.period)
    for (let i = 1; i < periods.length; i++) {
      expect(periods[i]).toBeGreaterThan(periods[i - 1])
    }
  })

  it('默认语料的周期比默认窗口小，长周期语料的周期比默认窗口大', () => {
    // 这是模块⑩「周期 vs 窗口」那组实验能够成立的前提
    expect(PRESET_CORPORA[0].period).toBeLessThan(DEFAULT_MODEL.blockSize)
    expect(PRESET_CORPORA[2].period).toBeGreaterThan(DEFAULT_MODEL.blockSize)
  })

  it('默认语料在默认配置下的字符种类不超过词表上限', () => {
    const vocab = buildVocab(PRESET_CORPORA[0].text, 96)
    expect(vocab.droppedKinds).toBe(0)
  })
})

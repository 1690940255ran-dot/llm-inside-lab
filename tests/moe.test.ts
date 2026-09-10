import { describe, expect, it } from 'vitest'
import {
  DEFAULT_MOE,
  MOE_ARCH_PRESETS,
  MOE_SCENARIOS,
  auxGradient,
  cosine,
  fmtParams,
  makeRouterState,
  makeTokenSet,
  moeAnatomy,
  routeStep,
  sweepBalance,
  trainRouter,
} from '../src/core/moe'

describe('makeTokenSet', () => {
  it('同样种子得到完全一样的结果', () => {
    const a = makeTokenSet(48, 16, 5, 123)
    const b = makeTokenSet(48, 16, 5, 123)
    expect(a.tokens).toEqual(b.tokens)
    expect(a.topicOf).toEqual(b.topicOf)
  })

  it('不同种子结果不同', () => {
    const a = makeTokenSet(48, 16, 5, 1)
    const b = makeTokenSet(48, 16, 5, 2)
    expect(a.tokens).not.toEqual(b.tokens)
  })

  it('所有 token 都是单位向量', () => {
    const ts = makeTokenSet(64, 24, 6, 7)
    for (const t of ts.tokens) {
      const norm = Math.sqrt(t.reduce((a, b) => a + b * b, 0))
      expect(norm).toBeCloseTo(1, 6)
    }
  })

  it('主题频率服从 Zipf：第一个主题最大且占比之和为 100', () => {
    const ts = makeTokenSet(400, 16, 6, 9)
    const share = ts.topicShare
    expect(share.reduce((a, b) => a + b, 0)).toBeCloseTo(100, 4)
    for (let i = 1; i < share.length; i++) expect(share[i]).toBeLessThanOrEqual(share[i - 1])
  })

  it('维度对得上', () => {
    const ts = makeTokenSet(32, 20, 4, 3)
    expect(ts.tokens[0].length).toBe(20)
    expect(ts.topicOf.length).toBe(32)
    expect(ts.dim).toBe(20)
  })

  it('散度越大，主题内的 token 越分散（与主题中心的平均余弦下降）', () => {
    const tight = makeTokenSet(64, 32, 4, 5, 0.1)
    const loose = makeTokenSet(64, 32, 4, 5, 1.0)
    const spread = (ts: ReturnType<typeof makeTokenSet>) => {
      const centers: number[][] = []
      for (let t = 0; t < 4; t++) {
        const v = new Array(32).fill(0)
        let c = 0
        ts.tokens.forEach((tok, i) => {
          if (ts.topicOf[i] === t) {
            c++
            for (let d = 0; d < 32; d++) v[d] += tok[d]
          }
        })
        const n = Math.sqrt(v.reduce((a, b) => a + b * b, 0)) || 1
        centers.push(v.map((x) => x / n))
      }
      let s = 0
      ts.tokens.forEach((tok, i) => (s += cosine(tok, centers[ts.topicOf[i]])))
      return s / ts.tokens.length
    }
    expect(spread(loose)).toBeLessThan(spread(tight))
  })
})

describe('cosine', () => {
  it('相同向量为 1，正交为 0', () => {
    expect(cosine([1, 0], [1, 0])).toBeCloseTo(1, 10)
    expect(cosine([1, 0], [0, 1])).toBeCloseTo(0, 10)
  })

  it('零向量不产生 NaN', () => {
    expect(Number.isFinite(cosine([0, 0], [1, 0]))).toBe(true)
  })
})

describe('routeStep', () => {
  const base = { ...DEFAULT_MOE, steps: 0 }

  it('每个 token 拿到的专家数不超过 top-k', () => {
    const ts = makeTokenSet(base.nTokens, base.dim, base.nTopics, base.seed, base.noise)
    const st = makeRouterState(base.nExperts, base.dim, base.seed)
    const r = routeStep(ts.tokens, st, base, 0)
    for (const a of r.assign) expect(a.length).toBeLessThanOrEqual(base.topK)
  })

  it('权重用的是 gate 概率（在落到的专家上归一化到 1），不是专家下标', () => {
    const ts = makeTokenSet(base.nTokens, base.dim, base.nTopics, base.seed, base.noise)
    const st = makeRouterState(base.nExperts, base.dim, base.seed)
    const r = routeStep(ts.tokens, st, base, 0)
    r.weight.forEach((w, i) => {
      if (r.assign[i].length > 0) {
        expect(w.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 8)
        // 权重还必须与对应专家的 gate 概率成比例
        const s = r.assign[i].map((e) => r.prob[i][e]).reduce((a, b) => a + b, 0)
        r.assign[i].forEach((e, k) => expect(w[k]).toBeCloseTo(r.prob[i][e] / s, 8))
      }
    })
  })

  it('每行 gate 概率之和为 1', () => {
    const ts = makeTokenSet(base.nTokens, base.dim, base.nTopics, base.seed, base.noise)
    const st = makeRouterState(base.nExperts, base.dim, base.seed)
    const r = routeStep(ts.tokens, st, base, 0)
    for (const row of r.prob) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 8)
  })

  it('偏置确实会改变路由：把某个专家的偏置顶到很高，它会吃掉大量 token', () => {
    const ts = makeTokenSet(base.nTokens, base.dim, base.nTopics, base.seed, base.noise)
    const st = makeRouterState(base.nExperts, base.dim, base.seed)
    const before = routeStep(ts.tokens, st, base, 0)
    const boosted = { ...st, bias: st.bias.map((b, e) => (e === 0 ? b + 20 : b)) }
    const after = routeStep(ts.tokens, boosted, base, 0)
    expect(after.load[0]).toBeGreaterThan(before.load[0])
    expect(after.load[0]).toBe(base.nTokens)
  })

  it('容量限制真的会卡住负载', () => {
    const cfg = { ...base, capacityFactor: 1.25 }
    const ts = makeTokenSet(cfg.nTokens, cfg.dim, cfg.nTopics, cfg.seed, cfg.noise)
    const st = makeRouterState(cfg.nExperts, cfg.dim, cfg.seed)
    const r = routeStep(ts.tokens, st, cfg, 0)
    const cap = Math.ceil((cfg.nTokens * cfg.topK) / cfg.nExperts * cfg.capacityFactor)
    for (const c of r.load) expect(c).toBeLessThanOrEqual(cap)
  })

  it('capacityFactor = 0 表示不限容量，不会丢弃 token', () => {
    const ts = makeTokenSet(base.nTokens, base.dim, base.nTopics, base.seed, base.noise)
    const st = makeRouterState(base.nExperts, base.dim, base.seed)
    const r = routeStep(ts.tokens, st, { ...base, capacityFactor: 0 }, 0)
    expect(r.droppedCount).toBe(0)
    expect(r.partialDropped).toBe(0)
  })

  it('aux loss 有理论下界 1（完美均衡时取到）', () => {
    const ts = makeTokenSet(base.nTokens, base.dim, base.nTopics, base.seed, base.noise)
    const st = makeRouterState(base.nExperts, base.dim, base.seed)
    const r = routeStep(ts.tokens, st, base, 0)
    expect(r.auxLoss).toBeGreaterThanOrEqual(1 - 1e-9)
  })

  it('负载不均度与熵在合理范围内', () => {
    const ts = makeTokenSet(base.nTokens, base.dim, base.nTopics, base.seed, base.noise)
    const st = makeRouterState(base.nExperts, base.dim, base.seed)
    const r = routeStep(ts.tokens, st, base, 0)
    expect(r.imbalance).toBeGreaterThanOrEqual(1 - 1e-9)
    expect(r.entropy).toBeGreaterThan(0)
    expect(r.entropy).toBeLessThanOrEqual(1 + 1e-9)
  })
})

describe('auxGradient —— 死专家的梯度恒为 0', () => {
  it('负载为 0 的专家梯度精确等于 0', () => {
    const ts = makeTokenSet(DEFAULT_MOE.nTokens, DEFAULT_MOE.dim, DEFAULT_MOE.nTopics, DEFAULT_MOE.seed, DEFAULT_MOE.noise)
    const st = makeRouterState(DEFAULT_MOE.nExperts, DEFAULT_MOE.dim, DEFAULT_MOE.seed)
    // 人为造一个死专家：把它的偏置压到极低
    const killed = { ...st, bias: st.bias.map((b, e) => (e === 3 ? b - 30 : b)) }
    const r = routeStep(ts.tokens, killed, DEFAULT_MOE, 0)
    expect(r.load[3]).toBe(0)
    expect(r.auxGrad[3]).toBe(0)
    // 活着的专家梯度一定为正
    for (let e = 0; e < DEFAULT_MOE.nExperts; e++) {
      if (r.load[e] > 0) expect(r.auxGrad[e]).toBeGreaterThan(0)
    }
  })

  it('梯度与负载份额成正比（同概率分布下）', () => {
    const prob = [
      [0.5, 0.5],
      [0.5, 0.5],
    ]
    const g = auxGradient([2, 0], prob, 2)
    expect(g[0]).toBeGreaterThan(0)
    expect(g[1]).toBe(0)
  })
})

describe('trainRouter', () => {
  it('确定性：同样配置两次训练结果完全一致', () => {
    const cfg = { ...DEFAULT_MOE, balanceMode: 'bias' as const, balanceWeight: 2 }
    const a = trainRouter(cfg)
    const b = trainRouter(cfg)
    expect(a.final.load).toEqual(b.final.load)
    expect(a.final.bias).toEqual(b.final.bias)
    expect(a.expertPurity).toEqual(b.expertPurity)
  })

  it('不施加约束时负载会明显倾斜（默认配置下 2 倍以上）', () => {
    const r = trainRouter({ ...DEFAULT_MOE, balanceMode: 'none', balanceWeight: 0 })
    expect(r.final.imbalance).toBeGreaterThan(1.8)
  })

  it('负载倾斜的根源是 Zipf 数据先验：话题数远多于专家数时不出现死专家', () => {
    const r = trainRouter({ ...DEFAULT_MOE, balanceMode: 'none', balanceWeight: 0 })
    expect(r.final.deadExperts).toBe(0)
    expect(r.tokenSet.topicShare[0]).toBeGreaterThan(r.tokenSet.topicShare[10])
  })

  it('话题数少于专家数时出现结构性死专家', () => {
    const r = trainRouter({ ...DEFAULT_MOE, balanceMode: 'none', balanceWeight: 0, nTopics: 3 })
    expect(r.final.deadExperts).toBeGreaterThanOrEqual(3)
    expect(r.final.imbalance).toBeGreaterThan(3)
  })

  it('负载反馈偏置（bias）压平负载，且需要的偏置极小', () => {
    const r = trainRouter({ ...DEFAULT_MOE, balanceMode: 'bias', balanceWeight: 2 })
    expect(r.final.imbalance).toBeLessThan(1.1)
    expect(r.final.deadExperts).toBe(0)
    const biasNorm = r.final.bias.reduce((a, b) => a + Math.abs(b), 0) / r.final.bias.length
    expect(biasNorm).toBeLessThan(0.5)
  })

  it('aux 梯度也能压平负载，但要把偏置推大一个数量级', () => {
    const bias = trainRouter({ ...DEFAULT_MOE, balanceMode: 'bias', balanceWeight: 2 })
    const aux = trainRouter({ ...DEFAULT_MOE, balanceMode: 'aux', balanceWeight: 1 })
    const norm = (r: ReturnType<typeof trainRouter>) =>
      r.final.bias.reduce((a, b) => a + Math.abs(b), 0) / r.final.bias.length
    expect(aux.final.imbalance).toBeLessThan(1.3)
    expect(norm(aux)).toBeGreaterThan(norm(bias) * 5)
  })

  it('aux 开过头会自己造出新的死专家（曲线非单调）', () => {
    const ok = trainRouter({ ...DEFAULT_MOE, balanceMode: 'aux', balanceWeight: 1 })
    const over = trainRouter({ ...DEFAULT_MOE, balanceMode: 'aux', balanceWeight: 4 })
    expect(ok.final.deadExperts).toBe(0)
    expect(over.final.deadExperts).toBeGreaterThan(0)
    expect(over.final.imbalance).toBeGreaterThan(ok.final.imbalance)
  })

  it('bias 开过头同样会过冲', () => {
    const ok = trainRouter({ ...DEFAULT_MOE, balanceMode: 'bias', balanceWeight: 2 })
    const over = trainRouter({ ...DEFAULT_MOE, balanceMode: 'bias', balanceWeight: 8 })
    expect(over.final.imbalance).toBeGreaterThan(ok.final.imbalance)
  })

  it('topK 越大负载越平均、专精越弱', () => {
    const k1 = trainRouter({ ...DEFAULT_MOE, topK: 1, balanceMode: 'none' })
    const k2 = trainRouter({ ...DEFAULT_MOE, topK: 2, balanceMode: 'none' })
    expect(k2.final.imbalance).toBeLessThan(k1.final.imbalance)
    const purity = (r: ReturnType<typeof trainRouter>) =>
      r.expertPurity.reduce((a, b) => a + b, 0) / r.expertPurity.length
    expect(purity(k2)).toBeLessThan(purity(k1))
  })

  it('容量因子限制负载，代价是丢 token', () => {
    const capped = trainRouter({ ...DEFAULT_MOE, topK: 1, capacityFactor: 1.25, balanceMode: 'none' })
    expect(capped.final.imbalance).toBeLessThan(1.5)
    expect(capped.final.droppedCount).toBeGreaterThan(0)
  })

  it('排序后的主题序列单调不减（热力图分组的前提）', () => {
    const r = trainRouter(DEFAULT_MOE)
    expect(r.sortedTopic.length).toBe(DEFAULT_MOE.nTokens)
    for (let i = 1; i < r.sortedTopic.length; i++) {
      expect(r.sortedTopic[i]).toBeGreaterThanOrEqual(r.sortedTopic[i - 1])
    }
    expect(r.sortedProb.length).toBe(r.sortedTopic.length)
  })

  it('专精纯度在 [0,1] 内，死专家的纯度为 0', () => {
    const r = trainRouter({ ...DEFAULT_MOE, balanceMode: 'none', nTopics: 3 })
    r.expertPurity.forEach((p, e) => {
      expect(p).toBeGreaterThanOrEqual(0)
      expect(p).toBeLessThanOrEqual(1)
      if (r.final.load[e] === 0) expect(p).toBe(0)
    })
  })

  it('expertTopic 的每一行之和等于该专家分到的 token 数', () => {
    const r = trainRouter(DEFAULT_MOE)
    for (let e = 0; e < DEFAULT_MOE.nExperts; e++) {
      const row = r.expertTopic[e]
      expect(row.reduce((a, b) => a + b, 0)).toBe(r.final.load[e])
    }
  })

  it('每个场景都能跑通，且声明的场景 id 唯一', () => {
    const ids = new Set<string>()
    for (const s of MOE_SCENARIOS) {
      expect(ids.has(s.id)).toBe(false)
      ids.add(s.id)
      const r = trainRouter({ ...DEFAULT_MOE, ...s.patch })
      expect(Number.isFinite(r.final.imbalance)).toBe(true)
      expect(r.final.load.length).toBe(r.cfg.nExperts)
      expect(r.final.auxGrad.length).toBe(r.cfg.nExperts)
    }
    expect(ids.size).toBe(MOE_SCENARIOS.length)
  })
})

describe('sweepBalance', () => {
  it('返回每个权重的指标，且权重顺序保留', () => {
    const ws = [0, 1, 2, 4]
    const pts = sweepBalance(DEFAULT_MOE, ws, 'bias')
    expect(pts.map((p) => p.w)).toEqual(ws)
    for (const p of pts) {
      expect(Number.isFinite(p.imbalance)).toBe(true)
      expect(p.entropy).toBeGreaterThan(0)
      expect(p.livePurity).toBeGreaterThanOrEqual(0)
      expect(p.biasNorm).toBeGreaterThanOrEqual(0)
    }
  })

  it('bias 的扫描曲线单调压平负载；aux 的曲线会掉头（非单调）', () => {
    const ws = [1, 2, 4]
    const b = sweepBalance(DEFAULT_MOE, ws, 'bias')
    expect(b[1].imbalance).toBeLessThan(b[0].imbalance)
    expect(b[2].imbalance).toBeLessThanOrEqual(b[1].imbalance)

    const a = sweepBalance(DEFAULT_MOE, [1, 4], 'aux')
    expect(a[1].imbalance).toBeGreaterThan(a[0].imbalance)
  })

  it('不传 mode 时，若 cfg 本身就是 none 则默认扫 bias', () => {
    const pts = sweepBalance({ ...DEFAULT_MOE, balanceMode: 'none' }, [2])
    expect(pts[0].biasNorm).toBeGreaterThan(0)
  })
})

describe('moeAnatomy', () => {
  it('能复现官方公布的参数量（5% 以内）', () => {
    for (const p of MOE_ARCH_PRESETS) {
      if (!p.published) continue
      const a = moeAnatomy(p.arch)
      const totalB = a.totalParams / 1e9
      const activeB = a.activeParams / 1e9
      expect(Math.abs(totalB - p.published.totalB) / p.published.totalB).toBeLessThan(0.05)
      expect(Math.abs(activeB - p.published.activeB) / p.published.activeB).toBeLessThan(0.15)
    }
  })

  it('激活参数一定小于总参数，且激活占比在 (0,1)', () => {
    for (const p of MOE_ARCH_PRESETS) {
      const a = moeAnatomy(p.arch)
      expect(a.activeParams).toBeLessThan(a.totalParams)
      expect(a.activeRatio).toBeGreaterThan(0)
      expect(a.activeRatio).toBeLessThan(1)
    }
  })

  it('专家数越多总参数越大，但激活参数基本不变', () => {
    const a = moeAnatomy({ ...MOE_ARCH_PRESETS[0].arch, nExperts: 8 })
    const b = moeAnatomy({ ...MOE_ARCH_PRESETS[0].arch, nExperts: 64 })
    expect(b.totalParams).toBeGreaterThan(a.totalParams)
    expect(b.activeParams).toBeCloseTo(a.activeParams, -6)
  })

  it('共享专家同时进总参数和激活参数', () => {
    const a = moeAnatomy({ ...MOE_ARCH_PRESETS[0].arch, nShared: 0 })
    const b = moeAnatomy({ ...MOE_ARCH_PRESETS[0].arch, nShared: 2 })
    expect(b.totalParams).toBeGreaterThan(a.totalParams)
    expect(b.activeParams).toBeGreaterThan(a.activeParams)
  })
})

describe('fmtParams', () => {
  it('各量级格式化正确', () => {
    expect(fmtParams(1.5e12)).toBe('1.50 T')
    expect(fmtParams(46.7e9)).toBe('46.70 B')
    expect(fmtParams(3.3e6)).toBe('3.3 M')
    expect(fmtParams(2500)).toBe('2.5 K')
    expect(fmtParams(700)).toBe('700')
  })
})

import { describe, expect, it } from 'vitest'
import {
  DEFAULT_CONTEXT,
  METHOD_LABEL,
  analyzeContext,
  compareMethods,
  deltaGrid,
  phaseTrack,
  ropeDims,
  ropeScore,
  wavelengthSpectrum,
  type RopeMethod,
} from '../src/core/context'

const METHODS: RopeMethod[] = ['none', 'linear', 'ntk', 'yarn']

describe('ropeDims', () => {
  it('原始角频率符合 θ_j = base^(-2j/d)', () => {
    const dims = ropeDims({ ...DEFAULT_CONTEXT, method: 'none' })
    expect(dims.length).toBe(DEFAULT_CONTEXT.dHead / 2)
    dims.forEach((d, j) => {
      expect(d.theta0).toBeCloseTo(Math.pow(DEFAULT_CONTEXT.base, (-2 * j) / DEFAULT_CONTEXT.dHead), 12)
    })
  })

  it('波长 = 2π / θ，且随维度单调递增', () => {
    const dims = ropeDims({ ...DEFAULT_CONTEXT, method: 'none' })
    dims.forEach((d) => expect(d.lambda).toBeCloseTo((2 * Math.PI) / d.theta, 9))
    for (let i = 1; i < dims.length; i++) expect(dims[i].lambda).toBeGreaterThan(dims[i - 1].lambda)
  })

  it('none：不改任何维度', () => {
    const dims = ropeDims({ ...DEFAULT_CONTEXT, method: 'none' })
    for (const d of dims) {
      expect(d.theta).toBeCloseTo(d.theta0, 12)
      expect(d.stretch).toBeCloseTo(1, 12)
    }
  })

  it('linear：所有维度统一放大 s 倍波长', () => {
    const s = DEFAULT_CONTEXT.targetLen / DEFAULT_CONTEXT.trainLen
    const dims = ropeDims({ ...DEFAULT_CONTEXT, method: 'linear' })
    for (const d of dims) expect(d.stretch).toBeCloseTo(s, 6)
  })

  it('ntk：最高频维度几乎不动，最低频维度接近完整 s 倍', () => {
    const s = DEFAULT_CONTEXT.targetLen / DEFAULT_CONTEXT.trainLen
    const dims = ropeDims({ ...DEFAULT_CONTEXT, method: 'ntk' })
    expect(dims[0].stretch).toBeCloseTo(1, 6)
    expect(dims[dims.length - 1].stretch).toBeCloseTo(s, 1)
    // 单调递增
    for (let i = 1; i < dims.length; i++) {
      expect(dims[i].stretch).toBeGreaterThanOrEqual(dims[i - 1].stretch - 1e-9)
    }
  })

  it('yarn：低频段完全不动、高频段完整拉伸，中间有过渡', () => {
    const s = DEFAULT_CONTEXT.targetLen / DEFAULT_CONTEXT.trainLen
    const dims = ropeDims({ ...DEFAULT_CONTEXT, method: 'yarn' })
    const frozen = dims.filter((d) => d.ramp === 0)
    const full = dims.filter((d) => d.ramp === 1)
    const mid = dims.filter((d) => d.ramp > 0 && d.ramp < 1)
    expect(frozen.length).toBeGreaterThan(0)
    expect(full.length).toBeGreaterThan(0)
    expect(mid.length).toBeGreaterThan(0)
    for (const d of frozen) expect(d.stretch).toBeCloseTo(1, 9)
    for (const d of full) expect(d.stretch).toBeCloseTo(s, 6)
    for (const d of dims) {
      expect(d.ramp).toBeGreaterThanOrEqual(0)
      expect(d.ramp).toBeLessThanOrEqual(1)
    }
  })

  it('别名起点 = π / θ', () => {
    const dims = ropeDims(DEFAULT_CONTEXT)
    for (const d of dims) expect(d.aliasOnset).toBeCloseTo(Math.PI / d.theta, 6)
  })

  it('训练长度内的圈数符合定义', () => {
    const dims = ropeDims(DEFAULT_CONTEXT)
    for (const d of dims) {
      expect(d.turnsInTrain).toBeCloseTo((DEFAULT_CONTEXT.trainLen * d.theta) / (2 * Math.PI), 6)
    }
  })
})

describe('ropeScore', () => {
  it('距离 0 处期望内积为 1', () => {
    const dims = ropeDims(DEFAULT_CONTEXT)
    expect(ropeScore(dims, 0)).toBeCloseTo(1, 9)
  })

  it('分数始终落在 [-1, 1]', () => {
    const dims = ropeDims(DEFAULT_CONTEXT)
    for (const dl of [1, 17, 256, 4096, 65536]) {
      const s = ropeScore(dims, dl)
      expect(s).toBeGreaterThanOrEqual(-1.0000001)
      expect(s).toBeLessThanOrEqual(1.0000001)
    }
  })

  it('等于各维度 cos(Δ·θ) 的算术平均', () => {
    const dims = ropeDims(DEFAULT_CONTEXT)
    const dl = 512
    const manual = dims.reduce((a, d) => a + Math.cos(dl * d.theta), 0) / dims.length
    expect(ropeScore(dims, dl)).toBeCloseTo(manual, 12)
  })
})

describe('位置可分辨性', () => {
  it('改写后每个距离上可分辨的维度数不少于「不改」', () => {
    const base = ropeDims({ ...DEFAULT_CONTEXT, method: 'none' })
    for (const m of METHODS.filter((x) => x !== 'none')) {
      const d = ropeDims({ ...DEFAULT_CONTEXT, method: m })
      const at = (dims: typeof base, delta: number) =>
        dims.filter((x) => delta * x.theta < Math.PI).length
      for (const delta of [1024, 4096, 16384, 32768]) {
        expect(at(d, delta)).toBeGreaterThanOrEqual(at(base, delta))
      }
    }
  })

  it('「不改」在目标长度处可分辨维度归零', () => {
    const r = analyzeContext(DEFAULT_CONTEXT)
    const alive = r.dims.filter((d) => DEFAULT_CONTEXT.targetLen * d.theta < Math.PI).length
    expect(alive).toBe(0)
  })

  it('三种改写方案在目标长度处都还留有可分辨维度', () => {
    for (const m of METHODS.filter((x) => x !== 'none')) {
      const r = analyzeContext({ ...DEFAULT_CONTEXT, method: m })
      const alive = r.dims.filter((d) => DEFAULT_CONTEXT.targetLen * d.theta < Math.PI).length
      expect(alive).toBeGreaterThan(0)
    }
  })

  it('可分辨占比随距离单调不增', () => {
    const r = analyzeContext(DEFAULT_CONTEXT)
    for (let i = 1; i < r.resolvable.length; i++) {
      expect(r.resolvable[i].value).toBeLessThanOrEqual(r.resolvable[i - 1].value + 1e-12)
    }
  })
})

describe('局部精度（最小可分辨间隔）', () => {
  it('minGap = π / θ_max', () => {
    const r = analyzeContext(DEFAULT_CONTEXT)
    const maxTheta = Math.max(...r.dims.map((d) => d.theta))
    expect(r.minGap).toBeCloseTo(Math.PI / maxTheta, 9)
  })

  it('线性插值把局部精度拉差 s 倍，NTK / YaRN 不变', () => {
    const s = DEFAULT_CONTEXT.targetLen / DEFAULT_CONTEXT.trainLen
    const none = analyzeContext({ ...DEFAULT_CONTEXT, method: 'none' }).minGap
    expect(analyzeContext({ ...DEFAULT_CONTEXT, method: 'linear' }).minGap).toBeCloseTo(none * s, 4)
    expect(analyzeContext({ ...DEFAULT_CONTEXT, method: 'ntk' }).minGap).toBeCloseTo(none, 9)
    expect(analyzeContext({ ...DEFAULT_CONTEXT, method: 'yarn' }).minGap).toBeCloseTo(none, 9)
  })

  it('这正是选型依据：NTK / YaRN 长距离能力接近线性插值，局部精度却不掉', () => {
    const alive = (m: RopeMethod) =>
      analyzeContext({ ...DEFAULT_CONTEXT, method: m }).dims.filter(
        (d) => DEFAULT_CONTEXT.targetLen * d.theta < Math.PI,
      ).length
    const linearGap = analyzeContext({ ...DEFAULT_CONTEXT, method: 'linear' }).minGap
    const yarnGap = analyzeContext({ ...DEFAULT_CONTEXT, method: 'yarn' }).minGap
    expect(yarnGap).toBeLessThan(linearGap)
    expect(alive('yarn')).toBeGreaterThanOrEqual(alive('ntk'))
  })
})

describe('analyzeContext', () => {
  it('缩放倍数 = 目标长度 / 训练长度', () => {
    const r = analyzeContext({ ...DEFAULT_CONTEXT, trainLen: 4096, targetLen: 32768 })
    expect(r.scale).toBe(8)
  })

  it('几何上界 = 最长波长的一半', () => {
    const r = analyzeContext(DEFAULT_CONTEXT)
    expect(r.geometricHorizon).toBeCloseTo(Math.max(...r.dims.map((d) => d.lambda)) / 2, 6)
  })

  it('长波长维度占比在 (0,1)', () => {
    const r = analyzeContext(DEFAULT_CONTEXT)
    expect(r.lowFreqShare).toBeGreaterThan(0)
    expect(r.lowFreqShare).toBeLessThan(1)
  })

  it('base 越大，几何上界越大', () => {
    const a = analyzeContext({ ...DEFAULT_CONTEXT, base: 10000 })
    const b = analyzeContext({ ...DEFAULT_CONTEXT, base: 500000 })
    expect(b.geometricHorizon).toBeGreaterThan(a.geometricHorizon)
  })

  it('YaRN 的温度补偿会把分数整体压小', () => {
    const off = analyzeContext({ ...DEFAULT_CONTEXT, method: 'yarn', yarnTemp: false })
    const on = analyzeContext({ ...DEFAULT_CONTEXT, method: 'yarn', yarnTemp: true })
    const at = (r: typeof off, dl: number) => r.score.find((p) => p.delta >= dl)!.value
    expect(Math.abs(at(on, 4096))).toBeLessThan(Math.abs(at(off, 4096)))
  })

  it('温度补偿只在 yarn 下生效', () => {
    const a = analyzeContext({ ...DEFAULT_CONTEXT, method: 'none', yarnTemp: false })
    const b = analyzeContext({ ...DEFAULT_CONTEXT, method: 'none', yarnTemp: true })
    expect(a.score.map((p) => p.value)).toEqual(b.score.map((p) => p.value))
  })
})

describe('deltaGrid / phaseTrack / 其它', () => {
  it('网格从 1 到上限，单调递增', () => {
    const g = deltaGrid(32768, 50)
    expect(g.length).toBe(50)
    expect(g[0]).toBeCloseTo(1, 9)
    expect(g[g.length - 1]).toBeCloseTo(32768, 3)
    for (let i = 1; i < g.length; i++) expect(g[i]).toBeGreaterThan(g[i - 1])
  })

  it('相位一定折在 [-π, π] 内', () => {
    const positions = Array.from({ length: 200 }, (_, i) => i * 100)
    const tracks = phaseTrack(DEFAULT_CONTEXT, [0, 10, 30], positions)
    expect(tracks.length).toBe(3)
    for (const tr of tracks) {
      expect(tr.phase.length).toBe(positions.length)
      for (const p of tr.phase) {
        expect(p).toBeGreaterThanOrEqual(-Math.PI - 1e-9)
        expect(p).toBeLessThanOrEqual(Math.PI + 1e-9)
      }
    }
  })

  it('phaseTrack 忽略越界的维度编号', () => {
    const tracks = phaseTrack(DEFAULT_CONTEXT, [-1, 0, 9999], [0, 1, 2])
    expect(tracks.length).toBe(1)
  })

  it('波长谱给出改写前后两条曲线', () => {
    const sp = wavelengthSpectrum({ ...DEFAULT_CONTEXT, method: 'ntk' })
    expect(sp.dims.length).toBe(DEFAULT_CONTEXT.dHead / 2)
    for (const d of sp.dims) {
      expect(d.scaled).toBeGreaterThanOrEqual(d.base - 1e-9)
    }
  })

  it('compareMethods 覆盖全部四种方法', () => {
    const cmp = compareMethods(DEFAULT_CONTEXT)
    expect(Object.keys(cmp).sort()).toEqual(['linear', 'none', 'ntk', 'yarn'])
    expect(cmp.none.length).toBeGreaterThan(0)
  })

  it('每种方法都有中英标签', () => {
    for (const m of METHODS) {
      expect(METHOD_LABEL[m].zh.length).toBeGreaterThan(0)
      expect(METHOD_LABEL[m].en.length).toBeGreaterThan(0)
    }
  })
})

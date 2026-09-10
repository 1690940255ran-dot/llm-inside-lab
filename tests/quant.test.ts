import { describe, expect, it } from 'vitest'
import {
  DEFAULT_QUANT,
  NF4_CODEBOOK,
  bytesPerWeight,
  compareSchemes,
  fmtGB,
  makeWeights,
  memoryTable,
  quantize,
} from '../src/core/quant'

describe('makeWeights', () => {
  it('确定性', () => {
    expect(makeWeights(DEFAULT_QUANT)).toEqual(makeWeights(DEFAULT_QUANT))
  })

  it('按配置插入指定个数的离群值', () => {
    const w = makeWeights({ ...DEFAULT_QUANT, outlierScale: 40, outlierCount: 3 })
    expect(w.filter((x) => Math.abs(x) > 10).length).toBe(3)
  })

  it('离群值个数为 0 时不产生尖峰', () => {
    const w = makeWeights({ ...DEFAULT_QUANT, outlierCount: 0 })
    expect(Math.max(...w.map(Math.abs))).toBeLessThan(10)
  })

  it('长度正确', () => {
    expect(makeWeights({ ...DEFAULT_QUANT, n: 128 }).length).toBe(128)
  })
})

describe('quantize — 对称', () => {
  it('还原值一定落在 scale 的整数倍上', () => {
    const r = quantize({ ...DEFAULT_QUANT, mode: 'tensor', scheme: 'sym' })
    const scale = r.scales[0]
    for (let i = 0; i < r.dequant.length; i++) {
      expect(r.dequant[i] / scale).toBeCloseTo(r.codes[i], 6)
    }
  })

  it('码字不超过位宽能表示的范围', () => {
    for (const bits of [2, 3, 4, 8]) {
      const r = quantize({ ...DEFAULT_QUANT, bits, mode: 'tensor', scheme: 'sym' })
      const qmax = Math.pow(2, bits - 1) - 1
      for (const c of r.codes) {
        expect(c).toBeLessThanOrEqual(qmax)
        expect(c).toBeGreaterThanOrEqual(-qmax - 1)
      }
    }
  })

  it('有效档位数不超过理论档位数', () => {
    const r = quantize({ ...DEFAULT_QUANT, bits: 4, scheme: 'sym' })
    expect(r.usedLevels).toBeLessThanOrEqual(r.totalLevels)
    expect(r.totalLevels).toBe(16)
  })
})

describe('quantize — 非对称', () => {
  it('码字落在 [0, 2^bits-1]', () => {
    const bits = 4
    const r = quantize({ ...DEFAULT_QUANT, bits, scheme: 'asym', mode: 'tensor' })
    for (const c of r.codes) {
      expect(c).toBeGreaterThanOrEqual(0)
      expect(c).toBeLessThanOrEqual(Math.pow(2, bits) - 1)
    }
  })

  it('还原值 = (code - zero) * scale', () => {
    const r = quantize({ ...DEFAULT_QUANT, scheme: 'asym', mode: 'tensor' })
    const scale = r.scales[0]
    const zero = r.zeros[0]
    for (let i = 0; i < r.dequant.length; i++) {
      expect(r.dequant[i]).toBeCloseTo((r.codes[i] - zero) * scale, 6)
    }
  })
})

describe('quantize — NF4', () => {
  it('还原值一定是码本元素乘以 scale', () => {
    const r = quantize({ ...DEFAULT_QUANT, scheme: 'nf4', mode: 'tensor' })
    const scale = r.scales[0]
    for (let i = 0; i < r.dequant.length; i++) {
      expect(r.dequant[i]).toBeCloseTo(NF4_CODEBOOK[r.codes[i]] * scale, 6)
    }
  })

  it('最多用到 16 个档位', () => {
    const r = quantize({ ...DEFAULT_QUANT, scheme: 'nf4' })
    expect(r.totalLevels).toBe(16)
    expect(r.usedLevels).toBeLessThanOrEqual(16)
  })

  it('对无离群值的高斯权重，NF4 比对称 INT4 更准', () => {
    const cfg = { ...DEFAULT_QUANT, outlierCount: 0, mode: 'group' as const, groupSize: 32 }
    const sym = quantize({ ...cfg, scheme: 'sym' })
    const nf4 = quantize({ ...cfg, scheme: 'nf4' })
    expect(nf4.snrDb).toBeGreaterThan(sym.snrDb)
  })
})

describe('离群值的杀伤力', () => {
  it('离群值越大，per-tensor 的有效档位越少（单调不增）', () => {
    const at = (s: number) =>
      quantize({ ...DEFAULT_QUANT, outlierScale: s, outlierCount: 2, mode: 'tensor' }).usedLevels
    const series = [1, 10, 20, 40, 80].map(at)
    for (let i = 1; i < series.length; i++) expect(series[i]).toBeLessThanOrEqual(series[i - 1])
    expect(series[0]).toBeGreaterThan(series[series.length - 1])
  })

  it('离群值把有效档位打到远低于理论值', () => {
    const r = quantize({ ...DEFAULT_QUANT, bits: 4, outlierScale: 40, outlierCount: 2, mode: 'tensor' })
    expect(r.usedLevels).toBeLessThan(10)
    expect(r.usedLevels).toBeLessThanOrEqual(r.totalLevels / 2)
  })

  it('去掉离群值后有效档位显著回升', () => {
    const withOut = quantize({ ...DEFAULT_QUANT, outlierScale: 40, outlierCount: 2, mode: 'tensor' })
    const without = quantize({ ...DEFAULT_QUANT, outlierScale: 40, outlierCount: 0, mode: 'tensor' })
    expect(without.usedLevels).toBeGreaterThan(withOut.usedLevels)
  })

  it('细粒度分组能大幅挽回离群值造成的精度损失', () => {
    const base = { ...DEFAULT_QUANT, outlierScale: 40, outlierCount: 2 }
    const t = quantize({ ...base, mode: 'tensor' })
    const g = quantize({ ...base, mode: 'group', groupSize: 32 })
    expect(g.snrDb).toBeGreaterThan(t.snrDb + 3)
  })
})

describe('位数与精度', () => {
  it('无离群值时，位数越多 SQNR 越高', () => {
    const at = (bits: number) =>
      quantize({ ...DEFAULT_QUANT, bits, outlierCount: 0, mode: 'tensor', scheme: 'sym' }).snrDb
    const s = [2, 3, 4, 6, 8].map(at)
    for (let i = 1; i < s.length; i++) expect(s[i]).toBeGreaterThan(s[i - 1])
  })

  it('每多一位大约多 6 dB（相邻两档至少 +4 dB）', () => {
    const at = (bits: number) =>
      quantize({ ...DEFAULT_QUANT, bits, outlierCount: 0, mode: 'tensor', scheme: 'sym' }).snrDb
    for (let b = 3; b <= 6; b++) expect(at(b) - at(b - 1)).toBeGreaterThan(4)
  })
})

describe('误差指标', () => {
  it('MSE 与最大误差非负，且最大误差 ≥ RMSE', () => {
    const r = quantize(DEFAULT_QUANT)
    expect(r.mse).toBeGreaterThanOrEqual(0)
    expect(r.maxAbsErr).toBeGreaterThanOrEqual(0)
    expect(r.maxAbsErr).toBeGreaterThanOrEqual(Math.sqrt(r.mse))
  })

  it('误差确实是原始值与还原值之差', () => {
    const r = quantize(DEFAULT_QUANT)
    let se = 0
    for (let i = 0; i < r.values.length; i++) se += (r.values[i] - r.dequant[i]) ** 2
    expect(r.mse).toBeCloseTo(se / r.values.length, 10)
  })

  it('排序下标能还原成升序', () => {
    const r = quantize(DEFAULT_QUANT)
    const sorted = r.order.map((i) => r.values[i])
    for (let i = 1; i < sorted.length; i++) expect(sorted[i]).toBeGreaterThanOrEqual(sorted[i - 1])
  })
})

describe('bytesPerWeight / memoryTable', () => {
  it('裸位数换算正确', () => {
    expect(bytesPerWeight(16, 'tensor', 0)).toBe(2)
    expect(bytesPerWeight(8, 'tensor', 0)).toBe(1)
    expect(bytesPerWeight(4, 'tensor', 0)).toBe(0.5)
  })

  it('group-wise 要把 scale/zero 的元数据算进去', () => {
    expect(bytesPerWeight(4, 'group', 128)).toBeCloseTo(0.5 + 4 / 128, 9)
    expect(bytesPerWeight(4, 'group', 32)).toBeCloseTo(0.5 + 4 / 32, 9)
    expect(bytesPerWeight(4, 'group', 32)).toBeGreaterThan(bytesPerWeight(4, 'group', 128))
  })

  it('7B 的 FP16 约 13 GB，4bit group=128 约 3.5 GB', () => {
    const t = memoryTable(7e9)
    const fp16 = t.find((x) => x.label === 'FP16')!
    const g128 = t.find((x) => x.label === 'INT4 group=128')!
    expect(fp16.bytes / 1024 ** 3).toBeCloseTo(13.04, 1)
    expect(g128.bytes / 1024 ** 3).toBeGreaterThan(3.2)
    expect(g128.bytes / 1024 ** 3).toBeLessThan(3.8)
  })

  it('字节数随位数单调下降', () => {
    const t = memoryTable(1e9)
    const byLabel = (l: string) => t.find((x) => x.label === l)!.bytes
    expect(byLabel('FP16')).toBeGreaterThan(byLabel('INT8'))
    expect(byLabel('INT8')).toBeGreaterThan(byLabel('INT4 group=128'))
    expect(byLabel('INT4 group=128')).toBeGreaterThan(byLabel('INT3 group=128'))
  })
})

describe('fmtGB', () => {
  it('各量级格式化正确', () => {
    expect(fmtGB(2 * 1024 ** 3)).toBe('2.00 GB')
    expect(fmtGB(512 * 1024 ** 2)).toBe('512.0 MB')
    expect(fmtGB(4096)).toBe('4.0 KB')
  })
})

describe('compareSchemes', () => {
  it('返回全部七种方案，且 SQNR 有序', () => {
    const list = compareSchemes(DEFAULT_QUANT)
    expect(list.length).toBe(7)
    const get = (id: string) => list.find((x) => x.id === id)!
    // 有离群值时，per-tensor 4bit 明显差于 group-wise
    expect(get('g4').snrDb).toBeGreaterThan(get('t4').snrDb)
    // 位数越多越好
    expect(get('t8').snrDb).toBeGreaterThan(get('t4').snrDb)
  })

  it('字节/权重一列与 bytesPerWeight 一致', () => {
    const list = compareSchemes(DEFAULT_QUANT)
    const g4 = list.find((x) => x.id === 'g4')!
    expect(g4.bytesPerWeight).toBeCloseTo(bytesPerWeight(4, 'group', 32), 9)
  })
})

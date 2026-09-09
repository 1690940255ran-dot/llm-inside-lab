import { describe, it, expect } from 'vitest'
import {
  sampleNext,
  argmax,
  entropyOf,
  candidateCount,
  DEFAULT_SAMPLING,
} from '../src/core/sampling'

/** logits = ln(p) 时 softmax(logits) === p，用来构造精确已知的分布 */
const logitsFor = (probs: number[]) => probs.map((p) => Math.log(p))
const P = [0.5, 0.3, 0.15, 0.05]

const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0)

describe('sampleNext · 基本性质', () => {
  it('base 就是 softmax(logits)，且与构造的分布一致', () => {
    const s = sampleNext(logitsFor(P), DEFAULT_SAMPLING, 1)
    P.forEach((p, i) => expect(s.base[i]).toBeCloseTo(p, 10))
    expect(sum(s.base)).toBeCloseTo(1, 10)
  })

  it('final 始终是合法分布', () => {
    const s = sampleNext(logitsFor(P), { temperature: 0.7, topK: 3, topP: 0.9 }, 42)
    expect(sum(s.final)).toBeCloseTo(1, 10)
    s.final.forEach((p) => expect(p).toBeGreaterThanOrEqual(0))
  })

  it('被过滤掉的位置概率恰好为 0，且 kept 与之对应', () => {
    const s = sampleNext(logitsFor(P), { temperature: 1, topK: 2, topP: 1 }, 7)
    s.final.forEach((p, i) => expect(p > 0).toBe(s.kept[i]))
  })

  it('抽中的下标一定在候选集内', () => {
    for (let seed = 0; seed < 25; seed++) {
      const s = sampleNext(logitsFor(P), { temperature: 1.2, topK: 2, topP: 1 }, seed)
      expect(s.kept[s.chosen]).toBe(true)
    }
  })

  it('同一 seed 结果完全可复现', () => {
    const cfg = { temperature: 1.3, topK: 0, topP: 0.95 }
    expect(sampleNext(logitsFor(P), cfg, 123).chosen).toBe(
      sampleNext(logitsFor(P), cfg, 123).chosen,
    )
  })
})

describe('温度', () => {
  it('T < 1 拉大差距（最大概率变高、熵变低）', () => {
    const s = sampleNext(logitsFor(P), { ...DEFAULT_SAMPLING, temperature: 0.5 }, 1)
    expect(Math.max(...s.tempered)).toBeGreaterThan(Math.max(...s.base))
    expect(entropyOf(s.tempered)).toBeLessThan(entropyOf(s.base))
  })

  it('T > 1 抹平差距（最大概率变低、熵变高）', () => {
    const s = sampleNext(logitsFor(P), { ...DEFAULT_SAMPLING, temperature: 2 }, 1)
    expect(Math.max(...s.tempered)).toBeLessThan(Math.max(...s.base))
    expect(entropyOf(s.tempered)).toBeGreaterThan(entropyOf(s.base))
  })

  it('T = 1 时 tempered === base', () => {
    const s = sampleNext(logitsFor(P), DEFAULT_SAMPLING, 1)
    s.base.forEach((p, i) => expect(s.tempered[i]).toBeCloseTo(p, 12))
  })

  it('T 极低时退化为贪心：无论 seed 都选 argmax', () => {
    for (const seed of [0, 5, 99, 12345]) {
      const s = sampleNext(logitsFor(P), { temperature: 0.01, topK: 0, topP: 1 }, seed)
      expect(s.chosen).toBe(argmax(P))
    }
  })

  it('温度不改变概率排序', () => {
    const s = sampleNext(logitsFor(P), { ...DEFAULT_SAMPLING, temperature: 0.3 }, 1)
    const order = (xs: number[]) =>
      xs.map((p, i) => [p, i] as const).sort((a, b) => b[0] - a[0]).map(([, i]) => i)
    expect(order(s.tempered)).toEqual(order(s.base))
  })
})

describe('top-k', () => {
  it('恰好保留 k 个候选', () => {
    for (const k of [1, 2, 3]) {
      const s = sampleNext(logitsFor(P), { temperature: 1, topK: k, topP: 1 }, 3)
      expect(candidateCount(s)).toBe(k)
    }
  })

  it('保留的是概率最大的那 k 个', () => {
    const s = sampleNext(logitsFor(P), { temperature: 1, topK: 2, topP: 1 }, 3)
    expect(s.kept).toEqual([true, true, false, false])
  })

  it('k = 0 或 k >= 词表大小时不启用', () => {
    expect(candidateCount(sampleNext(logitsFor(P), { temperature: 1, topK: 0, topP: 1 }, 3))).toBe(4)
    expect(candidateCount(sampleNext(logitsFor(P), { temperature: 1, topK: 9, topP: 1 }, 3))).toBe(4)
  })

  it('k = 1 等价于贪心', () => {
    const s = sampleNext(logitsFor(P), { temperature: 1, topK: 1, topP: 1 }, 777)
    expect(s.chosen).toBe(argmax(P))
  })
})

describe('top-p（核采样）', () => {
  it('保留累计概率刚跨过 p 的最少候选', () => {
    // 0.5 -> 0.8 跨过 0.7，所以是 2 个
    const s = sampleNext(logitsFor(P), { temperature: 1, topK: 0, topP: 0.7 }, 3)
    expect(candidateCount(s)).toBe(2)
    expect(s.kept).toEqual([true, true, false, false])
  })

  it('p 恰好等于第一个概率时只留 1 个（>= 阈值即停）', () => {
    const s = sampleNext(logitsFor(P), { temperature: 1, topK: 0, topP: 0.5 }, 3)
    expect(candidateCount(s)).toBe(1)
  })

  it('p 越大候选集单调不减', () => {
    let prev = 0
    for (const p of [0.4, 0.6, 0.8, 0.95, 1]) {
      const n = candidateCount(sampleNext(logitsFor(P), { temperature: 1, topK: 0, topP: p }, 3))
      expect(n).toBeGreaterThanOrEqual(prev)
      prev = n
    }
  })

  it('p = 1 时不启用', () => {
    expect(candidateCount(sampleNext(logitsFor(P), DEFAULT_SAMPLING, 3))).toBe(4)
  })

  it('top-k 与 top-p 叠加时取更严格的那个', () => {
    const s = sampleNext(logitsFor(P), { temperature: 1, topK: 3, topP: 0.7 }, 3)
    expect(candidateCount(s)).toBe(2)
  })
})

describe('argmax / entropyOf', () => {
  it('argmax 取最大值下标', () => {
    expect(argmax([0.1, 0.7, 0.2])).toBe(1)
    expect(argmax([0.9])).toBe(0)
  })

  it('argmax 平局取最靠前的', () => {
    expect(argmax([0.5, 0.5])).toBe(0)
  })

  it('one-hot 分布熵为 0', () => {
    expect(entropyOf([1, 0, 0, 0])).toBeCloseTo(0, 12)
  })

  it('均匀分布熵为 ln(n)', () => {
    expect(entropyOf([0.25, 0.25, 0.25, 0.25])).toBeCloseTo(Math.log(4), 12)
  })

  it('均匀分布是熵最大的分布', () => {
    expect(entropyOf([0.25, 0.25, 0.25, 0.25])).toBeGreaterThan(entropyOf(P))
  })
})

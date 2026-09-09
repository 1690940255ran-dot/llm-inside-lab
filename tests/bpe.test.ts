import { describe, it, expect } from 'vitest'
import {
  pretokenize,
  trainBPE,
  encode,
  encodeByChar,
  computeStats,
} from '../src/core/bpe'

describe('pretokenize', () => {
  it('把连续汉字、连续字母数字各自聚成一个单元', () => {
    expect(pretokenize('注意力abc123')).toEqual(['注意力', 'abc123'])
  })

  it('标点独立成单元，不污染词表', () => {
    expect(pretokenize('a,b')).toEqual(['a', ',', 'b'])
  })

  it('空白独立成单元并被保留', () => {
    expect(pretokenize('hi there')).toEqual(['hi', ' ', 'there'])
  })

  it('中英切换处断开', () => {
    expect(pretokenize('模型model')).toEqual(['模型', 'model'])
  })

  it('拼接所有单元必须还原原文（无损预分词）', () => {
    const text = '大模型 LLM 的 attention, 很妙!\n换行也算'
    expect(pretokenize(text).join('')).toBe(text)
  })
})

describe('trainBPE', () => {
  it('真的从语料里学合并，而不是硬编码', () => {
    // ab 出现 4 次，是唯一高频对，第一条合并必须是 a+b
    const m = trainBPE('ab ab ab ab', 5)
    expect(m.merges.length).toBeGreaterThan(0)
    expect(m.merges[0].a).toBe('a')
    expect(m.merges[0].b).toBe('b')
    expect(m.merges[0].count).toBe(4)
    expect(m.merges[0].rank).toBe(0)
  })

  it('同一语料两次训练结果完全一致（可复现）', () => {
    const a = trainBPE('the cat sat on the mat, the cat ran', 30)
    const b = trainBPE('the cat sat on the mat, the cat ran', 30)
    expect(a.merges.map((m) => m.a + m.b)).toEqual(b.merges.map((m) => m.a + m.b))
    expect(a.idToToken).toEqual(b.idToToken)
  })

  it('合并次数不超过上限，且 rank 严格递增', () => {
    const m = trainBPE(undefined, 20)
    expect(m.merges.length).toBeLessThanOrEqual(20)
    for (let i = 0; i < m.merges.length; i++) expect(m.merges[i].rank).toBe(i)
  })

  it('出现次数少于 2 的对不再合并（提前收敛）', () => {
    // 每个字符只出现一次，没有任何频次 >= 2 的相邻对
    const m = trainBPE('abcdef', 50)
    expect(m.merges.length).toBe(0)
  })

  it('语料越大 / 合并越多，词表单调变大', () => {
    const few = trainBPE(undefined, 20)
    const many = trainBPE(undefined, 200)
    expect(many.idToToken.length).toBeGreaterThan(few.idToToken.length)
  })

  it('词表里每个 token 的 id 与 tokenToId 双向一致', () => {
    const m = trainBPE(undefined, 60)
    m.idToToken.forEach((tok, i) => expect(m.tokenToId.get(tok)).toBe(i))
  })
})

describe('encode', () => {
  it('token 拼接必须还原原文（分词无损）', () => {
    const m = trainBPE(undefined, 120)
    const text = '大模型的注意力机制 attention is all you need.'
    const { tokens } = encode(text, m)
    expect(tokens.map((t) => t.text).join('')).toBe(text)
  })

  it('学过的高频对会被合并成一个 token', () => {
    const m = trainBPE('ab ab ab ab', 5)
    const { tokens } = encode('ab', m)
    expect(tokens).toHaveLength(1)
    expect(tokens[0].text).toBe('ab')
    expect(tokens[0].oov).toBe(false)
  })

  it('语料里没出现过的字符标为 OOV', () => {
    const m = trainBPE('ab ab ab ab', 5)
    const { tokens } = encode('z', m)
    expect(tokens[0].oov).toBe(true)
    expect(tokens[0].id).toBe(-1)
  })

  it('合并过程逐步记录，且符号数每步减一', () => {
    const m = trainBPE('abab abab abab', 10)
    const { stepsByUnit } = encode('abab', m)
    const steps = stepsByUnit[0]
    expect(steps.length).toBeGreaterThan(1)
    for (let i = 1; i < steps.length; i++) {
      expect(steps[i].symbols.length).toBe(steps[i - 1].symbols.length - 1)
      expect(steps[i].pair).toBeDefined()
    }
  })

  it('BPE 的 token 数不多于字符级', () => {
    const m = trainBPE(undefined, 160)
    const text = '注意力机制是大模型的核心 attention'
    expect(encode(text, m).tokens.length).toBeLessThanOrEqual(encodeByChar(text).length)
  })
})

describe('computeStats', () => {
  it('压缩率 = 字符数 / token 数', () => {
    const m = trainBPE(undefined, 160)
    const text = 'the transformer architecture'
    const { tokens } = encode(text, m)
    const s = computeStats(tokens, text)
    expect(s.charCount).toBe([...text].length)
    expect(s.tokenCount).toBe(tokens.length)
    expect(s.compression).toBeCloseTo(s.charCount / s.tokenCount, 10)
    expect(s.compression).toBeGreaterThanOrEqual(1)
  })

  it('token 数为 0 时不除零', () => {
    const s = computeStats([], 'abc')
    expect(Number.isFinite(s.compression)).toBe(true)
  })
})

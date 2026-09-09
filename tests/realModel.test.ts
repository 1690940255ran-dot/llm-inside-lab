import { describe, it, expect } from 'vitest'
import {
  REAL_MODELS,
  specOf,
  prettifyLabel,
  buildTokenLabels,
  alignLabels,
  softmaxTop,
  summarizeKV,
  explainError,
  toArray,
  pickThreads,
  pickDevice,
  isNodeRuntime,
  mergeFragments,
  topTokenLabel,
} from '../src/core/realModel'

/**
 * 这个文件里的大部分断言都是「回归测试」——每一条都对应一个真实踩过的坑，
 * 详见 scripts/probe-attentions.mjs 与 scripts/probe-tokenizers.mjs 的实测输出。
 */

describe('REAL_MODELS 配置（回归：dtype 与仓库文件名必须对得上）', () => {
  it('每个模型都显式声明 dtype', () => {
    for (const m of REAL_MODELS) {
      expect(m.dtype, `${m.id} 缺少 dtype`).toBeTruthy()
    }
  })

  it('Xenova/gpt2 不能用 q8 —— 该仓库没有 model_quantized.onnx，会 404', () => {
    const gpt2 = specOf('Xenova/gpt2')
    expect(gpt2).toBeDefined()
    // 实测：hf-mirror 上 Xenova/gpt2/onnx/ 只有 model_int8 / model_uint8 / model_q4 / model_fp16 / model.onnx
    expect(gpt2!.dtype).not.toBe('q8')
    expect(gpt2!.dtype).toBe('int8')
  })

  it('模型 id 不重复，且都带体积说明（避免用户误点 500MB 下载）', () => {
    const ids = REAL_MODELS.map((m) => m.id)
    expect(new Set(ids).size).toBe(ids.length)
    for (const m of REAL_MODELS) expect(m.size).toMatch(/\d+\s*MB/)
  })

  it('默认（第一个）应当是体积最小的那个', () => {
    const mb = (s: string) => parseFloat(s)
    const sizes = REAL_MODELS.map((m) => mb(m.size))
    expect(sizes[0]).toBe(Math.min(...sizes))
  })

  it('中英文标签都提供，界面切语言不会露出空白', () => {
    for (const m of REAL_MODELS) {
      expect(m.label.length).toBeGreaterThan(0)
      expect(m.labelEn.length).toBeGreaterThan(0)
    }
  })

  it('specOf 查不到时返回 undefined 而不是抛错', () => {
    expect(specOf('not/exists')).toBeUndefined()
  })
})

describe('prettifyLabel', () => {
  it('空串回退成 #id', () => {
    expect(prettifyLabel('', 1234)).toBe('#1234')
  })

  it('普通文本原样返回', () => {
    expect(prettifyLabel('注意力', 1)).toBe('注意力')
    expect(prettifyLabel(' capital', 2)).toBe(' capital')
  })

  it('纯空白变成可见记号（否则热力图上是一片空白）', () => {
    expect(prettifyLabel(' ', 1)).toBe('␣')
    expect(prettifyLabel('   ', 1)).toBe('␣␣␣')
  })

  it('换行 / 制表符换成可见记号', () => {
    expect(prettifyLabel('\n', 1)).toBe('⏎')
    expect(prettifyLabel('a\tb', 1)).toBe('a⇥b')
  })

  it('字节级 BPE 的替换字符原样保留（这是真实行为，不该隐藏）', () => {
    // 实测：GPT-2 对「注」逐 id 解码就是 U+FFFD 碎片
    expect(prettifyLabel('\uFFFD', 37345)).toBe('\uFFFD')
  })
})

describe('buildTokenLabels（回归：P1 标签错位）', () => {
  const decode = (map: Record<number, string>) => (id: number) => map[id] ?? ''

  it('标签数量恒等于 ids 数量 —— 不做任何 BOS 假设', () => {
    const ids = [785, 6722, 315]
    const labels = buildTokenLabels(ids, decode({ 785: 'The', 6722: ' capital', 315: ' of' }))
    expect(labels).toEqual(['The', ' capital', ' of'])
    expect(labels).toHaveLength(ids.length)
  })

  it('无 BOS 的模型（Qwen2.5）第一个标签就是真实内容，不会被塞进 <s>', () => {
    // 实测 Qwen2.5-0.5B：'注意力机制是大模型的核心' -> 6 个 token，首个 id 108260 = '注意力'
    const ids = [108260, 45350, 20412, 26288, 104949, 105205]
    const labels = buildTokenLabels(
      ids,
      decode({
        108260: '注意力',
        45350: '机制',
        20412: '是',
        26288: '大',
        104949: '模型',
        105205: '的核心',
      }),
    )
    expect(labels[0]).toBe('注意力')
    expect(labels).not.toContain('<s>')
    expect(labels.join('')).toBe('注意力机制是大模型的核心')
  })

  it('解码函数抛错时单个标签降级为 #id，不影响其他标签', () => {
    const labels = buildTokenLabels([1, 2, 3], (id) => {
      if (id === 2) throw new Error('boom')
      return `t${id}`
    })
    expect(labels).toEqual(['t1', '#2', 't3'])
  })

  it('空输入返回空数组', () => {
    expect(buildTokenLabels([], () => 'x')).toEqual([])
  })
})

describe('alignLabels（回归：绝不在开头补 BOS）', () => {
  it('长度相等时原样返回', () => {
    const a = ['x', 'y']
    expect(alignLabels(a, 2)).toEqual(['x', 'y'])
  })

  it('偏长时截断尾部', () => {
    expect(alignLabels(['a', 'b', 'c'], 2)).toEqual(['a', 'b'])
  })

  it('偏短时在【末尾】补占位，首个标签绝不被顶掉', () => {
    const out = alignLabels(['a', 'b'], 4)
    expect(out[0]).toBe('a') // 这一条就是旧实现 tokens.unshift('<s>') 的反例
    expect(out[1]).toBe('b')
    expect(out).toHaveLength(4)
    expect(out.slice(2).every((s) => s.startsWith('?'))).toBe(true)
  })

  it('n <= 0 时不炸', () => {
    expect(alignLabels(['a'], 0)).toEqual(['a'])
    expect(alignLabels(['a'], -3)).toEqual(['a'])
  })
})

describe('softmaxTop', () => {
  it('按概率降序返回前 k 个', () => {
    const { top } = softmaxTop([1, 5, 3, 2], 3)
    expect(top.map((t) => t.id)).toEqual([1, 2, 3])
    for (let i = 1; i < top.length; i++) expect(top[i].p).toBeLessThanOrEqual(top[i - 1].p)
  })

  it('概率在完整词表上归一化，前 k 个之和 <= 1', () => {
    const { top } = softmaxTop([1, 2, 3, 4, 5], 2)
    expect(top.reduce((a, b) => a + b.p, 0)).toBeLessThanOrEqual(1 + 1e-12)
  })

  it('数值稳定：极大 logits 不产生 NaN/Infinity', () => {
    // 实测 GPT-2 int8 的 logits 在 -110 附近，另一些模型能到 +1000，必须先减最大值
    const { top, entropy } = softmaxTop([1000, 999, 998], 3)
    for (const t of top) expect(Number.isFinite(t.p)).toBe(true)
    expect(Number.isFinite(entropy)).toBe(true)
    expect(top[0].p).toBeGreaterThan(0)
  })

  it('极大负 logits 同样稳定', () => {
    const { top } = softmaxTop([-109.489, -109.541, -109.713], 3)
    expect(top[0].p).toBeGreaterThan(0.3)
    expect(top.reduce((a, b) => a + b.p, 0)).toBeCloseTo(1, 6)
  })

  it('均匀分布的熵 = log2(n)', () => {
    const { entropy } = softmaxTop([0, 0, 0, 0], 4)
    expect(entropy).toBeCloseTo(2, 10)
  })

  it('one-hot 分布熵趋近 0', () => {
    const { entropy } = softmaxTop([100, 0, 0], 3)
    expect(entropy).toBeCloseTo(0, 6)
  })

  it('k 超出长度时自动收敛到长度', () => {
    expect(softmaxTop([1, 2], 99).top).toHaveLength(2)
  })

  it('空输入返回空结果', () => {
    expect(softmaxTop([], 5)).toEqual({ top: [], entropy: 0 })
  })
})

describe('summarizeKV（真实 present.* 形状统计）', () => {
  it('12 层 GPT-2：24 个 [1,12,5,64] 张量 -> 12 层、正确字节数', () => {
    const dims = Array.from({ length: 24 }, () => [1, 12, 5, 64])
    const kv = summarizeKV(dims)!
    expect(kv.nLayers).toBe(12)
    expect(kv.nKVHeads).toBe(12)
    expect(kv.seqLen).toBe(5)
    expect(kv.headDim).toBe(64)
    expect(kv.elems).toBe(24 * 12 * 5 * 64)
    expect(kv.bytes).toBe(kv.elems * 4)
  })

  it('GQA 模型的 KV 头数少于 Query 头数，统计按实际形状走', () => {
    // Qwen2.5-0.5B：14 个 Q 头，但只有 2 个 KV 头
    const dims = Array.from({ length: 48 }, () => [1, 2, 8, 64])
    const kv = summarizeKV(dims)!
    expect(kv.nLayers).toBe(24)
    expect(kv.nKVHeads).toBe(2)
  })

  it('自定义每元素字节数（fp16 减半）', () => {
    const dims = [[1, 4, 2, 8], [1, 4, 2, 8]]
    expect(summarizeKV(dims, 2)!.bytes).toBe(summarizeKV(dims, 4)!.bytes / 2)
  })

  it('没有合法形状时返回 null 而不是 0/NaN', () => {
    expect(summarizeKV([])).toBeNull()
    expect(summarizeKV([[1, 2]])).toBeNull()
  })

  it('忽略维数不对的条目', () => {
    const kv = summarizeKV([[1, 4, 2, 8], [1, 2], [1, 4, 2, 8]])!
    expect(kv.elems).toBe(2 * 1 * 4 * 2 * 8)
  })
})

describe('explainError（把库的原始报错翻译成能指导操作的话）', () => {
  it('tokenizer_class 报错指向 CORS / 防盗链，并提示刷新页面（探测结果被 memoize，原地重试无效）', () => {
    // 这就是 remotePathTemplate 被错误覆盖 或 hf-mirror 剥 CORS 头 时用户看到的报错
    const msg = explainError(
      new Error("Cannot read properties of undefined (reading 'tokenizer_class')"),
    )
    expect(msg).toContain('配置文件')
    expect(msg).toContain('Referer')
    expect(msg).toContain('刷新页面')
  })

  it('权重缺失报错指向 dtype 配错，并带上文件名', () => {
    const msg = explainError(
      new Error('Could not locate file: "https://hf-mirror.com/Xenova/gpt2/resolve/main/onnx/model_quantized.onnx".'),
    )
    expect(msg).toContain('dtype')
    expect(msg).toContain('model_quantized.onnx')
  })

  it('网络类报错提到 Referer 防盗链与刷新重试', () => {
    const msg = explainError(new Error('Failed to fetch'))
    expect(msg).toContain('Referer')
    expect(msg).toContain('镜像')
    expect(msg).toContain('刷新页面')
  })

  it('未知报错原样透出，不吞信息', () => {
    expect(explainError(new Error('something odd'))).toBe('something odd')
  })

  it('非 Error 对象不会炸', () => {
    expect(typeof explainError('plain string')).toBe('string')
    expect(typeof explainError(null)).toBe('string')
  })
})

describe('topTokenLabel（回归：中文 prompt 的 top 候选整列 `` 没法分辨）', () => {
  it('字节碎片候选显示 #id，而不是千篇一律的 ``', () => {
    expect(topTokenLabel('\uFFFD', 123456)).toBe('#123456')
    expect(topTokenLabel('\uFFFD', 789)).toBe('#789')
  })

  it('正常候选原样显示', () => {
    expect(topTokenLabel(' token', 1)).toBe(' token')
    expect(topTokenLabel('的', 2)).toBe('的')
  })

  it('空白候选仍走可见记号', () => {
    expect(topTokenLabel(' ', 3)).toBe('␣')
  })
})

describe('mergeFragments（字节碎片的合并解码）', () => {
  it('连续的 `` 碎片合并后能解出真实字符', () => {
    // 模拟 GPT-2 把「注」切成 3 个 UTF-8 字节 token：单独解全是，合起来解出「注」
    const ids = [100, 101, 102]
    const decode = (part: number[]) =>
      part.length === 1 ? '\uFFFD' : part.length === 3 ? '注' : '\uFFFD'
    const out = mergeFragments(ids, decode)
    expect(out).toHaveLength(1)
    expect(out[0].ids).toEqual([100, 101, 102])
    expect(out[0].label).toBe('注')
  })

  it('解不回来的坏字节逐 id 保留，不伪造可读文本', () => {
    const decode = () => '\uFFFD' // 怎么解都是
    const out = mergeFragments([1, 2], decode)
    expect(out).toHaveLength(2)
    expect(out.map((t) => t.label)).toEqual(['\uFFFD', '\uFFFD'])
    expect(out.map((t) => t.ids)).toEqual([[1], [2]])
  })

  it('正常 token 不受影响（一一对应）', () => {
    const decode = (part: number[]) => ({ 5: 'The', 6: ' capital' })[part[0]] ?? ''
    const out = mergeFragments([5, 6], decode)
    expect(out.map((t) => t.label)).toEqual(['The', ' capital'])
    expect(out.every((t) => t.ids.length === 1)).toBe(true)
  })

  it('碎片段与正常 token 混排时各自正确分组', () => {
    // 模拟 GPT-2 的「注意力是大模型的核心」：碎片段 + 独立字 + 碎片段
    const bytes = [10, 11, 12] // -> '注'
    const decode = (part: number[]) => {
      if (part.length > 1) return '注'
      if (part[0] === 20) return '是'
      if (part[0] === 10 || part[0] === 11 || part[0] === 12) return '\uFFFD'
      if (part[0] === 30) return '大'
      return ''
    }
    const out = mergeFragments([...bytes, 20, 30], decode)
    expect(out.map((t) => t.label)).toEqual(['注', '是', '大'])
    expect(out[0].ids).toEqual(bytes)
  })

  it('空输入返回空数组', () => {
    expect(mergeFragments([], () => 'x')).toEqual([])
  })

  it('解码函数抛错时按坏字节处理，不炸', () => {
    const out = mergeFragments([1], () => {
      throw new Error('boom')
    })
    expect(out).toHaveLength(1)
    expect(out[0].label).toBe('#1') // 空 label 走 prettifyLabel 的 #id 回退
  })

  it('末尾单个碎片不会被误并（没有相邻碎片可拼）', () => {
    const decode = (part: number[]) => (part[0] === 9 ? '\uFFFD' : 'a')
    const out = mergeFragments([9], decode)
    expect(out).toHaveLength(1)
    expect(out[0].label).toBe('\uFFFD')
  })
})

describe('pickDevice / isNodeRuntime（回归：两个 onnxruntime 认的后端名不一样）', () => {
  it('测试跑在 Node 里，应识别为 Node 运行时', () => {
    expect(isNodeRuntime()).toBe(true)
  })

  it('Node 下必须返回 cpu —— 传 wasm 会被 onnxruntime-node 拒绝', () => {
    // 实测报错：Unsupported device: "wasm". Should be one of: dml, webgpu, cpu.
    expect(pickDevice()).toBe('cpu')
  })
})

describe('pickThreads（回归：GitHub Pages 不发 COOP/COEP，多线程 WASM 会挂）', () => {
  it('非跨源隔离时锁单线程 —— 这就是 GitHub Pages 的情况', () => {
    expect(pickThreads(true, false, 16)).toBe(1)
  })

  it('没有 SharedArrayBuffer 时锁单线程', () => {
    expect(pickThreads(false, true, 16)).toBe(1)
  })

  it('两个条件都满足才开多线程，且上限 4', () => {
    expect(pickThreads(true, true, 16)).toBe(4)
    expect(pickThreads(true, true, 2)).toBe(2)
  })

  it('核数异常时至少返回 1，不返回 0 / NaN', () => {
    expect(pickThreads(true, true, 0)).toBe(1)
    expect(pickThreads(true, true, NaN)).toBe(1)
    expect(pickThreads(true, true, -8)).toBe(1)
  })
})

describe('toArray', () => {
  it('Tensor 风格对象走 tolist()', () => {
    expect(toArray({ tolist: () => [[1, 2]] })).toEqual([[1, 2]])
  })

  it('普通数组原样返回', () => {
    expect(toArray([1, 2, 3])).toEqual([1, 2, 3])
  })

  it('null / undefined 不抛错', () => {
    expect(toArray(null)).toBeNull()
    expect(toArray(undefined)).toBeUndefined()
  })
})

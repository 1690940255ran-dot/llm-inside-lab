/**
 * 集成测试：真的下载权重、真的跑一次前向。
 *
 * 默认跳过 —— 它要从网络下 100 MB 以上的权重，不该在每次 `npm test` 时跑。
 * 手动执行：
 *     REAL_MODEL_TEST=1 npm test
 *     REAL_MODEL_TEST=1 REAL_MODEL_ID=onnx-community/Qwen2.5-0.5B-Instruct npm test
 *
 * 它验证的是浏览器里那条完整链路（load → runNextToken），只不过后端换成 onnxruntime-node。
 * 之所以值得存在：注意力那条路走不通就是在这一层才暴露出来的，
 * 纯单测覆盖不到"库到底会返回什么"。
 */
import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import { load, runNextToken, runTokenize, unload, isReady, getDevice } from '../src/core/realModel'

const ENABLED = process.env.REAL_MODEL_TEST === '1'
const MODEL_ID = process.env.REAL_MODEL_ID || 'onnx-community/SmolLM2-135M-Instruct-ONNX'
const MIRROR = process.env.HF_HOST || 'https://hf-mirror.com'

describe.skipIf(!ENABLED)('真实模型端到端（需要网络）', () => {
  beforeAll(async () => {
    await load(MODEL_ID, MIRROR, (pct, file) => {
      if (pct === 100 && file) console.log(`  [downloaded] ${file}`)
    })
  }, 600_000)

  afterAll(() => unload())

  it('加载成功并选出推理后端', () => {
    expect(isReady()).toBe(true)
    // Node 下必须是 cpu：onnxruntime-node 不认 'wasm'
    expect(getDevice()).toBe('cpu')
  })

  it('分词标签数量与 ids 一致，且没有 <s> 占位', async () => {
    const r = await runTokenize('The capital of France is')
    expect(r.ids.length).toBe(r.tokens.length)
    expect(r.tokens).not.toContain('<s>')
    expect(r.tokens.join('')).toContain('capital')
  })

  it('前向返回合法的真实分布', async () => {
    const r = await runNextToken('The capital of France is', 10)
    // 词表规模应当是真实量级
    expect(r.vocabSize).toBeGreaterThan(10_000)
    expect(r.logits.length).toBe(r.vocabSize)
    // top 按概率降序，且都是有限数
    expect(r.top.length).toBe(10)
    for (let i = 1; i < r.top.length; i++) {
      expect(r.top[i].p).toBeLessThanOrEqual(r.top[i - 1].p)
    }
    for (const tk of r.top) {
      expect(Number.isFinite(tk.p)).toBe(true)
      expect(Number.isFinite(tk.logit)).toBe(true)
      expect(tk.label.length).toBeGreaterThan(0)
    }
    // 熵应当落在 (0, log2(V)) 内
    expect(r.entropy).toBeGreaterThan(0)
    expect(r.entropy).toBeLessThan(Math.log2(r.vocabSize))
    console.log(
      `  [top5] ${r.top.slice(0, 5).map((x) => `${JSON.stringify(x.label)} ${(x.p * 100).toFixed(1)}%`).join('  ')}`,
    )
  }, 120_000)

  it('KV cache 形状能被真实读出来', async () => {
    const r = await runNextToken('The capital of France is', 5)
    expect(r.kv).not.toBeNull()
    expect(r.kv!.nLayers).toBeGreaterThan(0)
    expect(r.kv!.seqLen).toBe(r.ids.length)
    expect(r.kv!.bytes).toBeGreaterThan(0)
    console.log(
      `  [kv] ${r.kv!.nLayers} layers × ${r.kv!.nKVHeads} kv-heads × ${r.kv!.seqLen} × ${r.kv!.headDim} = ${(r.kv!.bytes / 1024).toFixed(1)} KB`,
    )
  }, 120_000)

  it('序列变长时 KV 占用线性增长（这是 KV Cache 模块的核心论点）', async () => {
    const short = await runNextToken('The capital', 3)
    const long = await runNextToken('The capital of France is a very large city', 3)
    expect(long.kv!.seqLen).toBeGreaterThan(short.kv!.seqLen)
    const perTokShort = short.kv!.bytes / short.kv!.seqLen
    const perTokLong = long.kv!.bytes / long.kv!.seqLen
    // 每 token 增量应当基本恒定
    expect(perTokLong).toBeCloseTo(perTokShort, -1)
  }, 180_000)
})

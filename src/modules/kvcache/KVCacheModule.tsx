/**
 * 模块五：KV Cache
 *
 * 推理加速里最"划算"的一招，也是长上下文最大的显存开销来源。
 * 这个模块把两件事量化出来：
 *   1. 省了多少计算（O(m·n²) → O(n² + m·n)）
 *   2. 付了多少显存（随序列长度线性增长，且乘上 batch）
 */
import { useMemo, useState } from 'react'
import { Card, Segmented, Slider, Stats } from '../../components/Controls'
import { LineChart } from '../../components/LineChart'
import { Principle } from '../../components/Principle'
import {
  DEFAULT_KV,
  analyzeKV,
  fmtBytes,
  fmtFlops,
  type KVCacheConfig,
} from '../../core/kvcache'
import { useLang, type Lang } from '../../i18n'

const PRESETS = [
  { zh: 'Qwen2.5-7B（GQA）', en: 'Qwen2.5-7B (GQA)', patch: { nLayers: 28, nHeads: 28, nKVHeads: 4, dHead: 128 } },
  { zh: 'Llama3-8B（GQA）', en: 'Llama3-8B (GQA)', patch: { nLayers: 32, nHeads: 32, nKVHeads: 8, dHead: 128 } },
  { zh: 'GPT-3 175B（MHA）', en: 'GPT-3 175B (MHA)', patch: { nLayers: 96, nHeads: 96, nKVHeads: 96, dHead: 128 } },
]

const PRECISION = [
  { value: 2, label: 'FP16' },
  { value: 1, label: 'FP8' },
  { value: 0.5, label: 'INT4' },
]

const zh = {
  h2: '⑤ KV Cache：用显存换计算量',
  lead1: '解码时每生成一个 token，注意力都要看一遍前面所有 token 的 K 和 V。但这些 K、V 上一次已经算过了 —— 存起来复用，就能把每步的代价从 ',
  lead2: 'O(s²)',
  lead3: ' 降到 ',
  lead4: 'O(s)',
  lead5: '。代价是显存随序列长度线性增长。',
  principleTitle: '省在哪，又贵在哪？',
  d1a: '为什么长上下文这么吃显存？',
  d1b: '因为 KV Cache 是「序列长度 × batch × 层数 × KV头数 × d_head × 2 × 精度」的乘积，全是线性相乘。128K 上下文、batch 32 的 7B 模型，光 KV Cache 就能吃掉几十 GB —— 比模型权重本身还大。',
  d2a: 'GQA（分组查询注意力）为什么重要？',
  d2b: '它让多个 Query 头共享同一组 KV 头。当前配置下 KV 头数是 Query 头数的 1/',
  d2c: '，意味着 KV Cache 直接省了 ',
  d2d: '。这就是 Qwen、Llama3 这些新架构敢把上下文做到 128K 的底气。',
  d3: '工程上还有 MQA（KV 头数 = 1）、PagedAttention（分页管理，避免碎片）、以及干脆丢掉远距离 KV 的 StreamingLLM —— 但保留首 token 的 attention sink 是必须的，这一点你在注意力模块里已经看到过了。',
  warn: '这里用的是解析公式，只统计自注意力部分的计算量（不含 FFN、不含访存与算子开销），量级和趋势是对的，具体数字会和真实 profiler 有出入。',
  struct: '模型与推理配置',
  prompt: '提示长度 n',
  gen: '生成长度 m',
  layers: '层数 L',
  qHeads: 'Query 头数',
  kvHeads: 'KV 头数（GQA）',
  kvHint: '调小它 = GQA，显存直接成比例下降',
  dHead: 'd_head',
  batch: 'batch',
  precision: '缓存精度',
  costTitle: '注意力计算量：累计 FLOPs 随生成步数增长',
  costHint: '粉线是无缓存，绿线是有缓存',
  costNote: '两条线的间距会越拉越大 —— 这正是 O(m·n²) 与 O(m·n) 的差别。生成越长、提示越长，缓存的收益越夸张。',
  memTitle: 'KV Cache 显存随序列长度增长',
  memHint: '线性增长，但乘上了 batch',
  xCost: '已生成 token 数',
  yCost: '累计 FLOPs',
  xMem: '序列长度（token）',
  yMem: 'KV Cache 显存',
  numbers: '当前配置下的数字',
  prefill: 'prefill 计算量',
  decodeNo: 'decode（无缓存）',
  decodeYes: 'decode（有缓存）',
  speedup: '整体加速比',
  perToken: '每 token KV 占用',
  totalKV: 'KV 总显存',
  gqaSaving: 'GQA 节省',
  none: '无',
  dModel: 'd_model',
  noteSpeedup: '注意「整体加速比」里已经包含了无法省的 prefill，所以它不是纯 decode 的加速倍数。把生成长度 m 调大，你会看到加速比迅速上升 —— 因为可省的 decode 部分占比变高了。',
}

const en: typeof zh = {
  h2: '⑤ KV Cache: trading memory for compute',
  lead1: 'While decoding, every new token attends to the K and V of everything before it — but those were already computed last step. Cache them and the per-step cost drops from ',
  lead2: 'O(s²)',
  lead3: ' to ',
  lead4: 'O(s)',
  lead5: '. The price is memory that grows linearly with sequence length.',
  principleTitle: 'What it saves, and what it costs',
  d1a: 'Why long context eats so much memory?',
  d1b: 'Because KV cache size is the product of sequence length × batch × layers × KV heads × d_head × 2 × precision — all linear factors multiplied together. A 7B model at 128K context and batch 32 burns tens of GB on KV alone — more than the weights themselves.',
  d2a: 'Why GQA matters so much?',
  d2b: 'It lets several query heads share one set of K/V. With the current config, KV heads are 1/',
  d2c: ' of query heads, which cuts the cache by ',
  d2d: '. That is what lets Qwen and Llama3 push context to 128K.',
  d3: 'On top of that: MQA (a single KV head), PagedAttention (paged allocation to kill fragmentation), and StreamingLLM (drop distant KV entirely) — but all of them must keep the first token’s attention sink, which you already saw in the attention module.',
  warn: 'These are closed-form numbers covering the self-attention part only (no FFN, no memory traffic, no kernel overhead). Magnitudes and trends are right; exact figures will differ from a real profiler.',
  struct: 'Model and serving config',
  prompt: 'prompt length n',
  gen: 'generated length m',
  layers: 'layers L',
  qHeads: 'query heads',
  kvHeads: 'KV heads (GQA)',
  kvHint: 'smaller = GQA, memory drops proportionally',
  dHead: 'd_head',
  batch: 'batch',
  precision: 'cache precision',
  costTitle: 'Cumulative attention FLOPs vs generated tokens',
  costHint: 'pink = without cache, green = with cache',
  costNote: 'The gap widens without bound — that is O(m·n²) versus O(m·n). The longer the generation and the longer the prompt, the more absurd the saving.',
  memTitle: 'KV cache memory vs sequence length',
  memHint: 'linear in length, multiplied by batch',
  xCost: 'tokens generated',
  yCost: 'cumulative FLOPs',
  xMem: 'sequence length (tokens)',
  yMem: 'KV cache memory',
  numbers: 'Numbers for the current config',
  prefill: 'prefill FLOPs',
  decodeNo: 'decode (no cache)',
  decodeYes: 'decode (with cache)',
  speedup: 'overall speedup',
  perToken: 'KV bytes per token',
  totalKV: 'total KV memory',
  gqaSaving: 'GQA saving',
  none: 'none',
  dModel: 'd_model',
  noteSpeedup: 'The speedup includes the prefill you cannot avoid, so it is not the pure decode speedup. Raise the generated length m and it climbs fast — the avoidable decode part becomes a bigger share.',
}

const DICT: Record<Lang, typeof zh> = { zh, en }

export function KVCacheModule() {
  const { lang } = useLang()
  const c = DICT[lang]
  const [cfg, setCfg] = useState<KVCacheConfig>(DEFAULT_KV)
  const report = useMemo(() => analyzeKV(cfg), [cfg])
  const patch = (p: Partial<KVCacheConfig>) => setCfg((prev) => ({ ...prev, ...p }))

  const costLines = useMemo(
    () => [
      {
        name: lang === 'zh' ? '无 KV Cache（每步重算整段）' : 'no KV cache (recompute everything)',
        color: '#d4537e',
        points: report.series.map((s) => ({ x: s.step, y: s.noCache })),
      },
      {
        name: lang === 'zh' ? '有 KV Cache' : 'with KV cache',
        color: '#1d9e75',
        points: report.series.map((s) => ({ x: s.step, y: s.withCache })),
      },
    ],
    [report, lang],
  )

  const maxLen = Math.max(cfg.nPrompt + cfg.nGen, 2048)
  const memLines = useMemo(() => {
    const batches = [1, 8, 32]
    const colors = ['#1d9e75', '#534ab7', '#ba7517']
    return batches.map((b, i) => ({
      name: `batch = ${b}`,
      color: colors[i],
      points: Array.from({ length: 40 }, (_, k) => {
        const len = Math.round(((k + 1) / 40) * maxLen)
        return { x: len, y: report.bytesPerToken * len * b }
      }),
    }))
  }, [report, maxLen])

  const gqaSaving = cfg.nHeads > 0 ? cfg.nHeads / Math.max(cfg.nKVHeads, 1) : 1

  return (
    <div>
      <div className="module-head">
        <h2>{c.h2}</h2>
        <div className="lead">
          {c.lead1}
          <strong>{c.lead2}</strong>
          {c.lead3}
          <strong>{c.lead4}</strong>
          {c.lead5}
        </div>
      </div>

      <Principle
        title={c.principleTitle}
        formula={`n = prompt length, m = generated length, d = d_head × heads, L = layers

no cache: step t re-runs the whole forward pass, s = n + t
    per step ≈ 4·s²·d·L      total ≈ Σ(t=1..m) 4·(n+t)²·d·L   → O(m·n²)

with cache:
    prefill (prompt, once)   = 4·n²·d·L
    decode  (new query only) = 4·(n+t)·d·L                    → O(n² + m·n)

memory: KV bytes per token
    2 (K and V) × L × KV heads × d_head × bytes per param
    times (n + m) × batch`}
        analogy={
          lang === 'zh'
            ? '没有缓存时，你每写一个字都要把前面整篇文章重读一遍；有了缓存，你读完一遍就给每个段落贴上便签，之后写新字只需要扫一眼便签。快是快了，但便签本身要占地方——文章越长、同时处理的人（batch）越多，便签就越堆越高，最后整张桌子放不下。'
            : 'Without a cache you re-read the entire article before writing each new word. With a cache you read it once, stick a sticky note on every paragraph, and afterwards just glance at the notes. Faster — but the notes take up space: the longer the article and the more people (batch) working at once, the higher the stack until the desk runs out.'
        }
        detail={
          <>
            <p>
              <strong>{c.d1a}</strong> {c.d1b}
            </p>
            <p>
              <strong>{c.d2a}</strong> {c.d2b}
              {gqaSaving.toFixed(0)}
              {c.d2c}
              {gqaSaving > 1 ? ((1 - 1 / gqaSaving) * 100).toFixed(0) : 0}
              {'%'}
              {c.d2d}
            </p>
            <p>{c.d3}</p>
          </>
        }
        warn={c.warn}
      />

      <Card title={c.struct}>
        <div className="chip-row" style={{ marginBottom: 12 }}>
          {PRESETS.map((p) => (
            <button key={p.zh} className="btn" onClick={() => patch(p.patch)}>
              {lang === 'zh' ? p.zh : p.en}
            </button>
          ))}
        </div>
        <div className="controls">
          <Slider
            label={c.prompt}
            value={cfg.nPrompt}
            min={64}
            max={8192}
            step={64}
            onChange={(v) => patch({ nPrompt: v })}
            format={(v) => `${v} tok`}
          />
          <Slider
            label={c.gen}
            value={cfg.nGen}
            min={16}
            max={1024}
            step={16}
            onChange={(v) => patch({ nGen: v })}
            format={(v) => `${v} tok`}
          />
          <Slider
            label={c.layers}
            value={cfg.nLayers}
            min={1}
            max={96}
            onChange={(v) => patch({ nLayers: v })}
            format={(v) => (lang === 'zh' ? `${v} 层` : `L${v}`)}
          />
          <Slider
            label={c.qHeads}
            value={cfg.nHeads}
            min={1}
            max={96}
            onChange={(v) => patch({ nHeads: v, nKVHeads: Math.min(cfg.nKVHeads, v) })}
            format={(v) => (lang === 'zh' ? `${v} 头` : `${v}`)}
          />
          <Slider
            label={c.kvHeads}
            value={cfg.nKVHeads}
            min={1}
            max={Math.max(cfg.nHeads, 1)}
            onChange={(v) => patch({ nKVHeads: v })}
            format={(v) => (lang === 'zh' ? `${v} 头` : `${v}`)}
            hint={c.kvHint}
          />
          <Slider
            label={c.dHead}
            value={cfg.dHead}
            min={32}
            max={256}
            step={32}
            onChange={(v) => patch({ dHead: v })}
            format={(v) => String(v)}
          />
          <Slider
            label={c.batch}
            value={cfg.batch}
            min={1}
            max={64}
            onChange={(v) => patch({ batch: v })}
            format={(v) => String(v)}
          />
          <div className="control" style={{ minWidth: 180 }}>
            <label>
              <span>{c.precision}</span>
            </label>
            <Segmented
              value={cfg.bytesPerParam}
              onChange={(v) => patch({ bytesPerParam: v })}
              options={PRECISION}
            />
          </div>
        </div>
      </Card>

      <Card title={c.costTitle} hint={c.costHint} exportName="05-kvcache-flops">
        <LineChart
          lines={costLines}
          xLabel={c.xCost}
          yLabel={c.yCost}
          yFormat={fmtFlops}
          xFormat={(v) => v.toFixed(0)}
        />
        <div className="note" style={{ marginTop: 8 }}>
          {c.costNote}
        </div>
      </Card>

      <Card title={c.memTitle} hint={c.memHint} exportName="05-kvcache-memory">
        <LineChart
          lines={memLines}
          xLabel={c.xMem}
          yLabel={c.yMem}
          yFormat={fmtBytes}
          xFormat={(v) => v.toFixed(0)}
        />
      </Card>

      <Card title={c.numbers}>
        <Stats
          items={[
            { k: c.prefill, v: fmtFlops(report.prefillFlops) },
            { k: c.decodeNo, v: fmtFlops(report.decodeNoCache) },
            { k: c.decodeYes, v: fmtFlops(report.decodeWithCache) },
            { k: c.speedup, v: report.speedup.toFixed(2) + '×' },
          ]}
        />
        <div style={{ marginTop: 12 }}>
          <Stats
            items={[
              { k: c.perToken, v: fmtBytes(report.bytesPerToken) },
              {
                k: `${c.totalKV}（${cfg.batch}×${cfg.nPrompt + cfg.nGen}）`,
                v: fmtBytes(report.cacheBytes),
              },
              { k: c.gqaSaving, v: gqaSaving > 1 ? `${((1 - 1 / gqaSaving) * 100).toFixed(0)}%` : c.none },
              { k: c.dModel, v: String(cfg.nHeads * cfg.dHead) },
            ]}
          />
        </div>
        <div className="note" style={{ marginTop: 10 }}>
          {c.noteSpeedup}
        </div>
      </Card>
    </div>
  )
}

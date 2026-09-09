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

const PRESETS: { label: string; patch: Partial<KVCacheConfig> }[] = [
  { label: 'Qwen2.5-7B（GQA）', patch: { nLayers: 28, nHeads: 28, nKVHeads: 4, dHead: 128 } },
  { label: 'Llama3-8B（GQA）', patch: { nLayers: 32, nHeads: 32, nKVHeads: 8, dHead: 128 } },
  { label: 'GPT-3 175B（MHA）', patch: { nLayers: 96, nHeads: 96, nKVHeads: 96, dHead: 128 } },
]

const PRECISION = [
  { value: 2, label: 'FP16' },
  { value: 1, label: 'FP8' },
  { value: 0.5, label: 'INT4' },
]

export function KVCacheModule() {
  const [cfg, setCfg] = useState<KVCacheConfig>(DEFAULT_KV)
  const report = useMemo(() => analyzeKV(cfg), [cfg])
  const patch = (p: Partial<KVCacheConfig>) => setCfg((c) => ({ ...c, ...p }))

  // 计算量曲线：累计 FLOPs vs 已生成 token 数
  const costLines = useMemo(
    () => [
      {
        name: '无 KV Cache（每步重算整段）',
        color: '#d4537e',
        points: report.series.map((s) => ({ x: s.step, y: s.noCache })),
      },
      {
        name: '有 KV Cache',
        color: '#1d9e75',
        points: report.series.map((s) => ({ x: s.step, y: s.withCache })),
      },
    ],
    [report],
  )

  // 显存曲线：KV Cache 大小 vs 序列长度，画三条不同 batch
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
        <h2>⑤ KV Cache：用显存换计算量</h2>
        <div className="lead">
          解码时每生成一个 token，注意力都要看一遍前面所有 token 的 K 和 V。
          但这些 K、V 上一次已经算过了 —— 存起来复用，就能把每步的代价从 <strong>O(s²)</strong> 降到{' '}
          <strong>O(s)</strong>。代价是显存随序列长度线性增长。
        </div>
      </div>

      <Principle
        title="省在哪，又贵在哪？"
        formula={`设 n = 提示长度，m = 生成长度，d = d_head × 头数，L = 层数

【无缓存】第 t 步要跑一整次前向，序列长 s = n + t
    单步代价 ≈ 4·s²·d·L        累计 ≈ Σ(t=1..m) 4·(n+t)²·d·L   → O(m·n²)

【有缓存】
    prefill（一次性算完提示）     = 4·n²·d·L
    decode（每步只算新 token 的 Q）= 4·(n+t)·d·L               → O(n² + m·n)

【显存】每 token 的 KV 占用
    2（K 和 V）× L × KV头数 × d_head × 每参数字节数
    再乘上 (n + m) × batch`}
        analogy="没有缓存时，你每写一个字都要把前面整篇文章重读一遍；有了缓存，你读完一遍就给每个段落贴上便签，之后写新字只需要扫一眼便签。快是快了，但便签本身要占地方——文章越长、同时处理的人（batch）越多，便签就越堆越高，最后整张桌子放不下。"
        detail={
          <>
            <p>
              <strong>为什么长上下文这么吃显存？</strong> 因为 KV Cache 是「序列长度 × batch × 层数 ×
              KV头数 × d_head × 2 × 精度」的乘积，全是线性相乘。128K 上下文、batch 32 的 7B 模型，
              光 KV Cache 就能吃掉几十 GB —— 比模型权重本身还大。
            </p>
            <p>
              <strong>GQA（分组查询注意力）为什么重要？</strong> 它让多个 Query 头共享同一组 KV 头。
              当前配置下 KV 头数是 Query 头数的 1/{gqaSaving.toFixed(0)}，意味着 KV Cache 直接省了{' '}
              {gqaSaving > 1 ? ((1 - 1 / gqaSaving) * 100).toFixed(0) : 0}%。
              这就是 Qwen、Llama3 这些新架构敢把上下文做到 128K 的底气。
            </p>
            <p>
              工程上还有 MQA（KV 头数 = 1）、PagedAttention（分页管理，避免碎片）、
              以及干脆丢掉远距离 KV 的 StreamingLLM —— 但保留首 token 的 attention sink 是必须的，
              这一点你在注意力模块里已经看到过了。
            </p>
          </>
        }
        warn="这里用的是解析公式，只统计自注意力部分的计算量（不含 FFN、不含访存与算子开销），量级和趋势是对的，具体数字会和真实 profiler 有出入。"
      />

      <Card title="模型与推理配置">
        <div className="chip-row" style={{ marginBottom: 12 }}>
          {PRESETS.map((p) => (
            <button key={p.label} className="btn" onClick={() => patch(p.patch)}>
              {p.label}
            </button>
          ))}
        </div>
        <div className="controls">
          <Slider
            label="提示长度 n"
            value={cfg.nPrompt}
            min={64}
            max={8192}
            step={64}
            onChange={(v) => patch({ nPrompt: v })}
            format={(v) => `${v} tok`}
          />
          <Slider
            label="生成长度 m"
            value={cfg.nGen}
            min={16}
            max={1024}
            step={16}
            onChange={(v) => patch({ nGen: v })}
            format={(v) => `${v} tok`}
          />
          <Slider
            label="层数 L"
            value={cfg.nLayers}
            min={1}
            max={96}
            onChange={(v) => patch({ nLayers: v })}
            format={(v) => `${v} 层`}
          />
          <Slider
            label="Query 头数"
            value={cfg.nHeads}
            min={1}
            max={96}
            onChange={(v) => patch({ nHeads: v, nKVHeads: Math.min(cfg.nKVHeads, v) })}
            format={(v) => `${v} 头`}
          />
          <Slider
            label="KV 头数（GQA）"
            value={cfg.nKVHeads}
            min={1}
            max={Math.max(cfg.nHeads, 1)}
            onChange={(v) => patch({ nKVHeads: v })}
            format={(v) => `${v} 头`}
            hint="调小它 = GQA，显存直接成比例下降"
          />
          <Slider
            label="d_head"
            value={cfg.dHead}
            min={32}
            max={256}
            step={32}
            onChange={(v) => patch({ dHead: v })}
            format={(v) => String(v)}
          />
          <Slider
            label="batch"
            value={cfg.batch}
            min={1}
            max={64}
            onChange={(v) => patch({ batch: v })}
            format={(v) => String(v)}
          />
          <div className="control" style={{ minWidth: 180 }}>
            <label>
              <span>缓存精度</span>
            </label>
            <Segmented
              value={cfg.bytesPerParam}
              onChange={(v) => patch({ bytesPerParam: v })}
              options={PRECISION}
            />
          </div>
        </div>
      </Card>

      <Card title="注意力计算量：累计 FLOPs 随生成步数增长" hint="虚线/粉线是无缓存，绿线是有缓存">
        <LineChart
          lines={costLines}
          xLabel="已生成 token 数"
          yLabel="累计 FLOPs"
          yFormat={fmtFlops}
          xFormat={(v) => v.toFixed(0)}
        />
        <div className="note" style={{ marginTop: 8 }}>
          两条线的间距会越拉越大 —— 这正是 O(m·n²) 与 O(m·n) 的差别。生成越长、提示越长，缓存的收益越夸张。
        </div>
      </Card>

      <Card title="KV Cache 显存随序列长度增长" hint="线性增长，但乘上了 batch">
        <LineChart
          lines={memLines}
          xLabel="序列长度（token）"
          yLabel="KV Cache 显存"
          yFormat={fmtBytes}
          xFormat={(v) => v.toFixed(0)}
        />
      </Card>

      <Card title="当前配置下的数字">
        <Stats
          items={[
            { k: 'prefill 计算量', v: fmtFlops(report.prefillFlops) },
            { k: 'decode（无缓存）', v: fmtFlops(report.decodeNoCache) },
            { k: 'decode（有缓存）', v: fmtFlops(report.decodeWithCache) },
            { k: '整体加速比', v: report.speedup.toFixed(2) + '×' },
          ]}
        />
        <div style={{ marginTop: 12 }}>
          <Stats
            items={[
              { k: '每 token KV 占用', v: fmtBytes(report.bytesPerToken) },
              { k: `KV 总显存（${cfg.batch}×${cfg.nPrompt + cfg.nGen}）`, v: fmtBytes(report.cacheBytes) },
              { k: 'GQA 节省', v: gqaSaving > 1 ? `${((1 - 1 / gqaSaving) * 100).toFixed(0)}%` : '无' },
              { k: 'd_model', v: String(cfg.nHeads * cfg.dHead) },
            ]}
          />
        </div>
        <div className="note" style={{ marginTop: 10 }}>
          注意「整体加速比」里已经包含了无法省的 prefill，所以它不是纯 decode 的加速倍数。
          把生成长度 m 调大，你会看到加速比迅速上升 —— 因为可省的 decode 部分占比变高了。
        </div>
      </Card>
    </div>
  )
}

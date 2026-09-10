/**
 * 模块八：量化
 *
 * 三张图讲清楚三件事：
 *   1. 分布 × 档位 —— 一张离群值的图能顶一段文字：档位本来铺得好好的，
 *      一个 40σ 的尖峰一来，整个 scale 被撑开，绝大多数权重挤在同一个台阶上
 *   2. 量化台阶 —— 排序之后原始值和还原值叠在一起，台阶有多粗一眼看得见
 *   3. 方案对比 —— per-tensor / per-channel / group-wise / NF4 在同一份数据上比 SQNR
 */
import { useMemo, useState } from 'react'
import { Card, Segmented, Slider, Stats } from '../../components/Controls'
import { BarList } from '../../components/Heatmap'
import { Principle } from '../../components/Principle'
import {
  DEFAULT_QUANT,
  compareSchemes,
  fmtGB,
  memoryTable,
  quantize,
  type QuantConfig,
  type QuantMode,
  type QuantScheme,
} from '../../core/quant'
import { useLang, type Lang } from '../../i18n'

const zh = {
  h2: '⑧ 量化：把 16 位压成 4 位，代价是什么',
  lead1: '推理的瓶颈常常不是算力而是显存带宽。量化把每个权重从 16 位压到 4 位，模型立刻小 4 倍、也能塞进更小的卡。代价是权重被舍入到有限个台阶上，而',
  lead2: '这一步的误差有多大，几乎完全由离群值决定',
  lead3: '—— 一个 40 倍大的权重，就能把整个张量的精度吃掉。',

  principleTitle: '对称量化在算什么？为什么一个离群值就崩？',
  d1a: '对称量化的全部秘密就在那个 scale 上。',
  d1b: '它先用 max|x| 定出「最大刻度」，把 [-max|x|, max|x|] 均匀切成 2^(bits-1) 份，然后每个权重除以 scale、四舍五入到最近的整数，还原时再乘回来。所以 scale 是被**最大的那个权重**决定的。',
  d2a: '这就是离群值的杀伤力。',
  d2b: '真实 LLM 的权重近似高斯分布，但存在极少数特别大的「离群特征」。一旦有 40σ 的尖峰，scale 就被撑到 40 倍，剩下的权重只能挤在 0 附近的一两个台阶上 —— 有效档位数从 15 掉到 5 甚至 3。这不是精度慢慢变差，而是断崖式失效。',
  d3a: '三种解法，本质都是「别让全体共用一个 scale」。',
  d3b: 'per-channel 按输出通道各自算 scale；group-wise 进一步把通道切成小组（典型 group=32/64/128），每组一个 scale；NF4 则换掉均匀台阶本身，改用按标准正态分位数摆放的非均匀码本，让 0 附近的台阶更密。前两种是工程上最有效的，NF4 是在 4 bit 这个极低预算下的额外优化。',
  warn: '权重数据是按高斯分布加离群值确定性生成的，不是某个具体模型的真实张量；但量化、反量化、误差统计、字节开销全是按真实公式算的，你也可以用滑块亲手复现 4 bit 崩溃的过程。另外注意：加了离群值之后 SNR 会失真（离群值把方差本身也撑大了，所以 SNR 会在中间触底又回升），判断精度损失请看有效档位数和台阶图。',

  outlierTitle: '离群值实验：一个尖峰能毁掉多少精度',
  outlierHint: '把所有滑块都留着，只动离群值幅度',
  outlierScale: '离群值幅度（倍标准差）',
  outlierCount: '离群值个数',
  outlierNote: '看「有效档位数」那一列：从 15/16 一路掉到 3/16。15 个台阶变成了 3 个，意味着绝大多数权重都被塞进了同一个值 —— 这一步的信息基本丢光了。',

  configTitle: '量化配置',
  bits: '位数',
  scheme: '方案',
  sym: '对称',
  asym: '非对称',
  nf4: 'NF4 码本',
  mode: '分组粒度',
  tensor: 'per-tensor',
  channel: 'per-channel',
  group: 'group-wise',
  groupSize: 'group 大小',
  nChannels: '通道数',
  n: '权重个数',

  histTitle: '分布 × 量化档位',
  histHint: '柱子是权重分布，竖线是档位落在哪里',
  histNote: '柱子是权重的直方图，竖线是当前配置下所有档位落在的数值。最理想的情况是竖线均匀铺满直方图的形状；出现离群值时你会看到柱子全部挤在最左边一条细缝里，而竖线稀稀拉拉铺满整个横轴 —— 右边的档位根本没有权重落上去，白白浪费。',
  histLevels: '档位数（去重后）',
  groupCaveat: 'group-wise / per-channel 模式下每个分组都有自己的 scale，这里画的是所有分组档位的并集，密集程度本身就反映了精度。',

  stepTitle: '量化台阶：原始值 vs 还原值',
  stepHint: '按数值排序后叠在一起，台阶有多粗一眼看得见',
  yRange: '纵轴范围',
  yAll: '全部',
  yRobust: '中间 96%（裁掉离群值）',
  stepNote: '灰线是原始权重，紫线是量化再还原的结果。紫线的每一个水平段就是一个台阶。用「中间 96%」视图能看清主体权重的台阶有多粗：位数越低台阶越粗，4 bit 时一共只有 16 级。',

  cmpTitle: '七种方案的实测对比',
  cmpHint: '同一份权重，同样的离群值，只改量化策略',
  cmpNote: '这张表就是这个领域的技术路线图：per-tensor 到 per-channel/group-wise 是最大的那一步收益（本页实测 +6~7 dB），NF4 在 4 bit 这个预算下再挤出一点（对高斯权重，非均匀码本比均匀台阶更合身）。非对称量化在数值范围明显不对称时会更好。字节/权重这一列才是显存账单：裸的 4 bit 是 0.5 字节，但 group-wise 每组要额外存 scale 和 zero point，group=32 时实际是 0.625 字节 —— 精度是用显存换来的。',
  snr: 'SQNR',
  levelsCol: '有效档位',
  bytesCol: '字节/权重',
  baseline: '基准',

  memTitle: '换算到显存：7B 模型要占多少',
  memHint: '字节/权重 × 参数量，含量化元数据',
  memNote: '这一列直接对应你会在 HuggingFace 上看到的模型文件大小：7B 的 FP16 约 13 GB，而 group=128 的 4 bit 量化约 3.5 GB —— 这就是为什么单张 8GB 显卡能跑 4 bit 的 7B，却装不下 FP16 的 7B。注意 group 越小精度越好，但元数据开销越大（group=32 比 group=128 多约 18% 体积）。',
  modelSize: '参数量',
}

const en: typeof zh = {
  h2: '⑧ Quantization: what does 16-bit to 4-bit actually cost',
  lead1: 'Inference is often limited by memory bandwidth, not FLOPs. Quantization squeezes each weight from 16 bits to 4, making the model four times smaller and able to fit on smaller cards. The price is that weights get rounded onto a finite set of steps — and ',
  lead2: 'how big that error is depends almost entirely on outliers',
  lead3: '. A single weight 40 times larger than the rest can eat the whole tensor\u2019s precision.',

  principleTitle: 'What does symmetric quantization compute, and why does one outlier break it?',
  d1a: 'The whole secret of symmetric quantization is that one scale.',
  d1b: 'It takes max|x| as the largest tick, divides [-max|x|, max|x|] into 2^(bits-1) even slices, then rounds each weight divided by the scale to the nearest integer and multiplies back on dequantization. So the scale is decided by the single largest weight.',
  d2a: 'That is exactly why outliers are so destructive.',
  d2b: 'Real LLM weights are roughly Gaussian but contain a handful of very large outlier features. One 40σ spike stretches the scale by 40x, so every other weight is crammed into one or two steps near zero — the number of effective levels collapses from 15 to 5 or even 3. This is not gradual degradation; it is a cliff.',
  d3a: 'All three fixes amount to the same idea: don\u2019t let everything share one scale.',
  d3b: 'per-channel gives each output channel its own scale; group-wise cuts channels into small groups (typically 32/64/128) with one scale per group; NF4 replaces the even steps themselves with a non-uniform codebook placed at the quantiles of a standard normal, so steps are denser near zero. The first two give the big engineering win; NF4 is a further optimisation at the extreme 4-bit budget.',
  warn: 'The weights are generated deterministically as a Gaussian plus outliers, not taken from a specific model; but the quantization, dequantization, error statistics and byte overhead all follow the real formulas, and you can reproduce the 4-bit cliff yourself with the sliders. Note also that SNR becomes misleading once outliers are present (they inflate the variance itself, so SNR bottoms out then rises again) — judge precision loss by the effective level count and the step plot.',

  outlierTitle: 'The outlier experiment: how much can one spike cost?',
  outlierHint: 'leave every other slider alone and move only this one',
  outlierScale: 'outlier magnitude (std devs)',
  outlierCount: 'number of outliers',
  outlierNote: 'Watch the effective-level count: it falls from 15/16 down to 3/16. Fifteen steps become three, meaning almost every weight is forced onto the same value — that information is essentially gone.',

  configTitle: 'Quantization settings',
  bits: 'bits',
  scheme: 'scheme',
  sym: 'symmetric',
  asym: 'asymmetric',
  nf4: 'NF4 codebook',
  mode: 'granularity',
  tensor: 'per-tensor',
  channel: 'per-channel',
  group: 'group-wise',
  groupSize: 'group size',
  nChannels: 'channels',
  n: 'weights',

  histTitle: 'Distribution × quantization levels',
  histHint: 'bars are the weights, vertical lines are where the levels land',
  histNote: 'The bars are a histogram of the weights; the vertical lines are every level the current configuration lands on. Ideally the lines spread themselves evenly across the shape of the histogram. With an outlier you see the bars crushed into one thin sliver on the left while the lines are sprinkled across the whole axis — the right-hand levels have no weights on them at all, and that range is wasted.',
  histLevels: 'levels (deduplicated)',
  groupCaveat: 'In group-wise / per-channel modes every group has its own scale, so this shows the union of all groups\u2019 levels — how dense they look is itself a measure of precision.',

  stepTitle: 'Quantization steps: original vs restored',
  stepHint: 'sorted and overlaid, so the step width is visible at a glance',
  yRange: 'y range',
  yAll: 'everything',
  yRobust: 'middle 96% (outliers clipped)',
  stepNote: 'The grey line is the original weights, the purple line is what comes back after quantize-dequantize. Every flat segment of the purple line is one step. The middle-96% view shows how coarse the steps are for the bulk: fewer bits means wider steps, and at 4 bits there are only 16 of them.',

  cmpTitle: 'Seven schemes, measured on the same tensor',
  cmpHint: 'same weights, same outliers, only the strategy changes',
  cmpNote: 'This table is the roadmap of the field: per-tensor to per-channel/group-wise is the single biggest win (measured here at +6 to +7 dB), and NF4 squeezes out a little more at the 4-bit budget (a non-uniform codebook fits Gaussian weights better than even steps). Asymmetric quantization does better when the value range is clearly lopsided. The bytes-per-weight column is the actual memory bill: raw 4 bits is 0.5 bytes, but group-wise also stores a scale and zero point per group, so group=32 really costs 0.625 bytes — precision paid for in memory.',
  snr: 'SQNR',
  levelsCol: 'effective levels',
  bytesCol: 'bytes/weight',
  baseline: 'baseline',

  memTitle: 'Converting to memory: what a 7B model occupies',
  memHint: 'bytes per weight × parameter count, including quant metadata',
  memNote: 'This column maps directly onto the file sizes you see on HuggingFace: 7B at FP16 is about 13 GB, while group=128 4-bit is about 3.5 GB — which is why a single 8 GB card can run a 4-bit 7B but cannot hold an FP16 one. Smaller groups are more accurate but cost more metadata (group=32 is roughly 18% larger than group=128).',
  modelSize: 'parameters',
}

const DICT: Record<Lang, typeof zh> = { zh, en }

const SCHEMES: { value: QuantScheme; key: 'sym' | 'asym' | 'nf4' }[] = [
  { value: 'sym', key: 'sym' },
  { value: 'asym', key: 'asym' },
  { value: 'nf4', key: 'nf4' },
]

const MODES: { value: QuantMode; key: 'tensor' | 'channel' | 'group' }[] = [
  { value: 'tensor', key: 'tensor' },
  { value: 'channel', key: 'channel' },
  { value: 'group', key: 'group' },
]

/** 直方图 + 档位竖线 */
function HistWithLevels(props: {
  values: number[]
  levels: number[]
  bins?: number
  height?: number
  levelsLabel: string
}) {
  const W = 620
  const H = props.height ?? 220
  const padL = 44
  const padR = 14
  const padT = 16
  const padB = 30
  const bins = props.bins ?? 48

  const values = props.values
  if (values.length === 0) return null
  const mn = Math.min(...values)
  const mx = Math.max(...values)
  const span = mx - mn || 1

  const counts = new Array(bins).fill(0)
  for (const v of values) {
    const b = Math.min(bins - 1, Math.max(0, Math.floor(((v - mn) / span) * bins)))
    counts[b]++
  }
  const maxCount = Math.max(...counts, 1)

  const sx = (v: number) => padL + ((v - mn) / span) * (W - padL - padR)
  const sy = (c: number) => H - padB - (c / maxCount) * (H - padB - padT)
  const barW = (W - padL - padR) / bins

  const shown = props.levels.slice(0, 240)

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }}>
      <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--border-strong)" strokeWidth={1} />
      {counts.map((c, i) =>
        c > 0 ? (
          <rect
            key={i}
            x={sx(mn + (i / bins) * span)}
            y={sy(c)}
            width={Math.max(1, barW - 0.6)}
            height={H - padB - sy(c)}
            fill="var(--text-3)"
            opacity={0.35}
          />
        ) : null,
      )}
      {shown.map((lv, i) => (
        <line
          key={i}
          x1={sx(lv)}
          y1={padT}
          x2={sx(lv)}
          y2={H - padB}
          stroke="var(--accent)"
          strokeWidth={0.8}
          opacity={0.5}
        />
      ))}
      <text x={padL} y={H - padB + 16} fontSize={10} fill="var(--text-3)">
        {mn.toFixed(1)}
      </text>
      <text x={W - padR} y={H - padB + 16} textAnchor="end" fontSize={10} fill="var(--text-3)">
        {mx.toFixed(1)}
      </text>
      <text x={padL} y={padT - 3} fontSize={10} fill="var(--text-3)">
        {props.levelsLabel}
      </text>
    </svg>
  )
}

/** 排序后的原始值 vs 量化台阶 */
function StepPlot(props: {
  order: number[]
  values: number[]
  dequant: number[]
  robust: boolean
  height?: number
}) {
  const W = 620
  const H = props.height ?? 240
  const padL = 56
  const padR = 14
  const padT = 14
  const padB = 30

  const n = props.order.length
  if (n === 0) return null

  const vs = props.order.map((i) => props.values[i])
  let lo = Math.min(...vs)
  let hi = Math.max(...vs)
  if (props.robust) {
    const loIdx = Math.floor(n * 0.02)
    const hiIdx = Math.ceil(n * 0.98) - 1
    lo = vs[Math.max(0, loIdx)]
    hi = vs[Math.min(n - 1, hiIdx)]
    if (hi - lo < 1e-6) {
      lo -= 1
      hi += 1
    }
  }
  const span = hi - lo || 1

  const sx = (k: number) => padL + (k / Math.max(1, n - 1)) * (W - padL - padR)
  const sy = (v: number) => H - padB - Math.min(1, Math.max(0, (v - lo) / span)) * (H - padB - padT)

  const path = (get: (i: number, k: number) => number) =>
    props.order.map((i, k) => `${k === 0 ? 'M' : 'L'}${sx(k).toFixed(1)},${sy(get(i, k)).toFixed(1)}`).join(' ')

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }}>
      <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--border-strong)" strokeWidth={1} />
      {[0, 0.25, 0.5, 0.75, 1].map((t, i) => (
        <g key={i}>
          <line
            x1={padL}
            y1={H - padB - t * (H - padB - padT)}
            x2={W - padR}
            y2={H - padB - t * (H - padB - padT)}
            stroke="var(--border)"
            strokeWidth={0.6}
          />
          <text
            x={padL - 6}
            y={H - padB - t * (H - padB - padT)}
            textAnchor="end"
            dominantBaseline="central"
            fontSize={10}
            fill="var(--text-3)"
          >
            {(lo + t * span).toFixed(1)}
          </text>
        </g>
      ))}
      <path d={path((i) => props.values[i])} fill="none" stroke="var(--text-3)" strokeWidth={1} opacity={0.55} />
      <path d={path((i) => props.dequant[i])} fill="none" stroke="var(--accent)" strokeWidth={2} />
      <text x={padL} y={H - padB + 16} fontSize={10} fill="var(--text-3)">
        0
      </text>
      <text x={W - padR} y={H - padB + 16} textAnchor="end" fontSize={10} fill="var(--text-3)">
        {`n = ${n}`}
      </text>
    </svg>
  )
}

export function QuantModule() {
  const { lang } = useLang()
  const c = DICT[lang]
  const [cfg, setCfg] = useState<QuantConfig>(DEFAULT_QUANT)
  const [robust, setRobust] = useState<0 | 1>(1)
  const [nParams, setNParams] = useState(7)

  const patch = (p: Partial<QuantConfig>) => setCfg((prev) => ({ ...prev, ...p }))

  const result = useMemo(() => quantize(cfg), [cfg])
  const comparison = useMemo(() => compareSchemes(cfg), [cfg])

  // 所有分组档位的并集（去重）—— group / per-channel 模式下每个组各有自己的 scale
  const levels = useMemo(() => {
    const set = new Set<string>()
    const out: number[] = []
    for (let i = 0; i < result.dequant.length; i++) {
      const v = result.dequant[i]
      const k = v.toFixed(6)
      if (!set.has(k)) {
        set.add(k)
        out.push(v)
      }
    }
    out.sort((a, b) => a - b)
    return out
  }, [result])

  const mem = useMemo(() => memoryTable(nParams * 1e9), [nParams])

  const cmpItems = comparison.map((x) => ({
    label: lang === 'zh' ? x.label : x.labelEn,
    value: x.snrDb >= 119 ? 120 : x.snrDb,
  }))

  return (
    <div>
      <div className="module-head">
        <h2>{c.h2}</h2>
        <div className="lead">
          {c.lead1}
          <strong>{c.lead2}</strong>
          {c.lead3}
        </div>
      </div>

      <Principle
        title={c.principleTitle}
        formula={`对称量化 (per-tensor / per-channel / per-group)
  scale = max|x| / (2^(bits-1) - 1)
  code  = clamp( round(x / scale), -2^(bits-1), 2^(bits-1) - 1 )
  x'    = code · scale

非对称量化
  scale = (max - min) / (2^bits - 1),   zero = round(-min / scale)
  code  = clamp( round(x / scale) + zero, 0, 2^bits - 1 )
  x'    = (code - zero) · scale

NF4（非均匀码本，16 级）
  x'    = codebook[ argmin_k |x/scale - codebook[k]| ] · scale

误差   SQNR(dB) = 10 · log10( var(x) / MSE )`}
        analogy={
          lang === 'zh'
            ? '一把尺子量所有东西。尺子的最大刻度由最长的那个东西决定（max|x|）。如果量尺子的人里混进一根电线杆，刻度就得拉到电线杆那么长，那么量手指的时候，所有手指都会落在同一格上 —— 这就是 4 bit 被离群值毁掉的场景。解法有三个层次：给每个房间配一把尺子（per-channel）、给每个抽屉配一把（group-wise）、或者干脆换一把刻度不均匀、细密处更密的尺子（NF4）。'
            : 'One ruler for everything. Its largest tick is set by the longest object you measure (max|x|). Let a lamppost into the room and the scale stretches to lamppost size — now every finger lands on the same tick. That is a 4-bit tensor destroyed by one outlier. There are three fixes: one ruler per room (per-channel), one per drawer (group-wise), or simply a ruler whose ticks are uneven and denser where it matters (NF4).'
        }
        detail={
          <>
            <p>
              <strong>{c.d1a}</strong> {c.d1b}
            </p>
            <p>
              <strong>{c.d2a}</strong> {c.d2b}
            </p>
            <p>
              <strong>{c.d3a}</strong> {c.d3b}
            </p>
          </>
        }
        warn={c.warn}
      />

      <Card title={c.outlierTitle} hint={c.outlierHint}>
        <div className="controls" style={{ marginBottom: 12 }}>
          <Slider
            label={c.outlierScale}
            value={cfg.outlierScale}
            min={1}
            max={80}
            onChange={(v) => patch({ outlierScale: v })}
            format={(v) => `${v}σ`}
          />
          <Slider
            label={c.outlierCount}
            value={cfg.outlierCount}
            min={0}
            max={8}
            onChange={(v) => patch({ outlierCount: v })}
            format={(v) => String(v)}
          />
        </div>
        <BarList
          items={[
            { label: `${c.histLevels} (${lang === 'zh' ? '离群' : 'outlier'} ${cfg.outlierScale}σ × ${cfg.outlierCount})`, value: result.usedLevels },
            { label: `${c.histLevels} (4 bit ${lang === 'zh' ? '无离群' : 'no outlier'})`, value: quantize({ ...cfg, outlierCount: 0 }).usedLevels },
          ]}
          max={result.totalLevels}
          format={(v) => `${v.toFixed(0)} / ${result.totalLevels}`}
        />
        <div className="note" style={{ marginTop: 10 }}>{c.outlierNote}</div>
      </Card>

      <Card title={c.configTitle}>
        <div className="controls">
          <Slider
            label={c.bits}
            value={cfg.bits}
            min={2}
            max={8}
            onChange={(v) => patch({ bits: v })}
            format={(v) => `${v} bit`}
          />
          <div className="control" style={{ minWidth: 200 }}>
            <label>
              <span>{c.scheme}</span>
            </label>
            <Segmented
              value={cfg.scheme}
              onChange={(v) => patch({ scheme: v })}
              options={SCHEMES.map((s) => ({ value: s.value, label: c[s.key] }))}
            />
          </div>
          <div className="control" style={{ minWidth: 240 }}>
            <label>
              <span>{c.mode}</span>
            </label>
            <Segmented
              value={cfg.mode}
              onChange={(v) => patch({ mode: v })}
              options={MODES.map((s) => ({ value: s.value, label: c[s.key] }))}
            />
          </div>
          <Slider
            label={c.groupSize}
            value={cfg.groupSize}
            min={8}
            max={256}
            step={8}
            onChange={(v) => patch({ groupSize: v })}
            format={(v) => String(v)}
            disabled={cfg.mode !== 'group'}
          />
          <Slider
            label={c.nChannels}
            value={cfg.nChannels}
            min={2}
            max={32}
            onChange={(v) => patch({ nChannels: v })}
            format={(v) => String(v)}
            disabled={cfg.mode !== 'channel'}
          />
          <Slider
            label={c.n}
            value={cfg.n}
            min={64}
            max={1024}
            step={32}
            onChange={(v) => patch({ n: v })}
            format={(v) => String(v)}
          />
        </div>
      </Card>

      <Card title={c.histTitle} hint={c.histHint} exportName="08-quant-histogram">
        <HistWithLevels
          values={result.values}
          levels={levels}
          levelsLabel={`${c.histLevels}: ${levels.length}`}
        />
        <Stats
          items={[
            { k: c.histLevels, v: `${result.usedLevels} / ${result.totalLevels}` },
            { k: c.snr, v: result.snrDb >= 119 ? '≥ 120 dB' : `${result.snrDb.toFixed(1)} dB` },
            { k: 'MSE', v: result.mse.toFixed(4) },
            { k: lang === 'zh' ? '最大绝对误差' : 'max abs error', v: result.maxAbsErr.toFixed(3) },
          ]}
        />
        <div className="note" style={{ marginTop: 10 }}>{c.histNote}</div>
        <div className="note" style={{ marginTop: 4 }}>{c.groupCaveat}</div>
      </Card>

      <Card title={c.stepTitle} hint={c.stepHint} exportName="08-quant-steps">
        <div style={{ marginBottom: 10 }}>
          <div className="control" style={{ minWidth: 260 }}>
            <label>
              <span>{c.yRange}</span>
            </label>
            <Segmented
              value={robust}
              onChange={(v) => setRobust(v)}
              options={[
                { value: 0, label: c.yAll },
                { value: 1, label: c.yRobust },
              ]}
            />
          </div>
        </div>
        <StepPlot order={result.order} values={result.values} dequant={result.dequant} robust={robust === 1} />
        <div className="note" style={{ marginTop: 8 }}>{c.stepNote}</div>
      </Card>

      <Card title={c.cmpTitle} hint={c.cmpHint} exportName="08-quant-comparison">
        <BarList items={cmpItems} max={120} format={(v) => (v >= 119.5 ? '∞' : `${v.toFixed(1)} dB`)} />
        <table className="tbl" style={{ marginTop: 14 }}>
          <thead>
            <tr>
              <th>{c.scheme}</th>
              <th>{c.snr}</th>
              <th>{c.levelsCol}</th>
              <th>{c.bytesCol}</th>
            </tr>
          </thead>
          <tbody>
            {comparison.map((x) => (
              <tr key={x.id}>
                <td>{lang === 'zh' ? x.label : x.labelEn}</td>
                <td>{x.snrDb >= 119 ? '∞' : `${x.snrDb.toFixed(1)} dB`}</td>
                <td>{x.usedLevels === 0 ? '—' : x.usedLevels}</td>
                <td>{x.bytesPerWeight.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="note" style={{ marginTop: 10 }}>{c.cmpNote}</div>
      </Card>

      <Card title={c.memTitle} hint={c.memHint} exportName="08-quant-memory">
        <div className="controls" style={{ marginBottom: 12 }}>
          <Slider
            label={c.modelSize}
            value={nParams}
            min={0.5}
            max={70}
            step={0.5}
            onChange={setNParams}
            format={(v) => `${v} B`}
          />
        </div>
        <table className="tbl">
          <thead>
            <tr>
              <th>{lang === 'zh' ? '格式' : 'format'}</th>
              <th>{c.bytesCol}</th>
              <th>{lang === 'zh' ? '显存占用' : 'memory'}</th>
            </tr>
          </thead>
          <tbody>
            {mem.map((m) => (
              <tr key={m.label}>
                <td>{m.label}</td>
                <td>{m.bytesPerWeight.toFixed(3)}</td>
                <td>{fmtGB(m.bytes)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="note" style={{ marginTop: 10 }}>{c.memNote}</div>
      </Card>
    </div>
  )
}

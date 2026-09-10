/**
 * 模块九：长上下文外推
 *
 * 一个只在 4K 上训过的模型，为什么喂 32K 就崩？这个模块把几何层面的原因画出来：
 *   1. 相位缠绕 —— 位置走远之后每个维度的相位开始来回折返，不同位置变得无法区分
 *   2. 可分辨维度数 —— none 在目标长度处直接归零
 *   3. 波长谱 —— 看清 NTK / YaRN 到底改了哪些维度（只拉低频，保住高频）
 *   4. 两个指标的权衡 —— 线性插值换来了长距离，却把局部精度丢了 8 倍
 */
import { useMemo, useState } from 'react'
import { Card, Segmented, Slider, Stats } from '../../components/Controls'
import { LineChart } from '../../components/LineChart'
import { Principle } from '../../components/Principle'
import {
  DEFAULT_CONTEXT,
  METHOD_LABEL,
  analyzeContext,
  phaseTrack,
  ropeDims,
  type ContextConfig,
  type RopeMethod,
} from '../../core/context'
import { useLang, type Lang } from '../../i18n'

const METHODS: RopeMethod[] = ['none', 'linear', 'ntk', 'yarn']
const MCOLOR: Record<RopeMethod, string> = {
  none: '#d4537e',
  linear: '#ba7517',
  ntk: '#534ab7',
  yarn: '#1d9e75',
}

function fmtDelta(v: number): string {
  if (v >= 1000) return (v / 1000).toFixed(v >= 10000 ? 0 : 1) + 'K'
  return v.toFixed(0)
}

const zh = {
  h2: '⑨ 长上下文外推：为什么 4K 训出来的模型喂 32K 就崩',
  lead1: 'RoPE 把位置编成一组不同频率的旋转。短波长的维度转得快，负责「这两个词挨着」；长波长的维度转得慢，负责「它们在文章的两端」。问题在于位置走得足够远之后，',
  lead2: '每个维度的相位都会绕圈回到原点',
  lead3: '，两个相距很远的 token 会拿到几乎一样的编码 —— 模型再也分不清它们差多远。',

  principleTitle: '几何上到底发生了什么？',
  d1a: '每个维度就是一个模 2π 的计数器。',
  d1b: '第 j 个维度对的角频率是 θ_j = base^(-2j/d)，波长 λ_j = 2π/θ_j。位置 t 的相位是 t·θ_j。当两个位置的距离 Δ 满足 Δ·θ_j ≥ π 时，这个维度的相位差就绕过了半圈 —— 它再也无法唯一地表示 Δ，这叫别名（aliasing）。距离越远，发生别名的维度越多；本页默认配置下，Δ 到 32768 时 64 个维度**全部**别名，几何上位置信息归零。',
  d2a: '所以三种方案都是在「把波长拉长」。',
  d2b: '线性插值（PI）最直接：位置统一除以 s，等价于把所有权重都乘 s，相位差压到原来的 1/s。NTK-aware 换了思路：不改位置，改 base。因为 θ_j = base^(-2j/d)，把 base 乘上 s^(d/(d-2)) 之后，高维（低频、长波长）被拉伸到接近 s 倍，低维（高频、短波长）几乎不动 —— 长距离覆盖和局部精度同时保住。YaRN 在 NTK 基础上再分频段：波长小于 L/β_fast 的维度完全不动，大于 L/β_slow 的完整拉伸，中间平滑过渡。',
  d3: '这也是为什么业界普遍用 NTK / YaRN 而不是线性插值：线性插值把高频维度的波长也一起拉长了 s 倍，模型就分不清「相邻两个词」和「隔了 s 个词」。看下面那张权衡表，线性插值的最小可分辨间隔会直接涨 8 倍。',
  warn: '这里只做几何分析：相位、波长、别名、期望内积，全部按公式算出来。但真实模型外推失败还牵涉训练时学到的注意力模式 —— 几何上「能区分」不等于模型「会用」。所以下面的几何上界（最长波长的一半，默认配置约 27K）会明显偏乐观，实际 4K 训出的模型往往几千 token 就开始退化。这一点不能含糊。',

  phaseTitle: '相位缠绕：位置走多远就开始认不出来了',
  phaseHint: '几个代表维度的相位折到 [-π, π] 之后的样子',
  phaseNote: '横轴是位置，纵轴是这个位置的相位（已折回 -π~π）。曲线越密，说明这个维度转得越快、越早失去区分能力。每个维度的「折返点」之间的距离就是它的波长 —— 当波长远远小于你想外推的距离时，同一个相位值会对应一大堆位置。切换上面的方法，你会看到这些锯齿被整体拉疏。',
  dimLabel: '维度',

  resolvTitle: '还有多少维度没失去区分能力',
  resolvHint: '纵轴是未发生别名的维度占比，横轴是对数刻度的相对距离',
  scoreHint: '纵轴是 RoPE 的期望注意力分数',
  metric: '指标',
  metricResolv: '未别名维度占比',
  metricScore: '期望注意力分数',
  resolvNote: '这张图是全文最直接的判据。「不改」的曲线在 Δ = 目标长度处掉到 0 —— 意思是这个距离上没有任何一个维度还能唯一表示位置，模型拿到的是纯粹的噪声。三条改写方案都把曲线整体往右推，其中线性插值和 YaRN 在目标长度处保住的维度最多。',
  scoreNote: '纵轴是 (1/n)·Σ cos(Δ·θ_j)，也就是随机单位 q/k 下 RoPE 点积的解析期望，文献里说的「RoPE 远程衰减」就是它。注意它是**振荡**的，不是单调衰减 —— 所以别只看某一个点。真正要紧的是：在训练长度（竖线）之后，曲线还有没有结构。分数在 0 附近抖动意味着远距离的注意力 logit 大家差不多，注意力变成了均匀分布。',
  trainMark: '训练长度',

  specTitle: '波长谱：NTK 和 YaRN 到底改了哪些维度',
  specHint: '纵轴是对数刻度的波长，横轴是维度编号',
  specNote: '每一条横向差距就是一个维度被拉长了多少倍。「线性插值」是一条水平的整体上移（所有维度一视同仁 ×s）；「NTK」是一条斜线 —— 低编号的高频维度几乎没动，高编号的低频维度被拉满；「YaRN」是折线 —— 两头是平的，中间过渡，所以它能更精确地控制「哪些维度该保精度、哪些该管长距离」。',
  jLabel: '维度编号',

  tradeTitle: '两个指标的权衡：这就是选型的全部依据',
  tradeHint: '局部精度和长距离能力天生对立',
  tradeNote: '左边一列是局部精度（能区分的最小间隔，越小越好），右边一列是长距离能力（目标长度处还没别名的维度数，越多越好）。「不改」是长距离直接归零；线性插值买到了长距离，但局部精度掉了 s 倍（默认配置下 3.14 → 25.13）；NTK 和 YaRN 两个指标都不塌 —— 这就是它们成为业界默认选择的原因。',
  minGap: '最小可分辨间隔',
  aliveAtTarget: '目标长度处可用维度',
  horizon: '几何上界（最长波长的一半）',
  lowFreq: '长波长维度占比（4K 内转不到一圈）',

  configTitle: '配置',
  dHead: 'head 维度 d',
  trainLen: '训练长度',
  targetLen: '目标长度',
  base: 'RoPE base',
  method: '外推方法',
  betaFast: 'β_fast（高频边界）',
  betaSlow: 'β_slow（低频边界）',
  yarnTemp: 'YaRN 注意力温度补偿',
  yarnTempHint: '把 logits 乘 1/(0.1·ln s + 1) 补偿缩放带来的熵下降',
  lowFreqHint: '波长长于训练长度的那些维度，它们是长距离信息的载体',
  scaleFactor: '缩放倍数 s',
  dimsCount: '维度对数量',
}

const en: typeof zh = {
  h2: '⑨ Context extension: why a 4K model breaks at 32K',
  lead1: 'RoPE encodes position as a set of rotations at different frequencies. Short-wavelength dimensions spin fast and say "these two words are adjacent"; long-wavelength ones spin slowly and say "they sit at opposite ends of the document". The problem is that once positions go far enough, ',
  lead2: 'every dimension\u2019s phase wraps back around',
  lead3: ', so two tokens very far apart receive almost identical encodings — and the model can no longer tell how far apart they are.',

  principleTitle: 'What is actually happening, geometrically?',
  d1a: 'Each dimension is a counter modulo 2π.',
  d1b: 'Dimension pair j has angular frequency θ_j = base^(-2j/d) and wavelength λ_j = 2π/θ_j. The phase at position t is t·θ_j. Once a distance Δ satisfies Δ·θ_j ≥ π, that dimension\u2019s phase difference has wrapped past half a turn and can no longer represent Δ uniquely — that is aliasing. The farther apart the positions, the more dimensions alias; at the default settings here, by Δ = 32768 **all 64** dimensions have aliased and the geometric position signal is gone.',
  d2a: 'So all three methods amount to stretching the wavelengths.',
  d2b: 'Linear interpolation (PI) is the most direct: divide all positions by s, which is the same as multiplying every wavelength by s and compressing phase differences to 1/s. NTK-aware takes another route: leave positions alone and change the base. Since θ_j = base^(-2j/d), multiplying the base by s^(d/(d-2)) stretches the high-index (low-frequency, long-wavelength) dimensions close to s× while leaving the low-index (high-frequency, short-wavelength) ones almost untouched — preserving long-range coverage and local precision at the same time. YaRN goes one step further, splitting by wavelength band: dimensions with λ below L/β_fast stay completely untouched, those above L/β_slow are fully stretched, and the band between ramps smoothly.',
  d3: 'This is why the field settled on NTK / YaRN rather than linear interpolation: PI stretches the high-frequency wavelengths by s as well, so the model can no longer tell "two adjacent words" from "two words s apart". In the trade-off table below, PI\u2019s minimum resolvable gap jumps 8×.',
  warn: 'This is a geometric analysis only: phases, wavelengths, aliasing and the expected inner product all come from the formulas. But real extrapolation failure also involves the attention patterns learned during training — "geometrically distinguishable" does not mean "the model uses it". So the geometric horizon below (half the longest wavelength, roughly 27K at the defaults) is noticeably optimistic; a model trained at 4K typically degrades after a few thousand tokens. That caveat matters.',

  phaseTitle: 'Phase wrapping: how far before it can no longer tell',
  phaseHint: 'a few representative dimensions, phase folded into [-π, π]',
  phaseNote: 'The x axis is position and the y axis is that position\u2019s phase (folded back into -π..π). Denser curves mean the dimension spins faster and loses discriminative power sooner. The spacing between fold-backs is exactly the wavelength — when the wavelength is far smaller than the distance you want to extrapolate to, one phase value corresponds to a whole crowd of positions. Switch the method above and watch these sawteeth get stretched apart.',
  dimLabel: 'dim',

  resolvTitle: 'How many dimensions keep their discriminative power',
  resolvHint: 'y is the share of dimensions not yet aliased, x is distance on a log scale',
  scoreHint: 'y is RoPE\u2019s expected attention score',
  metric: 'metric',
  metricResolv: 'share of un-aliased dimensions',
  metricScore: 'expected attention score',
  resolvNote: 'This is the most direct test in the whole module. The "none" curve falls to zero at Δ = target length — meaning that at that distance not a single dimension can still represent position uniquely, and the model receives pure noise. All three rewrites push the curve to the right, with linear interpolation and YaRN retaining the most dimensions at the target length.',
  scoreNote: 'The y axis is (1/n)·Σ cos(Δ·θ_j), the analytic expectation of a RoPE dot product for random unit q/k — what the literature calls RoPE\u2019s long-term decay. Note that it **oscillates** rather than decaying monotonically, so never read a single point. What matters is whether the curve still has structure beyond the training length (the vertical line). Scores jittering around zero mean distant attention logits are all alike, and attention degenerates into a uniform distribution.',
  trainMark: 'training length',

  specTitle: 'Wavelength spectrum: which dimensions do NTK and YaRN actually touch',
  specHint: 'y is wavelength on a log scale, x is the dimension index',
  specNote: 'The vertical gap at each index is how much that dimension got stretched. Linear interpolation is a uniform upward shift (every dimension ×s). NTK is a slanted line — the low-index high-frequency dimensions barely move while the high-index low-frequency ones stretch fully. YaRN is a piecewise line — flat at both ends with a transition between, which is precisely how it controls which dimensions keep precision and which handle long range.',
  jLabel: 'dim index',

  tradeTitle: 'The trade-off: this is the entire basis for choosing',
  tradeHint: 'local precision and long-range reach are inherently opposed',
  tradeNote: 'The left column is local precision (the smallest distinguishable gap — smaller is better); the right is long-range reach (un-aliased dimensions at the target length — more is better). "none" loses long range entirely. Linear interpolation buys the long range but gives up s× of local precision (3.14 → 25.13 at the defaults). NTK and YaRN collapse on neither — which is why they became the default choice.',
  minGap: 'min resolvable gap',
  aliveAtTarget: 'usable dims at target',
  horizon: 'geometric horizon (half the longest wavelength)',
  lowFreq: 'long-wavelength share (under 1 turn within 4K)',

  configTitle: 'Settings',
  dHead: 'head dim d',
  trainLen: 'training length',
  targetLen: 'target length',
  base: 'RoPE base',
  method: 'extension method',
  betaFast: 'β_fast (high-freq edge)',
  betaSlow: 'β_slow (low-freq edge)',
  yarnTemp: 'YaRN attention temperature',
  yarnTempHint: 'scale logits by 1/(0.1·ln s + 1) to compensate the entropy drop from stretching',
  lowFreqHint: 'dimensions whose wavelength exceeds the training length — they carry long-range information',
  scaleFactor: 'scale factor s',
  dimsCount: 'dimension pairs',
}

const DICT: Record<Lang, typeof zh> = { zh, en }

/** 相位缠绕图：多个维度的相位折到 [-π, π] */
function PhasePlot(props: {
  tracks: { j: number; phase: number[]; theta: number }[]
  positions: number[]
  trainLen: number
  dimLabel: string
  trainMark: string
  height?: number
}) {
  const W = 620
  const H = props.height ?? 240
  const padL = 44
  const padR = 14
  const padT = 16
  const padB = 30

  const tMax = props.positions[props.positions.length - 1] || 1
  const sx = (t: number) => padL + (t / tMax) * (W - padL - padR)
  const sy = (p: number) => H - padB - ((p + Math.PI) / (2 * Math.PI)) * (H - padB - padT)

  const colors = ['#534ab7', '#1d9e75', '#ba7517', '#d4537e', '#185fa5']

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }}>
      <line x1={padL} y1={sy(Math.PI)} x2={W - padR} y2={sy(Math.PI)} stroke="var(--border)" strokeWidth={0.6} />
      <line x1={padL} y1={sy(0)} x2={W - padR} y2={sy(0)} stroke="var(--border)" strokeWidth={0.6} />
      <line
        x1={padL}
        y1={sy(-Math.PI)}
        x2={W - padR}
        y2={sy(-Math.PI)}
        stroke="var(--border)"
        strokeWidth={0.6}
      />
      <text x={padL - 6} y={sy(Math.PI)} textAnchor="end" dominantBaseline="central" fontSize={10} fill="var(--text-3)">
        π
      </text>
      <text x={padL - 6} y={sy(0)} textAnchor="end" dominantBaseline="central" fontSize={10} fill="var(--text-3)">
        0
      </text>
      <text x={padL - 6} y={sy(-Math.PI)} textAnchor="end" dominantBaseline="central" fontSize={10} fill="var(--text-3)">
        -π
      </text>

      {props.trainLen <= tMax && (
        <>
          <line
            x1={sx(props.trainLen)}
            y1={padT}
            x2={sx(props.trainLen)}
            y2={H - padB}
            stroke="var(--text-3)"
            strokeWidth={0.8}
            strokeDasharray="4 3"
          />
          <text
            x={sx(props.trainLen) + 4}
            y={padT + 8}
            fontSize={10}
            fill="var(--text-3)"
          >
            {props.trainMark}
          </text>
        </>
      )}

      {props.tracks.map((tr, idx) => (
        <polyline
          key={tr.j}
          points={props.positions.map((t, k) => `${sx(t).toFixed(1)},${sy(tr.phase[k]).toFixed(1)}`).join(' ')}
          fill="none"
          stroke={colors[idx % colors.length]}
          strokeWidth={1.1}
          opacity={0.85}
        />
      ))}

      <text x={padL} y={H - padB + 16} fontSize={10} fill="var(--text-3)">
        0
      </text>
      <text x={W - padR} y={H - padB + 16} textAnchor="end" fontSize={10} fill="var(--text-3)">
        {fmtDelta(tMax)}
      </text>
    </svg>
  )
}

export function ContextModule() {
  const { lang } = useLang()
  const c = DICT[lang]
  const [cfg, setCfg] = useState<ContextConfig>(DEFAULT_CONTEXT)
  const [metric, setMetric] = useState<0 | 1>(0)

  const patch = (p: Partial<ContextConfig>) => setCfg((prev) => ({ ...prev, ...p }))

  const report = useMemo(() => analyzeContext(cfg), [cfg])
  const byMethod = useMemo(
    () => METHODS.map((m) => ({ m, r: analyzeContext({ ...cfg, method: m }) })),
    [cfg],
  )

  const half = Math.floor(cfg.dHead / 2)
  const jList = useMemo(() => {
    const set = [0, Math.floor(half * 0.25), Math.floor(half * 0.5), Math.floor(half * 0.75), half - 1]
    return Array.from(new Set(set)).filter((j) => j >= 0 && j < half)
  }, [half])

  const positions = useMemo(() => {
    const n = 900
    return Array.from({ length: n }, (_, i) => (i / (n - 1)) * cfg.targetLen)
  }, [cfg.targetLen])

  const tracks = useMemo(() => phaseTrack(cfg, jList, positions), [cfg, jList, positions])

  // 以 log10(Δ) 作横轴，这样从 1 到 32K 的衰减过程才看得清
  const logLines = useMemo(
    () =>
      byMethod.map(({ m, r }) => {
        const src = metric === 0 ? r.resolvable : r.score
        return {
          name: METHOD_LABEL[m][lang],
          color: MCOLOR[m],
          points: src.map((p) => ({ x: Math.log10(p.delta), y: p.value })),
        }
      }),
    [byMethod, metric, lang],
  )

  const specLines = useMemo(() => {
    return METHODS.map((m) => {
      const d = ropeDims({ ...cfg, method: m })
      return {
        name: METHOD_LABEL[m][lang],
        color: MCOLOR[m],
        points: d.map((x, i) => ({ x: i, y: Math.log10(x.lambda) })),
        dashed: m !== cfg.method,
      }
    })
  }, [cfg, lang])

  const aliveAtTarget = (m: RopeMethod) =>
    byMethod.find((x) => x.m === m)!.r.dims.filter((d) => cfg.targetLen * d.theta < Math.PI).length

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
        formula={`RoPE 角频率    θ_j = base^(-2j/d)          j = 0 .. d/2-1
波长            λ_j = 2π / θ_j
位置 t 的相位    φ_j(t) = t · θ_j      （模 2π）
别名条件         Δ · θ_j ≥ π          该维度无法再唯一表示距离 Δ

linear (PI)     θ_j' = θ_j / s                          s = L_target / L_train
ntk             base' = base · s^(d/(d-2))  →  θ_j' = base'^(-2j/d)
yarn            ramp_j = clamp( (λ_j - L/β_fast) / (L/β_slow - L/β_fast), 0, 1 )
                θ_j' = θ_j / s^(ramp_j)

期望注意力分数   score(Δ) = (1/n) · Σ_j cos(Δ · θ_j)`}
        analogy={
          lang === 'zh'
            ? '一架有 64 根指针的钟。每根指针转速不同，转得快的负责报告「秒」，转得慢的负责报告「年」。读时间就是同时看 64 根指针的位置 —— 组合起来就能唯一确定时刻。但如果这架钟只在 4 千秒内被校准过，现在要它报 3 万 2 千秒，那些转得快的指针早就绕了无数圈，早就分不清是这一圈还是上一圈。要么把指针全调慢 s 倍（线性插值，但那连「差几秒」也分不清了），要么只把慢指针调慢、快指针不动（NTK / YaRN）。'
            : 'A clock with 64 hands. Each spins at a different speed: the fast ones report seconds, the slow ones report years. Reading the time means looking at all 64 positions at once — the combination uniquely determines the moment. But if the clock was only calibrated for the first 4,000 seconds and you now ask it about second 32,000, the fast hands have wrapped countless times and can no longer tell this turn from the last. You can slow every hand by s (linear interpolation — but then even "a few seconds apart" becomes unreadable), or slow only the slow hands and leave the fast ones alone (NTK / YaRN).'
        }
        detail={
          <>
            <p>
              <strong>{c.d1a}</strong> {c.d1b}
            </p>
            <p>
              <strong>{c.d2a}</strong> {c.d2b}
            </p>
            <p>{c.d3}</p>
          </>
        }
        warn={c.warn}
      />

      <Card title={c.configTitle}>
        <div className="controls">
          <div className="control" style={{ minWidth: 300 }}>
            <label>
              <span>{c.method}</span>
            </label>
            <Segmented
              value={cfg.method}
              onChange={(v) => patch({ method: v })}
              options={METHODS.map((m) => ({ value: m, label: METHOD_LABEL[m][lang] }))}
            />
          </div>
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
            label={c.trainLen}
            value={cfg.trainLen}
            min={512}
            max={8192}
            step={512}
            onChange={(v) => patch({ trainLen: v })}
            format={(v) => fmtDelta(v)}
          />
          <Slider
            label={c.targetLen}
            value={cfg.targetLen}
            min={2048}
            max={131072}
            step={2048}
            onChange={(v) => patch({ targetLen: v })}
            format={(v) => fmtDelta(v)}
          />
          <Slider
            label={c.base}
            value={cfg.base}
            min={1000}
            max={1000000}
            step={1000}
            onChange={(v) => patch({ base: v })}
            format={(v) => v.toLocaleString('en-US')}
          />
          <Slider
            label={c.betaFast}
            value={cfg.betaFast}
            min={1}
            max={128}
            onChange={(v) => patch({ betaFast: v })}
            format={(v) => String(v)}
            disabled={cfg.method !== 'yarn'}
          />
          <Slider
            label={c.betaSlow}
            value={cfg.betaSlow}
            min={1}
            max={64}
            onChange={(v) => patch({ betaSlow: v })}
            format={(v) => String(v)}
            disabled={cfg.method !== 'yarn'}
          />
          <div className="control" style={{ minWidth: 220 }}>
            <label title={c.yarnTempHint}>
              <span>{c.yarnTemp}</span>
            </label>
            <Segmented
              value={cfg.yarnTemp ? 1 : 0}
              onChange={(v) => patch({ yarnTemp: v === 1 })}
              options={[
                { value: 0, label: lang === 'zh' ? '关' : 'off' },
                { value: 1, label: lang === 'zh' ? '开' : 'on' },
              ]}
            />
          </div>
        </div>
      </Card>

      <Card title={c.phaseTitle} hint={c.phaseHint} exportName="09-context-phase-wrap">
        <div className="chip-row" style={{ marginBottom: 8 }}>
          {tracks.map((tr, i) => (
            <span
              key={tr.j}
              style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-2)' }}
            >
              <span
                style={{
                  width: 14,
                  height: 3,
                  borderRadius: 2,
                  display: 'inline-block',
                  background: ['#534ab7', '#1d9e75', '#ba7517', '#d4537e', '#185fa5'][i % 5],
                }}
              />
              {`${c.dimLabel} ${tr.j}`}
            </span>
          ))}
        </div>
        <PhasePlot
          tracks={tracks}
          positions={positions}
          trainLen={cfg.trainLen}
          dimLabel={c.dimLabel}
          trainMark={c.trainMark}
        />
        <div className="note" style={{ marginTop: 8 }}>{c.phaseNote}</div>
      </Card>

      <Card title={c.resolvTitle} hint={metric === 0 ? c.resolvHint : c.scoreHint} exportName="09-context-aliasing">
        <div style={{ marginBottom: 10 }}>
          <div className="control" style={{ minWidth: 260 }}>
            <label>
              <span>{c.metric}</span>
            </label>
            <Segmented
              value={metric}
              onChange={(v) => setMetric(v)}
              options={[
                { value: 0, label: c.metricResolv },
                { value: 1, label: c.metricScore },
              ]}
            />
          </div>
        </div>
        <LineChart
          lines={logLines}
          xLabel={`Δ (log10)　—　${c.trainMark} = ${fmtDelta(cfg.trainLen)}`}
          yLabel={metric === 0 ? '0 — 1' : '-1 — 1'}
          height={280}
          yMin={metric === 0 ? 0 : -1}
          yFormat={(v) => v.toFixed(2)}
          xFormat={(v) => fmtDelta(Math.pow(10, v))}
        />
        <div className="note" style={{ marginTop: 8 }}>{metric === 0 ? c.resolvNote : c.scoreNote}</div>
      </Card>

      <Card title={c.specTitle} hint={c.specHint} exportName="09-context-wavelength">
        <LineChart
          lines={specLines}
          xLabel={c.jLabel}
          yLabel="log10(λ)"
          height={260}
          yFormat={(v) => fmtDelta(Math.pow(10, v))}
          xFormat={(v) => v.toFixed(0)}
        />
        <div className="note" style={{ marginTop: 8 }}>{c.specNote}</div>
      </Card>

      <Card title={c.tradeTitle} hint={c.tradeHint} exportName="09-context-tradeoff">
        <table className="tbl">
          <thead>
            <tr>
              <th>{c.method}</th>
              <th>{c.minGap}</th>
              <th>{c.aliveAtTarget}</th>
              <th>{c.scaleFactor}</th>
            </tr>
          </thead>
          <tbody>
            {byMethod.map(({ m, r }) => (
              <tr key={m} style={m === cfg.method ? { background: 'var(--accent-soft)' } : undefined}>
                <td>{METHOD_LABEL[m][lang]}</td>
                <td>{r.minGap.toFixed(2)}</td>
                <td>{`${aliveAtTarget(m)} / ${r.dims.length}`}</td>
                <td>{`×${r.scale.toFixed(1)}`}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <Stats
          items={[
            { k: c.horizon, v: fmtDelta(report.geometricHorizon) },
            { k: c.lowFreq, v: `${(report.lowFreqShare * 100).toFixed(1)}%` },
            { k: c.dimsCount, v: String(report.dims.length) },
            { k: c.base, v: cfg.base.toLocaleString('en-US') },
          ]}
        />
        <div className="note" style={{ marginTop: 10 }}>{c.tradeNote}</div>
        <div className="warn" style={{ marginTop: 8 }}>{c.warn}</div>
      </Card>
    </div>
  )
}

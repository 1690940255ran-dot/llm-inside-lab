/**
 * 模块七：MoE（稀疏专家）
 *
 * 这个模块想让三件事变得可见：
 *   1. 路由器真的会自己长出分工（token 按主题排好序之后，热力图上是块对角的）
 *   2. 分工天然不均 —— 根源是语料话题频率本身服从 Zipf，不是路由器的锅
 *   3. 治它有两个流派，作用在同一组参数（每专家一个偏置）上，差别只在信号：
 *      aux loss 用损失梯度（死专家那一项精确为 0，而且梯度在最优处不归零，
 *      所以曲线抖、开大了会自己造出新死专家）；负载反馈偏置用实测负载当信号
 *      （均衡了偏差自动归零，需要的偏置比 aux 小一个数量级）
 *
 * 所有数字都来自 core/moe.ts 的真实计算，页面上给的实测值也是跑出来的。
 */
import { useMemo, useState } from 'react'
import { Card, Segmented, Slider, Stats } from '../../components/Controls'
import { BarList, Heatmap, HeatLegend } from '../../components/Heatmap'
import { LineChart } from '../../components/LineChart'
import { Principle } from '../../components/Principle'
import {
  DEFAULT_MOE,
  MOE_ARCH_PRESETS,
  MOE_SCENARIOS,
  fmtParams,
  moeAnatomy,
  sweepBalance,
  trainRouter,
  type BalanceMode,
  type MoEConfig,
} from '../../core/moe'
import { heatColor } from '../../core/color'
import { useLang, type Lang } from '../../i18n'

const zh = {
  h2: '⑦ MoE：稀疏专家',
  lead1: '稠密模型每读一个 token，所有参数都要参与计算。MoE 换了个思路：把 FFN 拆成几十上百个专家，每个 token 只走其中 ',
  lead2: 'top-k 个',
  lead3: '。于是总参数可以做到几百 B，而每个 token 实际动用的只有几十 B —— 这就是 ',
  lead4: '671B 的 DeepSeek-V3 能跑在几十 B 稠密模型的成本上',
  lead5: '的原因。但它有个著名的副作用：专家负载会严重倾斜，甚至出现永远拿不到 token 的「死专家」。',

  principleTitle: '路由器怎么选专家？为什么会选歪？',
  d1a: '路由就是一次小型的注意力。',
  d1b: '每个专家有一个路由向量，token 和它做点积、除温度、过 softmax 得到 gate 概率，取 top-k 就选完了。gate 概率还会当作权重，把被选中专家的输出加权求和。所以「专家」不是真的分类器，只是一组和 token 对齐的方向。',
  d2a: '但数据本来就是歪的。',
  d2b: '真实语料的话题频率接近 Zipf 分布。本页的 token 也按 Zipf 生成主题，最热门的主题占了 ',
  d2c: ' 的 token。路由按相似度分流，负载自然就跟着数据倾斜 —— 这一层跟路由器学得好不好没关系，是数据的先验。',
  d3a: '于是有两条治它的路子，而且它们作用在同一个地方。',
  d3b: '现代 MoE 都给每个专家配一个 logit 偏置 b_e（路由 logit = cos/T + b_e），均衡力改的就是它。区别在「用什么当信号」：aux loss 用辅助损失的解析梯度 ∂L/∂b_e = n_E·f_e·avg[p(1-p)]，其中 f_e 是分到的 token 占比 —— 死专家的 f_e = 0，所以这一项精确等于 0，均衡力对它没有任何直接作用。DeepSeek-V3 干脆不用梯度，直接看实测负载偏差做负反馈，负载平了偏差就归零、自动停手。',

  formulaTitle: '',
  warn: '这里的 token 向量和路由权重是按确定性种子造出来的，不是真实语料上训出来的模型。但路由的每一小步——softmax、top-k、容量淘汰、aux loss 梯度、偏置反馈、专家专精更新——都是按真实公式算的，负载倾斜和死专家是这套机制自己涌现出来的，不是画出来的。',

  scenarioTitle: '六个实测场景（数字都是跑出来的）',
  scenarioHint: '点一下切换，再自己拖滑块验证。场景会把配置重置回默认值，方便对照复现',
  configTitle: '配置',
  nExperts: '专家总数',
  topK: '每 token 激活数 top-k',
  nTokens: 'token 数',
  nTopics: '潜在主题数',
  nTopicsHint: '主题数少于专家数时，会出现结构性的死专家 —— 均衡力也只能让它们轮流上岗',
  noise: '主题内散度',
  noiseHint: '越大主题越糊，路由越难学出分工',
  steps: '路由训练步数',
  lr: '路由学习率',
  balanceMode: '均衡方式',
  modeNone: '不治',
  modeAux: 'aux 梯度',
  modeBias: '负载反馈',
  modeHint: '两种方式都只改偏置 b_e，唯一区别是信号来源',
  balance: '均衡强度',
  balanceHintAux: 'aux loss 权重。实测 1 左右最好，开到 4 会自己造出死专家',
  balanceHintBias: '反馈增益。实测 2 就能把倾斜压到 1.06，开到 8 会过冲',
  balanceHintNone: '当前是「不治」，调它没有效果 —— 先把上面的均衡方式切到 aux 或负载反馈',
  capacity: '容量因子',
  capacityHint: '1.0 = 严格平均配额，超出的 token 被丢弃；0 = 不限容量',

  heatTitle: '路由热力图：token 真的自己分好了组',
  heatHint: '按主题排序后能看到块对角 —— 这就是专家专精',
  heatMode: '着色方式',
  heatProb: '绝对 gate 概率',
  heatNorm: '按行归一（看清选了谁）',
  topic: '主题',
  heatNote: '每行是一个 token，每列是一个专家，颜色是 gate 概率。按主题排序后，同一主题的 token 会集中流向同一批专家，形成明显的块状结构；这就是「专家学会了分工」的样子，它是路由自己长出来的，没有人告诉它哪个专家该管哪个主题。',
  clarityNote: '注意绝对概率普遍偏小（softmax 要在 8 个专家间分配），所以「绝对概率」模式看着偏暗，切到「按行归一」更好看路由决策。',

  loadTitle: '专家负载：不治 vs 当前配置',
  loadHint: '左：完全不施加约束；右：当前配置',
  loadNone: '不施加任何约束',
  loadNow: '当前配置',
  experts: '专家',

  gradTitle: '证据：死专家的 aux 梯度精确等于 0',
  gradHint: '左＝负载，右＝辅助损失对偏置的梯度',
  gradLoad: '负载（分到的 token 数）',
  gradGrad: '∂L/∂b_e（aux 对偏置的梯度）',
  gradNote: '这是这个模块最硬的一条证据。辅助损失的梯度按定义正比于 f_e（该专家分到的 token 占比），死专家的 f_e = 0，所以柱子上就是精确的 0.0000 —— 均衡力推不动它。反观偏置反馈：死专家的负载偏差是 -1（最负），偏置被顶到最高。两者都作用在同一个 b_e 上，差别完全来自信号。',

  sweepTitle: '两种流派的曲线形状完全不同',
  sweepHint: '同一个横轴（均衡强度），四条线',
  sweepNote: '紫色是 aux 梯度，青色是负载反馈，实线是负载均衡度（1/倾斜，越高越平），虚线是路由熵。aux 的实线是抖的、非单调的：权重 4 和 16 反而比 1 更差，因为它靠损失梯度推偏置，而那个梯度在负载已经均衡的位置也不会归零（f_e 被当作常数 stop-grad 掉），于是它一路把偏置推到 8.85、36.2，顺手把路由向量学好的分工冲掉，还自己造出一个新的死专家。负载反馈则是一条平滑下降曲线，偏置只要 0.30 就够 —— 它是负反馈，均衡了偏差就归零，自动停手。这也是 DeepSeek-V3 弃用 aux loss 梯度、改用偏置反馈的原因。',
  evennessLabel: '均衡度 1/倾斜（越高越平）',
  entropyLabel: '路由熵（越高越均匀）',
  auxTag: 'aux 梯度',
  biasTag: '负载反馈',

  statTitle: '当前配置的关键数字',
  imbalance: '负载倾斜 max/mean',
  dead: '死专家数',
  dropped: '完全丢弃的 token',
  partial: '少拿到专家的 token',
  entropy: '路由熵',
  auxLoss: 'aux loss',
  auxFloor: '完美均衡下界',
  purity: '平均专精纯度',
  biasNorm: '偏置绝对值均值 |b|',
  biasUnit: 'logit 单位',
  expertMapTitle: '每个专家主要管哪个主题',
  purityHint: '纯度 = 该专家收到的 token 里第一主题占比。点击专家可看它在各主题上的分布',

  budgetTitle: '参数量预算：总参数和激活参数为什么差一个数量级',
  budgetHint: '换成真实模型的超参，可以核对官方公布的参数量',
  preset: '机型预设',
  published: '官方公布',
  calculated: '本页按公式算出',
  total: '总参数',
  active: '激活参数',
  ratio: '激活占比',
  expertShare: '专家参数占比',
  budgetNote: '激活参数 = 稠密部分（词表嵌入 + 注意力 + LayerNorm，每个 token 都要走）+ 激活的专家。总参数里绝大多数的专家参数只有被选中时才参与计算，所以显存要按总参数算，算力按激活参数算。注意 DeepSeek-V3 用的是 MLA 而不是标准多头注意力，本页的公式套不上它，所以预设里没放它（官方数字是 671B 总 / 37B 激活）。',
}

const en: typeof zh = {
  h2: '⑦ MoE: sparse experts',
  lead1: 'A dense model runs every parameter for every token. MoE does something else: split the FFN into dozens or hundreds of experts and let each token visit only ',
  lead2: 'its top-k of them',
  lead3: '. Total parameters can then reach hundreds of billions while each token only touches tens of billions — which is why ',
  lead4: 'DeepSeek-V3 at 671B runs at the cost of a dense model tens of times smaller',
  lead5: '. But it has a famous side effect: expert load skews badly, and some experts become permanently dead — never receiving a single token.',

  principleTitle: 'How does the router pick, and why does it pick badly?',
  d1a: 'Routing is a tiny attention.',
  d1b: 'Each expert has a router vector; a token dot-products against it, divides by a temperature and runs softmax to get gate probabilities, then top-k finishes the selection. Those probabilities also serve as weights when summing the selected experts. An expert is not a real classifier — it is just a direction that tokens align with.',
  d2a: 'But the data is already skewed.',
  d2b: 'Topic frequency in real text follows a Zipf law. The tokens here are generated the same way, and the hottest topic takes ',
  d2c: ' of all tokens. Routing by similarity therefore inherits the skew — that layer has nothing to do with how well the router learns; it is a prior of the data.',
  d3a: 'So there are two ways to fight it, and they act on exactly the same place.',
  d3b: 'Modern MoE gives every expert a logit bias b_e (routing logit = cos/T + b_e), and balancing only touches that. The difference is the signal. The aux loss uses its analytic gradient ∂L/∂b_e = n_E·f_e·avg[p(1-p)], where f_e is the share of tokens dispatched — for a dead expert f_e = 0, so that term is exactly zero and balancing exerts no direct force on it. DeepSeek-V3 dropped the gradient entirely and feeds the measured load deviation back instead: once load is even the deviation goes to zero and it stops on its own.',

  formulaTitle: '',
  warn: 'The token vectors and router weights here are generated from a deterministic seed, not trained on real text. But every step of the routing — softmax, top-k, capacity eviction, the auxiliary-loss gradient, the bias feedback, the expert update — follows the real formulas, and the load skew and dead experts emerge from that machinery rather than being drawn.',

  scenarioTitle: 'Six measured scenarios (numbers come from the code)',
  scenarioHint: 'click to switch, then drag the sliders yourself. A scenario resets the config to defaults so the numbers can be reproduced',
  configTitle: 'Configuration',
  nExperts: 'total experts',
  topK: 'experts per token (top-k)',
  nTokens: 'tokens',
  nTopics: 'latent topics',
  nTopicsHint: 'when topics are fewer than experts you get structurally dead experts — balancing can only rotate who is on duty',
  noise: 'within-topic spread',
  noiseHint: 'larger = fuzzier topics, harder to specialise',
  steps: 'router training steps',
  lr: 'router learning rate',
  balanceMode: 'balancing mode',
  modeNone: 'untreated',
  modeAux: 'aux gradient',
  modeBias: 'load feedback',
  modeHint: 'both only move the bias b_e — the only difference is the signal',
  balance: 'balancing strength',
  balanceHintAux: 'aux loss weight. Measured sweet spot is around 1; pushing to 4 creates a new dead expert',
  balanceHintBias: 'feedback gain. A gain of 2 already brings skew down to 1.06; 8 overshoots',
  balanceHintNone: 'you are in "untreated" mode, so this slider does nothing — switch the mode above to aux or load feedback first',
  capacity: 'capacity factor',
  capacityHint: '1.0 = a strict even quota, overflow tokens get dropped; 0 = unlimited',

  heatTitle: 'Routing heatmap: the tokens really do sort themselves',
  heatHint: 'sorted by topic it shows block diagonals — that is specialisation',
  heatMode: 'colour scale',
  heatProb: 'absolute gate probability',
  heatNorm: 'row-normalised (see who got picked)',
  topic: 'topic',
  heatNote: 'Each row is a token, each column an expert, and the colour is the gate probability. Sorted by topic, tokens of the same topic flow to the same few experts, producing an unmistakable block structure. That is what "the experts learned to divide the work" looks like — and the router grew it on its own; nobody told it which expert should own which topic.',
  clarityNote: 'Absolute probabilities look dim because softmax has to spread mass across 8 experts. Switch to row-normalised to read the routing decisions more easily.',

  loadTitle: 'Expert load: untreated vs current config',
  loadHint: 'left: no constraint at all; right: current config',
  loadNone: 'no constraint at all',
  loadNow: 'current config',
  experts: 'experts',

  gradTitle: 'The evidence: a dead expert\u2019s aux gradient is exactly zero',
  gradHint: 'left = load, right = the auxiliary loss gradient w.r.t. the bias',
  gradLoad: 'load (tokens received)',
  gradGrad: '∂L/∂b_e (aux gradient w.r.t. bias)',
  gradNote: 'This is the hardest piece of evidence in this module. The auxiliary-loss gradient is by definition proportional to f_e, the share of tokens that expert received. A dead expert has f_e = 0, so its bar is exactly 0.0000 — balancing simply cannot push it. Load feedback instead sees a dead expert as the most negative deviation (−1) and drives its bias to the top. Both act on the same b_e; the entire difference comes from the signal.',

  sweepTitle: 'The two schools have completely different curves',
  sweepHint: 'one x-axis (balancing strength), four lines',
  sweepNote: 'Purple is the aux gradient, teal is load feedback; solid is load evenness (1/skew, higher is flatter) and dashed is routing entropy. The aux solid line is noisy and non-monotonic: weights 4 and 16 are worse than 1, because it pushes the bias with a loss gradient that does not vanish even at the balanced optimum (f_e is stop-grad\u2019ed to a constant). It drives the bias to 8.85 and then 36.2, washes out the specialisation the router vectors had learned, and creates a brand-new dead expert along the way. Load feedback is a smooth declining curve that needs a bias of only 0.30 — it is negative feedback, so the deviation goes to zero once load is even and it stops by itself. This is why DeepSeek-V3 dropped the aux-loss gradient in favour of bias feedback.',
  evennessLabel: 'evenness 1/skew (higher = flatter)',
  entropyLabel: 'routing entropy (higher = more even)',
  auxTag: 'aux gradient',
  biasTag: 'load feedback',

  statTitle: 'Key numbers for the current config',
  imbalance: 'load skew max/mean',
  dead: 'dead experts',
  dropped: 'tokens fully dropped',
  partial: 'tokens missing an expert',
  entropy: 'routing entropy',
  auxLoss: 'aux loss',
  auxFloor: 'perfectly balanced floor',
  purity: 'mean specialisation purity',
  biasNorm: 'mean |bias| |b|',
  biasUnit: 'logit units',
  expertMapTitle: 'Which topic each expert mainly covers',
  purityHint: 'purity = share of an expert\u2019s tokens in its top topic. Hover an expert for its full topic distribution',

  budgetTitle: 'Parameter budget: why total and active differ by an order of magnitude',
  budgetHint: 'load a real model\u2019s hyperparameters and check against the published count',
  preset: 'presets',
  published: 'published',
  calculated: 'computed here from the formula',
  total: 'total parameters',
  active: 'active parameters',
  ratio: 'active share',
  expertShare: 'expert share of total',
  budgetNote: 'Active parameters = the dense part (token embeddings + attention + LayerNorm, needed by every token) + the activated experts. Most of the total parameters are expert weights that only participate when selected, so memory is sized by the total and compute by the active count. Note DeepSeek-V3 uses MLA rather than standard multi-head attention, so the formula here does not fit it — that is why it is not a preset (published figures: 671B total / 37B active).',
}

const DICT: Record<Lang, typeof zh> = { zh, en }

/** 扫描均衡强度的取样点。aux 要覆盖到 16 才能看见它掉头，负载反馈 8 就过冲了 */
const SWEEP_W = [0, 0.5, 1, 2, 4, 8, 16]

const AUX_COLOR = 'var(--accent)'
const BIAS_COLOR = '#1d9e75'

export function MoEModule() {
  const { lang } = useLang()
  const c = DICT[lang]
  const [cfg, setCfg] = useState<MoEConfig>(DEFAULT_MOE)
  const [scenario, setScenario] = useState<string | null>('collapse')
  const [heatNorm, setHeatNorm] = useState<0 | 1>(1)
  const [presetId, setPresetId] = useState(MOE_ARCH_PRESETS[0].id)

  const patch = (p: Partial<MoEConfig>) => {
    setScenario(null)
    setCfg((prev) => ({ ...prev, ...p }))
  }

  const applyScenario = (id: string) => {
    const s = MOE_SCENARIOS.find((x) => x.id === id)
    if (!s) return
    setScenario(id)
    setCfg({ ...DEFAULT_MOE, ...s.patch })
  }

  const report = useMemo(() => trainRouter(cfg), [cfg])

  // 结构相关的参数变了才需要重跑基线 / 扫描，避免拖顶部滑块时反复重算
  const structKey = [
    cfg.nExperts,
    cfg.topK,
    cfg.nTokens,
    cfg.nTopics,
    cfg.dim,
    cfg.noise,
    cfg.steps,
    cfg.lr,
    cfg.seed,
  ].join('|')

  const baseline = useMemo(
    () => trainRouter({ ...cfg, balanceMode: 'none' as BalanceMode, balanceWeight: 0, capacityFactor: 0 }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [structKey],
  )

  const sweep = useMemo(
    () => ({
      aux: sweepBalance({ ...cfg, balanceWeight: 0, capacityFactor: 0 }, SWEEP_W, 'aux'),
      bias: sweepBalance({ ...cfg, balanceWeight: 0, capacityFactor: 0 }, SWEEP_W, 'bias'),
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [structKey],
  )

  // 热力图：按主题排序，每个主题抽样显示，控制总行数避免太长
  const heat = useMemo(() => {
    const perTopic: number[][] = Array.from({ length: cfg.nTopics }, () => [])
    for (const i of report.orderByTopic) perTopic[report.tokenSet.topicOf[i]].push(i)
    const active = perTopic.map((idxs, t) => ({ t, idxs })).filter((g) => g.idxs.length > 0)
    const MAX_ROWS = 132
    const perCap = Math.max(1, Math.min(8, Math.floor(MAX_ROWS / Math.max(1, active.length))))
    const matrix: number[][] = []
    const labels: string[] = []
    for (const g of active) {
      const step = Math.max(1, Math.ceil(g.idxs.length / perCap))
      const picked = g.idxs.filter((_, k) => k % step === 0).slice(0, perCap)
      picked.forEach((i, k) => {
        const row = report.final.prob[i].slice()
        matrix.push(heatNorm ? row.map((v) => v / Math.max(...row, 1e-6)) : row)
        labels.push(k === 0 ? `T${g.t}×${g.idxs.length}` : '')
      })
    }
    return { matrix, labels, shownTopics: active.length }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [report, cfg.nTopics, heatNorm])

  const colLabels = useMemo(
    () => Array.from({ length: cfg.nExperts }, (_, e) => `E${e}`),
    [cfg.nExperts],
  )

  const anatomy = useMemo(() => {
    const p = MOE_ARCH_PRESETS.find((x) => x.id === presetId) ?? MOE_ARCH_PRESETS[0]
    return { preset: p, a: moeAnatomy(p.arch) }
  }, [presetId])

  const pub = anatomy.preset.published
  const totalB = anatomy.a.totalParams / 1e9
  const activeB = anatomy.a.activeParams / 1e9

  const loadItems = (load: number[]) => load.map((v, e) => ({ label: `E${e}`, value: v }))

  const meanPurity =
    report.expertPurity.reduce((a, b) => a + b, 0) / Math.max(report.expertPurity.length, 1)
  const biasNorm =
    report.final.bias.reduce((a, b) => a + Math.abs(b), 0) / Math.max(report.final.bias.length, 1)

  const balanceHint =
    cfg.balanceMode === 'aux' ? c.balanceHintAux : cfg.balanceMode === 'bias' ? c.balanceHintBias : c.balanceHintNone

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
        formula={`gate      p(i,e) = softmax_e( cos(x_i, router_e) / T + b_e )
route     assign_i = TopK_e p(i,e)          (容量满则顺延，全满则该 token 被丢弃)
output    y_i = Σ_{e ∈ assign_i} w(i,e) · Expert_e(x_i),  w = 重新归一化后的 gate

专精力    router_e ← router_e + lr · w · (x_i − cos·router_e)      质心式更新
          ⇒ 专家被拉向"选中自己的那些 token"，分工自己长出来

均衡力   只改偏置 b_e（两种流派作用在同一个参数上）
  aux     ∂L/∂b_e = n_E · f_e · avg_i[ p_e(1−p_e) ]   L = n_E·Σ f_e·P_e
          死专家 f_e = 0 ⇒ 梯度恒为 0，推不动
  bias    b_e ← b_e − gain · (load_e/mean − 1)
          死专家偏差最负 ⇒ 偏置被顶到最高，重新上岗

参数      total  = dense + (nExperts + nShared) · expertFFN
          active = dense + (topK     + nShared) · expertFFN`}
        analogy={
          lang === 'zh'
            ? '一个有 100 位专科医生的大诊室。病人（token）进门先被分诊台（router）看一眼，然后只去最对口的 2 位医生那里 —— 这就是「总共有 100 位医生的知识，每次只花 2 位的力气」。问题在于分诊台是学出来的、而病人天然扎堆：感冒的一下子来了一半人，那两位医生排到门外，剩下 80 位医生闲坐着。治它有两条路。一条是给分诊台加个「别把人往一处送」的评分（aux loss），但这条评分有个毛病：对那些一个病人都没分到的医生，它的评分为零，所以推不动人家；而且分诊台已经均衡了它还在推，推过头反而把好不容易形成的专长搅乱。另一条是每个医生门口挂个牌子，忙的往下调、闲的往上调（负载反馈），牌子只要微调一点点就够了，而且病人一均衡它自己就不动了。'
            : 'A clinic with 100 specialists. A patient (token) is glanced at by the triage desk (router) and then visits only the 2 most relevant doctors — that is "the knowledge of 100 doctors, at the cost of 2". The trouble is that triage is learned while patients arrive in clumps: half of them show up with a cold, that queue spills into the corridor, and the other 80 doctors sit idle. There are two fixes. One is to give the desk a score that penalises sending everyone to the same place (aux loss) — but that score reads zero for a doctor nobody was sent to, so it cannot push them at all, and it keeps pushing even after triage is balanced, eventually scrambling the specialisation that had formed. The other is a small sign at each door, nudged down when busy and up when idle (load feedback): it only needs a tiny adjustment, and once patients are spread out the sign stops moving on its own.'
        }
        detail={
          <>
            <p>
              <strong>{c.d1a}</strong> {c.d1b}
            </p>
            <p>
              <strong>{c.d2a}</strong> {c.d2b}
              {(report.tokenSet.topicShare[0] ?? 0).toFixed(0)}
              {'%'}
              {c.d2c}
            </p>
            <p>
              <strong>{c.d3a}</strong> {c.d3b}
            </p>
          </>
        }
        warn={c.warn}
      />

      <Card title={c.scenarioTitle} hint={c.scenarioHint}>
        <div className="chip-row">
          {MOE_SCENARIOS.map((s) => (
            <button
              key={s.id}
              className={scenario === s.id ? 'btn primary' : 'btn'}
              onClick={() => applyScenario(s.id)}
            >
              {lang === 'zh' ? s.label : s.labelEn}
            </button>
          ))}
        </div>
      </Card>

      <Card title={c.configTitle}>
        <div className="controls">
          <Slider
            label={c.nExperts}
            value={cfg.nExperts}
            min={2}
            max={16}
            onChange={(v) => patch({ nExperts: v, topK: Math.min(cfg.topK, v) })}
            format={(v) => String(v)}
          />
          <Slider
            label={c.topK}
            value={cfg.topK}
            min={1}
            max={Math.min(4, cfg.nExperts)}
            onChange={(v) => patch({ topK: v })}
            format={(v) => String(v)}
          />
          <Slider
            label={c.nTokens}
            value={cfg.nTokens}
            min={64}
            max={384}
            step={16}
            onChange={(v) => patch({ nTokens: v })}
            format={(v) => String(v)}
          />
          <Slider
            label={c.nTopics}
            value={cfg.nTopics}
            min={2}
            max={64}
            onChange={(v) => patch({ nTopics: v })}
            format={(v) => String(v)}
            hint={c.nTopicsHint}
          />
          <Slider
            label={c.noise}
            value={cfg.noise}
            min={0.05}
            max={1.2}
            step={0.05}
            onChange={(v) => patch({ noise: v })}
            format={(v) => v.toFixed(2)}
            hint={c.noiseHint}
          />
          <Slider
            label={c.steps}
            value={cfg.steps}
            min={10}
            max={240}
            step={10}
            onChange={(v) => patch({ steps: v })}
            format={(v) => String(v)}
          />
          <Slider
            label={c.lr}
            value={cfg.lr}
            min={0.05}
            max={1}
            step={0.05}
            onChange={(v) => patch({ lr: v })}
            format={(v) => v.toFixed(2)}
          />
          <div className="control" style={{ minWidth: 220 }}>
            <label>
              <span>{c.balanceMode}</span>
            </label>
            <Segmented<BalanceMode>
              value={cfg.balanceMode}
              onChange={(v) => patch({ balanceMode: v })}
              options={[
                { value: 'none', label: c.modeNone },
                { value: 'aux', label: c.modeAux },
                { value: 'bias', label: c.modeBias },
              ]}
            />
            <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 4, lineHeight: 1.5 }}>
              {c.modeHint}
            </div>
          </div>
          <Slider
            label={c.balance}
            value={cfg.balanceWeight}
            min={0}
            max={cfg.balanceMode === 'aux' ? 16 : 8}
            step={0.25}
            onChange={(v) => patch({ balanceWeight: v })}
            format={(v) => v.toFixed(2)}
            hint={balanceHint}
          />
          <Slider
            label={c.capacity}
            value={cfg.capacityFactor}
            min={0}
            max={2.5}
            step={0.05}
            onChange={(v) => patch({ capacityFactor: v })}
            format={(v) => (v === 0 ? (lang === 'zh' ? '不限' : 'off') : v.toFixed(2))}
            hint={c.capacityHint}
          />
        </div>
      </Card>

      <Card title={c.heatTitle} hint={c.heatHint} exportName="07-moe-routing-heatmap">
        <div style={{ marginBottom: 10 }}>
          <div className="control" style={{ minWidth: 220 }}>
            <label>
              <span>{c.heatMode}</span>
            </label>
            <Segmented
              value={heatNorm}
              onChange={(v) => setHeatNorm(v)}
              options={[
                { value: 0, label: c.heatProb },
                { value: 1, label: c.heatNorm },
              ]}
            />
          </div>
        </div>
        <Heatmap
          matrix={heat.matrix}
          rowLabels={heat.labels}
          colLabels={colLabels}
          colorOf={(v) => heatColor(Math.pow(Math.max(0, Math.min(1, v)), 0.5))}
        />
        <HeatLegend min="0" max="1" colorOf={(v) => heatColor(Math.pow(v, 0.5))} />
        <div className="note" style={{ marginTop: 10 }}>{c.heatNote}</div>
        <div className="note" style={{ marginTop: 4 }}>{c.clarityNote}</div>
      </Card>

      <Card title={c.loadTitle} hint={c.loadHint} exportName="07-moe-expert-load">
        <div className="stats" style={{ marginBottom: 12 }}>
          <div className="stat">
            <div className="k">{c.loadNone}</div>
            <div className="v" style={{ color: 'var(--text-2)' }}>
              {`${lang === 'zh' ? '倾斜' : 'skew'} ${baseline.final.imbalance.toFixed(2)} · ${lang === 'zh' ? '死专家' : 'dead'} ${baseline.final.deadExperts}`}
            </div>
          </div>
          <div className="stat">
            <div className="k">{c.loadNow}</div>
            <div className="v" style={{ color: 'var(--accent)' }}>
              {`${lang === 'zh' ? '倾斜' : 'skew'} ${report.final.imbalance.toFixed(2)} · ${lang === 'zh' ? '死专家' : 'dead'} ${report.final.deadExperts}`}
            </div>
          </div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18 }}>
          <div>
            <div className="card-title" style={{ fontSize: 13, marginBottom: 6 }}>{c.loadNone}</div>
            <BarList items={loadItems(baseline.final.load)} format={(v) => v.toFixed(0)} />
          </div>
          <div>
            <div className="card-title" style={{ fontSize: 13, marginBottom: 6 }}>{c.loadNow}</div>
            <BarList items={loadItems(report.final.load)} format={(v) => v.toFixed(0)} />
          </div>
        </div>
      </Card>

      <Card title={c.gradTitle} hint={c.gradHint} exportName="07-moe-aux-gradient">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18 }}>
          <div>
            <div className="card-title" style={{ fontSize: 13, marginBottom: 6 }}>{c.gradLoad}</div>
            <BarList items={loadItems(report.final.load)} format={(v) => v.toFixed(0)} />
          </div>
          <div>
            <div className="card-title" style={{ fontSize: 13, marginBottom: 6 }}>{c.gradGrad}</div>
            <BarList
              items={report.final.auxGrad.map((v, e) => ({ label: `E${e}`, value: v }))}
              format={(v) => v.toFixed(4)}
            />
          </div>
        </div>
        <div className="note" style={{ marginTop: 10 }}>{c.gradNote}</div>
      </Card>

      <Card title={c.sweepTitle} hint={c.sweepHint} exportName="07-moe-balance-sweep">
        <LineChart
          lines={[
            {
              name: `${c.auxTag} · ${c.evennessLabel}`,
              color: AUX_COLOR,
              points: sweep.aux.map((p) => ({ x: p.w, y: 1 / Math.max(p.imbalance, 1e-6) })),
            },
            {
              name: `${c.biasTag} · ${c.evennessLabel}`,
              color: BIAS_COLOR,
              points: sweep.bias.map((p) => ({ x: p.w, y: 1 / Math.max(p.imbalance, 1e-6) })),
            },
            {
              name: `${c.auxTag} · ${c.entropyLabel}`,
              color: AUX_COLOR,
              dashed: true,
              points: sweep.aux.map((p) => ({ x: p.w, y: p.entropy })),
            },
            {
              name: `${c.biasTag} · ${c.entropyLabel}`,
              color: BIAS_COLOR,
              dashed: true,
              points: sweep.bias.map((p) => ({ x: p.w, y: p.entropy })),
            },
          ]}
          xLabel={c.balance}
          yLabel="0 — 1"
          yFormat={(v) => v.toFixed(2)}
          xFormat={(v) => v.toFixed(1)}
        />
        <div className="note" style={{ marginTop: 8 }}>{c.sweepNote}</div>
      </Card>

      <Card title={c.statTitle}>
        <Stats
          items={[
            { k: c.imbalance, v: report.final.imbalance.toFixed(2) + '×' },
            { k: c.dead, v: `${report.final.deadExperts} / ${cfg.nExperts}` },
            { k: c.entropy, v: report.final.entropy.toFixed(3) },
            { k: c.auxLoss, v: `${report.final.auxLoss.toFixed(2)} (${c.auxFloor} 1.00)` },
          ]}
        />
        <div style={{ marginTop: 12 }}>
          <Stats
            items={[
              { k: c.dropped, v: String(report.final.droppedCount) },
              { k: c.partial, v: String(report.final.partialDropped) },
              { k: c.purity, v: meanPurity.toFixed(2) },
              { k: c.biasNorm, v: `${biasNorm.toFixed(2)} (${c.biasUnit})` },
            ]}
          />
        </div>
      </Card>

      <Card title={c.expertMapTitle} hint={c.purityHint}>
        <div className="chip-row">
          {report.expertTopTopic.map((t, e) => (
            <span
              key={e}
              className="token"
              style={{
                borderColor: report.final.load[e] === 0 ? 'var(--border)' : 'var(--accent)',
                color: report.final.load[e] === 0 ? 'var(--text-3)' : 'var(--text)',
                opacity: report.final.load[e] === 0 ? 0.5 : 1,
              }}
              title={`E${e}: ${report.expertTopic[e].join(' / ')}`}
            >
              E{e} → {t < 0 ? (lang === 'zh' ? '死' : 'dead') : `T${t}`} · {report.expertPurity[e].toFixed(2)}
            </span>
          ))}
        </div>
      </Card>

      <Card title={c.budgetTitle} hint={c.budgetHint} exportName="07-moe-param-budget">
        <div className="chip-row" style={{ marginBottom: 12 }}>
          {MOE_ARCH_PRESETS.map((p) => (
            <button
              key={p.id}
              className={presetId === p.id ? 'btn primary' : 'btn'}
              onClick={() => setPresetId(p.id)}
            >
              {lang === 'zh' ? p.name : p.nameEn}
            </button>
          ))}
        </div>
        <Stats
          items={[
            { k: c.total, v: fmtParams(anatomy.a.totalParams) },
            { k: c.active, v: fmtParams(anatomy.a.activeParams) },
            {
              k: c.ratio,
              v: `${(anatomy.a.activeRatio * 100).toFixed(1)}%`,
            },
            {
              k: c.expertShare,
              v: `${(anatomy.a.expertShare * 100).toFixed(1)}%`,
            },
          ]}
        />
        {pub && (
          <div style={{ marginTop: 12 }}>
            <Stats
              items={[
                {
                  k: `${c.published} · ${lang === 'zh' ? '总' : 'total'}`,
                  v: `${pub.totalB} B`,
                },
                {
                  k: `${c.published} · ${lang === 'zh' ? '激活' : 'active'}`,
                  v: `${pub.activeB} B`,
                },
                {
                  k: `${c.calculated} · ${lang === 'zh' ? '总' : 'total'}`,
                  v: `${totalB.toFixed(2)} B`,
                },
                {
                  k: `${c.calculated} · ${lang === 'zh' ? '激活' : 'active'}`,
                  v: `${activeB.toFixed(2)} B`,
                },
              ]}
            />
          </div>
        )}
        <div className="note" style={{ marginTop: 10 }}>{c.budgetNote}</div>
      </Card>
    </div>
  )
}

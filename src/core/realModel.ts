/**
 * 真实模型推理（可选功能）
 *
 * 用 @huggingface/transformers 在浏览器里（WebGPU 优先，回退 WASM）加载一个 ONNX 小模型，
 * 让页面上的数字从"模拟"变成"真的从权重里算出来的"。
 *
 * ⚠️ 关于「真实注意力热力图」—— 做不到，这是实测结论，不是猜测。
 *   scripts/probe-attentions.mjs 在 Node 里用同一个库跑了 Xenova/gpt2，ONNX 的输出签名是：
 *       outputNames: logits  (+24 present.*)
 *       HAS ATTENTION OUTPUT? NO
 *       out.attentions = undefined
 *   decoder-only 模型的 ONNX 导出只吐 logits 和 present.*（KV cache），注意力概率矩阵
 *   在图内部就被 fuse 掉了，根本没有对外的输出节点。而 transformers.js 的 getAttentions()
 *   只识别 cross_/encoder_/decoder_attentions.*（whisper 那类 seq2seq 的用法），
 *   对 CausalLM 传 output_attentions: true 是空操作。
 *   → 所以注意力模块保持"确定性模拟"，并在界面上说清楚原因；
 *     真实权重用在**它真能覆盖的地方**：下一个 token 的概率分布、以及 KV cache 的实测形状。
 *
 * 设计原则：
 *   1. 动态 import —— 不打进主包，不用这个功能的人完全不会下载这部分代码
 *   2. 全程 try/catch，失败要给出人看得懂的原因（大概率是网络 / 镜像）
 *   3. 推理完全在本地，文本不会发到任何服务器
 *   4. 纯计算部分（解码标签、softmax top-k、KV 统计）抽成纯函数，可脱离浏览器做单测
 */
export type RealStatus = 'idle' | 'loading' | 'ready' | 'error'

/** 推理后端。名字必须和当前环境的 onnxruntime 对得上，见 pickDevice() 注释。 */
export type Device = 'webgpu' | 'wasm' | 'cpu'

export interface RealModelSpec {
  id: string
  /**
   * 必须和仓库里实际存在的文件对得上。
   * transformers.js 的映射：fp32→model.onnx, q8→model_quantized.onnx,
   * int8→model_int8.onnx, uint8→model_uint8.onnx, q4→model_q4.onnx, fp16→model_fp16.onnx
   * Xenova/gpt2 这种老仓库【没有】model_quantized.onnx，用 q8 会 404 —— 必须用 int8。
   */
  dtype: 'fp32' | 'fp16' | 'q8' | 'int8' | 'uint8' | 'q4'
  label: string
  labelEn: string
  /** 实测 content-length，不是估算 */
  size: string
  note?: string
  noteEn?: string
}

export const REAL_MODELS: RealModelSpec[] = [
  {
    id: 'onnx-community/SmolLM2-135M-Instruct-ONNX',
    dtype: 'q8',
    label: 'SmolLM2-135M · 30 层 × 9 头',
    labelEn: 'SmolLM2-135M · 30 layers × 9 heads',
    size: '129 MB',
    note: '2024 · 体积最小，加载最快，推荐先用它',
    noteEn: '2024 · smallest download, fastest to start — begin here',
  },
  {
    id: 'onnx-community/LFM2-350M-ONNX',
    dtype: 'q4',
    label: 'LFM2-350M · 16 块混合卷积+门控注意力',
    labelEn: 'LFM2-350M · 16 hybrid conv + gated-attention blocks',
    size: '280 MB',
    note: '2025 新架构（Liquid AI）：不用标准自注意力做主计算，是「注意力不是唯一出路」的活例子',
    noteEn: '2025 architecture (Liquid AI): standard self-attention is not the main compute — living proof attention is not the only way',
  },
  {
    id: 'onnx-community/Qwen3-0.6B-ONNX',
    dtype: 'q8',
    label: 'Qwen3-0.6B · 28 层 × 16Q/8KV 头 · 中文',
    labelEn: 'Qwen3-0.6B · 28 layers × 16Q/8KV heads · Chinese',
    size: '589 MB',
    note: '2025 · 唯一真正支持中文的选项；16 个 Query 头共享 8 个 KV 头，GQA 的现成教材',
    noteEn: '2025 · the only option that truly handles Chinese; 16 query heads share 8 KV heads — GQA on display',
  },
  {
    id: 'Xenova/gpt2',
    dtype: 'int8',
    label: 'GPT-2 (124M) · 12 层 × 12 头 · 2019 经典对照',
    labelEn: 'GPT-2 (124M) · 12 layers × 12 heads · 2019 classic',
    size: '268 MB',
    note: '留在列表里是为了对照：它把中文按 UTF-8 字节切分（12 个字 → 23 个 token），而上面的现代模型只要几个 —— 一眼看出 BPE 词表的进步',
    noteEn: 'Kept for contrast: it splits Chinese into UTF-8 bytes (12 chars → 23 tokens) while the modern models above need a handful — BPE progress at a glance',
  },
]

/** 为什么没有"真实注意力"，界面上要如实说明 */
export const ATTENTION_UNAVAILABLE = {
  zh: 'decoder-only 模型的 ONNX 导出只有 logits 和 present.*（KV cache）两类输出，注意力概率矩阵在计算图内部就被融合掉了，没有对外的输出节点；transformers.js 的 output_attentions 只对 whisper 那类 seq2seq 生效。所以浏览器里拿不到真实注意力权重 —— 这个结论可以用仓库里的 scripts/probe-attentions.mjs 复现。',
  en: 'The ONNX export of a decoder-only model exposes only logits and present.* (KV cache). The attention probability matrix is fused inside the graph with no output node, and transformers.js only honours output_attentions for whisper-style seq2seq models. So real attention weights are simply not reachable from the browser — reproduce it yourself with scripts/probe-attentions.mjs.',
}

export interface TopToken {
  id: number
  label: string
  logit: number
  p: number
}

export interface KVMeasurement {
  nLayers: number
  nKVHeads: number
  seqLen: number
  headDim: number
  /** K + V 全层元素总数 */
  elems: number
  /** 按 fp32 折算的字节数 */
  bytes: number
}

export interface RealNextToken {
  /** 输入 token 的显示标签，长度 === input_ids 长度 */
  tokens: string[]
  ids: number[]
  /** 面向展示的分组视图：连续的 `` 字节碎片已合并解码 */
  display: DisplayToken[]
  /**
   * 最后一个位置的完整 logits（长度 = 词表大小）。
   * 留着它，采样模块的温度 / top-k / top-p 就能直接作用在真实分布上，
   * 而不是只能看我们挑好的 top-k。
   */
  logits: number[]
  top: TopToken[]
  /** 完整分布的熵（bits），衡量模型这一步有多"犹豫" */
  entropy: number
  vocabSize: number
  kv: KVMeasurement | null
  device: Device
  ms: number
}

export interface RealTokenization {
  ids: number[]
  tokens: string[]
  /** 面向展示的分组视图：连续的 `` 字节碎片已合并解码 */
  display: DisplayToken[]
  vocabSize: number
}

interface Session {
  tokenizer: any
  model: any
  modelId: string
  device: Device
}

let session: Session | null = null
let status: RealStatus = 'idle'
let lastError = ''

export function getStatus(): RealStatus {
  return status
}
export function getError(): string {
  return lastError
}
export function getModelId(): string {
  return session?.modelId ?? ''
}
export function getDevice(): Device | '' {
  return session?.device ?? ''
}
export function isReady(): boolean {
  return status === 'ready' && session !== null
}

export function unload(): void {
  session = null
  status = 'idle'
  lastError = ''
}

/** 当前是否运行在 Node（集成测试环境），而不是浏览器 */
export function isNodeRuntime(): boolean {
  return (
    typeof window === 'undefined' &&
    typeof process !== 'undefined' &&
    !!(process as any)?.versions?.node
  )
}

/**
 * 决定推理后端。
 *
 * 坑：两个 onnxruntime 认的名字不一样 ——
 *   浏览器 onnxruntime-web  ：wasm / webgpu
 *   Node   onnxruntime-node ：cpu / dml / webgpu（传 'wasm' 会直接报
 *                             Unsupported device: "wasm". Should be one of: dml, webgpu, cpu.）
 * 集成测试跑在 Node 里，所以必须分环境返回，否则同一份代码只能在一边跑。
 */
export function pickDevice(): Device {
  if (isNodeRuntime()) return 'cpu'
  const gpu = typeof navigator !== 'undefined' ? (navigator as any).gpu : null
  return gpu ? 'webgpu' : 'wasm'
}

/**
 * 决定 WASM 后端用几个线程。
 *
 * 多线程 WASM 依赖 SharedArrayBuffer，而浏览器只在页面「跨源隔离」（crossOriginIsolated）
 * 时才提供它 —— 那要求服务器发 COOP/COEP 响应头。GitHub Pages 这类静态托管【不发】这两个头，
 * 所以线上必然拿不到 SharedArrayBuffer。不显式处理的话，onnxruntime-web 可能在初始化
 * 多线程 worker 时直接报错；锁成单线程后最坏结果只是"慢一些"，功能仍然可用。
 *
 * 抽成纯函数是为了能直接单测这个判定逻辑。
 */
export function pickThreads(hasSharedArrayBuffer: boolean, crossOriginIsolated: boolean, cores: number): number {
  if (!hasSharedArrayBuffer || !crossOriginIsolated) return 1
  return Math.max(1, Math.min(4, Math.floor(cores) || 1))
}

export function specOf(modelId: string): RealModelSpec | undefined {
  return REAL_MODELS.find((m) => m.id === modelId)
}

/**
 * 给全局 fetch 装一层「网络错误自动重试」。
 *
 * 为什么必须在这一层修：transformers.js 对元数据探测（get_file_metadata）失败会静默吞掉
 * 并把 exists:false memoize 进模块实例（浏览器实测复现：控制台
 * "Unable to fetch file metadata ... Failed to fetch" → tokenizer_class 报错，
 * 且之后点多少次重新加载都不发请求）。hf-mirror 偶发掐断连接（curl 实测多次 code=000），
 * 一次抖动 = 本次会话永久失败。在 fetch 层透明重试到成功，memoize 就永远不会缓存失败结果。
 *
 * 只重试「建立连接阶段」的失败；调用方主动 abort 不重试；响应体流式传输中途失败不归它管。
 */
export function installRetryingFetch(attempts = 3, timeoutMs = 20_000): void {
  const g = globalThis as any
  if (g.__hfRetryingFetchInstalled) return
  const orig = g.fetch.bind(g)
  g.fetch = async (input: any, init?: any) => {
    let lastErr: unknown
    for (let i = 0; i < attempts; i++) {
      const ctrl = new AbortController()
      const timer = setTimeout(() => ctrl.abort(), timeoutMs)
      try {
        // 调用方自己给了 signal 就尊重它，不加超时
        return await orig(input, init?.signal ? init : { ...init, signal: ctrl.signal })
      } catch (e) {
        lastErr = e
        if (init?.signal?.aborted) throw e
        await new Promise((r) => setTimeout(r, 600 * (i + 1)))
      } finally {
        clearTimeout(timer)
      }
    }
    throw lastErr
  }
  g.__hfRetryingFetchInstalled = true
}

export async function load(
  modelId: string,
  mirrorHost: string,
  onProgress?: (pct: number, file?: string) => void,
): Promise<void> {
  status = 'loading'
  lastError = ''
  try {
    const tf: any = await import('@huggingface/transformers')
    if (mirrorHost) {
      // 只换 host，绝对不要动 remotePathTemplate。
      // 它的默认值 '{model}/resolve/{revision}/' 会作用在【每一个】文件上，
      // 一旦改成 '.../onnx/'，根目录的 config.json / tokenizer_config.json 就会 404，
      // 表现为 "Cannot read properties of undefined (reading 'tokenizer_class')"。
      // 权重在 onnx/ 子目录这件事，transformers.js 自己会拼，不需要我们帮忙。
      tf.env.remoteHost = mirrorHost.endsWith('/') ? mirrorHost : mirrorHost + '/'
    }
    tf.env.allowRemoteModels = true
    tf.env.allowLocalModels = false

    // 网络抖动重试：必须在 from_pretrained 之前装好（见函数注释）
    installRetryingFetch()

    // 静态托管（GitHub Pages）拿不到 SharedArrayBuffer，必须锁单线程，否则多线程 WASM 会挂
    const wasmCfg = tf.env?.backends?.onnx?.wasm
    if (wasmCfg) {
      wasmCfg.numThreads = pickThreads(
        typeof SharedArrayBuffer !== 'undefined',
        (globalThis as any).crossOriginIsolated === true,
        typeof navigator !== 'undefined' ? ((navigator as any).hardwareConcurrency ?? 1) : 1,
      )
    }

    const progress = (p: any) => {
      if (!onProgress || !p || typeof p.progress !== 'number') return
      onProgress(Math.round(p.progress), p.file)
    }

    const device = pickDevice()
    const dtype = specOf(modelId)?.dtype ?? 'q8'

    const tokenizer = await tf.AutoTokenizer.from_pretrained(modelId, {
      progress_callback: progress,
    })
    const model = await tf.AutoModelForCausalLM.from_pretrained(modelId, {
      dtype,
      device,
      progress_callback: progress,
    })

    session = { tokenizer, model, modelId, device }
    status = 'ready'
  } catch (e: any) {
    session = null
    status = 'error'
    lastError = explainError(e)
    throw e
  }
}

/** 把库里的原始报错翻译成能指导下一步操作的话 */
export function explainError(e: any): string {
  const msg: string = e?.message ?? String(e ?? 'unknown error')
  if (/tokenizer_class|Cannot read properties of undefined/i.test(msg)) {
    return `配置文件没下下来（${msg}）。常见原因：① 镜像站对跨域请求有限制（hf-mirror 会剥掉带第三方 Referer 请求的 CORS 头，浏览器页面默认就带 Referer，所以 curl 通、浏览器挂）；② transformers.js 会把探测失败的结果缓存在页面会话里，改完镜像地址后【刷新页面】再重试才有效。`
  }
  if (/Could not locate file/i.test(msg)) {
    const m = msg.match(/onnx\/([\w.]+)/)
    return `仓库里没有这个权重文件${m ? `（${m[1]}）` : ''}，说明 dtype 和仓库实际文件名对不上。`
  }
  if (/Failed to fetch|NetworkError|CORS|ERR_|timeout/i.test(msg)) {
    return `网络请求失败（${msg}）。按可能性排查：① Windows 系统代理开着但代理软件没在运行（设置 → 网络 → 代理，实测本机常见 127.0.0.1:10809 类端口）——Chrome 会走死代理而 curl 直连，表现为"命令行通、浏览器挂"；② hf-mirror 对带第三方 Referer 的浏览器请求会剥掉 CORS 头（防盗链）；③ 页面会话内探测结果被缓存，【刷新页面】后再重试或换镜像地址。`
  }
  return msg
}

// ---------------------------------------------------------------------------
// 纯函数区：不依赖 transformers.js，可直接单测
// ---------------------------------------------------------------------------

/** 把 Tensor / 嵌套数组统一成普通数组 */
export function toArray(x: any): any {
  return x?.tolist ? x.tolist() : x
}

/**
 * 把单个 token 的解码结果变成适合上屏的标签。
 * 空串回退成 #id；纯空白用 ␣ 标出来（否则热力图上是一片看不见的空格）；
 * 换行 / 制表符换成可见记号。
 */
export function prettifyLabel(raw: string, id: number): string {
  if (raw === '') return `#${id}`
  const shown = raw.replace(/\n/g, '⏎').replace(/\r/g, '␍').replace(/\t/g, '⇥')
  if (shown.trim() === '') return '␣'.repeat(shown.length)
  return shown
}

/**
 * 由 input_ids 逐个解码出显示标签。
 *
 * 为什么不用 tokenizer.tokenize(text)：
 *   1. 它返回的是原始子词串，字节级 BPE 下是 "Ġcapital" / "æ³¨æĦıåĬĽ" 这种伪影，中文全是乱码；
 *      逐 id 解码得到的是 " capital" / "注意力"，可读性完全不同。
 *   2. 它不含 special token，长度可能比 input_ids 短。旧实现用 tokens.unshift('<s>') 补齐，
 *      等于假设"缺的那个一定在开头、一定是 BOS" —— GPT-2 成立，Qwen2.5（默认不加 BOS）不成立，
 *      一旦不成立整张热力图的标签会静默错位一行。
 *   逐 id 解码从源头保证 labels.length === ids.length，不需要任何假设。
 *
 * @param decodeOne 注入的单 id 解码函数，方便脱离 transformers.js 做单测
 */
export function buildTokenLabels(ids: number[], decodeOne: (id: number) => string): string[] {
  return ids.map((id) => {
    let s: string
    try {
      s = decodeOne(id) ?? ''
    } catch {
      s = ''
    }
    return prettifyLabel(s, id)
  })
}

/**
 * 标签长度与矩阵维度对齐。
 * 正常情况下两者相等；不相等说明模型行为超出预期，此时在【末尾】补占位，
 * 而不是在开头塞 BOS —— 后者会把所有标签静默移位，属于越帮越忙。
 */
export function alignLabels(labels: string[], n: number): string[] {
  if (n <= 0 || labels.length === n) return labels
  if (labels.length > n) return labels.slice(0, n)
  const out = labels.slice()
  while (out.length < n) out.push(`?${out.length}`)
  return out
}

/** 一个面向展示的 token 组：可能对应多个原始 id（字节碎片合并的情况） */
export interface DisplayToken {
  /** 组内所有原始 id，按顺序 */
  ids: number[]
  label: string
}

/**
 * 把连续的 `` 字节碎片合并起来整体解码。
 *
 * 背景：字节级 BPE（GPT-2 系）会把「注」切成 3 个 UTF-8 字节 token，
 * 逐 id 解码每个都是 U+FFFD（），一串碎片诚实但没法读。
 * 但把这些 id 【合在一起】传给 tokenizer.decode，字节序列拼回去就能解出「注」。
 *
 * 规则：
 *   - 找出连续的「单独解码为 」的 id 段；
 *   - 整段一起解码，若结果里没有  → 用整段结果作为一个 DisplayToken；
 *   - 若仍是 （说明真的是坏字节，比如半个字符）→ 逐 id 保留碎片，不造假。
 *
 * @param decode 接收 id 数组、返回解码字符串的函数（生产环境传 tokenizer.decode）
 */
export function mergeFragments(ids: number[], decode: (part: number[]) => string): DisplayToken[] {
  const out: DisplayToken[] = []
  let i = 0
  while (i < ids.length) {
    let one = ''
    try {
      one = decode([ids[i]]) ?? ''
    } catch {
      one = ''
    }
    if (one !== '\uFFFD') {
      out.push({ ids: [ids[i]], label: prettifyLabel(one, ids[i]) })
      i++
      continue
    }
    // 收集连续的碎片段
    let j = i
    while (j < ids.length) {
      let d = ''
      try {
        d = decode([ids[j]]) ?? ''
      } catch {
        d = ''
      }
      if (d !== '\uFFFD') break
      j++
    }
    let merged = ''
    try {
      merged = decode(ids.slice(i, j)) ?? ''
    } catch {
      merged = ''
    }
    if (merged && !merged.includes('\uFFFD')) {
      out.push({ ids: ids.slice(i, j), label: merged })
    } else {
      // 真是解不回来的坏字节，原样保留，不伪造可读文本
      for (let k = i; k < j; k++) out.push({ ids: [ids[k]], label: '\uFFFD' })
    }
    i = j
  }
  return out
}

/**
 * top 候选的显示标签。
 *
 * ⚠️ 与分词 chips 不同：候选是孤立的单个 id，没有「相邻碎片」可以合并，
 * 所以字节碎片在这里只能显示 #id（浏览器实测：中文 prompt 的 top 候选
 * 逐 id 解码全是 U+FFFD，整列 `` 没法分辨是这次跑出来的真 bug）。
 */
export function topTokenLabel(raw: string, id: number): string {
  if (raw === '\uFFFD') return `#${id}`
  return prettifyLabel(raw, id)
}

/** 数值稳定的 softmax + 取前 k 个。返回的 p 是在【完整词表】上归一化后的概率。 */
export function softmaxTop(logits: number[], k: number): { top: TopToken[]; entropy: number } {
  if (!logits.length) return { top: [], entropy: 0 }
  let max = -Infinity
  for (const v of logits) if (v > max) max = v
  let Z = 0
  const exps = new Float64Array(logits.length)
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - max)
    exps[i] = e
    Z += e
  }
  let entropy = 0
  for (let i = 0; i < exps.length; i++) {
    const p = exps[i] / Z
    if (p > 0) entropy -= p * Math.log2(p)
  }
  const idx = Array.from(logits.keys())
  // 只需要前 k 个，用部分选择比全排序省一个数量级（词表 5 万 ~ 15 万）
  const kk = Math.min(k, idx.length)
  for (let i = 0; i < kk; i++) {
    let best = i
    for (let j = i + 1; j < idx.length; j++) {
      if (logits[idx[j]] > logits[idx[best]]) best = j
    }
    const tmp = idx[i]
    idx[i] = idx[best]
    idx[best] = tmp
  }
  const top: TopToken[] = idx.slice(0, kk).map((id) => ({
    id,
    label: `#${id}`,
    logit: logits[id],
    p: exps[id] / Z,
  }))
  return { top, entropy }
}

/**
 * 从 present.* 输出的形状统计真实 KV cache 占用。
 * dims 形如 [batch, nKVHeads, seqLen, headDim]，K 和 V 各一份，所以层数 = 条目数 / 2。
 */
export function summarizeKV(dimsList: number[][], bytesPerElem = 4): KVMeasurement | null {
  const valid = dimsList.filter((d) => d && d.length === 4)
  if (!valid.length) return null
  let elems = 0
  for (const d of valid) elems += d[0] * d[1] * d[2] * d[3]
  const [, nKVHeads, seqLen, headDim] = valid[0]
  return {
    nLayers: Math.max(1, Math.round(valid.length / 2)),
    nKVHeads,
    seqLen,
    headDim,
    elems,
    bytes: elems * bytesPerElem,
  }
}

// ---------------------------------------------------------------------------
// 推理入口
// ---------------------------------------------------------------------------

/** 只跑分词器（不需要权重，几 MB 就够），用于分词模块对比真实词表 */
export async function runTokenize(text: string): Promise<RealTokenization> {
  if (!session) throw new Error('model not loaded')
  const { tokenizer } = session
  const encoded = await tokenizer(text)
  const ids: number[] = ((toArray(encoded.input_ids)?.[0] ?? []) as any[]).map(Number)
  const labels = alignLabels(
    buildTokenLabels(ids, (id) => safeDecode(tokenizer, [id])),
    ids.length,
  )
  return {
    ids,
    tokens: labels,
    display: mergeFragments(ids, (part) => safeDecode(tokenizer, part)),
    vocabSize: tokenizer.model?.vocab?.length ?? 0,
  }
}

/**
 * 跑一次真实前向，返回下一个 token 的真实概率分布 + KV cache 实测形状。
 * 这是真实权重在浏览器里【确实拿得到】的东西（见文件头注释）。
 */
export async function runNextToken(text: string, topK = 12): Promise<RealNextToken> {
  if (!session) throw new Error('model not loaded')
  const { tokenizer, model, device } = session

  const encoded = await tokenizer(text)
  const ids: number[] = ((toArray(encoded.input_ids)?.[0] ?? []) as any[]).map(Number)
  if (!ids.length) throw new Error('分词结果为空，换一段文本试试')

  const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now()
  const out: any = await model(encoded)
  const ms = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0

  if (!out?.logits) throw new Error('模型没有返回 logits')
  // logits 形状 [batch, seqLen, vocab]，我们只要最后一个位置
  const logitsAll = toArray(out.logits)[0]
  const last: number[] = (logitsAll[logitsAll.length - 1] as any[]).map(Number)

  const { top, entropy } = softmaxTop(last, topK)
  for (const item of top) {
    item.label = topTokenLabel(safeDecode(tokenizer, [item.id]), item.id)
  }

  // present.*.key / .value 的形状就是真实 KV cache 占用
  const dimsList: number[][] = []
  for (const key of Object.keys(out)) {
    if (/^present\./.test(key) && Array.isArray(out[key]?.dims)) {
      dimsList.push(out[key].dims.map(Number))
    }
  }

  return {
    tokens: alignLabels(
      buildTokenLabels(ids, (id) => safeDecode(tokenizer, [id])),
      ids.length,
    ),
    ids,
    display: mergeFragments(ids, (part) => safeDecode(tokenizer, part)),
    logits: last,
    top,
    entropy,
    vocabSize: last.length,
    kv: summarizeKV(dimsList),
    device,
    ms,
  }
}

function safeDecode(tokenizer: any, ids: number[]): string {
  try {
    return tokenizer.decode(ids) ?? ''
  } catch {
    return ''
  }
}

/**
 * 真实模型推理（可选功能）
 *
 * 默认页面用的是确定性模拟。这个模块提供"真的跑一个小模型"的能力：
 * 用 @huggingface/transformers 在浏览器里（WebGPU 优先，回退 WASM）加载一个 ONNX 小模型，
 * 取出 output_attentions，于是注意力热力图里显示的就不再是模拟，而是真实权重算出来的结果。
 *
 * 设计原则：
 *   1. 动态 import —— 不打进主包，不加载它的人完全不会下载这部分代码
 *   2. 全程 try/catch，失败要给出人看得懂的原因（大概率是网络 / 镜像）
 *   3. 推理完全在本地，文本不会发到任何服务器
 */
export type RealStatus = 'idle' | 'loading' | 'ready' | 'error'

export const REAL_MODELS = [
  { id: 'Xenova/gpt2', label: 'GPT-2 (124M) · 12 层 × 12 头', size: '≈ 90 MB' },
  { id: 'onnx-community/SmolLM2-135M-Instruct', label: 'SmolLM2 (135M) · 30 层 × 9 头', size: '≈ 100 MB' },
  { id: 'onnx-community/Qwen2.5-0.5B-Instruct', label: 'Qwen2.5 (0.5B) · 24 层 × 14 头 · 支持中文', size: '≈ 350 MB' },
] as const

export interface RealAttention {
  tokens: string[]
  nLayers: number
  nHeads: number
  /** weights[layer][head][i][j] */
  weights: number[][][][]
}

interface Session {
  tokenizer: any
  model: any
  modelId: string
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
export function isReady(): boolean {
  return status === 'ready' && session !== null
}

export function unload(): void {
  session = null
  status = 'idle'
  lastError = ''
}

/** 探测浏览器是否支持 WebGPU，决定推理后端 */
function pickDevice(): 'webgpu' | 'wasm' {
  const gpu = (navigator as any).gpu
  return gpu ? 'webgpu' : 'wasm'
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
      tf.env.remoteHost = mirrorHost
      tf.env.remotePathTemplate = '{model}/resolve/{revision}/onnx/'
    }
    tf.env.allowRemoteModels = true
    tf.env.allowLocalModels = false

    const progress = (p: any) => {
      if (!onProgress || !p || typeof p.progress !== 'number') return
      onProgress(Math.round(p.progress), p.file)
    }

    const tokenizer = await tf.AutoTokenizer.from_pretrained(modelId, {
      progress_callback: progress,
    })
    const model = await tf.AutoModelForCausalLM.from_pretrained(modelId, {
      dtype: 'q8',
      device: pickDevice(),
      progress_callback: progress,
    })

    session = { tokenizer, model, modelId }
    status = 'ready'
  } catch (e: any) {
    session = null
    status = 'error'
    lastError =
      e?.message ??
      'unknown error (most likely a network / CDN issue — try switching the mirror host)'
    throw e
  }
}

/** 跑一次前向，返回真实注意力权重 */
/** 把 Tensor / 嵌套数组统一成普通数组 */
function toArray(x: any): any {
  return x?.tolist ? x.tolist() : x
}

export async function runAttention(text: string): Promise<RealAttention> {
  if (!session) throw new Error('model not loaded')
  const { tokenizer, model } = session

  const encoded = tokenizer(text, { add_special_tokens: true })
  const forward: any = (model as any).forward ?? model
  const out: any = await forward(encoded.input_ids ?? encoded, { output_attentions: true })

  const atten = out.attentions ?? out.attn ?? null
  if (!atten) {
    throw new Error('this model does not expose attentions')
  }

  const weights: number[][][][] = atten.map((layerTensor: any) => {
    const l = toArray(layerTensor)
    const heads = l[0] // 取 batch 0
    return (heads as any[]).map((h: any) => h as number[][])
  })

  let tokens: string[]
  try {
    tokens = tokenizer.tokenize(text)
    if (!tokens.length) throw new Error('empty')
  } catch {
    const ids: number[] = (toArray(encoded.input_ids)?.[0] ?? []) as number[]
    tokens = ids.map((i) => `#${i}`)
  }
  // tokenize() 可能不含 special token，这里做一次长度对齐
  const n = weights[0]?.[0]?.length ?? tokens.length
  while (tokens.length < n) tokens.unshift('<s>')
  tokens = tokens.slice(0, n)

  return {
    tokens,
    nLayers: weights.length,
    nHeads: weights[0]?.length ?? 0,
    weights,
  }
}


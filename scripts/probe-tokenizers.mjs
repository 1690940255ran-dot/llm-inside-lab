/**
 * 轻量探测：验证「逐 id 解码」比 tokenize() + unshift('<s>') 更可靠。
 *
 * 只下 tokenizer（几 MB），不下权重，几秒就能跑完。
 * 重点看 ids.length 和 tokenize().length 是否相等 —— 不相等的模型上，
 * 旧实现的 unshift('<s>') 就会让热力图标签整体错位。
 */
const MIRROR = process.env.HF_HOST || 'https://hf-mirror.com/'
const tf = await import('@huggingface/transformers')
tf.env.remoteHost = MIRROR
tf.env.allowRemoteModels = true

const MODELS = [
  'Xenova/gpt2',
  'onnx-community/SmolLM2-135M-Instruct-ONNX',
  'onnx-community/Qwen2.5-0.5B-Instruct',
]
const TEXTS = ['The capital of France is', '注意力机制是大模型的核心']

for (const id of MODELS) {
  console.log(`\n${'='.repeat(72)}\n${id}`)
  let tok
  try {
    tok = await tf.AutoTokenizer.from_pretrained(id)
  } catch (e) {
    console.log('  LOAD FAILED:', e.message.slice(0, 140))
    continue
  }
  for (const text of TEXTS) {
    const enc = await tok(text)
    const ids = enc.input_ids.tolist()[0].map(Number)
    let pieces = []
    try {
      pieces = tok.tokenize(text)
    } catch (e) {
      pieces = ['<tokenize() threw>']
    }
    const decoded = ids.map((i) => tok.decode([i]))
    const mismatch = ids.length !== pieces.length
    console.log(`  text        : ${JSON.stringify(text)}`)
    console.log(`  ids.length  : ${ids.length}   tokenize().length: ${pieces.length}  ${mismatch ? '<-- MISMATCH，旧实现会错位' : '(一致)'}`)
    console.log(`  first id    : ${ids[0]} -> ${JSON.stringify(decoded[0])}`)
    console.log(`  per-id      : ${decoded.map((s) => JSON.stringify(s)).join(' ')}`)
    console.log(`  tokenize()  : ${pieces.map((s) => JSON.stringify(s)).join(' ')}`)
  }
}

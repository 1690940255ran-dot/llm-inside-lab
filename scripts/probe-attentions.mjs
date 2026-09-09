/**
 * 探测脚本：真实模型到底能不能吐出注意力权重？
 *
 * 为什么要有这个脚本：
 *   浏览器里 runAttention() 依赖 out.attentions，但 transformers.js 的 getAttentions()
 *   只识别 ONNX 输出名 cross_/encoder_/decoder_attentions.*（seq2seq 用法）。
 *   decoder-only 的 ONNX 导出通常只有 logits + present.*，根本没有注意力输出。
 *   与其在浏览器里猜，不如在 Node 里用同一个库跑一次，把 ONNX 的输出名单打印出来。
 *
 * 用法：node scripts/probe-attentions.mjs [modelId]
 */
const MIRROR = process.env.HF_HOST || 'https://hf-mirror.com/'
const MODEL = process.argv[2] || 'Xenova/gpt2'
const DTYPE = process.argv[3] || 'q8'

const tf = await import('@huggingface/transformers')

tf.env.remoteHost = MIRROR
// 注意：这里【不】设置 remotePathTemplate。默认值 "{model}/resolve/{revision}/"
// 对所有文件生效，transformers.js 自己知道权重在 onnx/ 子目录。
tf.env.allowRemoteModels = true

console.log('[env] remoteHost         =', tf.env.remoteHost)
console.log('[env] remotePathTemplate =', tf.env.remotePathTemplate)
console.log('[probe] model =', MODEL)

const t0 = Date.now()
const tokenizer = await tf.AutoTokenizer.from_pretrained(MODEL)
console.log(`[ok] tokenizer loaded in ${Date.now() - t0}ms`)

const model = await tf.AutoModelForCausalLM.from_pretrained(MODEL, { dtype: DTYPE })
console.log(`[ok] model loaded in ${Date.now() - t0}ms`)

// ---- 关键：ONNX session 的输入/输出名单 ----
const sessions = model.sessions ?? {}
for (const [name, sess] of Object.entries(sessions)) {
  console.log(`\n[session] ${name}`)
  console.log('  inputNames :', sess.inputNames?.join(', '))
  const outs = sess.outputNames ?? []
  // present.* 有几十个，折叠一下只看有没有 attention
  const folded = outs.filter((o) => !/^present/.test(o))
  console.log(`  outputNames: ${folded.join(', ')}  (+${outs.length - folded.length} present.*)`)
  console.log('  HAS ATTENTION OUTPUT?', outs.some((o) => /atten/i.test(o)) ? 'YES' : 'NO')
}

// ---- 跑一次前向，看 JS 层拿到什么 ----
const text = 'The capital of France is'
const encoded = await tokenizer(text)
const ids = encoded.input_ids.tolist()[0]
console.log('\n[tokens] ids =', ids)
console.log('[tokens] per-id decode =', ids.map((id) => JSON.stringify(tokenizer.decode([id]))).join(' '))
console.log('[tokens] tokenize()    =', JSON.stringify(tokenizer.tokenize(text)))

const out = await model(encoded, { output_attentions: true })
console.log('\n[forward] output keys =', Object.keys(out))
console.log('[forward] out.attentions =', out.attentions === undefined ? 'undefined' : typeof out.attentions)
if (out.logits) console.log('[forward] logits dims =', out.logits.dims)

// ---- 真实 next-token 分布（这个一定拿得到）----
if (out.logits) {
  const dims = out.logits.dims
  const V = dims[dims.length - 1]
  const flat = out.logits.tolist()[0].at(-1)
  const top = flat
    .map((v, i) => [i, v])
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
  const max = top[0][1]
  const exps = flat.map((v) => Math.exp(v - max))
  const Z = exps.reduce((a, b) => a + b, 0)
  console.log(`\n[logits] vocab=${V}, top-5 next tokens for ${JSON.stringify(text)}:`)
  for (const [i, v] of top) {
    console.log(`  ${JSON.stringify(tokenizer.decode([i]))}  logit=${v.toFixed(3)}  p=${(Math.exp(v - max) / Z * 100).toFixed(2)}%`)
  }
}

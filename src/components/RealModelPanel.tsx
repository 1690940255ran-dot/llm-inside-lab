/**
 * 真实模型面板：在浏览器里真跑一个小模型，展示【真实的】下一个 token 分布。
 *
 * 为什么这个面板在生成模块、而不在注意力模块：
 *   decoder-only 模型的 ONNX 导出只有 logits 和 present.*（KV cache）两类输出，
 *   注意力概率矩阵没有对外输出节点 —— 详见 core/realModel.ts 文件头与 scripts/probe-attentions.mjs。
 *   拿得到的是 logits 和 KV 形状，所以真实权重就用在这两处，不做做不到的承诺。
 *
 * 亮点：温度 / top-k / top-p 三个旋钮由父组件传进来，直接作用在真实 logits 上，
 *      也就是说采样机制那一套代码（core/sampling.ts，已单测）跑的是真模型的分布。
 */
import { useState } from 'react'
import { Card, Stats } from './Controls'
import {
  REAL_MODELS,
  getStatus,
  load,
  runNextToken,
  specOf,
  unload,
  explainError,
  type RealNextToken,
  type RealStatus,
} from '../core/realModel'
import { sampleNext, candidateCount, type SamplingConfig } from '../core/sampling'
import { useLang } from '../i18n'

const zh = {
  run: '跑一次真实前向',
  running: '推理中…',
  rerun: '重新推理',
  inputTokens: '真实分词结果',
  tokenNote: '逐 token 解码；连续的字节碎片已合并解码（悬停看原始 id）。字节级 BPE 的碎片合并前后对比，本身就是「为什么 GPT-2 处理中文贵」的活教材',
  distTitle: '真实 logits 的 top 候选',
  distNote: '概率在完整词表上归一化；灰掉的行是被你当前的 top-k / top-p 截断掉的',
  colRank: '#',
  colToken: 'token',
  colLogit: 'logit',
  colP: '原始 p',
  colFinal: '采样后 p',
  colState: '状态',
  cut: '被截断',
  chosen: '✓ 抽中',
  kept: '保留',
  statVocab: '词表大小',
  statEntropy: '分布熵',
  statDevice: '推理后端',
  statLatency: '前向耗时',
  statKept: '保留候选数',
  kvTitle: 'KV Cache 实测占用',
  kvNote:
    '这些数字直接读自模型输出的 present.*.key / .value 张量形状，不是估算。注意 KV 头数可能小于 Query 头数 —— 那就是 GQA 在省显存。',
  kvLayers: '层数',
  kvHeads: 'KV 头数',
  kvSeq: '序列长度',
  kvDim: '每头维度',
  kvBytes: 'KV 总占用 (fp32)',
  kvPerToken: '每 token 增量',
  attnWhy: '为什么没有真实注意力热力图',
  firstLoad: '首次加载要下完整权重，慢是正常的；之后走浏览器缓存会快很多。',
  cached: '（已缓存的模型会秒开）',
}

const en: typeof zh = {
  run: 'Run a real forward pass',
  running: 'Running…',
  rerun: 'Run again',
  inputTokens: 'Real tokenization',
  tokenNote: 'Decoded per token id; consecutive byte fragments are merged and decoded together (hover for raw ids). The before/after of fragment merging is itself a live lesson in why GPT-2 is expensive for Chinese',
  distTitle: 'Top candidates from real logits',
  distNote: 'Probabilities are normalised over the full vocabulary; greyed rows are cut by your current top-k / top-p',
  colRank: '#',
  colToken: 'token',
  colLogit: 'logit',
  colP: 'raw p',
  colFinal: 'p after sampling',
  colState: 'state',
  cut: 'cut',
  chosen: '✓ drawn',
  kept: 'kept',
  statVocab: 'vocab size',
  statEntropy: 'distribution entropy',
  statDevice: 'backend',
  statLatency: 'forward latency',
  statKept: 'kept candidates',
  kvTitle: 'Measured KV cache footprint',
  kvNote:
    'These numbers are read straight off the shapes of the present.*.key / .value tensors the model returns — not estimates. Note that KV heads may be fewer than query heads: that is GQA saving memory.',
  kvLayers: 'layers',
  kvHeads: 'KV heads',
  kvSeq: 'sequence length',
  kvDim: 'per-head dims',
  kvBytes: 'total KV (fp32)',
  kvPerToken: 'per extra token',
  attnWhy: 'Why there is no real attention heatmap',
  firstLoad: 'The first load downloads the full weights, so slowness is expected; later loads hit the browser cache.',
  cached: '(cached models open instantly)',
}

const DICT = { zh, en }

function fmtBytes(b: number): string {
  if (b < 1024) return `${b} B`
  if (b < 1024 ** 2) return `${(b / 1024).toFixed(1)} KB`
  return `${(b / 1024 ** 2).toFixed(2)} MB`
}

export function RealModelPanel(props: { text: string; cfg: SamplingConfig; seed: number }) {
  const { lang, t } = useLang()
  const c = DICT[lang]

  const [status, setStatus] = useState<RealStatus>(getStatus())
  const [progress, setProgress] = useState(0)
  const [progressFile, setProgressFile] = useState('')
  const [error, setError] = useState('')
  const [modelId, setModelId] = useState<string>(REAL_MODELS[0].id)
  const [mirror, setMirror] = useState('https://hf-mirror.com')
  const [data, setData] = useState<RealNextToken | null>(null)
  const [busy, setBusy] = useState(false)

  const spec = specOf(modelId)

  const handleLoad = async () => {
    setStatus('loading')
    setProgress(0)
    setError('')
    setData(null)
    try {
      await load(modelId, mirror, (p, f) => {
        setProgress(p)
        if (f) setProgressFile(f)
      })
      setStatus('ready')
    } catch (e: any) {
      setStatus('error')
      setError(explainError(e))
    }
  }

  const handleUnload = () => {
    unload()
    setData(null)
    setStatus('idle')
    setError('')
  }

  const handleRun = async () => {
    if (!props.text.trim()) return
    setBusy(true)
    setError('')
    try {
      setData(await runNextToken(props.text, 14))
    } catch (e: any) {
      setError(explainError(e))
    } finally {
      setBusy(false)
    }
  }

  // 把父组件的采样参数作用在【真实 logits】上，复用已单测的 core/sampling.ts
  const step = data ? sampleNext(data.logits, props.cfg, props.seed) : null

  return (
    <Card title={t('realSection')} hint={status === 'ready' ? t('realReady') : undefined}>
      <div className="note" style={{ marginBottom: 10 }}>
        {t('realIntro')}
      </div>

      <div className="controls">
        <div className="control" style={{ minWidth: 260 }}>
          <label>
            <span>{t('modelLabel')}</span>
          </label>
          <select
            value={modelId}
            disabled={status === 'loading' || status === 'ready'}
            onChange={(e) => setModelId(e.target.value)}
            style={{
              font: 'inherit',
              fontSize: 12.5,
              padding: '6px 8px',
              border: '1px solid var(--border-strong)',
              borderRadius: 6,
              background: 'var(--surface)',
              color: 'var(--text)',
            }}
          >
            {REAL_MODELS.map((m) => (
              <option key={m.id} value={m.id}>
                {(lang === 'zh' ? m.label : m.labelEn)} · {m.size}
              </option>
            ))}
          </select>
        </div>
        <div className="control" style={{ minWidth: 220 }}>
          <label>
            <span>{t('mirrorLabel')}</span>
          </label>
          <input
            type="text"
            value={mirror}
            disabled={status === 'loading' || status === 'ready'}
            onChange={(e) => setMirror(e.target.value)}
            style={{ font: 'inherit', fontSize: 12.5, padding: '6px 8px' }}
          />
        </div>
        {status !== 'ready' ? (
          <button className="btn primary" onClick={handleLoad} disabled={status === 'loading'}>
            {status === 'loading' ? `${t('loading')} ${progress}%` : t('loadModel')}
          </button>
        ) : (
          <>
            <button className="btn primary" onClick={handleRun} disabled={busy}>
              {busy ? c.running : data ? c.rerun : c.run}
            </button>
            <button className="btn" onClick={handleUnload}>
              {t('unload')}
            </button>
          </>
        )}
      </div>

      {spec && (
        <div className="note" style={{ marginTop: 8 }}>
          <code>{spec.id}</code> · dtype <code>{spec.dtype}</code> ·{' '}
          {lang === 'zh' ? spec.note : spec.noteEn}
        </div>
      )}

      {status === 'loading' && (
        <div className="note" style={{ marginTop: 8 }}>
          {c.firstLoad} {progressFile && <code>{progressFile}</code>}
        </div>
      )}

      <div className="note" style={{ marginTop: 8 }}>
        {t('realNote')}
        {'　'}
        {t('mirrorHint')}
        {c.cached}
      </div>

      {status === 'error' && (
        <div className="warn" style={{ marginTop: 10 }}>
          {t('realFailed')}：{error}
        </div>
      )}
      {status === 'ready' && error && (
        <div className="warn" style={{ marginTop: 10 }}>
          {error}
        </div>
      )}

      {data && step && (
        <>
          <div style={{ marginTop: 16 }}>
            <div className="card-title" style={{ marginBottom: 6 }}>
              <span>{c.inputTokens}</span>
              <span className="hint">{c.tokenNote}</span>
            </div>
            <div className="chip-row tight">
              {data.display.map((g, i) => (
                <span
                  key={i}
                  className="token"
                  title={g.ids.length > 1 ? `ids ${g.ids.join(', ')}（${g.ids.length} 个字节碎片合并解码）` : `id ${g.ids[0]}`}
                >
                  <span>{g.label}</span>
                </span>
              ))}
            </div>
          </div>

          <Stats
            items={[
              { k: c.statVocab, v: data.vocabSize.toLocaleString() },
              { k: c.statEntropy, v: `${data.entropy.toFixed(2)} bits` },
              { k: c.statKept, v: String(candidateCount(step)) },
              { k: c.statDevice, v: data.device.toUpperCase() },
              { k: c.statLatency, v: `${data.ms.toFixed(0)} ms` },
            ]}
          />

          <div style={{ marginTop: 16 }}>
            <div className="card-title" style={{ marginBottom: 6 }}>
              <span>{c.distTitle}</span>
              <span className="hint">{c.distNote}</span>
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table className="tbl">
                <thead>
                  <tr>
                    <th>{c.colRank}</th>
                    <th>{c.colToken}</th>
                    <th>{c.colLogit}</th>
                    <th>{c.colP}</th>
                    <th>{c.colFinal}</th>
                    <th>{c.colState}</th>
                  </tr>
                </thead>
                <tbody>
                  {data.top.map((tk, i) => {
                    const kept = step.kept[tk.id]
                    const isChosen = step.chosen === tk.id
                    return (
                      <tr
                        key={tk.id}
                        style={{
                          opacity: kept ? 1 : 0.4,
                          background: isChosen ? 'var(--accent-soft)' : undefined,
                        }}
                      >
                        <td>{i + 1}</td>
                        <td>
                          <code>{tk.label}</code>
                        </td>
                        <td>{tk.logit.toFixed(2)}</td>
                        <td>{(tk.p * 100).toFixed(2)}%</td>
                        <td>{((step.final[tk.id] ?? 0) * 100).toFixed(2)}%</td>
                        <td>{isChosen ? c.chosen : kept ? c.kept : c.cut}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {data.kv && (
            <div style={{ marginTop: 16 }}>
              <div className="card-title" style={{ marginBottom: 6 }}>
                <span>{c.kvTitle}</span>
              </div>
              <Stats
                items={[
                  { k: c.kvLayers, v: String(data.kv.nLayers) },
                  { k: c.kvHeads, v: String(data.kv.nKVHeads) },
                  { k: c.kvSeq, v: String(data.kv.seqLen) },
                  { k: c.kvDim, v: String(data.kv.headDim) },
                  { k: c.kvBytes, v: fmtBytes(data.kv.bytes) },
                  {
                    k: c.kvPerToken,
                    v: fmtBytes(Math.round(data.kv.bytes / Math.max(data.kv.seqLen, 1))),
                  },
                ]}
              />
              <div className="note" style={{ marginTop: 8 }}>
                {c.kvNote}
              </div>
            </div>
          )}
        </>
      )}
    </Card>
  )
}

/**
 * 模块一：分词（Tokenizer）
 *
 * 这是整个链路的入口。很多人对"大模型读不懂字符、只读数字"没有实感，
 * 所以这里把「文本 → 预分词单元 → 字符 → 反复合并 → token id」的全过程摊开给你看，
 * 并且提供一个逐步播放的合并动画。
 */
import { useEffect, useMemo, useState } from 'react'
import { Card, Segmented, Slider, Stats } from '../../components/Controls'
import { Principle } from '../../components/Principle'
import { TokenChips } from '../../components/Tokens'
import {
  computeStats,
  encode,
  encodeByChar,
  pretokenize,
  trainBPE,
  type EncodeStep,
} from '../../core/bpe'
import { DEFAULT_CORPUS, SAMPLE_TEXTS } from '../../core/corpus'
import { useLang, type Lang } from '../../i18n'

const zh = {
  h2: '① 分词：把文本切成模型认识的 token',
  lead1: '模型不认识汉字和字母，只认识整数编号。分词器干的活就是：把一段文本切成一个个',
  lead2: '子词单元',
  lead3:
    '，再查表换成 id。同一个词在不同上下文里可能被切成不同的 token，而中文往往「一字一 token」——这就是为什么中文更费 token。',
  principleTitle: 'BPE 到底在干什么？',
  p1: '「合并次数」就是一个',
  p1b: '旋钮',
  p1c: '：次数越多，词表越大，单个 token 越长，同样的句子需要的 token 越少（更省上下文、更快），但嵌入矩阵也越大、越容易过拟合。GPT-2 大约是 5 万词表、几万次合并。',
  p2: '注意观察：中文几乎学不出多字词，因为每个汉字在语料里都很常见但组合太分散；而英文能明显学出 the、ing、tion 这类子词。',
  warn: '本模块跑的是真正的 BPE 算法，合并表是从内置语料（约 2 KB）现场学出来的，不是预先写死的结果。语料很小，所以学出来的词表比真实模型小得多——这恰恰方便你看清每一个合并是怎么来的。',
  inputHint: '改一个字，下面的 token 会立刻重算',
  settings: '分词设置',
  mode: '分词方式',
  modeBpe: 'BPE 子词',
  modeChar: '字符级（对照）',
  merges: 'BPE 合并次数',
  mergesHint: '决定词表大小',
  useCustom: '换成自定义语料',
  useDefault: '用回内置语料',
  customNote: '粘贴你自己的语料，观察词表会怎么变（语料里反复出现的组合会被优先学成 token）',
  result: '分词结果',
  resultHint: '色块下方的小数字是 token id；虚线框表示这个字符没在训练语料里出现过（OOV）',
  tokenCount: 'token 数',
  charCount: '字符数',
  avgLen: '平均 token 长度',
  unit: '字',
  vocabSize: '词表大小',
  mergeCount: '实际合并数',
  preUnits: '预分词单元共',
  preUnits2: '个',
  animTitle: '合并过程动画',
  animHint: '看一个单元是怎么从散字符被粘成 token 的',
  mergeOf: '本次合并',
  priority: '优先级',
  initial: '初始状态：全是单字符',
  noMerge: '当前文本里没有发生任何合并 —— 试试输入更长的英文单词，或者把合并次数调高。',
  rulesTitle: '学到的合并规则（优先级从高到低）',
  rulesHint: '这就是 BPE 学到的全部知识：都是从语料频次里统计出来的',
  noRule: '合并次数为 0，词表里只有基础字符。',
}

const en: typeof zh = {
  h2: '① Tokenization: cutting text into tokens the model knows',
  lead1: 'A model understands neither Hanzi nor Latin letters, only integers. The tokenizer cuts text into ',
  lead2: 'subword units',
  lead3:
    ' and maps each to an id. The same word can split differently depending on context, and Chinese is often one token per character — which is exactly why Chinese burns more tokens.',
  principleTitle: 'What is BPE actually doing?',
  p1: 'The merge count is a single ',
  p1b: 'knob',
  p1c: ': more merges means a larger vocabulary, longer tokens, and fewer tokens per sentence (cheaper context, faster) — but also a bigger embedding matrix and more overfitting risk. GPT-2 sits at roughly 50k vocab and tens of thousands of merges.',
  p2: 'Watch this: Chinese barely learns multi-character words, because each character is common but the combinations are spread thin; English visibly learns subwords like the, ing, tion.',
  warn: 'This module runs a real BPE algorithm — the merge table is learned on the fly from the built-in corpus (about 2 KB), not hardcoded. The corpus is tiny, so the vocabulary is far smaller than a real model’s, which is exactly what makes every merge traceable.',
  inputHint: 'Change one character and every token below recomputes instantly',
  settings: 'Tokenizer settings',
  mode: 'Mode',
  modeBpe: 'BPE subword',
  modeChar: 'Character-level (baseline)',
  merges: 'BPE merge count',
  mergesHint: 'Determines vocabulary size',
  useCustom: 'Use my own corpus',
  useDefault: 'Back to built-in corpus',
  customNote:
    'Paste your own text and watch the vocabulary change — combinations that repeat in the corpus get learned first.',
  result: 'Tokenization result',
  resultHint: 'The small number under each chip is the token id; a dashed border means the character was never seen in the training corpus (OOV).',
  tokenCount: 'tokens',
  charCount: 'characters',
  avgLen: 'avg token length',
  unit: 'chars',
  vocabSize: 'vocabulary',
  mergeCount: 'merges learned',
  preUnits: 'pre-tokens:',
  preUnits2: '',
  animTitle: 'Merge animation',
  animHint: "Watch one unit get glued together from loose characters",
  mergeOf: 'merge',
  priority: 'priority',
  initial: 'initial state: all single characters',
  noMerge:
    'Nothing merged in the current text — try a longer English word, or raise the merge count.',
  rulesTitle: 'Learned merge rules (highest priority first)',
  rulesHint: 'This is everything BPE knows, and it is all counted from corpus frequency.',
  noRule: 'Merge count is 0 — the vocabulary is just the base characters.',
}

const DICT: Record<Lang, typeof zh> = { zh, en }

export function TokenizerModule() {
  const { lang, t } = useLang()
  const c = DICT[lang]

  const [text, setText] = useState(SAMPLE_TEXTS[0].text)
  const [mode, setMode] = useState<'bpe' | 'char'>('bpe')
  const [numMerges, setNumMerges] = useState(160)
  const [corpus, setCorpus] = useState(DEFAULT_CORPUS)
  const [useCustomCorpus, setUseCustomCorpus] = useState(false)

  const model = useMemo(
    () => trainBPE(useCustomCorpus ? corpus : DEFAULT_CORPUS, numMerges),
    [numMerges, corpus, useCustomCorpus],
  )

  const result = useMemo(() => encode(text, model), [text, model])
  const tokens = mode === 'bpe' ? result.tokens : encodeByChar(text)
  const stats = useMemo(() => computeStats(tokens, text), [tokens, text])
  const units = useMemo(() => pretokenize(text), [text])

  const animated = useMemo(() => {
    const idx = result.stepsByUnit.findIndex((s) => s.length > 1)
    return idx >= 0 ? result.stepsByUnit[idx] : null
  }, [result])

  const [stepIdx, setStepIdx] = useState(0)
  const [playing, setPlaying] = useState(false)

  useEffect(() => {
    setStepIdx(0)
  }, [animated])

  useEffect(() => {
    if (!playing || !animated) return
    const timer = setInterval(() => {
      setStepIdx((s) => {
        if (s >= animated.length - 1) {
          setPlaying(false)
          return s
        }
        return s + 1
      })
    }, 900)
    return () => clearInterval(timer)
  }, [playing, animated])

  const curStep: EncodeStep | undefined = animated?.[Math.min(stepIdx, animated.length - 1)]

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
        formula={`1. pre-tokenize: split text into non-crossing units (CJK runs / alnum runs / single punctuation)\n2. init: break every unit into single characters\n3. count: find the most frequent adjacent pair (a, b)\n4. merge: glue every (a, b) into ab, record it, go back to 3\n5. repeat N times → a ranked merge table\n\nTo encode new text: apply merges from highest to lowest priority.`}
        analogy={
          lang === 'zh'
            ? '像小时候玩的拼字游戏：一开始只有 26 个字母，老师让你把最常见的字母组合（比如 th、ing）粘成一块新积木，反复粘几百次，最后你就有一盒子大小不一的积木。写字时能用大积木就别用小积木，句子就变短了。'
            : 'Like a spelling game: you start with 26 letters, and the teacher makes you glue the most common pairs (th, ing) into bigger blocks, a few hundred times over. You end up with a box of blocks of all sizes. Write with the biggest block that fits and your sentence gets shorter.'
        }
        detail={
          <>
            <p>
              {c.p1}
              <strong>{c.p1b}</strong>
              {c.p1c}
            </p>
            <p>{c.p2}</p>
          </>
        }
        warn={c.warn}
      />

      <Card title={t('inputText')} hint={c.inputHint}>
        <textarea rows={3} value={text} onChange={(e) => setText(e.target.value)} />
        <div className="chip-row" style={{ marginTop: 10 }}>
          {SAMPLE_TEXTS.map((s) => (
            <button key={s.label} className="btn" onClick={() => setText(s.text)}>
              {lang === 'zh' ? s.label : s.labelEn}
            </button>
          ))}
          <button className="btn" onClick={() => setText('')}>
            {t('clear')}
          </button>
        </div>
      </Card>

      <Card title={c.settings}>
        <div className="controls">
          <div className="control" style={{ minWidth: 180 }}>
            <label>
              <span>{c.mode}</span>
            </label>
            <Segmented
              value={mode}
              onChange={setMode}
              options={[
                { value: 'bpe', label: c.modeBpe },
                { value: 'char', label: c.modeChar },
              ]}
            />
          </div>
          <Slider
            label={c.merges}
            value={numMerges}
            min={0}
            max={400}
            step={10}
            onChange={setNumMerges}
            format={(v) => (lang === 'zh' ? `${v} 次` : `${v}`)}
            hint={c.mergesHint}
          />
          <button className="btn" onClick={() => setUseCustomCorpus((v) => !v)}>
            {useCustomCorpus ? c.useDefault : c.useCustom}
          </button>
        </div>
        {useCustomCorpus && (
          <div style={{ marginTop: 12 }}>
            <div className="note" style={{ marginBottom: 6 }}>
              {c.customNote}
            </div>
            <textarea rows={5} value={corpus} onChange={(e) => setCorpus(e.target.value)} />
          </div>
        )}
      </Card>

      <Card title={c.result} hint={c.resultHint} exportName="01-tokenizer-result">
        <TokenChips tokens={tokens} />
        <div style={{ marginTop: 14 }}>
          <Stats
            items={[
              { k: c.tokenCount, v: String(stats.tokenCount) },
              { k: c.charCount, v: String(stats.charCount) },
              { k: c.avgLen, v: stats.avgTokenLength.toFixed(2) + ' ' + c.unit },
              { k: c.vocabSize, v: String(model.idToToken.length) },
              { k: c.mergeCount, v: String(model.numMerges) },
            ]}
          />
        </div>
        <div className="note" style={{ marginTop: 10 }}>
          {c.preUnits} {units.length} {c.preUnits2}
          {units.slice(0, 12).map((u, i) => (
            <code key={i} style={{ marginRight: 4 }}>
              {u === ' ' ? '␣' : u}
            </code>
          ))}
          {units.length > 12 ? ' …' : ''}
        </div>
      </Card>

      <Card title={c.animTitle} hint={c.animHint}>
        {animated ? (
          <>
            <div className="merge-stage">
              {(curStep?.symbols ?? []).map((s, i) => (
                <span key={i} className={`sym${s.length > 1 ? ' joined' : ''}`}>
                  {s === ' ' ? '␣' : s}
                  {s.length > 1 && <span className="mark">✓</span>}
                </span>
              ))}
            </div>
            <div className="controls" style={{ marginTop: 12 }}>
              <button className="btn primary" onClick={() => setPlaying((p) => !p)}>
                {playing ? t('pause') : t('play')}
              </button>
              <button
                className="btn"
                onClick={() => setStepIdx((s) => Math.max(0, s - 1))}
                disabled={stepIdx === 0}
              >
                {t('prevStep')}
              </button>
              <button
                className="btn"
                onClick={() => setStepIdx((s) => Math.min((animated?.length ?? 1) - 1, s + 1))}
                disabled={stepIdx >= animated.length - 1}
              >
                {t('nextStep')}
              </button>
              <span className="note">
                {stepIdx + 1} / {animated.length} {t('stepProgress')}
                {curStep?.pair
                  ? `　${c.mergeOf}：${curStep.pair[0]} + ${curStep.pair[1]}（${c.priority} #${curStep.rank}）`
                  : `　${c.initial}`}
              </span>
            </div>
          </>
        ) : (
          <div className="note">{c.noMerge}</div>
        )}
      </Card>

      <Card title={c.rulesTitle} hint={c.rulesHint}>
        <div className="chip-row">
          {model.merges.slice(0, 24).map((m) => (
            <span
              key={m.rank}
              className="token"
              style={{ background: 'var(--surface-2)', color: 'var(--text)' }}
            >
              <span>
                {m.a === ' ' ? '␣' : m.a} + {m.b === ' ' ? '␣' : m.b} → <b>{m.a + m.b}</b>
              </span>
              <span className="tid">×{m.count}</span>
            </span>
          ))}
          {model.merges.length === 0 && <span className="note">{c.noRule}</span>}
        </div>
      </Card>
    </div>
  )
}

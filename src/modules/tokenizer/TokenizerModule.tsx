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

export function TokenizerModule() {
  const [text, setText] = useState(SAMPLE_TEXTS[0].text)
  const [mode, setMode] = useState<'bpe' | 'char'>('bpe')
  const [numMerges, setNumMerges] = useState(160)
  const [corpus, setCorpus] = useState(DEFAULT_CORPUS)
  const [useCustomCorpus, setUseCustomCorpus] = useState(false)

  // 合并次数 / 语料变化时重新训练 BPE（几十毫秒，同步算完即可）
  const model = useMemo(
    () => trainBPE(useCustomCorpus ? corpus : DEFAULT_CORPUS, numMerges),
    [numMerges, corpus, useCustomCorpus],
  )

  const result = useMemo(() => encode(text, model), [text, model])
  const tokens = mode === 'bpe' ? result.tokens : encodeByChar(text)
  const stats = useMemo(() => computeStats(tokens, text), [tokens, text])
  const units = useMemo(() => pretokenize(text), [text])

  // —— 合并动画：挑第一个发生过合并的单元，回放它的每一步 ——
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
        <h2>① 分词：把文本切成模型认识的 token</h2>
        <div className="lead">
          模型不认识汉字和字母，只认识整数编号。分词器干的活就是：把一段文本切成一个个<strong>子词单元</strong>，再查表换成 id。
          同一个词在不同上下文里可能被切成不同的 token，而中文往往「一字一 token」——这就是为什么中文更费 token。
        </div>
      </div>

      <Principle
        title="BPE 到底在干什么？"
        formula={`1. 预分词：把文本切成互不跨越的单元（连续汉字 / 连续英文数字 / 单个标点）\n2. 初始化：每个单元拆成单个字符\n3. 统计：找出语料里出现次数最多的相邻符号对 (a, b)\n4. 合并：把所有 (a, b) 粘成新符号 ab，记进合并表，回到第 3 步\n5. 重复 N 次 → 得到一个「合并优先级表」\n\n编码新文本时：按优先级从高到低，把字符依次粘回去`}
        analogy="像小时候玩的拼字游戏：一开始只有 26 个字母，老师让你把最常见的字母组合（比如 th、ing）粘成一块新积木，反复粘几百次，最后你就有一盒子大小不一的积木。写字时能用大积木就别用小积木，句子就变短了。"
        detail={
          <>
            <p>
              「合并次数」就是一个<strong>旋钮</strong>：次数越多，词表越大，单个 token 越长，同样的句子需要的 token
              越少（更省上下文、更快），但嵌入矩阵也越大、越容易过拟合。GPT-2 大约是 5 万词表、几万次合并。
            </p>
            <p>
              注意观察：中文几乎学不出多字词，因为每个汉字在语料里都很常见但组合太分散；而英文能明显学出
              <code>the</code>、<code>ing</code>、<code>tion</code> 这类子词。
            </p>
          </>
        }
        warn="本模块跑的是真正的 BPE 算法，合并表是从内置语料（约 2 KB）现场学出来的，不是预先写死的结果。语料很小，所以学出来的词表比真实模型小得多——这恰恰方便你看清每一个合并是怎么来的。"
      />

      <Card title="输入文本" hint="改一个字，下面的 token 会立刻重算">
        <textarea rows={3} value={text} onChange={(e) => setText(e.target.value)} />
        <div className="chip-row" style={{ marginTop: 10 }}>
          {SAMPLE_TEXTS.map((s) => (
            <button key={s.label} className="btn" onClick={() => setText(s.text)}>
              {s.label}
            </button>
          ))}
          <button className="btn" onClick={() => setText('')}>
            清空
          </button>
        </div>
      </Card>

      <Card title="分词设置">
        <div className="controls">
          <div className="control" style={{ minWidth: 180 }}>
            <label>
              <span>分词方式</span>
            </label>
            <Segmented
              value={mode}
              onChange={setMode}
              options={[
                { value: 'bpe', label: 'BPE 子词' },
                { value: 'char', label: '字符级（对照）' },
              ]}
            />
          </div>
          <Slider
            label="BPE 合并次数"
            value={numMerges}
            min={0}
            max={400}
            step={10}
            onChange={setNumMerges}
            format={(v) => `${v} 次`}
            hint="决定词表大小"
          />
          <button className="btn" onClick={() => setUseCustomCorpus((v) => !v)}>
            {useCustomCorpus ? '用回内置语料' : '换成自定义语料'}
          </button>
        </div>
        {useCustomCorpus && (
          <div style={{ marginTop: 12 }}>
            <div className="note" style={{ marginBottom: 6 }}>
              粘贴你自己的语料，观察词表会怎么变（语料里反复出现的组合会被优先学成 token）
            </div>
            <textarea rows={5} value={corpus} onChange={(e) => setCorpus(e.target.value)} />
          </div>
        )}
      </Card>

      <Card
        title="分词结果"
        hint="色块下方的小数字是 token id；虚线框表示这个字符没在训练语料里出现过（OOV）"
      >
        <TokenChips tokens={tokens} />
        <div style={{ marginTop: 14 }}>
          <Stats
            items={[
              { k: 'token 数', v: String(stats.tokenCount) },
              { k: '字符数', v: String(stats.charCount) },
              { k: '平均 token 长度', v: stats.avgTokenLength.toFixed(2) + ' 字' },
              { k: '词表大小', v: String(model.idToToken.length) },
              { k: '实际合并数', v: String(model.numMerges) },
            ]}
          />
        </div>
        <div className="note" style={{ marginTop: 10 }}>
          预分词单元共 {units.length} 个：{units.slice(0, 12).map((u, i) => (
            <code key={i} style={{ marginRight: 4 }}>
              {u === ' ' ? '␣' : u}
            </code>
          ))}
          {units.length > 12 ? ' …' : ''}
        </div>
      </Card>

      <Card title="合并过程动画" hint="看一个单元是怎么从散字符被粘成 token 的">
        {animated ? (
          <>
            <div className="merge-stage">
              {(curStep?.symbols ?? []).map((s, i) => (
                <span
                  key={i}
                  className={`sym${s.length > 1 ? ' joined' : ''}`}
                >
                  {s === ' ' ? '␣' : s}
                  {s.length > 1 && <span className="mark">✓</span>}
                </span>
              ))}
            </div>
            <div className="controls" style={{ marginTop: 12 }}>
              <button className="btn primary" onClick={() => setPlaying((p) => !p)}>
                {playing ? '暂停' : '播放'}
              </button>
              <button
                className="btn"
                onClick={() => setStepIdx((s) => Math.max(0, s - 1))}
                disabled={stepIdx === 0}
              >
                上一步
              </button>
              <button
                className="btn"
                onClick={() => setStepIdx((s) => Math.min((animated?.length ?? 1) - 1, s + 1))}
                disabled={stepIdx >= animated.length - 1}
              >
                下一步
              </button>
              <span className="note">
                第 {stepIdx + 1} / {animated.length} 步
                {curStep?.pair ? `　本次合并：${curStep.pair[0]} + ${curStep.pair[1]}（优先级 #${curStep.rank}）` : '　初始状态：全是单字符'}
              </span>
            </div>
          </>
        ) : (
          <div className="note">当前文本里没有发生任何合并 —— 试试输入更长的英文单词，或者把合并次数调高。</div>
        )}
      </Card>

      <Card title="学到的合并规则（优先级从高到低）" hint="这就是 BPE 学到的全部知识：都是从语料频次里统计出来的">
        <div className="chip-row">
          {model.merges.slice(0, 24).map((m) => (
            <span key={m.rank} className="token" style={{ background: 'var(--surface-2)', color: 'var(--text)' }}>
              <span>
                {m.a === ' ' ? '␣' : m.a} + {m.b === ' ' ? '␣' : m.b} → <b>{m.a + m.b}</b>
              </span>
              <span className="tid">×{m.count}</span>
            </span>
          ))}
          {model.merges.length === 0 && <span className="note">合并次数为 0，词表里只有基础字符。</span>}
        </div>
      </Card>
    </div>
  )
}

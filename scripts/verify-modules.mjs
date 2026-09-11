/**
 * 站点深检：逐个模块切换 + 导出 PNG 并校验非空白 + 抓控制台报错。
 * 用法: node scripts/verify-modules.mjs [url]
 */
import WebSocket from 'file:///C:/Users/cj169/.workbuddy/binaries/node/workspace/node_modules/ws/index.js'
import { spawn } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

const URL_TARGET = process.argv[2] || 'https://1690940255ran-dot.github.io/llm-inside-lab/'
const PORT = 9700 + Math.floor(Math.random() * 200)
const OUT_DIR = path.join('tmp', 'verify-modules')
const CHROME = process.env.CHROME || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
const log = (...a) => console.log(`[${new Date().toISOString().slice(11, 19)}]`, ...a)
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true })

const profile = path.join(os.tmpdir(), `llm-viz-mod-${Date.now()}`)
const chrome = spawn(CHROME, [
  '--remote-debugging-port=' + PORT, '--no-first-run', '--no-default-browser-check',
  '--headless=new', '--disable-gpu', '--no-proxy-server', '--hide-scrollbars',
  '--user-data-dir=' + profile, '--window-size=1440,1000', '--no-sandbox', 'about:blank',
], { stdio: 'ignore' })

let wsBase = null
for (let i = 0; i < 60; i++) {
  await sleep(400)
  try {
    const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()
    const page = list.find((t) => t.type === 'page' && t.webSocketDebuggerUrl)
    if (page) { wsBase = page.webSocketDebuggerUrl; break }
  } catch {}
}
if (!wsBase) { chrome.kill(); throw new Error('no CDP') }
const ws = new WebSocket(wsBase, { maxPayload: 256 * 1024 * 1024 })
await new Promise((res, rej) => { ws.once('open', res); ws.once('error', rej) })

let id = 0
const pending = new Map()
const errs = []
ws.on('message', (raw) => {
  const m = JSON.parse(raw)
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); return }
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') {
    errs.push(m.params.args.map((a) => a.value ?? a.description ?? '').join(' ').slice(0, 200))
  }
  if (m.method === 'Runtime.exceptionThrown') errs.push('EXC ' + (m.params.exceptionDetails.text || '').slice(0, 200))
})
function send(method, params = {}) {
  return new Promise((res, rej) => {
    const mid = ++id
    pending.set(mid, (m) => (m.error ? rej(new Error(method + ': ' + JSON.stringify(m.error))) : res(m.result)))
    ws.send(JSON.stringify({ id: mid, method, params }))
    setTimeout(() => { if (pending.has(mid)) { pending.delete(mid); rej(new Error(method + ' timeout')) } }, 90000)
  })
}
const evalJS = async (expr, awaitPromise = false) => {
  const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise })
  if (r.exceptionDetails) throw new Error('eval: ' + JSON.stringify(r.exceptionDetails).slice(0, 300))
  return r.result.value
}
await send('Runtime.enable'); await send('Page.enable')
await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false })
await send('Page.navigate', { url: URL_TARGET })
await sleep(7000)

const clickNav = (i) => evalJS(`(() => {
  const bs = [...document.querySelectorAll('.nav-item')].filter(b=>b.textContent.trim())
  const b = bs[${i}]; if (!b) return null
  b.click(); return b.textContent.trim()
})()`)

const navLabels = await evalJS(`[...document.querySelectorAll('.nav-item')].map(b=>b.textContent.trim()).filter(Boolean)`)
log('nav items (' + navLabels.length + '):', JSON.stringify(navLabels))
if (navLabels.length < 10) log('!! 期望至少 10 个模块，实际 ' + navLabels.length)

const results = []
for (let i = 0; i < navLabels.length; i++) {
  const clicked = await clickNav(i)
  if (!clicked) continue
  await sleep(2400)
  const info = await evalJS(`(async () => {
    const btn = document.querySelector('.btn-export')
    const out = { i: ${i}, label: ${JSON.stringify(clicked)}, exportBtn: !!btn,
                  cards: document.querySelectorAll('.card').length,
                  svgs: document.querySelectorAll('svg').length,
                  textLen: document.body.innerText.length,
                  overflow: document.documentElement.scrollWidth > window.innerWidth + 1 }
    if (btn) {
      out.labelBefore = btn.textContent.trim()
      btn.click()
      await new Promise(r => setTimeout(r, 2500))
      out.labelAfter = btn.textContent.trim()
    }
    return out
  })()`, true)
  results.push(info)
  log(JSON.stringify(info))
}

/* ---- 新模块针对性断言 ---- */
const checks = []
const navIndex = (kw) => navLabels.findIndex((l) => l.includes(kw))

// ⑦ MoE：切到「专家多于簇」场景 → aux 梯度应出现精确的 0.0000；
//   再逐个切均衡模式，确认三种模式都能渲染不崩
if (navIndex('MoE') >= 0) {
  await clickNav(navIndex('MoE'))
  await sleep(2500)
  const moe = await evalJS(`(async () => {
    const out = {}
    const chips = [...document.querySelectorAll('.chip-row button')]
    const labels = chips.map(b=>b.textContent.trim())
    out.scenarios = labels.filter(t=>/^[①②③④⑤⑥]/.test(t))
    const starved = chips.find(b => b.textContent.includes('专家多于簇') || b.textContent.includes('more experts'))
    out.hasStarvedScenario = !!starved
    if (starved) { starved.click(); await new Promise(r=>setTimeout(r,2200)) }
    out.hasGradZero = document.body.innerText.includes('0.0000')

    // 均衡模式 segmented：找含"不治"/"untreated"的那一组
    const modeLabels = []
    const svgPerMode = []
    const found = { none: null, aux: null, bias: null }
    for (const bg of document.querySelectorAll('.segmented')) {
      const btns = [...bg.querySelectorAll('button')]
      const txt = btns.map(b=>b.textContent.trim())
      if (txt.some(t=>t==='不治'||t==='untreated'||t==='untreated'.toUpperCase())) {
        btns.forEach((b,i)=>{ if(i<3) found[i===0?'none':i===1?'aux':'bias']=b })
      }
    }
    for (const key of ['none','aux','bias']) {
      const b = found[key]
      if (!b) { modeLabels.push(key+':MISSING'); continue }
      b.click()
      await new Promise(r=>setTimeout(r,1400))
      modeLabels.push(key+':ok')
      svgPerMode.push(document.querySelectorAll('svg').length)
    }
    out.modeSwitch = modeLabels
    out.svgPerMode = svgPerMode
    // 扫描图（四条折线）必须真的画出来，且 stroke 被解析成实际颜色而不是空值
    const polys = [...document.querySelectorAll('svg polyline')]
    out.polylineCount = polys.length
    out.strokeResolved = polys.map(p => getComputedStyle(p).stroke).slice(0, 4)
    out.polylinePoints = polys.map(p => (p.getAttribute('points') || '').length).slice(0, 4)
    out.textLen = document.body.innerText.length
    out.overflow = document.documentElement.scrollWidth > window.innerWidth + 1
    return out
  })()`, true)
  checks.push({ module: 'MoE', ...moe })
  log('MoE check:', JSON.stringify(moe))

  // 这里原来有一段"明暗主题各跑一遍"的检查，已经删掉：
  // 站点是**单一浅色主题**（global.css 里没有任何 dark / data-theme / prefers-color-scheme），
  // 所以那个检查永远只会打印 foundThemeBtn:false —— 看着像"查过了"，
  // 其实一次都没生效。假的安全感比没有检查更糟，不如删掉。
}

// ⑧ 量化：直方图/对比表在，且切方案不报错
if (navIndex('量化') >= 0 || navIndex('quant') >= 0) {
  const qi = navIndex('量化') >= 0 ? navIndex('量化') : navIndex('quant')
  await clickNav(qi)
  await sleep(2500)
  const q = await evalJS(`(async () => {
    const out = { svgs: document.querySelectorAll('svg').length, tables: document.querySelectorAll('table').length }
    const segs = [...document.querySelectorAll('.segmented button')]
    out.segCount = segs.length
    if (segs[1]) { segs[1].click(); await new Promise(r=>setTimeout(r,1200)) }
    out.textLen = document.body.innerText.length
    out.hasSQNR = document.body.innerText.includes('SQNR')
    return out
  })()`, true)
  checks.push({ module: 'Quant', ...q })
  log('Quant check:', JSON.stringify(q))
}

// ⑨ 上下文外推：四种方法切换 + 相位图
if (navIndex('上下文') >= 0 || navIndex('Context') >= 0) {
  const ci = navIndex('上下文') >= 0 ? navIndex('上下文') : navIndex('Context')
  await clickNav(ci)
  await sleep(2500)
  const cx = await evalJS(`(async () => {
    const out = {}
    /**
     * 关键：页面里有好几组 .segmented，其中一组是**语言切换（中文 / English）**。
     * 早先这里无脑点遍所有 segmented button，结果把界面语言切成了英文，
     * 于是后面所有"按中文 label 读 DOM"的断言全部静默返回 null ——
     * 看起来像"检查过了"，其实什么都没查到。所以这里必须只取目标那一组。
     * 外推方法的名字本身永远是英文（none / linear interpolation / NTK-aware / YaRN），
     * 所以用它们来定位是语言无关的。
     */
    let target = null
    for (const g of document.querySelectorAll('.segmented')) {
      const txt = [...g.querySelectorAll('button')].map(b => b.textContent.trim())
      if (txt.includes('中文') || txt.includes('English')) continue
      if (txt.some(t => /RoPE|interpolation|YaRN|NTK|vanilla/i.test(t))) target = g
    }
    out.segLabels = target ? [...target.querySelectorAll('button')].map(b=>b.textContent.trim()) : []
    const marks = []
    if (target) {
      for (const b of target.querySelectorAll('button')) {
        b.click()
        await new Promise(r=>setTimeout(r,900))
        marks.push(document.querySelectorAll('svg').length)
      }
    }
    out.svgCountsPerMethod = marks
    // 语言没被切走（切走了后面所有中文断言都会失效）
    out.stillChinese = /[\\u4e00-\\u9fa5]/.test(document.body.innerText)
    out.textLen = document.body.innerText.length
    return out
  })()`, true)
  checks.push({ module: 'Context', ...cx })
  log('Context check:', JSON.stringify(cx))
  if (!cx.stillChinese) log('!! 界面语言被切成了非中文，后续断言会失效')
}

/**
 * ⑩ 迷你 GPT：这一条和别的不一样——它真的点「开始训练」，等训练跑完，
 * 再断言 loss 确实降下来了。别的模块只需要"能渲染"，这个模块必须"真算过"。
 *
 * 注意：轮询必须放在 Node 侧，不能塞进一次 Runtime.evaluate ——
 * 单次 evaluate 有 90 秒上限，而无 GPU 的 headless 里跑满 1200 步可能更久，
 * 塞进去会撞 "Runtime.evaluate timeout"（我们第一版就是这么挂的）。
 */
const trainIdx = navIndex('迷你') >= 0 ? navIndex('迷你') : navIndex('mini')
if (trainIdx >= 0) {
  await clickNav(trainIdx)
  await sleep(3000)

  /**
   * 每次只读一小段 DOM，快速返回。
   * label 一律给中英两份 —— 免得哪天语言被切走，断言又静默变成 null。
   */
  const readTrain = () =>
    evalJS(`(() => {
      const stat = (labels) => {
        const want = [].concat(labels)
        for (const s of document.querySelectorAll('.stats .stat')) {
          const k = s.querySelector('.k')
          if (k && want.indexOf(k.textContent.trim()) >= 0) {
            const v = s.querySelector('.v')
            return v ? v.textContent.trim() : null
          }
        }
        return null
      }
      return {
        step: stat(['当前步', 'step']),
        lossNow: stat(['当前 loss', 'current loss']),
        lossInit: stat(['初始 loss', 'initial loss']),
        lossBest: stat(['最优 loss', 'best loss']),
        samples: document.querySelectorAll('.sample-row').length,
        attnCompare: !!document.querySelector('.two-col'),
        stillChinese: /[\\u4e00-\\u9fa5]/.test(document.body.innerText),
      }
    })()`)

  const open = await evalJS(`(async () => {
    const out = {}
    const stat = (labels) => {
      const want = [].concat(labels)
      for (const s of document.querySelectorAll('.stats .stat')) {
        const k = s.querySelector('.k')
        if (k && want.indexOf(k.textContent.trim()) >= 0) {
          const v = s.querySelector('.v')
          return v ? v.textContent.trim() : null
        }
      }
      return null
    }
    out.langChinese = /[\\u4e00-\\u9fa5]/.test(document.body.innerText)
    out.exportButtons = document.querySelectorAll('.btn-export').length
    out.paramCount = stat(['参数量', 'parameters'])
    out.flopsPerToken = stat(['每 token FLOPs', 'FLOPs / token'])
    out.corpusPresets = document.querySelectorAll('.chip-row .chip').length
    out.hasUpload = !!document.querySelector('input[type=file]')
    out.hasTextarea = !!document.querySelector('textarea')
    out.hasBaselineNote = document.body.innerText.includes('随机猜测基线') ||
                          document.body.innerText.includes('random-guess baseline')
    const startBtn = [...document.querySelectorAll('.btn.primary')]
      .find(b => /开始训练|Start training/.test(b.textContent))
    out.foundStart = !!startBtn
    if (startBtn) startBtn.click()
    return out
  })()`, true)

  let last = { step: null, lossNow: null, lossInit: null, samples: 0, attnCompare: false }
  let done = false
  for (let k = 0; k < 75; k++) {
    await sleep(2000)
    const cur = await readTrain()
    if (cur) {
      last = { ...last, ...cur }
      if (cur.samples) last.samples = Math.max(last.samples, cur.samples)
      if (cur.attnCompare) last.attnCompare = true
      if (cur.step && /^\d+ \/ \d+$/.test(cur.step)) {
        const [a, b] = cur.step.split('/').map((x) => parseInt(x.trim(), 10))
        if (a >= b) { done = true; break }
      }
    }
  }

  const tr = { ...open, ...last, reachedEnd: done }
  const li = parseFloat(tr.lossInit)
  const ln = parseFloat(tr.lossNow)
  tr.lossDropped = Number.isFinite(li) && Number.isFinite(ln) && ln < li * 0.5
  checks.push({ module: 'Train', ...tr })
  log('Train check:', JSON.stringify(tr))
  if (!tr.foundStart) log('!! 没找到「开始训练」按钮')
  if (!tr.lossDropped) log('!! 训练后 loss 没有明显下降（init=' + tr.lossInit + ' now=' + tr.lossNow + '）')
  if (!tr.reachedEnd) log('!! 训练没有跑到终点，step=' + tr.step)
  if (tr.exportButtons < 4) log('!! 期望 4 个导出按钮，实际 ' + tr.exportButtons)
  if (!tr.attnCompare) log('!! 注意力训练前后对比（.two-col）没出现')
}

/* ---- ⑩ 英文模式漏翻扫描 ----
 * 起因：core/minigpt.ts 的 paramBreakdown() 曾经直接返回中文显示名，
 * 于是切到英文界面时「这笔账有多大」那张条形图会露出中文。
 * 这类 bug 只有真浏览器能抓到 —— 单元测试看到的是数据，不是 DOM 文本。
 * 排除掉本来就该是中文的地方：语料输入框、模型采样输出、热力图里的字符行标、
 * 语言切换按钮本身。
 */
if (navIndex('迷你') >= 0) {
  await clickNav(navIndex('迷你'))
  await sleep(2500)

  const en = await evalJS(`(async () => {
    const out = {}
    const seg = [...document.querySelectorAll('.segmented button')].find(b => b.textContent.trim() === 'English')
    out.foundSwitch = !!seg
    if (seg) { seg.click(); await new Promise(r => setTimeout(r, 2600)) }
    out.title = document.querySelector('.module-head h2') ? document.querySelector('.module-head h2').textContent : null

    const SKIP = ['textarea', '.sample-list', '.heat-wrap', '.segmented', '.nav']
    const bad = []
    document.querySelectorAll('body *').forEach((el) => {
      if (SKIP.some(s => el.closest(s))) return
      // 只取元素自己直接持有的文本节点，避免把容器的拼接文本算进来
      const own = [...el.childNodes].filter(n => n.nodeType === 3).map(n => n.textContent).join('')
      if (!/[\\u4e00-\\u9fa5]{2,}/.test(own)) return
      const cls = String(el.className).split(' ').filter(Boolean).join('.')
      bad.push(el.tagName.toLowerCase() + (cls ? '.' + cls : '') + '=' + own.trim().slice(0, 40))
    })
    out.untranslated = bad.slice(0, 8)

    const back = [...document.querySelectorAll('.segmented button')].find(b => b.textContent.trim() === '中文')
    if (back) { back.click(); await new Promise(r => setTimeout(r, 2200)) }
    out.titleZh = document.querySelector('.module-head h2') ? document.querySelector('.module-head h2').textContent : null
    return out
  })()`, true)

  checks.push({ module: 'Train-i18n', ...en })
  log('Train i18n check:', JSON.stringify(en))
  if (!en.foundSwitch) log('!! 没找到 English 语言切换按钮')
  if (!en.title || !/Train a mini GPT/i.test(en.title)) log('!! 英文模式下模块标题没翻译: ' + en.title)
  if (en.untranslated.length) log('!! 英文模式下有未翻译的界面文案: ' + en.untranslated.join(' | '))
  if (!en.titleZh || en.titleZh === en.title) log('!! 切回中文后标题没恢复: ' + en.titleZh)
}

fs.writeFileSync(path.join(OUT_DIR, 'modules.json'), JSON.stringify({ url: URL_TARGET, navLabels, results, checks, errs }, null, 2))
log('=== ERRORS (' + errs.length + ') ===')
errs.slice(0, 10).forEach((e) => log('  ' + e))
ws.close(); chrome.kill(); process.exit(0)

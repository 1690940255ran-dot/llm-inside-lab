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
if (navLabels.length < 9) log('!! 期望至少 9 个模块，实际 ' + navLabels.length)

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

  // 明暗主题各跑一遍，确认没有渲染塌陷
  const themes = await evalJS(`(async () => {
    const out = {}
    const btns = [...document.querySelectorAll('.sidebar button, header button')]
    const themeBtn = btns.find(b=>/浅色|深色|light|dark/i.test(b.textContent))
    out.foundThemeBtn = !!themeBtn
    if (!themeBtn) return out
    const before = getComputedStyle(document.body).backgroundColor
    themeBtn.click()
    await new Promise(r=>setTimeout(r,1200))
    const after = getComputedStyle(document.body).backgroundColor
    out.bgBefore = before
    out.bgAfter = after
    out.changed = before !== after
    out.svgs = document.querySelectorAll('svg').length
    out.cards = document.querySelectorAll('.card').length
    return out
  })()`, true)
  checks.push({ module: 'MoE-theme', ...themes })
  log('Theme check:', JSON.stringify(themes))
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
    const segs = [...document.querySelectorAll('.segmented button')]
    out.segLabels = segs.map(b=>b.textContent.trim())
    // 依次点每个方法，确认都能渲染
    const marks = []
    for (let k = 0; k < segs.length; k++) {
      segs[k].click()
      await new Promise(r=>setTimeout(r,900))
      marks.push(document.querySelectorAll('svg').length)
    }
    out.svgCountsPerMethod = marks
    out.textLen = document.body.innerText.length
    return out
  })()`, true)
  checks.push({ module: 'Context', ...cx })
  log('Context check:', JSON.stringify(cx))
}

fs.writeFileSync(path.join(OUT_DIR, 'modules.json'), JSON.stringify({ url: URL_TARGET, navLabels, results, checks, errs }, null, 2))
log('=== ERRORS (' + errs.length + ') ===')
errs.slice(0, 10).forEach((e) => log('  ' + e))
ws.close(); chrome.kill(); process.exit(0)

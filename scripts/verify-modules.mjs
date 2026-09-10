/**
 * 线上站点深检：逐模块切换 + 导出 PNG 并校验非空白。
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
await sleep(6000)

const navLabels = await evalJS(`[...document.querySelectorAll('nav button, [class*=sidebar] button, [class*=nav] button')].map(b=>b.textContent.trim()).filter(Boolean)`)
log('nav items:', JSON.stringify(navLabels.slice(0, 12)))

const results = []
for (let i = 0; i < Math.min(navLabels.length, 7); i++) {
  const clicked = await evalJS(`(() => {
    const bs = [...document.querySelectorAll('nav button, [class*=sidebar] button, [class*=nav] button')].filter(b=>b.textContent.trim())
    const b = bs[${i}]; if (!b) return null
    b.click(); return b.textContent.trim()
  })()`)
  if (!clicked) continue
  await sleep(2200)
  const info = await evalJS(`(async () => {
    const btn = document.querySelector('.btn-export')
    const out = { label: ${JSON.stringify(clicked)}, exportBtn: !!btn, btnText: btn ? btn.textContent.trim() : null,
                  cards: document.querySelectorAll('.card, [class*=card]').length,
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
fs.writeFileSync(path.join(OUT_DIR, 'modules.json'), JSON.stringify({ navLabels, results, errs }, null, 2))
log('=== ERRORS (' + errs.length + ') ===')
errs.slice(0, 10).forEach((e) => log('  ' + e))
ws.close(); chrome.kill(); process.exit(0)

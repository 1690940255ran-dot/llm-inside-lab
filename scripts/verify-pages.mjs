/**
 * 验证 GitHub Pages 上的线上站点：真实 Chrome 加载 + 截屏 + 控制台错误 + 关键元素检查。
 * 用法: node scripts/verify-pages.mjs [url]
 */
import WebSocket from 'file:///C:/Users/cj169/.workbuddy/binaries/node/workspace/node_modules/ws/index.js'
import { spawn } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

const URL_TARGET = process.argv[2] || 'https://1690940255ran-dot.github.io/llm-inside-lab/'
const PORT = 9500 + Math.floor(Math.random() * 300)
const OUT_DIR = process.env.OUT_DIR || path.join('tmp', 'verify')
const CHROME = process.env.CHROME || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'

const log = (...a) => console.log(`[${new Date().toISOString().slice(11, 19)}]`, ...a)
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true })

const profile = path.join(os.tmpdir(), `llm-viz-verify-${Date.now()}`)
const chrome = spawn(CHROME, [
  '--remote-debugging-port=' + PORT,
  '--no-first-run',
  '--no-default-browser-check',
  '--headless=new',
  '--disable-gpu',
  '--no-proxy-server',
  '--hide-scrollbars',
  '--user-data-dir=' + profile,
  '--window-size=1440,1000',
  '--no-sandbox',
  'about:blank',
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
if (!wsBase) { chrome.kill(); throw new Error('chrome did not expose CDP') }

const ws = new WebSocket(wsBase, { maxPayload: 128 * 1024 * 1024 })
await new Promise((res, rej) => { ws.once('open', res); ws.once('error', rej) })

let id = 0
const pending = new Map()
const consoleMsgs = []
const exceptions = []
const failedRequests = []

ws.on('message', (raw) => {
  const m = JSON.parse(raw)
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); return }
  if (m.method === 'Runtime.consoleAPICalled') {
    const t = m.params.type
    if (t === 'error' || t === 'warning') {
      consoleMsgs.push(`${t}: ` + m.params.args.map((a) => a.value ?? a.description ?? a.type).join(' ').slice(0, 300))
    }
  }
  if (m.method === 'Runtime.exceptionThrown') {
    exceptions.push((m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text || '').slice(0, 400))
  }
  if (m.method === 'Network.loadingFailed') {
    failedRequests.push(`${m.params.type} ${m.params.errorText} blocked=${m.params.blockedReason ?? '-'}`)
  }
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
  if (r.exceptionDetails) throw new Error('eval failed: ' + JSON.stringify(r.exceptionDetails).slice(0, 400))
  return r.result.value
}
const shot = async (file) => {
  const r = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true })
  fs.writeFileSync(file, Buffer.from(r.data, 'base64'))
  return fs.statSync(file).size
}

await send('Runtime.enable')
await send('Page.enable')
await send('Network.enable')
await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false })

log('navigating →', URL_TARGET)
await send('Page.navigate', { url: URL_TARGET })
await sleep(6000)

const report = {}
report.title = await evalJS('document.title')
report.rootChildren = await evalJS('document.getElementById("root")?.children.length ?? -1')
report.heroDemo = await evalJS('!!document.querySelector(".hero-demo-card")')
report.navButtons = await evalJS('document.querySelectorAll("nav button, .nav button, [class*=nav] button").length')
report.exportButtons = await evalJS('document.querySelectorAll(".btn-export").length')
report.modules = await evalJS('document.querySelectorAll("section").length')
report.bodyHeight = await evalJS('document.body.scrollHeight')
report.horizontalOverflow = await evalJS('document.documentElement.scrollWidth > window.innerWidth + 1')
report.langToggle = await evalJS('!!document.querySelector("[class*=lang], [class*=Lang]")')
report.textSample = await evalJS('document.body.innerText.slice(0, 200)')

log('screenshot (full page)')
report.shotSize = await shot(path.join(OUT_DIR, 'pages-full.png'))

log('switching to English…')
await evalJS(`(() => {
  const els = [...document.querySelectorAll('button, a')]
  const t = els.find(e => /^(EN|English)$/i.test(e.textContent.trim()))
  if (t) { t.click(); return true }
  return false
})()`)
await sleep(1500)
report.heroTitleEn = await evalJS(`document.querySelector('.hero-demo-card')?.innerText.slice(0,120) ?? ''`)
report.shotEn = await shot(path.join(OUT_DIR, 'pages-en.png'))

report.consoleErrors = consoleMsgs.filter((m) => m.startsWith('error'))
report.consoleWarnings = consoleMsgs.filter((m) => m.startsWith('warning'))
report.exceptions = exceptions
report.failedRequests = failedRequests

fs.writeFileSync(path.join(OUT_DIR, 'report.json'), JSON.stringify(report, null, 2))
log('=== REPORT ===')
log(JSON.stringify(report, null, 2))

ws.close()
chrome.kill()
process.exit(0)

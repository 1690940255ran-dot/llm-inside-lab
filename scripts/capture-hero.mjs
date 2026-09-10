/**
 * 抓首页动图的 N 帧存为 PNG。
 *
 * 用法（开发机上）：
 *   1) npx vite --port 5175 --strictPort  &    # 启动 dev server
 *   2) node scripts/capture-hero.mjs 5175    # 生成 frames/f000.png ~ f033.png
 *   3) python scripts/make-gif.py          # 拼成 docs/demo.gif
 *
 * 注意：本机系统代理是死的（127.0.0.1:10809 不通），
 * 必须给 chrome 传 --no-proxy-server，否则浏览器根本打不开 127.0.0.1:5175。
 *
 * 这两个脚本只是「为了 README 里那张 GIF」，开发流程里并不需要跑它们。
 */
import WebSocket from 'file:///C:/Users/cj169/.workbuddy/binaries/node/workspace/node_modules/ws/index.js'
import { spawn } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

const APP = process.argv[2] ? `http://127.0.0.1:${process.argv[2]}/` : 'http://127.0.0.1:5175/'
const PORT = 9333 + Math.floor(Math.random() * 200)
const FRAME_N = Number(process.env.FRAME_N ?? 34)
const FRAME_MS = Number(process.env.FRAME_MS ?? 180)
const FRAMES_DIR = process.env.FRAMES_DIR ?? path.join('tmp', 'frames')

const CHROME = process.env.CHROME ?? 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'

const log = (...a) => console.log(`[${new Date().toISOString().slice(11, 19)}]`, ...a)
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

fs.rmSync(FRAMES_DIR, { recursive: true, force: true })
fs.mkdirSync(FRAMES_DIR, { recursive: true })

const profile = path.join(os.tmpdir(), `llm-viz-hero-${Date.now()}`)
const chrome = spawn(
  CHROME,
  [
    '--remote-debugging-port=' + PORT,
    '--no-first-run',
    '--no-default-browser-check',
    '--headless=new',
    '--disable-gpu',
    '--no-proxy-server',
    '--hide-scrollbars',
    '--force-device-scale-factor=2',
    '--user-data-dir=' + profile,
    '--window-size=1280,900',
    '--no-sandbox',
    'about:blank',
  ],
  { stdio: 'ignore' },
)

let wsBase = null
for (let i = 0; i < 60; i++) {
  await sleep(400)
  try {
    const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()
    const page = list.find((t) => t.type === 'page' && t.webSocketDebuggerUrl)
    if (page) {
      wsBase = page.webSocketDebuggerUrl
      break
    }
  } catch {}
}
if (!wsBase) {
  chrome.kill()
  throw new Error('chrome did not expose CDP on port ' + PORT)
}

const ws = new WebSocket(wsBase, { maxPayload: 128 * 1024 * 1024 })
await new Promise((res, rej) => {
  ws.once('open', res)
  ws.once('error', rej)
})

let id = 0
const pending = new Map()
ws.on('message', (raw) => {
  const m = JSON.parse(raw)
  if (m.id && pending.has(m.id)) {
    pending.get(m.id)(m)
    pending.delete(m.id)
  }
})
function send(method, params = {}) {
  return new Promise((res, rej) => {
    const mid = ++id
    pending.set(mid, (m) => (m.error ? rej(new Error(method + ': ' + JSON.stringify(m.error))) : res(m.result)))
    ws.send(JSON.stringify({ id: mid, method, params }))
    setTimeout(() => {
      if (pending.has(mid)) {
        pending.delete(mid)
        rej(new Error(method + ' timeout'))
      }
    }, 60000)
  })
}
const evalJS = async (expr, awaitPromise = false) => {
  const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise })
  if (r.exceptionDetails) throw new Error('eval failed: ' + JSON.stringify(r.exceptionDetails).slice(0, 400))
  return r.result.value
}
const shot = async (file, clip) => {
  const r = await send('Page.captureScreenshot', { format: 'png', clip: { scale: 1, ...clip } })
  fs.writeFileSync(file, Buffer.from(r.data, 'base64'))
}

await send('Runtime.enable')
await send('Page.enable')
await send('Emulation.setDeviceMetricsOverride', {
  width: 1180,
  height: 900,
  deviceScaleFactor: 2,
  mobile: false,
})
await send('Page.navigate', { url: APP })
await sleep(3500)

const box = await evalJS(`(() => {
  const r = document.querySelector('.hero-demo-card').getBoundingClientRect()
  return { x: Math.round(r.x) - 6, y: Math.round(r.y) - 6, width: Math.round(r.width) + 12, height: Math.round(r.height) + 12 }
})()`)
log('hero box:', JSON.stringify(box))

for (let i = 0; i < FRAME_N; i++) {
  await shot(path.join(FRAMES_DIR, `f${String(i).padStart(3, '0')}.png`), box)
  await sleep(FRAME_MS)
}
log('frames captured:', FRAME_N, '→', FRAMES_DIR)

ws.close()
chrome.kill()
process.exit(0)
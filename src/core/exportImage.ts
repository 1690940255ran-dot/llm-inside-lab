/**
 * 零依赖的 DOM → PNG 导出
 *
 * 为什么不引 html-to-image：本项目一直坚持「不引 UI / 图形库」，
 * 而且这里要导出的东西其实很单纯 —— 一堆 div / table / svg + 一份已知的 CSS。
 *
 * 做法（也就是 html-to-image 的思路，但自己实现）：
 *   1. 把目标节点的克隆塞进一个 SVG 的 <foreignObject> 里
 *   2. 同一份 CSS 以 <style> 的形式内联进去，让类选择器继续生效
 *   3. :root 上定义的 CSS 变量在 foreignObject 里不会自动继承，
 *      所以先把它们抄成内联样式挂在最外层包裹节点上（var() 是继承的，抄一层就够）
 *   4. 祖先节点的 class 也按顺序套一层空 div —— 这样 `.card .hint` 这类
 *      后代选择器在克隆树里依然能匹配上
 *   5. 整个 SVG 转成 data: URL 喂给 <img>，画到 canvas，导出 PNG
 *
 * 注意：canvas 会被浏览器的尺寸上限约束（Chrome 单边 16384px），
 * 所以 scale 是按实际尺寸算出来的，不是写死的 2。
 */

/** 祖先包裹层要抹掉的盒模型属性，避免祖先的 padding/max-width 把内容压变形 */
const NEUTRALIZE =
  'display:block;margin:0;padding:0;border:0;max-width:none;min-width:0;width:auto;height:auto;max-height:none;position:static;top:auto;left:auto;transform:none;overflow:visible;'

/**
 * 收集页面里所有样式表的文本。
 *
 * 优先走 CSSOM（最准，能拿到浏览器解析后的规则）；读不到时（跨域样式表、
 * 或 `<link crossorigin>` 且服务端没给 CORS 头）退回 fetch 拿原始文本 ——
 * 同源 fetch 不受 crossorigin 属性影响，所以这条路基本都能成。
 *
 * 注意：绝不能"读不到就静默跳过"，那样会导出一张毫无样式的图，还看不出哪里错了。
 */
export async function collectCssText(doc: Document = document): Promise<string> {
  const parts: string[] = []
  for (const sheet of Array.from(doc.styleSheets)) {
    let rules: CSSRuleList | null = null
    try {
      rules = (sheet as CSSStyleSheet).cssRules
    } catch {
      rules = null // 跨域 / 不透明样式表
    }
    if (rules) {
      for (const rule of Array.from(rules)) parts.push(rule.cssText)
      continue
    }
    const href = sheet.href
    if (!href) continue
    try {
      const res = await fetch(href, { credentials: 'omit' })
      if (res.ok) parts.push(await res.text())
    } catch {
      /* 真拿不到，留给下面的兜底 */
    }
  }

  // 兜底：直接读 <style> 标签（有些环境里 styleSheets 还没同步好）
  if (parts.join('').trim().length < 32) {
    for (const el of Array.from(doc.querySelectorAll('style'))) parts.push(el.textContent ?? '')
  }
  return parts.join('\n')
}

const CUSTOM_PROP_DECL = /(^|[;{\s])(--[A-Za-z0-9_-]+)\s*:/g

/** 从 CSS 文本里挑出所有自定义属性名（比遍历 CSSOM 更通用，跨域场景也能用） */
export function customPropsFromText(css: string): string[] {
  const names = new Set<string>()
  let m: RegExpExecArray | null
  CUSTOM_PROP_DECL.lastIndex = 0
  while ((m = CUSTOM_PROP_DECL.exec(css))) names.add(m[2])
  return Array.from(names)
}

/**
 * 把 :root 上定义的 CSS 变量抄成一段内联样式。
 * 这些变量要挂到最外层包裹节点上，foreignObject 里的元素才能解析 var()。
 */
export function buildVarStyle(css: string, doc: Document = document): string {
  const cs = doc.defaultView
    ? doc.defaultView.getComputedStyle(doc.documentElement)
    : null
  if (!cs) return ''
  return customPropsFromText(css)
    .map((n) => {
      const v = cs.getPropertyValue(n).trim()
      return v ? `${n}:${v}` : ''
    })
    .filter(Boolean)
    .join(';')
}

/**
 * 按 canvas 尺寸上限算实际缩放比。
 * Chrome 的 canvas 单边上限是 16384px，总面积也有上限（约 2.7 亿像素），
 * 两个都得考虑，否则 toBlob 会拿到一张空白图。
 */
export function fitScale(
  width: number,
  height: number,
  wanted = 2,
  maxSide = 16384,
  maxArea = 2.6e8,
): number {
  if (!(width > 0) || !(height > 0)) return 1
  let s = Math.max(wanted, 0.1)
  s = Math.min(s, maxSide / width, maxSide / height)
  const area = width * height
  if (area > 0) s = Math.min(s, Math.sqrt(maxArea / area))
  return Math.max(s, 0.5)
}

/** 文件名净化：去掉路径分隔符和 Windows 保留字符 */
export function sanitizeFilename(name: string): string {
  const cleaned = name
    .replace(/[\\/:*?"<>|]/g, '-')
    .replace(/\s+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '')
  return cleaned || 'export'
}

/** 用祖先 class 套一层，让后代选择器继续命中 */
function wrapWithAncestors(node: HTMLElement, doc: Document): HTMLElement {
  const chain: string[] = []
  let p: HTMLElement | null = node.parentElement
  while (p && p !== doc.body) {
    const cls = typeof p.className === 'string' ? p.className.trim() : ''
    if (cls) chain.push(cls)
    p = p.parentElement
  }
  chain.reverse()

  let root: HTMLElement | null = null
  let inner: HTMLElement | null = null
  for (const cls of chain) {
    const el = doc.createElement('div')
    el.setAttribute('class', cls)
    el.setAttribute('style', NEUTRALIZE)
    if (inner) inner.appendChild(el)
    else root = el
    inner = el
  }
  return (root ?? inner) as HTMLElement
}

export interface ExportOptions {
  /** 导出倍率上限，默认 2（会被 canvas 尺寸上限往下压） */
  scale?: number
  /** 背景色，默认取 --surface，取不到就白色 */
  background?: string
  /** 文件名（不含扩展名） */
  filename?: string
}

export interface ExportResult {
  blob: Blob
  width: number
  height: number
  scale: number
}

/** 把 DOM 节点渲染成 PNG Blob */
export async function renderNodeToPng(
  node: HTMLElement,
  opts: ExportOptions = {},
): Promise<ExportResult> {
  const doc = node.ownerDocument
  const win = doc.defaultView
  if (!win) throw new Error('no window')

  const width = Math.max(node.scrollWidth, node.offsetWidth, 1)
  const height = Math.max(node.scrollHeight, node.offsetHeight, 1)
  const scale = fitScale(width, height, opts.scale ?? 2)

  const clone = node.cloneNode(true) as HTMLElement

  // 0) 先把样式表文本取出来（可能是异步的 fetch 兜底）
  const css = await collectCssText(doc)
  if (css.trim().length < 32) {
    throw new Error('拿不到页面样式（样式表跨域且不允许读取），导出会没有样式')
  }
  const varStyle = buildVarStyle(css, doc)

  // 1) 祖先外壳
  const shell = wrapWithAncestors(node, doc)
  const outer = doc.createElementNS('http://www.w3.org/1999/xhtml', 'div')
  const surface = doc.defaultView
    ? doc.defaultView.getComputedStyle(doc.documentElement).getPropertyValue('--surface').trim()
    : ''
  const bg = (opts.background ?? surface) || '#ffffff'
  outer.setAttribute(
    'style',
    [
      `width:${width}px`,
      `background:${bg}`,
      'margin:0',
      'padding:0',
      'box-sizing:border-box',
      varStyle,
    ]
      .filter(Boolean)
      .join(';'),
  )

  // 2) 内联同一份 CSS；:root 换成一个普通类，否则变量在 foreignObject 里不生效
  const styleEl = doc.createElementNS('http://www.w3.org/1999/xhtml', 'style')
  styleEl.textContent = css.replace(/:root\b/g, '.llm-export-root')
  outer.appendChild(styleEl)

  const inner = doc.createElementNS('http://www.w3.org/1999/xhtml', 'div')
  inner.setAttribute('class', 'llm-export-root')
  inner.setAttribute('style', `width:${width}px;background:${bg};margin:0;padding:0;`)
  inner.appendChild(clone)
  outer.appendChild(inner)

  if (shell) shell.appendChild(outer)
  const payload = shell ?? outer

  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" ` +
    `viewBox="0 0 ${width} ${height}">` +
    `<foreignObject x="0" y="0" width="100%" height="100%">` +
    new win.XMLSerializer().serializeToString(payload) +
    `</foreignObject></svg>`

  const url = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg)
  const img = new win.Image()
  img.decoding = 'sync'
  await new Promise<void>((resolve, reject) => {
    const timer = win.setTimeout(() => reject(new Error('图片渲染超时')), 15000)
    img.onload = () => {
      win.clearTimeout(timer)
      resolve()
    }
    img.onerror = () => {
      win.clearTimeout(timer)
      reject(new Error('浏览器拒绝渲染该 SVG（通常是内容里有外链资源）'))
    }
    img.src = url
  })

  const canvas = doc.createElement('canvas')
  canvas.width = Math.round(width * scale)
  canvas.height = Math.round(height * scale)
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('无法创建 canvas 上下文')
  ctx.fillStyle = bg
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'))
  if (!blob) throw new Error('canvas.toBlob 返回空（可能超出了浏览器画布上限）')
  return { blob, width: canvas.width, height: canvas.height, scale }
}

/** 触发浏览器下载 */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.style.display = 'none'
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  setTimeout(() => URL.revokeObjectURL(url), 2000)
}

/** 一步到位：节点 → PNG → 下载 */
export async function exportNodeAsPng(
  node: HTMLElement | null,
  opts: ExportOptions & { filename: string },
): Promise<ExportResult> {
  if (!node) throw new Error('导出目标不存在')
  const res = await renderNodeToPng(node, opts)
  downloadBlob(res.blob, `${sanitizeFilename(opts.filename)}.png`)
  return res
}

/** 表格 / 数组 → CSV 字符串（导出数据时用） */
export function toCsv(rows: (string | number)[][]): string {
  const esc = (v: string | number) => {
    const s = String(v)
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
  }
  return rows.map((r) => r.map(esc).join(',')).join('\r\n')
}

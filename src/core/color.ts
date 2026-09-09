/**
 * 配色：把数值映射成颜色
 * 站点统一浅色主题，热力图用「白 → 浅紫 → 深紫」的单色渐变，
 * 双向数值（比如余弦相似度 -1~1）用「蓝 → 白 → 红」的发散配色。
 */
function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

function mix(c1: number[], c2: number[], t: number): string {
  const r = Math.round(lerp(c1[0], c2[0], t))
  const g = Math.round(lerp(c1[1], c2[1], t))
  const b = Math.round(lerp(c1[2], c2[2], t))
  return `rgb(${r}, ${g}, ${b})`
}

/** 注意力权重 0~1 → 颜色 */
const HEAT_STOPS: { at: number; rgb: number[] }[] = [
  { at: 0.0, rgb: [252, 251, 249] },
  { at: 0.12, rgb: [238, 237, 254] },
  { at: 0.35, rgb: [175, 169, 236] },
  { at: 0.65, rgb: [127, 119, 221] },
  { at: 1.0, rgb: [60, 52, 137] },
]

export function heatColor(v: number): string {
  const t = Math.min(1, Math.max(0, v))
  for (let i = 0; i < HEAT_STOPS.length - 1; i++) {
    const a = HEAT_STOPS[i]
    const b = HEAT_STOPS[i + 1]
    if (t >= a.at && t <= b.at) {
      const k = b.at === a.at ? 0 : (t - a.at) / (b.at - a.at)
      return mix(a.rgb, b.rgb, k)
    }
  }
  return `rgb(${HEAT_STOPS[HEAT_STOPS.length - 1].rgb.join(',')})`
}

/** 双向数值 -1~1 → 蓝白红 */
export function divergingColor(v: number): string {
  const blue = [66, 122, 196]
  const white = [250, 250, 249]
  const red = [209, 78, 70]
  const t = Math.min(1, Math.max(-1, v))
  return t >= 0 ? mix(white, red, t) : mix(white, blue, -t)
}

/** 给第 i 个 token 分配一个稳定的分类色（用于 token 高亮） */
const TOKEN_COLORS = [
  ['#EEEDFE', '#534AB7'],
  ['#E1F5EE', '#0F6E56'],
  ['#FAEEDA', '#854F0B'],
  ['#FBEAF0', '#993556'],
  ['#E6F1FB', '#185FA5'],
  ['#EAF3DE', '#3B6D11'],
  ['#F1EFE8', '#5F5E5A'],
  ['#FCEBEB', '#A32D2D'],
]

export function tokenColor(i: number): { bg: string; fg: string } {
  const [bg, fg] = TOKEN_COLORS[i % TOKEN_COLORS.length]
  return { bg, fg }
}

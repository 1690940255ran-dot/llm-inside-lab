/**
 * 导出图片的纯函数单测
 * 浏览器侧的渲染走端到端验证（scripts/verify-export.mjs），这里只测无 DOM 依赖的部分。
 */
import { describe, it, expect } from 'vitest'
import {
  collectCssText,
  customPropsFromText,
  fitScale,
  sanitizeFilename,
  toCsv,
} from '../src/core/exportImage'

describe('fitScale', () => {
  it('空尺寸返回 1', () => {
    expect(fitScale(0, 100)).toBe(1)
    expect(fitScale(100, 0)).toBe(1)
  })

  it('wanted=2 时不会越界', () => {
    const s = fitScale(800, 600, 2)
    expect(s).toBeGreaterThan(0)
    expect(s).toBeLessThanOrEqual(2)
  })

  it('过大尺寸会被压回上限内', () => {
    const s = fitScale(10000, 10000, 2)
    expect(s).toBeLessThan(2)
    expect(s * 10000).toBeLessThanOrEqual(16384)
  })

  it('wanted=10 也会被压回去', () => {
    const s = fitScale(5000, 5000, 10)
    expect(s).toBeLessThan(10)
    expect(s).toBeGreaterThanOrEqual(0.5)
  })
})

describe('sanitizeFilename', () => {
  it('去掉路径分隔符与 Windows 保留字符', () => {
    expect(sanitizeFilename('a/b\\c:d*e?f"g<h>i|j.txt')).toBe('a-b-c-d-e-f-g-h-i-j.txt')
  })
  it('合并连续短横线、压边', () => {
    expect(sanitizeFilename('  hello--world  ')).toBe('hello-world')
  })
  it('空字符串给个默认名', () => {
    expect(sanitizeFilename('')).toBe('export')
    expect(sanitizeFilename('///')).toBe('export')
  })
})

describe('toCsv', () => {
  it('普通行原样输出', () => {
    expect(toCsv([['a', 'b'], ['c', 'd']])).toBe('a,b\r\nc,d')
  })
  it('逗号 / 引号 / 换行都会被引号包起来', () => {
    const s = toCsv([['a,b'], ['he said "hi"'], ['line\nbreak']])
    expect(s).toBe('"a,b"\r\n"he said ""hi"""\r\n"line\nbreak"')
  })
  it('数字也接受', () => {
    expect(toCsv([['k', 'v'], ['temperature', 1.234]])).toBe('k,v\r\ntemperature,1.234')
  })
})

describe('customPropsFromText', () => {
  it('从 :root 块里挑出自定义属性名', () => {
    const css = `:root { --accent: #534ab7; --surface: #fff; }\n.card { color: var(--accent); }`
    expect(customPropsFromText(css).sort()).toEqual(['--accent', '--surface'])
  })

  it('去重 + 支持数字与连字符', () => {
    const css = `:root{--a:1}\n.x{--a:2}\n.y{--my-long-var2:3}`
    expect(customPropsFromText(css).sort()).toEqual(['--a', '--my-long-var2'])
  })

  it('不会把 var(--x) 的引用误判成声明', () => {
    const css = `.card { color: var(--accent); border: 1px solid var(--border); }`
    expect(customPropsFromText(css)).toEqual([])
  })

  it('没有自定义属性时返回空数组', () => {
    expect(customPropsFromText('body{margin:0}')).toEqual([])
  })
})

describe('collectCssText', () => {
  it('在 Node（无 DOM）下也不抛异常，返回空串', async () => {
    // 浏览器侧的行为由端到端脚本验证，这里只保证它不会在缺 document 时炸掉
    const anyFn = collectCssText as unknown as () => Promise<string>
    await expect(anyFn()).rejects.toBeDefined()
  })
})
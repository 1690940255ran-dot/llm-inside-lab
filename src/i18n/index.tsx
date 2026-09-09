/**
 * 极简 i18n：一个 Context + 一个 t() 函数，不引任何库
 * 模块自己的长文案就近放在模块里（用 useLang 拿 lang 后取本地字典），
 * 跨模块复用的短文案走这里的 t()
 */
import { createContext, useContext, useMemo, useState, type ReactNode } from 'react'
import { COMMON, type CommonKey, type Lang } from './common'

export type { Lang }

interface LangValue {
  lang: Lang
  setLang: (l: Lang) => void
  t: (key: CommonKey) => string
}

const LangCtx = createContext<LangValue>({
  lang: 'zh',
  setLang: () => {},
  t: (k) => COMMON.zh[k],
})

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Lang>('zh')
  const value = useMemo<LangValue>(
    () => ({
      lang,
      setLang,
      t: (key) => COMMON[lang][key] ?? COMMON.zh[key],
    }),
    [lang],
  )
  return <LangCtx.Provider value={value}>{children}</LangCtx.Provider>
}

export function useLang(): LangValue {
  return useContext(LangCtx)
}

/** 模块内字典的取用助手：传入 { zh: {...}, en: {...} }，返回当前语言的那一份 */
export function pick<T>(dict: Record<Lang, T>, lang: Lang): T {
  return dict[lang]
}

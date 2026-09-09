/**
 * 原理说明卡：每个模块都配「数学在算什么」+「一句话类比」
 * 默认展开一条精简版，点开能看到公式和更完整的解释，
 * 避免一上来就是一堵公式墙把人劝退。
 */
import type { ReactNode } from 'react'
import { useLang } from '../i18n'

export function Principle(props: {
  title?: string
  formula?: string
  analogy: string
  detail?: ReactNode
  warn?: string
}) {
  const { t } = useLang()
  return (
    <details className="principle" open>
      <summary>{props.title ?? t('principleTitle')}</summary>
      <div className="principle-body">
        {props.formula && (
          <>
            <h4>{t('formula')}</h4>
            <div className="formula">{props.formula}</div>
          </>
        )}
        <h4>{t('analogy')}</h4>
        <div className="analogy">{props.analogy}</div>
        {props.detail && (
          <>
            <h4>{t('moreDetails')}</h4>
            <div>{props.detail}</div>
          </>
        )}
        {props.warn && <div className="warn" style={{ marginTop: 10 }}>{props.warn}</div>}
      </div>
    </details>
  )
}

import { useEffect, useState } from 'react'
import { MODULES } from './modules/registry'
import { useLang } from './i18n'

function Home() {
  const { lang, t } = useLang()
  return (
    <div>
      <div className="module-head">
        <h2>{t('homeTitle')}</h2>
        <div className="lead">{t('homeLead')}</div>
      </div>

      <div className="card">
        <div className="card-title">{t('homeHonestTitle')}</div>
        <p style={{ color: 'var(--text-2)', marginBottom: 8 }}>
          {t('homeHonest1a')}
          <strong>{t('homeHonest1b')}</strong>
          {t('homeHonest1c')}
          <strong>{t('homeHonest1d')}</strong>
          {t('homeHonest1e')}
        </p>
        <p style={{ color: 'var(--text-2)', marginBottom: 0 }}>
          {t('homeHonest2a')}
          <strong>{t('homeHonest2b')}</strong>
          {t('homeHonest2c')}
        </p>
      </div>

      <div className="hero-grid">
        {MODULES.map((m) => (
          <button
            key={m.id}
            className={`hero-card${m.status === 'soon' ? ' soon' : ''}`}
            disabled={m.status === 'soon'}
            onClick={() => window.dispatchEvent(new CustomEvent('goto', { detail: m.id }))}
          >
            <h3>
              {lang === 'zh' ? m.name : m.nameEn}
              {m.status === 'soon' && <span className="badge-todo">{t('planBadge')}</span>}
            </h3>
            <p>{lang === 'zh' ? m.blurb : m.blurbEn}</p>
          </button>
        ))}
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <div className="card-title">{t('homeOrderTitle')}</div>
        <p style={{ color: 'var(--text-2)', marginBottom: 0 }}>{t('homeOrder')}</p>
      </div>
    </div>
  )
}

export default function App() {
  const [active, setActive] = useState<string | null>(null)
  const { lang, setLang, t } = useLang()

  // 首页的模块卡片通过自定义事件跳转，避免为了这点功能引入路由库
  useEffect(() => {
    const handler = (e: Event) => setActive((e as CustomEvent<string>).detail)
    window.addEventListener('goto', handler)
    return () => window.removeEventListener('goto', handler)
  }, [])

  const current = MODULES.find((m) => m.id === active)

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <h1>{t('brandTitle')}</h1>
          <p>{t('brandSub')}</p>
        </div>

        <div className="segmented" style={{ width: '100%', marginBottom: 14 }}>
          <button
            className={lang === 'zh' ? 'active' : ''}
            style={{ flex: 1 }}
            onClick={() => setLang('zh')}
          >
            中文
          </button>
          <button
            className={lang === 'en' ? 'active' : ''}
            style={{ flex: 1 }}
            onClick={() => setLang('en')}
          >
            English
          </button>
        </div>

        <div className="nav-row">
          <button
            className={`nav-item${active === null ? ' active' : ''}`}
            onClick={() => setActive(null)}
          >
            {t('navHome')}
            <span className="nav-sub">{t('navHomeSub')}</span>
          </button>
          {MODULES.map((m) => (
            <button
              key={m.id}
              className={`nav-item${active === m.id ? ' active' : ''}`}
              disabled={m.status === 'soon'}
              style={m.status === 'soon' ? { opacity: 0.5, cursor: 'not-allowed' } : undefined}
              onClick={() => m.status === 'done' && setActive(m.id)}
            >
              {lang === 'zh' ? m.name : m.nameEn}
              {m.status === 'soon' && <span className="badge-todo">{t('planBadge')}</span>}
              <span className="nav-sub">{lang === 'zh' ? m.sub : m.subEn}</span>
            </button>
          ))}
        </div>
      </aside>

      <main className="main">
        <div className="main-inner">
          {current?.Component ? <current.Component /> : <Home />}
        </div>
      </main>
    </div>
  )
}

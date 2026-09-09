import { useEffect, useState } from 'react'
import { MODULES } from './modules/registry'

function Home() {
  return (
    <div>
      <div className="module-head">
        <h2>把大模型的黑盒拆开看</h2>
        <div className="lead">
          纯前端、中文、零付费依赖。每个模块都可以输入你自己的文本实时渲染，并配了「公式在算什么」和「一句话类比」。
        </div>
      </div>

      <div className="card">
        <div className="card-title">先说清楚一件事</div>
        <p style={{ color: 'var(--text-2)', marginBottom: 8 }}>
          本站不下载任何模型权重，所有数值都是<strong>确定性模拟</strong>出来的：同一个输入永远得到同一个结果。
          这么做是为了让它能离线、秒开、可交互；代价是它<strong>不是真实模型的前向结果</strong>。
        </p>
        <p style={{ color: 'var(--text-2)', marginBottom: 0 }}>
          但模拟的部分只有「学出来的权重矩阵」，<strong>算法流程是真的</strong>：BPE
          合并表真的从语料里统计出来，注意力的 Q/K/V 投影、缩放、因果掩码、softmax 走的是完整正确的计算路径，
          不同头呈现的模式也是文献里反复观察到的那几种。用它建立直觉，再用真实权重的项目（如
          transformer-explainer）去验证，是最省时间的学习路径。
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
              {m.name}
              {m.status === 'soon' && <span className="badge-todo">规划中</span>}
            </h3>
            <p>{m.blurb}</p>
          </button>
        ))}
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <div className="card-title">建议的浏览顺序</div>
        <p style={{ color: 'var(--text-2)', marginBottom: 0 }}>
          分词 → 嵌入与位置编码 → 多头注意力 → 采样生成 → KV Cache。
          前三步是理解后面一切的地基，尤其是注意力那一步，值得你把每个头都点开看一遍；
          后两步是"训练好的模型怎么被用来生成"，也是工程面试最爱问的部分。
        </p>
      </div>
    </div>
  )
}

export default function App() {
  const [active, setActive] = useState<string | null>(null)

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
          <h1>LLM 内部机制<br />可视化实验室</h1>
          <p>v0.1 · 纯前端 · 中文</p>
        </div>
        <div className="nav-row">
          <button
            className={`nav-item${active === null ? ' active' : ''}`}
            onClick={() => setActive(null)}
          >
            首页
            <span className="nav-sub">这个项目在做什么</span>
          </button>
          {MODULES.map((m) => (
            <button
              key={m.id}
              className={`nav-item${active === m.id ? ' active' : ''}`}
              disabled={m.status === 'soon'}
              style={m.status === 'soon' ? { opacity: 0.5, cursor: 'not-allowed' } : undefined}
              onClick={() => m.status === 'done' && setActive(m.id)}
            >
              {m.name}
              {m.status === 'soon' && <span className="badge-todo">规划中</span>}
              <span className="nav-sub">{m.sub}</span>
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

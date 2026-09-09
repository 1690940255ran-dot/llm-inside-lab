# LLM 内部机制可视化实验室

用可交互的方式把大语言模型的内部机制拆成直观模块。**中文界面 · 纯前端 · 零付费依赖 · 响应式**。

> 目标读者：想真正"看懂"大模型在算什么，而不是只会调 API 的人。

## 快速开始

```bash
npm install
npm run dev      # 打开 http://127.0.0.1:5173
```

```bash
npm run build    # 产物在 dist/，可直接丢到 GitHub Pages
```

## 已完成模块（v0.1）

| 模块 | 位置 | 说明 |
| --- | --- | --- |
| ① 分词 | `src/modules/tokenizer` | **真实的 BPE 算法**：合并表从内置语料现场学出来，带逐步合并动画、词表统计、字符级对照 |
| ② 嵌入与位置编码 | `src/modules/embedding` | 嵌入向量热图、正弦位置编码、位置相似度矩阵、RoPE 旋转演示、PCA 降维散点 |
| ③ 多头自注意力 | `src/modules/attention` | **主推模块**：可切层切头的注意力热力图、本层所有头一览、因果掩码、温度与距离衰减可调、逐行注意力分布 |

| ④ 逐 token 生成与采样 | `src/modules/generation` | 把抽签前的每一步摊开：原始概率 → 温度 → top-k → top-p → 抽中谁；概率来自真实的 bigram 统计模型 |
| ⑤ KV Cache 加速 | `src/modules/kvcache` | 量化「省了多少 FLOPs、付了多少显存」，含 GQA、batch、精度的影响曲线 |

| ⑥ Transformer 层间数据流 | `src/modules/flow` | 可逐子步推进的 Block 结构动画（LN → Attention → 残差 → FFN → 残差），配当前表示热图、残差贡献、层间相似度 |

六个模块全部可用。侧边栏顺序即建议的浏览顺序。

## 目录结构

```
src/
├── core/                 纯计算层，与 UI 完全解耦，可单独复用和测试
│   ├── random.ts         确定性伪随机（哈希 → 种子 → mulberry32）、softmax
│   ├── bpe.ts            真实的 BPE 训练 / 编码 / 合并过程记录
│   ├── corpus.ts         内置训练语料与示例文本
│   ├── embedding.ts      模拟嵌入向量、余弦相似度、PCA 降维
│   ├── positional.ts     正弦位置编码、RoPE 角度、位置相似度
│   ├── attention.ts      多头注意力模拟（头行为偏置 + 掩码 + 温度）
│   ├── ngram.ts          从语料统计的 bigram 语言模型（给采样模块提供真实概率分布）
│   ├── sampling.ts       温度 / top-k / top-p 采样
│   ├── kvcache.ts        KV Cache 的计算量与显存解析模型
│   ├── color.ts          热力图配色
│   └── sharedModel.ts    全局共用的分词模型
├── components/           通用 UI：滑块、分段选择、热力图、条形图、折线图、token 卡片、原理卡
├── modules/              各个教学模块（registry.ts 是注册表，加模块只改一处）
├── styles/global.css     全部样式，浅色主题 + 响应式
└── App.tsx               侧边栏 + 内容区，无路由库
```

新增模块只需两步：在 `src/modules/<name>/` 写好组件，然后在 `src/modules/registry.ts` 里加一项。

## 关于"真实性"的诚实说明

这是本项目最重要的设计取舍，写在首页上，也写在这里：

- **不下载任何模型权重**，因此所有数值都是**确定性模拟**的（同一个输入永远得到同一个结果）。好处是离线、秒开、可任意交互；代价是它不是真实模型的前向结果。
- **但算法流程是真的**：
  - BPE 的合并表是真的从语料频次里统计出来的，不是硬编码；
  - 注意力的 Q/K/V 投影、`/√d_k` 缩放、因果掩码、softmax 走的是完整正确的计算路径；
  - 不同头呈现的模式（前一个 token / 句首 attention sink / 标点 / 内容相似 / 稀疏激发）是文献中反复观察到的典型行为。
- 建议的学习路径：用本站建立直觉 → 用跑真实权重的项目（如 [transformer-explainer](https://github.com/poloclub/transformer-explainer)）验证。

后续计划提供「接入 transformers.js 跑真实小模型权重」的可选开关。

## 参考与致谢

同类优秀项目（本项目的定位是"中文 + 模块化教学 + 零依赖"，与之互补而非竞争）：

- [poloclub/transformer-explainer](https://github.com/poloclub/transformer-explainer) — 浏览器内实时跑 GPT-2，佐治亚理工
- [bbycroft/llm-viz](https://github.com/bbycroft/llm-viz) — 极细致的 3D 张量流动画
- [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) — 从零实现 LLM 的教科书级仓库
- [jalammar/ecco](https://github.com/jalammar/ecco) — Jupyter 内的语言模型可解释性工具

## 路线图

- [x] v0.1 分词 / 位置编码 / 多头注意力
- [x] v0.2 逐 token 生成与采样、KV Cache 加速
- [x] v0.3 Transformer 层间数据流（残差、LayerNorm、FFN 张量形状动画）
- [ ] v0.4 可选接入 transformers.js，跑真实小模型权重
- [ ] v0.5 中英双语界面 + 每个模块配套的讲解长文

## License

MIT

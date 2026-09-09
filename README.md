# LLM 内部机制可视化实验室

**用可交互的方式，把大语言模型的黑盒拆开看。**

中文界面（可切 English） · 纯前端 · 零付费依赖 · 响应式 · 六个模块全部可玩 · 可选加载真实模型权重

![Node](https://img.shields.io/badge/node-%E2%89%A518-339933)

![React](https://img.shields.io/badge/react-18-61dafb)

![TS](https://img.shields.io/badge/typescript-5-3178c6)
![i18n](https://img.shields.io/badge/i18n-%E4%B8%AD%2FEN-blue)



![License](https://img.shields.io/badge/license-MIT-green)

---

## 在线体验

部署到 GitHub Pages 后，地址形如：

```
https://<你的用户名>.github.io/llm-inside-lab/
```

本地跑：

```bash
npm install
npm run dev      # http://127.0.0.1:5173
```

```bash
npm run build    # 产物在 dist/，可直接静态托管
```

---

## 六个模块

| 模块                      | 你能玩到什么                                                    |
| ----------------------- | --------------------------------------------------------- |
| **① 分词**                | 真实的 BPE 算法，合并表从语料现场学出来。可调合并次数、可换自定义语料，带逐步合并动画和字符级对照       |
| **② 嵌入与位置编码**           | 嵌入向量热图、正弦位置编码、位置相似度矩阵、RoPE 旋转演示、PCA 降维散点                  |
| **③ 多头自注意力**            | **主力模块**：可切层切头的注意力热力图、本层所有头一览、因果掩码、温度与距离衰减实时可调、逐行注意力分布    |
| **④ 逐 token 生成与采样**     | 把抽签前的每一步摊开：原始 p → 温度 → top-k → top-p → 抽中谁，被截断的候选整行灰掉     |
| **⑤ KV Cache 加速**       | 量化「省了多少 FLOPs、付了多少显存」，含 GQA、batch、精度三个维度的影响曲线             |
| **⑥ Transformer 层间数据流** | 可逐子步推进的 Block 结构动画（LN→Attn→残差→LN→FFN→残差），配表示热图、残差贡献、层间相似度 |

侧边栏顺序即建议的浏览顺序：

```
分词 → 嵌入与位置编码 → 多头注意力 → 采样生成 → KV Cache → 层间数据流
```

前三步是理解后面一切的地基；后三步是"训练好的模型怎么被用来生成"，也是工程面试最爱问的部分。

---

## 配套长文

站点负责"看见"，长文负责"讲透"。每篇 1500 字左右，配公式、类比、常见误解和动手实验清单。

| 文章                                                       | 内容                                                                |
| -------------------------------------------------------- | ----------------------------------------------------------------- |
| [01 · 分词](./docs/01-tokenizer.md)                        | 为什么不能按字切也不能按词切；BPE 到底在统计什么；**中文为什么更费 token**                      |
| [02 · 嵌入与位置编码](./docs/02-embedding.md)                   | 自注意力为什么是排列等变的；正弦编码的频率设计；**RoPE 为什么天然表达相对距离**                      |
| [03 · 多头注意力](./docs/03-attention.md)                     | 为什么要除以 √d_k；多头的必要性；**attention sink 是什么以及它为什么能撑起 StreamingLLM**   |
| [04 · 采样](./docs/04-sampling.md)                         | 温度 vs top-p 的本质区别（改 logits 还是改支撑集）；低温度为什么会复读                      |
| [05 · KV Cache](./docs/05-kvcache.md)                    | O(m·n²) → O(n²+m·n) 的推导；**GQA 为什么是最划算的一刀**；prefill 与 decode 是两种负载 |
| [06 · Transformer Block](./docs/06-transformer-block.md) | pre-norm vs post-norm；残差流视角；**FFN 才是参数大头与知识存储处**                  |

---

## 中英双语

侧边栏顶部一键切换 中文 / English。所有 UI 文案、原理卡、公式注释都是双语的。

实现方式很轻：`src/i18n/` 一个 Context + 一个 `t()`，跨模块复用的短文案放 `common.ts`，
每个模块自己的长文案就近放在模块文件里（`const DICT = { zh, en }`），不引任何 i18n 库。

## 可选：加载真实模型权重

注意力模块可以切换成**真实权重**。用 [@huggingface/transformers](https://github.com/huggingface/transformers.js)
在浏览器里（WebGPU 优先，回退 WASM）跑一个小模型，取出 `output_attentions`：

| 模型 | 规模 | 说明 |
| --- | --- | --- |
| `Xenova/gpt2` | ≈ 90 MB | 12 层 × 12 头，英文，加载最快 |
| `onnx-community/SmolLM2-135M-Instruct` | ≈ 100 MB | 30 层 × 9 头 |
| `onnx-community/Qwen2.5-0.5B-Instruct` | ≈ 350 MB | 24 层 × 14 头，**支持中文** |

要点：

- **动态 import**，不加载就不会下载这部分代码（主包 92 kB gzip，transformers 单独分片）
- **全程本地推理**，文本不会发到任何服务器
- **镜像可配**：默认 `https://hf-mirror.com`，国外网络可改回 `https://huggingface.co`
- 真实模式下层数 / 头数 / 温度由模型本身决定（锁定），因果掩码仍可切换（只影响显示）

## 关于"真实性"的诚实说明

这是本项目最重要的设计取舍，写在首页上，也写在这里：

- **不下载任何模型权重**，所以所有数值都是**确定性模拟**的（同一个输入永远得到同一个结果）。好处是离线、秒开、可任意交互；代价是它不是真实模型的前向结果。
- **但算法流程是真的**：
  - BPE 的合并表是真的从语料频次里统计出来的，不是硬编码；
  - 注意力的 Q/K/V 投影、`/√d_k` 缩放、因果掩码、softmax 走的是完整正确的计算路径；
  - 采样三个参数的实现与 HF `transformers` 的 logits warper 逻辑一致；
  - KV Cache 的复杂度是解析推导，可以直接和 profiler 对照；
  - 不同头呈现的模式（前一个 token / 句首 sink / 标点 / 内容相似 / 稀疏激发）是文献中反复观察到的典型行为。

**用它建立直觉是安全的，用它引用具体数值是不行的。**

后续计划提供「接入 transformers.js 跑真实小模型权重」的可选开关。

---

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
│   ├── transformer.ts    Transformer Block 前向模拟（pre-norm + GELU FFN）
│   ├── ngram.ts          从语料统计的 bigram 语言模型（给采样模块提供真实分布）
│   ├── sampling.ts       温度 / top-k / top-p 采样
│   ├── kvcache.ts        KV Cache 的计算量与显存解析模型
│   ├── realModel.ts      可选：浏览器内跑真实 ONNX 模型取真实注意力
│   ├── color.ts          热力图配色
│   └── sharedModel.ts    全局共用的分词模型
├── components/           通用 UI：滑块、分段选择、热力图、条形图、折线图、token 卡片、原理卡
├── modules/              六个教学模块（registry.ts 是注册表）
├── i18n/                 极简双语：Context + t()，common.ts 放跨模块文案
├── styles/global.css     全部样式，浅色主题 + 响应式
└── App.tsx               侧边栏 + 内容区，无路由库
docs/                     六篇配套长文
.github/workflows/        GitHub Pages 自动部署
```

**新增一个模块只需两步**：在 `src/modules/<name>/` 写好组件，然后在 `src/modules/registry.ts` 里加一项。

---

## 部署到 GitHub Pages

仓库里已经带好了 workflow（`.github/workflows/deploy.yml`）。步骤：

1. 在仓库 Settings → Pages → Source 选 **GitHub Actions**
2. 推送到 `main` 分支，workflow 会自动构建并部署

`vite.config.ts` 里 `base: './'` 已设好，所以在任意子路径下都能正常加载。

---

## 参考与致谢

同类优秀项目（本项目的定位是"中文 + 模块化教学 + 零依赖"，与之互补而非竞争）：

- [poloclub/transformer-explainer](https://github.com/poloclub/transformer-explainer) — 浏览器内实时跑 GPT-2，佐治亚理工
- [bbycroft/llm-viz](https://github.com/bbycroft/llm-viz) — 极细致的 3D 张量流动画
- [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) — 从零实现 LLM 的教科书级仓库
- [jalammar/ecco](https://github.com/jalammar/ecco) — Jupyter 内的语言模型可解释性工具

## 路线图

- [x] v0.1 分词 / 位置编码 / 多头注意力
- [x] v0.2 逐 token 生成与采样、KV Cache 加速
- [x] v0.3 Transformer 层间数据流
- [x] v0.4 六篇配套中文长文 + Pages 自动部署
- [x] v0.5 可选接入 transformers.js，浏览器内跑真实小模型权重
- [x] v0.6 中英双语界面
- [ ] v0.7 首页 GIF 动图 + 每模块「导出图片」按钮
- [ ] v0.8 采样模块也接真实模型（真实 next-token 分布）

## License

MIT

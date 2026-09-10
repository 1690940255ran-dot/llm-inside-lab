# LLM 内部机制可视化实验室

**用可交互的方式，把大语言模型的黑盒拆开看。**

[English](./README_EN.md) · 简体中文

纯前端 · 零付费依赖 · 中英双语 · 响应式 · 六个模块全部可玩 · 可选加载真实模型权重

![Node](https://img.shields.io/badge/node-%E2%89%A518-339933)
![React](https://img.shields.io/badge/react-18-61dafb)
![TS](https://img.shields.io/badge/typescript-5-3178c6)
![tests](https://img.shields.io/badge/tests-117%20passing-brightgreen)
![i18n](https://img.shields.io/badge/i18n-%E4%B8%AD%2FEN-blue)
![docs](https://img.shields.io/badge/docs-%E4%B8%AD%2FEN%20%C3%976-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 在线体验

**👉 [https://1690940255ran-dot.github.io/llm-inside-lab/](https://1690940255ran-dot.github.io/llm-inside-lab/)**

打开即用，无需注册、无需后端、无需下载任何东西。

首页有一条自动播放的流水线动图（文本 → token → 注意力 → 下一个 token 的分布），
每一步都由 `src/core/` 里那套已单测的函数算出来：

![demo](./docs/demo.gif)

---

## 它解决什么问题

大模型原理的现有资料基本分成两类：**公式推导**（看得懂符号，建立不了直觉）和
**封装好的 demo**（拖两下滑块，不知道背后在算什么）。

这个站点想补上中间那一层——**把每一步的中间结果摊开给你看，并且允许你改动输入实时重算**：

- 想知道「温度到底改了什么」？候选表里能同时看到**原始 p / 温度后 p / 采样后 p** 三列，
  被 top-k / top-p 砍掉的候选整行变灰，只被温度压低的数字变小但还在。
- 想知道「多头注意力为什么要有多个头」？点开「本层所有头一览」，
  十几个缩略图的图案完全不同——比任何文字解释都直观。
- 想知道「中文为什么更费 token」？同一段内容的中英文版本各输一遍，直接看 token 数。

## 亮点

| | |
| --- | --- |
| **算法是真的，不是画出来的** | BPE 合并表真从语料频次里数出来；注意力走完整的 Q/K/V 投影 → `/√d_k` 缩放 → 因果掩码 → softmax；采样三参数与 HF `transformers` 的 logits warper 逻辑一致 |
| **可选跑真实模型** | 浏览器内（WebGPU / WASM）真跑 SmolLM2 / LFM2 / Qwen3，把温度 / top-k / top-p 直接作用在**真实 logits** 上，顺带读出 KV Cache 的实测张量形状 |
| **不骗人** | 哪里是模拟、哪里是真算，首页和每个模块都写清楚；拿不到的东西（真实注意力热力图）直接贴出实测证据说明为什么拿不到 |
| **零 UI / 图表库** | 热力图用 `<table>`、折线图用 SVG、导出 PNG 自己实现（`core/exportImage.ts`），主包 101 kB gzip |
| **117 个单测** | `core/` 全纯函数，可脱离浏览器断言；踩过的每个坑都有回归测试 |
| **中英双语** | 界面双语，六篇配套长文也有完整英文版，章节一一对应 |

---

## 六个模块

每个模块的主视觉卡片右上角都有「导出图片」按钮，一键导出 2 倍 PNG（纯浏览器本地生成）。
做 PPT / 笔记 / 汇报直接拿图。

| 模块 | 你能玩到什么 |
| --- | --- |
| **① 分词** | 真实的 BPE 算法，合并表从语料现场学出来。可调合并次数、可换自定义语料，带逐步合并动画和字符级对照 |
| **② 嵌入与位置编码** | 嵌入向量热图、正弦位置编码、位置相似度矩阵、RoPE 旋转演示、PCA 降维散点 |
| **③ 多头自注意力** | **主力模块**：可切层切头的注意力热力图、本层所有头一览、因果掩码、温度与距离衰减实时可调、逐行注意力分布 |
| **④ 逐 token 生成与采样** | 把抽签前的每一步摊开：原始 p → 温度 → top-k → top-p → 抽中谁，被截断的候选整行灰掉。**可选加载真实模型**，让同样的旋钮作用在真实 logits 上 |
| **⑤ KV Cache 加速** | 量化「省了多少 FLOPs、付了多少显存」，含 GQA、batch、精度三个维度的影响曲线 |
| **⑥ Transformer 层间数据流** | 可逐子步推进的 Block 结构动画（LN→Attn→残差→LN→FFN→残差），配表示热图、残差贡献、层间相似度 |

建议顺序：分词 → 嵌入与位置编码 → 多头注意力 → 采样生成 → KV Cache → 层间数据流。
前三步是理解后面一切的地基；后三步是「训练好的模型怎么被用来生成」，也是工程面试最爱问的部分。

---

## 快速开始

**前置**：Node.js ≥ 18、npm ≥ 9。

```bash
git clone https://github.com/1690940255ran-dot/llm-inside-lab.git
cd llm-inside-lab
npm install
npm run dev          # http://127.0.0.1:5173
```

其他命令：

```bash
npm run build        # 产物在 dist/，可直接静态托管
npm test             # 117 个单测，全部纯函数，不碰网络
npm run test:watch   # 开发时用
npm run probe:model  # 探测 ONNX 输出签名（就是它证明了拿不到真实注意力）
```

想重新生成首页那张 GIF：`npm run capture:hero` 抓帧，再 `npm run make:gif` 合成
（分别需要 `ws` 与 Pillow，详见 `scripts/`）。

---

## 配套长文

站点负责「看见」，长文负责「讲透」。每篇约 1500 字，配公式、类比、常见误解和动手实验清单。
**中英双语，章节编号一一对应。**

| 文章 | 内容 |
| --- | --- |
| [01 · 分词](./docs/01-tokenizer.md) | 为什么不能按字切也不能按词切；BPE 到底在统计什么；**中文为什么更费 token** |
| [02 · 嵌入与位置编码](./docs/02-embedding.md) | 自注意力为什么是排列等变的；正弦编码的频率设计；**RoPE 为什么天然表达相对距离** |
| [03 · 多头注意力](./docs/03-attention.md) | 为什么要除以 √d_k；多头的必要性；**attention sink 是什么以及它为什么能撑起 StreamingLLM** |
| [04 · 采样](./docs/04-sampling.md) | 温度 vs top-p 的本质区别（改 logits 还是改支撑集）；低温度为什么会复读 |
| [05 · KV Cache](./docs/05-kvcache.md) | O(m·n²) → O(n²+m·n) 的推导；**GQA 为什么是最划算的一刀**；prefill 与 decode 是两种负载 |
| [06 · Transformer Block](./docs/06-transformer-block.md) | pre-norm vs post-norm；残差流视角；**FFN 才是参数大头与知识存储处** |

**English mirror**：[`docs/en/`](./docs/en/README.md)——六篇长文的完整英文版，可中英对着读。

---

## 可选：加载真实模型权重

**采样模块**可以切换成**真实权重**。用
[@huggingface/transformers](https://github.com/huggingface/transformers.js) 在浏览器里
（WebGPU 优先，回退 WASM）跑一个小模型，把页面上的温度 / top-k / top-p 三个旋钮
直接作用在真实 logits 上——采样那套代码（`core/sampling.ts`，有单测）跑的是真模型的分布。

| 模型 | 实测体积 | dtype | 说明 |
| --- | --- | --- | --- |
| `onnx-community/SmolLM2-135M-Instruct-ONNX` | 129 MB | `q8` | 2024 · 30 层 × 9 头，体积最小，推荐先用它 |
| `onnx-community/LFM2-350M-ONNX` | 280 MB | `q4` | 2025 · Liquid AI 混合卷积+门控注意力，16 块——注意力不是唯一出路的活例子 |
| `onnx-community/Qwen3-0.6B-ONNX` | 589 MB | `q8` | 2025 · 28 层 × 16Q/8KV 头（GQA），**唯一真正支持中文的选项** |
| `Xenova/gpt2` | 268 MB | `int8` | 2019 经典对照：中文按 UTF-8 字节切分（12 字 → 23 token），与现代模型对比看 BPE 进步 |

体积全部是 API `blobs=true` 实测。主力选型全部是 2024-2025 年的模型；GPT-2 只作为教学对照保留。

要点：

- **动态 import**，不加载就不会下载这部分代码（主包 101 kB gzip，transformers 单独分片）
- **全程本地推理**，文本不会发到任何服务器
- **镜像可配**：默认 `https://hf-mirror.com`，国外网络可改回 `https://huggingface.co`
- **静态托管自动锁单线程**：GitHub Pages 不发 COOP/COEP 响应头 → 拿不到 `SharedArrayBuffer`
  → 多线程 WASM 不可用。代码里显式探测并降级为单线程，最坏结果是「慢」而不是「崩」

### ⚠️ 为什么没有「真实注意力热力图」

这是实测结论，不是没做。decoder-only 模型的 ONNX 导出**根本没有注意力输出节点**：

```bash
$ npm run probe:model -- Xenova/gpt2 int8

[session] model
  outputNames: logits  (+24 present.*)
  HAS ATTENTION OUTPUT? NO
[forward] out.attentions = undefined
```

注意力概率矩阵在计算图内部就被融合掉了，而 `transformers.js` 的 `getAttentions()` 只识别
`cross_/encoder_/decoder_attentions.*`（whisper 那类 seq2seq 的用法），对 CausalLM 传
`output_attentions: true` 是空操作。

所以真实权重用在**它确实能覆盖的地方**：

1. **下一个 token 的真实概率分布**（`logits` 拿得到）
2. **KV Cache 的实测张量形状**（`present.*.key/.value` 的 `dims` 拿得到，包括 GQA 下 KV 头数少于 Query 头数）

注意力模块保持确定性模拟，并在界面上贴出上面这段实测输出。想自己复现，跑 `npm run probe:model` 即可。

### 四个踩过的坑（都已写成回归测试）

| 坑 | 现象 | 修法 |
| --- | --- | --- |
| 覆盖 `env.remotePathTemplate` 加 `onnx/` 前缀 | `Cannot read properties of undefined (reading 'tokenizer_class')` | 只改 `env.remoteHost`。模板对**所有**文件生效，根目录的配置文件会 404 |
| `dtype: 'q8'` 配 `Xenova/gpt2` | `Could not locate file: .../onnx/model_quantized.onnx` | 老仓库只有 `model_int8.onnx`，改用 `dtype: 'int8'` |
| 用 `tokenize()` + `tokens.unshift('<s>')` 对齐标签 | 无 BOS 的模型（Qwen2.5）标签整体错位一行；中文全是 `æ³¨æĦıåĬĽ` 乱码 | 逐 id `tokenizer.decode([id])`，从源头保证长度一致且可读 |
| hf-mirror 防盗链（curl 通、浏览器挂） | 同一 URL：不带 Referer 返回 307 + CORS 头；带第三方 Referer 后 CORS 头被剥掉，浏览器 fetch 必被拦截。transformers.js v4 又把探测失败静默吞掉（`get_file_metadata` 返回 `exists: false`），最终才报出那个迷惑性的 `tokenizer_class` | 镜像站行为改不了，但可以在 fetch 层兜底：`installRetryingFetch()` 对连接失败自动重试；失败提示写明原因并提示**刷新页面**重试——探测结果在页面会话内被 memoize，原地点「重新加载」一个请求都不会发 |

第三条里的中文乱码对比很直观：

```
tokenize()      = "æ³¨æĦıåĬĽ" "æľºåĪ¶" "æĺ¯" ...
per-id decode   = "注意力" "机制" "是" "大" "模型" "的核心"
```

字节级 BPE 还有个后续问题：「注」会被切成 3 个 UTF-8 字节 token，逐 id 解码每个都是乱码。
本站的处理是把**连续的碎片合并起来整体解码**（`core/realModel.ts` 的 `mergeFragments`）；
真的解不回来的坏字节（比如半个字符）才原样保留碎片，不伪造可读文本。

---

## 关于「真实性」的诚实说明

这是本项目最重要的设计取舍，写在首页上，也写在这里：

- **默认不下载任何模型权重**，所有数值都是**确定性模拟**（同一输入永远同一结果）。
  好处是离线、秒开、可任意交互；代价是它不是真实模型的前向结果。
- **但算法流程是真的**：
  - BPE 的合并表真的从语料频次里数出来，不是硬编码；
  - 注意力的 Q/K/V 投影、`/√d_k` 缩放、因果掩码、softmax 走的是完整正确的路径；
  - 采样三个参数的实现与 HF `transformers` 的 logits warper 逻辑一致；
  - KV Cache 的复杂度是解析推导，可以直接和 profiler 对照；
  - 不同头呈现的模式（前一个 token / 句首 sink / 标点 / 内容相似 / 稀疏激发）是文献里反复观察到的典型行为。

**用它建立直觉是安全的，用它引用具体数值是不行的。**

---

## 与同类项目的区别

同类优秀项目很多，这个仓库的定位是**中文 + 模块化教学 + 零依赖 + 可验证**，与之互补而非竞争：

| 项目 | 侧重点 | 差异 |
| --- | --- | --- |
| [transformer-explainer](https://github.com/poloclub/transformer-explainer) | 浏览器内实时跑 GPT-2，聚焦单一模型的完整前向 | 它跑真实权重但只有一个模型、一条链路；本项目默认零下载、六个模块可拆开单独玩，且中英双语 |
| [bbycroft/llm-viz](https://github.com/bbycroft/llm-viz) | 极细致的 3D 张量流动画 | 视觉震撼但需要跟着导览走；本项目偏向「每个参数都能拧、拧完立刻看到数字怎么变」 |
| [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) | 从零用 PyTorch 实现并训练 | 它是「你要写代码」；本项目是「你不用写代码，但能看到每一步在算什么」，两者配合最好 |
| [jalammar/ecco](https://github.com/jalammar/ecco) | Jupyter 内的可解释性分析 | 面向研究者；本项目面向学习者和面试准备，不需要 Python 环境 |

---

## 目录结构

```
src/
├── core/                 纯计算层，与 UI 完全解耦，可单独复用和测试
│   ├── random.ts         确定性伪随机（哈希 → 种子 → mulberry32）、softmax
│   ├── bpe.ts            真实的 BPE 训练 / 编码 / 合并过程记录
│   ├── corpus.ts         内置训练语料与示例文本
│   ├── attention.ts      多头注意力（Q/K/V + √d_k 缩放 + 因果掩码 + softmax）
│   ├── embedding.ts      嵌入向量、余弦相似度、PCA
│   ├── positional.ts     正弦位置编码 / RoPE
│   ├── transformer.ts    Transformer Block 前向模拟（pre-norm + GELU FFN）
│   ├── ngram.ts          从语料统计的 bigram 语言模型（给采样模块提供真实分布）
│   ├── sampling.ts       温度 / top-k / top-p 采样
│   ├── kvcache.ts        KV Cache 的计算量与显存解析模型
│   ├── realModel.ts      可选：浏览器内跑真实 ONNX 模型（真实 logits + KV 形状）
│   ├── exportImage.ts    零依赖的 DOM → PNG 导出（SVG foreignObject + 内联 CSS）
│   ├── color.ts          热力图配色
│   └── sharedModel.ts    全局共用的分词模型
├── components/           通用 UI：滑块、分段选择、热力图、条形图、折线图、token 卡片、原理卡
│   ├── RealModelPanel.tsx  真实模型面板（模型选择 / 镜像 / 真实分布表 / KV 实测）
│   └── HeroDemo.tsx        首页流水线动图（由 core/ 的真实函数驱动，可暂停）
├── modules/              六个教学模块（registry.ts 是注册表）
├── i18n/                 极简双语：Context + t()，common.ts 放跨模块文案
├── styles/global.css     全部样式，浅色主题 + 响应式
└── App.tsx               侧边栏 + 内容区，无路由库
tests/                    vitest 单测 + 可选的端到端集成测试
scripts/
├── probe-attentions.mjs  探测 ONNX 输出签名（就是它证明了拿不到真实注意力）
├── probe-tokenizers.mjs  对比三个模型的分词与逐 id 解码结果
├── capture-hero.mjs      用 CDP 抓首页动图的帧（供 GIF 用，需 ws）
└── make-gif.py           把帧拼成 docs/demo.gif（需 Pillow）
docs/                     六篇配套长文（中文） + en/（英文版）
.github/workflows/        GitHub Pages 自动部署
```

**新增一个模块只需两步**：在 `src/modules/<name>/` 写好组件，然后在 `src/modules/registry.ts` 里加一项。

---

## 测试

```bash
npm test              # 117 个单测，全部纯函数，不碰网络

# 端到端（会真的下 129 MB 权重，默认跳过）
REAL_MODEL_TEST=1 npm test
REAL_MODEL_TEST=1 REAL_MODEL_ID=onnx-community/Qwen3-0.6B-ONNX npm test
```

`core/` 全部是纯函数，所以能脱离浏览器直接断言：BPE 的无损性与可复现性、采样三个参数的
数学性质（温度不改排序、top-p 取最小跨阈候选集、低温退化为贪心）、真实模型的标签对齐与
数值稳定性、导出工具的尺寸上限与文件名净化。上面那张「四个坑」表里的每一条都有对应的回归测试。

---

## 部署到 GitHub Pages

仓库里已经带好了 workflow（`.github/workflows/deploy.yml`）。步骤：

1. 在仓库 Settings → Pages → Source 选 **GitHub Actions**
2. 推送到 `main` 分支，workflow 会自动构建并部署

`vite.config.ts` 里 `base: './'` 已设好，所以在任意子路径下都能正常加载。

---

## 常见问题

**这些数字是真实模型算出来的吗？**
默认不是，是确定性模拟（详见「关于真实性的诚实说明」）。想看真算的，去采样模块点「加载真实模型」。

**为什么不干脆全部用真实模型？**
真实权重最小也要 129 MB，且中文支持好的 Qwen3-0.6B 要 589 MB。默认零下载是为了秒开、离线可用、
任意参数都能实时重算。两者都提供，让你自己选。

**为什么中文 token 数比英文多？**
因为 BPE 是按**语料频次**合并的，不是按「意思」。英文高频组合被充分合并，中文要么一字一 token，
要么得在词表里专门占位置。详见[长文 01](./docs/01-tokenizer.md)。

**真实模型加载失败怎么办？**
按顺序排查：① 先**刷新页面**再点一次（transformers.js 会把失败的探测结果 memoize 住，原地点重试一个请求都不发）；
② 检查系统代理——Windows 系统代理开着但代理软件没运行时，Chrome 会走死代理全部 `Failed to fetch`，而 curl 直连正常；
③ 换个镜像地址试试。

**导出的 PNG 为什么是纯前端生成的？**
`core/exportImage.ts` 用 SVG `foreignObject` + 内联同一份 CSS 自己实现，没引 html-to-image，
为的是守住「零 UI / 图形库」这条线。导出的是 2 倍图，尺寸会按浏览器 canvas 上限自动压缩。

---

## 参与贡献

欢迎提 Issue 和 PR。几个容易上手的方向：

- 补第七个模块（比如 MoE、RoPE 外推、量化）——在 `src/modules/` 下新建组件 + 在 `registry.ts` 加一项即可
- 补充测试，尤其是 `core/` 里还没覆盖的边界
- 校对英文文案（`docs/en/` 与各模块的 `en` 字典）
- 报告真实模型链路在你网络环境下的表现

提交前请跑 `npm test` 和 `npm run build`。

---

## 如何引用

如果这个项目对你的学习或研究有帮助，可以这样引用：

```bibtex
@software{llm_inside_lab,
  title  = {LLM 内部机制可视化实验室 / Inside the LLM: An Interactive Visualization Lab},
  author = {1690940255ran-dot},
  year   = {2026},
  url    = {https://github.com/1690940255ran-dot/llm-inside-lab},
  note   = {纯前端、中英双语的大语言模型内部机制交互可视化；六个模块 + 十二篇配套长文}
}
```

---

## 参考与致谢

- [poloclub/transformer-explainer](https://github.com/poloclub/transformer-explainer) — 浏览器内实时跑 GPT-2，佐治亚理工
- [bbycroft/llm-viz](https://github.com/bbycroft/llm-viz) — 极细致的 3D 张量流动画
- [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) — 从零实现 LLM 的教科书级仓库
- [jalammar/ecco](https://github.com/jalammar/ecco) — Jupyter 内的语言模型可解释性工具
- [@huggingface/transformers](https://github.com/huggingface/transformers.js) — 让浏览器里跑真实 ONNX 模型成为可能

每篇长文末尾还列了该主题的经典论文，见 [`docs/`](./docs/README.md)。

## 路线图

- [x] v0.1 分词 / 位置编码 / 多头注意力
- [x] v0.2 逐 token 生成与采样、KV Cache 加速
- [x] v0.3 Transformer 层间数据流
- [x] v0.4 六篇配套中文长文 + Pages 自动部署
- [x] v0.5 可选接入 transformers.js，浏览器内跑真实小模型权重
- [x] v0.6 中英双语界面
- [x] v0.6.1 `core/` 单测 + 真实模型端到端集成测试；修掉镜像路径、dtype、token 对齐三个 bug；
      真实权重从「注意力」改挂到「采样 + KV Cache」（因为 ONNX 拿不到注意力，有实测证据）
- [x] v0.7 首页流水线动图 + 每模块「导出图片」按钮（零依赖 PNG 导出）
- [x] v0.7 docs/ 六篇长文的英文版（`docs/en/`，与中文版章节一一对应）
- [x] v0.8 采样模块接真实模型——真实 next-token 分布已落地：温度 / top-k / top-p
      直接作用在真实 logits 上，顺带读出 KV Cache 的实测张量形状
- [ ] v0.9 MoE / 量化 / 长上下文外推等新模块
- [ ] v1.0 自定义语料上传 + 迷你模型训练可视化

## License

MIT

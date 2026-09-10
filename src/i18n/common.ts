/**
 * 通用文案字典（中 / 英）
 *
 * 每个模块自己的长文案放在各自模块里（就近维护，避免一个巨大的文件），
 * 这里只放跨模块复用的：品牌名、导航、通用按钮、原理卡的三个小标题等。
 */
export type Lang = 'zh' | 'en'

export const COMMON = {
  zh: {
    brandTitle: 'LLM 内部机制可视化实验室',
    brandSub: 'v1.0 · 纯前端 · 中英双语',
    navHome: '首页',
    navHomeSub: '这个项目在做什么',

    homeTitle: '把大模型的黑盒拆开看',
    homeLead:
      '纯前端、中英双语、零付费依赖。每个模块都可以输入你自己的文本实时渲染，并配了「公式在算什么」和「一句话类比」。',
    homeHonestTitle: '先说清楚一件事',
    homeHonest1a: '本站默认不下载任何模型权重，所有数值都是',
    homeHonest1b: '确定性模拟',
    homeHonest1c: '出来的：同一个输入永远得到同一个结果。这么做是为了让它能离线、秒开、可交互；代价是它',
    homeHonest1d: '不是真实模型的前向结果',
    homeHonest1e: '。',
    homeHonest2a: '但模拟的部分只有「学出来的权重矩阵」，',
    homeHonest2b: '算法流程是真的',
    homeHonest2c:
      '：BPE 合并表真的从语料里统计出来，注意力的 Q/K/V 投影、缩放、因果掩码、softmax 走的是完整正确的计算路径，不同头呈现的模式也是文献里反复观察到的那几种。采样模块里还可以选择性地加载一个真实小模型（129 MB 起），把温度 / top-k / top-p 直接作用在真实权重算出的分布上，顺带看到 KV Cache 的实测张量形状。另外有一处例外要特别说明：⑩ 模块里的权重**不模拟**，是在浏览器里当场训出来的（手写前向 / 反向 / AdamW），所以那一页的 loss 曲线、采样文本和注意力热力图都是一次真实训练运行的产物，不是画出来的示意图。',
    homeOrderTitle: '建议的浏览顺序',
    homeOrder:
      '分词 → 嵌入与位置编码 → 多头注意力 → 采样生成 → KV Cache → 层间数据流 → MoE → 量化 → 长上下文外推 → 从零训一个迷你 GPT。前六步是「模型内部长什么样」，建议按顺序走一遍，尤其是注意力那一步，值得你把每个头都点开看。中间三步是工程与部署侧最常被问到的三个话题：稀疏专家怎么省算力、量化怎么省显存、位置编码怎么撑住长上下文。最后一步换了个方向 —— 不再看一个训好的模型，而是从随机权重开始**真的训一个**：前面所有被当作既定事实的东西（学习率、warmup、初始化、梯度验证、上下文窗口该开多大）在那里都会变成要你自己负责的决策。',

    principleTitle: '这一步在算什么？',
    formula: '数学形式',
    analogy: '通俗类比',
    moreDetails: '再多说两句',

    inputText: '输入文本',
    clear: '清空',
    play: '播放',
    pause: '暂停',
    prevStep: '上一步',
    nextStep: '下一步',
    reset: '重置',
    stepProgress: '步',
    sample: '示例',
    planBadge: '规划中',
    enterText: '请输入文本。',

    exportImage: '导出图片',
    exportImageHint: '把这张卡导出成 PNG（2 倍图，纯浏览器本地生成）',
    exporting: '导出中…',
    exportOk: '已导出',
    exportFail: '导出失败',
    heroDemoTitle: '一分钟看懂整条链路',
    heroDemoHint: '自动播放：文本 → token → 嵌入 → 注意力 → 下一个 token 的分布',

    realSection: '真实模型：真实的下一个 token 分布（可选）',
    realIntro:
      '上面的分布来自内置语料统计出来的 bigram。点下面的按钮可以在浏览器里真正跑一个小模型，把上面那三个旋钮（温度 / top-k / top-p）直接作用在真实权重算出的 logits 上。首次需要下载 129 MB 起的权重，之后走浏览器缓存。',
    loadModel: '加载真实模型',
    loading: '加载中',
    unload: '卸载模型',
    realReady: '真实模型已就绪',
    realFailed: '加载失败',
    realNote:
      '真实权重由 @huggingface/transformers 在浏览器里用 WebGPU / WASM 推理，不会把你的文本发到任何服务器。国内网络建议先设好镜像地址。',
    mirrorLabel: '模型镜像地址',
    mirrorHint: '默认 hf-mirror.com；国外网络可改回 huggingface.co',
    modelLabel: '模型',
  },
  en: {
    brandTitle: 'Inside the LLM',
    brandSub: 'v1.0 · frontend only · bilingual',
    navHome: 'Home',
    navHomeSub: 'What this project is',

    homeTitle: 'Open the black box, one module at a time',
    homeLead:
      'Frontend-only, bilingual, no paid APIs. Every module renders live from your own text, and each comes with what the formula means and a one-sentence analogy.',
    homeHonestTitle: 'One thing up front',
    homeHonest1a: 'This site ships no model weights, and by default every number is a ',
    homeHonest1b: 'deterministic simulation',
    homeHonest1c:
      ': the same input always gives the same output. That buys offline, instant, fully interactive pages — at the cost of ',
    homeHonest1d: 'not being a real forward pass',
    homeHonest1e: '.',
    homeHonest2a: 'But only the learned weight matrices are simulated — ',
    homeHonest2b: 'the algorithms are real',
    homeHonest2c:
      ': BPE merges are counted from a real corpus, attention runs the full Q/K/V projection, scaling, causal masking and softmax, and the head patterns are the ones repeatedly reported in the literature. The sampling module can also optionally load a real small model (129 MB and up), so temperature / top-k / top-p act on a distribution produced by real weights — and you get the measured KV-cache tensor shapes for free. There is one exception worth stating explicitly: the weights in module ⑩ are **not** simulated — they are trained on the spot in your browser (hand-written forward / backward / AdamW), so that page\'s loss curve, sampled text and attention heatmaps are the output of a real training run, not a diagram.',
    homeOrderTitle: 'Suggested order',
    homeOrder:
      'Tokenizer → Embeddings & Positional Encoding → Multi-Head Attention → Sampling → KV Cache → Block data flow → MoE → Quantization → Context extension → Training a mini GPT from scratch. The first six are "what the model looks like inside" and are best taken in order — especially attention, where every head is worth opening. The middle three are the topics that come up most in engineering and deployment: how sparse experts save compute, how quantization saves memory, and how positional encoding holds up over long context. The last one reverses direction — instead of inspecting a trained model, you train one from random weights, and everything the earlier modules took as given (learning rate, warmup, initialisation, gradient validation, how wide the context window should be) becomes a decision you have to own.',

    principleTitle: 'What is being computed here?',
    formula: 'The math',
    analogy: 'Plain-language analogy',
    moreDetails: 'A bit more',

    inputText: 'Input text',
    clear: 'Clear',
    play: 'Play',
    pause: 'Pause',
    prevStep: 'Prev',
    nextStep: 'Next',
    reset: 'Reset',
    stepProgress: 'step',
    sample: 'Sample',
    planBadge: 'planned',
    enterText: 'Please enter some text.',

    exportImage: 'Export PNG',
    exportImageHint: 'Export this card as a PNG (2x, generated locally in your browser)',
    exporting: 'Exporting…',
    exportOk: 'saved',
    exportFail: 'export failed',
    heroDemoTitle: 'The whole pipeline in one minute',
    heroDemoHint: 'Auto-playing: text → tokens → embeddings → attention → next-token distribution',

    realSection: 'Real model: the actual next-token distribution (optional)',
    realIntro:
      'The distribution above comes from a bigram counted off the bundled corpus. Press the button to genuinely run a small model in your browser, so the three knobs above (temperature / top-k / top-p) act on logits produced by real weights. The first load downloads 129 MB or more, then it is browser-cached.',
    loadModel: 'Load real model',
    loading: 'Loading',
    unload: 'Unload model',
    realReady: 'Real model ready',
    realFailed: 'Load failed',
    realNote:
      'Real weights run entirely in your browser via @huggingface/transformers on WebGPU / WASM. Your text is never sent to a server. If you are behind the Great Firewall, set a mirror host first.',
    mirrorLabel: 'Model mirror host',
    mirrorHint: 'Defaults to hf-mirror.com; use huggingface.co elsewhere',
    modelLabel: 'Model',
  },
} as const

export type CommonKey = keyof (typeof COMMON)['zh']

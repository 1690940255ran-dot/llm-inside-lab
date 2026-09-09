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
    brandSub: 'v0.6 · 纯前端 · 中英双语',
    navHome: '首页',
    navHomeSub: '这个项目在做什么',

    homeTitle: '把大模型的黑盒拆开看',
    homeLead:
      '纯前端、中英双语、零付费依赖。每个模块都可以输入你自己的文本实时渲染，并配了「公式在算什么」和「一句话类比」。',
    homeHonestTitle: '先说清楚一件事',
    homeHonest1a: '本站不下载任何模型权重，默认所有数值都是',
    homeHonest1b: '确定性模拟',
    homeHonest1c: '出来的：同一个输入永远得到同一个结果。这么做是为了让它能离线、秒开、可交互；代价是它',
    homeHonest1d: '不是真实模型的前向结果',
    homeHonest1e: '。',
    homeHonest2a: '但模拟的部分只有「学出来的权重矩阵」，',
    homeHonest2b: '算法流程是真的',
    homeHonest2c:
      '：BPE 合并表真的从语料里统计出来，注意力的 Q/K/V 投影、缩放、因果掩码、softmax 走的是完整正确的计算路径，不同头呈现的模式也是文献里反复观察到的那几种。用它建立直觉，再用真实权重的项目去验证，是最省时间的学习路径。',
    homeOrderTitle: '建议的浏览顺序',
    homeOrder:
      '分词 → 嵌入与位置编码 → 多头注意力 → 采样生成 → KV Cache → 层间数据流。前三步是理解后面一切的地基，尤其是注意力那一步，值得你把每个头都点开看一遍；后三步是「训练好的模型怎么被用来生成」，也是工程面试最爱问的部分。',

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

    realSection: '真实模型（可选）',
    realIntro:
      '默认展示的是确定性模拟结果。点下面的按钮可以在浏览器里真正跑一个小模型，看到真实权重算出来的注意力。首次需要下载约 100 MB 权重。',
    loadModel: '加载真实模型',
    loading: '加载中',
    unload: '切回模拟模式',
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
    brandSub: 'v0.6 · frontend only · bilingual',
    navHome: 'Home',
    navHomeSub: 'What this project is',

    homeTitle: 'Open the black box, one module at a time',
    homeLead:
      'Frontend-only, bilingual, no paid APIs. Every module renders live from your own text, and each comes with what the formula means and a one-sentence analogy.',
    homeHonestTitle: 'One thing up front',
    homeHonest1a: 'This site ships no model weights. By default every number is a ',
    homeHonest1b: 'deterministic simulation',
    homeHonest1c:
      ': the same input always gives the same output. That buys offline, instant, fully interactive pages — at the cost of ',
    homeHonest1d: 'not being a real forward pass',
    homeHonest1e: '.',
    homeHonest2a: 'But only the learned weight matrices are simulated — ',
    homeHonest2b: 'the algorithms are real',
    homeHonest2c:
      ': BPE merges are counted from a real corpus, attention runs the full Q/K/V projection, scaling, causal masking and softmax, and the head patterns are the ones repeatedly reported in the literature. Build intuition here, then verify against a real-weight project.',
    homeOrderTitle: 'Suggested order',
    homeOrder:
      'Tokenizer → Embeddings & Positional Encoding → Multi-Head Attention → Sampling → KV Cache → Block data flow. The first three are the foundation; the last three cover how a trained model is actually used to generate, which is what engineering interviews love to ask.',

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

    realSection: 'Real model (optional)',
    realIntro:
      'By default you are looking at a deterministic simulation. Press the button to actually run a small model in your browser and see attention from real weights. First load downloads about 100 MB.',
    loadModel: 'Load real model',
    loading: 'Loading',
    unload: 'Back to simulation',
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

/**
 * 模块注册表
 * 新增一个模块只需要：在 modules/ 下写好组件，然后在这里加一项（中英两份文案）。
 * status = 'done' 表示已实现，'soon' 表示规划中（侧边栏会显示但不可点）。
 */
import type { ComponentType } from 'react'
import { TokenizerModule } from './tokenizer/TokenizerModule'
import { EmbeddingModule } from './embedding/EmbeddingModule'
import { AttentionModule } from './attention/AttentionModule'
import { GenerationModule } from './generation/GenerationModule'
import { KVCacheModule } from './kvcache/KVCacheModule'
import { FlowModule } from './flow/FlowModule'

export interface ModuleMeta {
  id: string
  name: string
  sub: string
  nameEn: string
  subEn: string
  blurb: string
  blurbEn: string
  status: 'done' | 'soon'
  Component: ComponentType | null
}

export const MODULES: ModuleMeta[] = [
  {
    id: 'tokenizer',
    name: '① 分词',
    sub: '文本 → token id',
    nameEn: '① Tokenization',
    subEn: 'text → token ids',
    status: 'done',
    Component: TokenizerModule,
    blurb: '真实的 BPE 算法，合并表从语料现场学出来，带逐步合并动画。',
    blurbEn:
      'A real BPE implementation: merges are counted from the corpus on the fly, with a step-by-step merge animation.',
  },
  {
    id: 'embedding',
    name: '② 嵌入与位置编码',
    sub: '给 token 坐标和座次',
    nameEn: '② Embeddings & Positional Encoding',
    subEn: 'coordinates and seat numbers',
    status: 'done',
    Component: EmbeddingModule,
    blurb: '嵌入向量热图、正弦位置编码、位置相似度、RoPE 旋转演示、PCA 降维。',
    blurbEn:
      'Embedding heatmap, sinusoidal encoding, positional similarity, RoPE rotation demo, PCA projection.',
  },
  {
    id: 'attention',
    name: '③ 多头自注意力',
    sub: '模型在看哪里',
    nameEn: '③ Multi-Head Attention',
    subEn: 'what the model looks at',
    status: 'done',
    Component: AttentionModule,
    blurb: '主力模块：可切层切头的注意力热力图，本层所有头一览，因果掩码与温度实时可调。',
    blurbEn:
      'Flagship module: per-layer / per-head attention heatmaps, all-heads overview, live causal-mask and temperature controls.',
  },
  {
    id: 'generation',
    name: '④ 逐 token 生成与采样',
    sub: '温度 / top-k / top-p',
    nameEn: '④ Token-by-Token Generation',
    subEn: 'temperature / top-k / top-p',
    status: 'done',
    Component: GenerationModule,
    blurb: '把抽签前的每一步摊开：原始概率 → 温度 → top-k → top-p → 抽中谁。',
    blurbEn:
      'Every step before the draw, laid open: raw p → temperature → top-k → top-p → what got picked.',
  },
  {
    id: 'kvcache',
    name: '⑤ KV Cache 加速',
    sub: '用显存换计算量',
    nameEn: '⑤ KV Cache',
    subEn: 'trading memory for compute',
    status: 'done',
    Component: KVCacheModule,
    blurb: '量化「省了多少 FLOPs、付了多少显存」，含 GQA、batch、精度的影响曲线。',
    blurbEn: 'Quantifies FLOPs saved and memory paid, with GQA, batch size and precision curves.',
  },
  {
    id: 'flow',
    name: '⑥ Transformer 层间数据流',
    sub: '一个 block 里发生了什么',
    nameEn: '⑥ Block Data Flow',
    subEn: 'what happens inside one block',
    status: 'done',
    Component: FlowModule,
    blurb: '可逐子步推进的 Block 结构动画，配表示热图、残差贡献与层间相似度。',
    blurbEn:
      'Step-by-step animation through one block, with state heatmap, residual contribution and layer similarity.',
  },
]

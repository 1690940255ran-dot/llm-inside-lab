/**
 * 模块注册表
 * 新增一个模块只需要：在 modules/ 下写好组件，然后在这里加一项。
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
  status: 'done' | 'soon'
  Component: ComponentType | null
  blurb: string
}

export const MODULES: ModuleMeta[] = [
  {
    id: 'tokenizer',
    name: '① 分词',
    sub: '文本 → token id',
    status: 'done',
    Component: TokenizerModule,
    blurb: '真实的 BPE 算法，合并表从语料现场学出来，带逐步合并动画。',
  },
  {
    id: 'embedding',
    name: '② 嵌入与位置编码',
    sub: '给 token 坐标和座次',
    status: 'done',
    Component: EmbeddingModule,
    blurb: '嵌入向量热图、正弦位置编码、位置相似度、RoPE 旋转演示、PCA 降维。',
  },
  {
    id: 'attention',
    name: '③ 多头自注意力',
    sub: '模型在看哪里',
    status: 'done',
    Component: AttentionModule,
    blurb: '主力模块：可切层切头的注意力热力图，所有头一览，温度/掩码/距离衰减可调。',
  },
  {
    id: 'generation',
    name: '④ 逐 token 生成与采样',
    sub: '温度 / top-k / top-p',
    status: 'done',
    Component: GenerationModule,
    blurb: '把抽签前的每一步摊开：原始概率 → 温度 → top-k → top-p → 抽中谁。',
  },
  {
    id: 'kvcache',
    name: '⑤ KV Cache 加速',
    sub: '用显存换计算量',
    status: 'done',
    Component: KVCacheModule,
    blurb: '量化省了多少 FLOPs、又付了多少显存，含 GQA 与精度的影响。',
  },
  {
    id: 'flow',
    name: '⑥ Transformer 层间数据流',
    sub: '一个 block 里发生了什么',
    status: 'done',
    Component: FlowModule,
    blurb: '可逐子步推进的 Block 结构动画：LN → Attention → 残差 → FFN → 残差，配表示热图与层间相似度。',
  },
]

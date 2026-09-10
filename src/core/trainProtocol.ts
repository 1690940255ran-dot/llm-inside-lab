/**
 * 训练线程的消息协议
 *
 * 训练循环必须放在 Web Worker 里：一次训练要跑上千步、十几秒到几十秒，
 * 放在主线程上会把页面彻底冻住（连「停止」按钮都点不动）。
 * 这里把两个方向的消息类型集中定义，主线程和 Worker 共用，改一处不会漏另一边。
 */
import type { ModelConfig, TrainConfig } from './minigpt'

/** 主线程 → Worker */
export type MainToWorker =
  | {
      type: 'init'
      /** 原始语料文本 */
      corpus: string
      model: ModelConfig
      train: TrainConfig
      /** 词表上限（超出部分映射到 UNK） */
      maxVocab: number
    }
  | { type: 'start' }
  | { type: 'stop' }
  /** 回到第 0 步（权重重新按种子初始化） */
  | { type: 'reset' }

/** Worker → 主线程 */
export type WorkerToMain =
  | {
      type: 'ready'
      vocabSize: number
      /** 词表里的字符（UNK 用 ␀ 表示） */
      vocabChars: string[]
      /** 因为超出词表上限而被折成 UNK 的字符种类数 */
      droppedKinds: number
      corpusChars: number
      paramCount: number
      flopsPerToken: number
      paramBreakdown: { name: string; size: number }[]
      /** 初始验证 loss（用于画曲线起点） */
      initialLoss: number
      textPerStepMs: number
    }
  | {
      type: 'progress'
      step: number
      totalSteps: number
      /** 最新一步的训练 loss（单个 batch，噪声大，仅供「抖动感」参考） */
      batchLoss: number
      gradNorm: number
      lr: number
      elapsedMs: number
      /** 固定验证集上的 loss 曲线 */
      trace: { step: number; loss: number }[]
      samples: { step: number; text: string }[]
      /** 注意力快照的版本号；-1 表示还没有 */
      attnStep: number
      attnVersion: number
      /** 只在版本变化时携带，避免每次进度都传 4000 个数 */
      attn?: number[][][]
      /** 每个头在「关注前一个位置」上的平均权重 */
      prevToken: number[]
      done: boolean
    }
  | { type: 'error'; message: string }

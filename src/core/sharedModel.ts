import { trainBPE } from './bpe'

/**
 * 全局共用一个分词模型（模块加载时训练一次，约几十毫秒），
 * 各个可视化模块都拿它来把用户输入切成 token。
 */
export const SHARED_MODEL = trainBPE(undefined, 200)

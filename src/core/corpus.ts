/**
 * 内置训练语料
 *
 * 分词模块要演示的是「真实的 BPE 算法」，而不是硬编码的假结果。
 * BPE 的合并表是从语料里「学」出来的，所以必须给它一份语料。
 * 这里内置一份中英混合的小语料（约 2 KB），页面加载时几百次合并只需几十毫秒。
 * 用户也可以在界面里粘贴自己的语料重新训练，观察词表会怎么变。
 */
export const DEFAULT_CORPUS = [
  // —— 中文部分：重复出现的高频术语会被 BPE 学成两字词 ——
  '大语言模型通过预测下一个 token 来学习语言规律。',
  '注意力机制让模型能够关注输入中不同位置的信息。',
  '每个注意力头会学习不同的模式，比如关注前一个词或者关注句首。',
  '多头注意力把每个头的输出拼接起来，再做一次线性变换。',
  '位置编码把顺序信息注入到词向量之中，让模型知道谁在前谁在后。',
  '分词器把文本切分成子词单元，子词平衡了词表大小和序列长度。',
  '嵌入向量把离散的 token 映射到连续的向量空间里。',
  '残差连接和层归一化让很深的网络也能够稳定训练。',
  '前馈网络对每一个位置单独做一次非线性变换。',
  '模型参数越多，需要的训练数据和算力也越多。',
  '梯度下降通过反向传播来更新模型的每一个参数。',
  '过拟合是指模型在训练集上表现很好，但在测试集上表现很差。',
  '温度参数控制生成时的随机性，温度越高分布越平滑，输出越发散。',
  '采样策略决定了每一个 token 是怎么从概率分布里选出来的。',
  '键值缓存保存已经算过的键和值，避免重复计算从而加速推理。',
  '预训练让模型学会通用的语言规律，微调让模型适配具体任务。',
  '注意力分数经过缩放和归一化之后变成一组权重，用来加权求和值向量。',
  '同一个词在不同的上下文里会得到不同的表示，这就是上下文相关的嵌入。',
  '模型的上下文长度决定了它一次最多能看多少 token。',
  '词表越大，同样的文本需要的 token 越少，但嵌入矩阵也越大。',
  '中文通常一个字就是一个 token，所以同样的意思中文会消耗更多的 token。',
  '推理阶段生成每一个 token 都需要一次完整的前向传播。',
  '训练数据是模型能力的上限，数据质量比数据规模更重要。',
  '注意力权重的熵越低，说明这个头越集中地关注少数几个位置。',
  '深层网络里靠后的层往往捕捉更抽象的语义，靠前的层更关注局部和位置。',

  // —— 英文部分：BPE 会自然地学出 "the" "ing" "atten" 这类子词 ——
  'the model predicts the next token given all previous tokens',
  'attention allows the model to focus on different positions of the input',
  'each attention head learns a different pattern in the sequence',
  'positional encoding injects order information into the embedding vector',
  'the tokenizer splits text into subword units using byte pair encoding',
  'softmax normalizes the attention scores into a probability distribution',
  'the key query and value are computed from the same hidden state',
  'multi head attention concatenates all heads and projects the result',
  'residual connections and layer normalization make deep networks trainable',
  'the temperature parameter controls the randomness of generation',
  'the kv cache stores keys and values to avoid repeated computation',
  'the transformer stacks many identical layers with attention and feed forward',
  'pretraining teaches language patterns and finetuning adapts to a task',
  'the embedding maps a discrete token into a continuous vector space',
  'gradient descent updates the model parameters through backpropagation',
  'overfitting means good performance on training data but poor generalization',
  'a longer context window lets the model see more tokens at once',
].join('\n')

/** 界面上给用户的示例文本（覆盖中英文、标点、数字，方便观察边界情况） */
export const SAMPLE_TEXTS: { label: string; labelEn: string; text: string }[] = [
  { label: '中英混排', labelEn: 'Mixed zh/en', text: '注意力机制让模型关注输入中不同位置的信息。' },
  { label: '英文长句', labelEn: 'English', text: 'the model predicts the next token given all previous tokens' },
  { label: '指代消解', labelEn: 'Coreference', text: '猫追着老鼠跑，因为它饿了。' },
  { label: '数字与符号', labelEn: 'Digits', text: 'GPT-4 有 1.8T 参数，上下文 128K token。' },
]

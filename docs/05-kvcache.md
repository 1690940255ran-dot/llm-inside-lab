# 05 · KV Cache：用显存换计算量

> 对应模块：站点左侧「⑤ KV Cache 加速」

## 一句话

解码时每生成一个 token，注意力都要看一遍前面所有 token 的 K 和 V——但这些上一次已经算过了。存起来复用，就能把每步的代价从 **O(s²)** 降到 **O(s)**。代价是显存随序列长度线性增长。

## 问题从哪来

自回归生成是这样的：

```
第 1 步：输入 [x₁]                    → 预测 x₂
第 2 步：输入 [x₁, x₂]                → 预测 x₃
第 3 步：输入 [x₁, x₂, x₃]            → 预测 x₄
...
```

朴素实现下，第 `t` 步要把长度为 `n + t` 的整个序列重新跑一遍前向。但注意：

- 第 `t` 步真正需要的，只是**最后一个位置的输出**（用来预测下一个 token）
- 前面所有位置的 `K` 和 `V`，在第 `t-1` 步已经算过一遍了

所以每一步都在做大量重复计算。

## 算一下省了多少

设 `n` = 提示长度，`m` = 生成长度，`d = d_head × 头数`，`L` = 层数。只统计注意力部分（每个矩阵乘 2 FLOPs/MAC，QKᵀ 和 AV 两次共 4 倍）：

**无缓存**：第 `t` 步要跑一整次前向，序列长 `s = n + t`

```
单步 = 4 · s² · d · L
累计 ≈ Σ(t=1..m) 4 · (n+t)² · d · L   →  O(m · n²)
```

**有缓存**：

```
prefill（一次性算完提示） = 4 · n² · d · L
decode 第 t 步（只算新 token 的 Q）= 4 · (n+t) · d · L
累计 = 4n²dL + Σ(t=1..m) 4(n+t)dL   →  O(n² + m · n)
```

复杂度从 **O(m·n²)** 降到 **O(n² + m·n)**。生成越长、提示越长，收益越夸张。

在站点的「累计 FLOPs」曲线里，两条线的间距会越拉越大——就是这个二次 vs 一次的差别。

## 付了多少显存

每一层、每一个 token，要缓存一份 K 和一份 V：

```
每 token 占用 = 2（K 和 V）× L × KV头数 × d_head × 每参数字节数
总占用        = 每 token 占用 × (n + m) × batch
```

注意这是**线性**的，但乘的东西很多：层数 × KV头数 × d_head × 序列长度 × batch。

举个具体的（Qwen2.5-7B 配置：L=28，KV头数=4，d_head=128，fp16）：

```
每 token = 2 × 28 × 4 × 128 × 2 字节 = 57,344 字节 ≈ 56 KB
128K 上下文 × batch 1  = 56 KB × 131072 ≈ 7.3 GB
128K 上下文 × batch 32 = ≈ 234 GB   ← 放不下
```

**这就是为什么长上下文服务这么贵。** KV Cache 经常比模型权重本身还大（7B 模型 fp16 权重约 15 GB）。

## GQA：最划算的一刀

注意公式里的"KV 头数"。传统 MHA（多头注意力）里，KV 头数 = Query 头数。但**多个 Query 头完全可以共享同一组 K/V**——这就是 GQA（Grouped Query Attention）。

| 架构 | Query 头数 | KV 头数 | KV Cache |
| --- | --- | --- | --- |
| MHA（GPT-3） | 96 | 96 | 100% |
| GQA（Qwen2.5-7B） | 28 | 4 | 14% |
| MQA | 32 | 1 | 3% |

**质量几乎不损失，显存直接除以一个系数。** 在站点里把 KV 头数从 28 拖到 4，看"KV 总显存"和"GQA 节省"两个数字的变化。

这就是新模型敢把上下文做到 128K 的底气。

## 其他省显存的手段

**量化 KV**：fp16 → fp8 → int4，直接按比例下降。站点里给了三档切换。质量损失通常可以接受。

**PagedAttention**（vLLM 的核心）：借鉴操作系统虚拟内存的分页思想，把 KV Cache 切成固定大小的块按需分配，解决碎片问题。显存利用率从 20~40% 提升到接近 100%。

**滑动窗口 + sink token**（StreamingLLM）：只保留最近的 K 个 token 加上首 token。前面提到的 attention sink 现象正是它的理论基础——首 token 不能丢，丢了模型就崩。

**MQA**：极端版 GQA，所有 Query 头共享 1 组 KV。显存最省，质量损失略大。

## 一个容易忽略的点：prefill 和 decode 是两种负载

这是推理优化的核心洞察：

| | prefill | decode |
| --- | --- | --- |
| 输入 | 整个提示（几千 token） | 1 个 token |
| 瓶颈 | 算力（compute-bound） | 显存带宽（memory-bound） |
| 并行度 | 高，能吃满 GPU | 极低，主要时间花在读 KV |
| 优化方向 | 算子融合、FlashAttention | 增大 batch、量化 KV |

**decode 阶段是 memory-bound 的**：每生成一个 token，都要把整个 KV Cache 从显存读一遍。所以 KV Cache 不只是占地方，它直接决定了生成速度。

这也解释了为什么**连续批处理（continuous batching）**这么重要——既然单序列 decode 吃不饱 GPU，那就把很多请求凑一起。

在站点里注意「整体加速比」这个数：它包含了无法省的 prefill，所以不等于纯 decode 的加速倍数。把生成长度 `m` 调大，加速比会迅速上升——因为可省的 decode 部分占比变高了。

## 常见误解

**"KV Cache 是可选的优化"** —— 在现代推理框架里它是默认项，不开的话长文本生成慢到不可接受。真正的"可选"是怎么管理它（分页、量化、窗口）。

**"显存只和模型大小有关"** —— 服务场景下，KV Cache 经常是显存的主要占用者，尤其在高并发 + 长上下文时。

**"量化只影响模型权重"** —— KV Cache 往往比权重量化收益更大，因为它更大、访存更频繁。

## 在站点上怎么玩

1. 按 Qwen2.5-7B 预设，看当前配置的显存数字
2. 把 KV 头数从 4 拖到 28（变成 MHA），看显存翻几倍
3. 把 batch 从 1 拖到 32，看显存怎么线性暴涨
4. 生成长度从 128 拖到 1024，看"整体加速比"上升
5. 精度从 FP16 切到 INT4，看显存除以 4
6. 观察显存曲线：它是**线性**的，但斜率被 batch 放大

## 延伸

- vLLM / PagedAttention：[Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- GQA：[GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- StreamingLLM：[Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)
- FlashAttention：[FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)

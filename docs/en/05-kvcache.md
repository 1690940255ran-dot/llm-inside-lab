# 05 · KV Cache: trading memory for compute

> Module on the site: sidebar **⑤ KV Cache**

## In one sentence

When decoding, every new token requires attention to look at all preceding tokens' K and V — but those were already computed in the previous step. Stash them and reuse, and the per-step cost drops from **O(s²)** to **O(s)**. The price is memory that grows linearly with sequence length.

## Where the problem comes from

Autoregressive generation works like this:

```
step 1: input [x₁]                    → predict x₂
step 2: input [x₁, x₂]                → predict x₃
step 3: input [x₁, x₂, x₃]            → predict x₄
...
```

In a naïve implementation, step `t` reruns the entire forward pass over the sequence of length `n + t`. But note:

- at step `t` the model genuinely needs only the **last position's output** (for predicting the next token)
- all preceding positions' `K` and `V` were already computed at step `t-1`

So every step is doing a huge amount of repeated work.

## How much is saved

Set `n` = prompt length, `m` = generation length, `d = d_head × num_heads`, `L` = layers. Count attention only (each matmul is 2 FLOPs/MAC; the two of QKᵀ and AV together are 4×):

**Without cache:** at step `t` you do a full forward with sequence length `s = n + t`

```
per-step = 4 · s² · d · L
total ≈ Σ(t=1..m) 4 · (n+t)² · d · L   →  O(m · n²)
```

**With cache:**

```
prefill (compute the prompt once) = 4 · n² · d · L
decode at step t (only the new token's Q) = 4 · (n+t) · d · L
total = 4n²dL + Σ(t=1..m) 4(n+t)dL    →  O(n² + m · n)
```

Complexity drops from **O(m · n²)** to **O(n² + m · n)**. The longer the generation and the longer the prompt, the bigger the win.

In the site's "cumulative FLOPs" chart, the gap between the two curves just keeps widening — that is the difference between quadratic and linear.

## What memory costs

For every layer and every token you stash one K and one V:

```
per-token cost = 2 (K and V) × L × KV heads × d_head × bytes per param
total cost     = per-token cost × (n + m) × batch
```

Note the **linear** scaling, but multiplied out by a lot: layers × KV heads × d_head × sequence × batch.

For concrete numbers (Qwen2.5-7B config: L = 28, KV heads = 4, d_head = 128, fp16):

```
per token = 2 × 28 × 4 × 128 × 2 bytes ≈ 56 KB
128K context × batch 1   ≈ 7.3 GB
128K context × batch 32  ≈ 234 GB   ← does not fit
```

**This is why long-context serving is so expensive.** KV cache is routinely larger than the model weights themselves (a 7B model in fp16 is about 15 GB).

## GQA: the most cost-effective cut

Look at "KV heads" in the formula. In classic MHA (multi-head attention) the number of KV heads equals the number of Query heads. But **many Query heads can perfectly well share the same K/V** — that is GQA (Grouped Query Attention).

| Architecture | Query heads | KV heads | KV cache |
| --- | --- | --- | --- |
| MHA (GPT-3) | 96 | 96 | 100% |
| GQA (Qwen2.5-7B) | 28 | 4 | 14% |
| MQA | 32 | 1 | 3% |

**Quality barely moves, memory drops by a factor.** On the site, drag KV heads from 28 down to 4 and watch the "total KV memory" and "GQA savings" numbers.

This is exactly why new models dare ship 128K context.

## Other memory-saving tricks

**KV quantisation:** fp16 → fp8 → int4 — drops memory proportionally. Quality loss is usually acceptable. The site offers three switchable tiers.

**PagedAttention** (the core of vLLM): borrows the OS virtual-memory idea of paging, slicing the KV cache into fixed-size blocks allocated on demand. Solves fragmentation; utilisation jumps from 20–40% to close to 100%.

**Sliding window + sink tokens** (StreamingLLM): keep only the latest K tokens plus the first token. The attention-sink phenomenon above is the theoretical foundation — the first token cannot be dropped without breaking the model.

**MQA:** the extreme end of GQA — every Query head shares one single KV. Maximum memory savings, somewhat larger quality cost.

## A subtlety that often gets missed: prefill and decode are two different workloads

This is the core insight of inference optimisation:

| | prefill | decode |
| --- | --- | --- |
| Input | the whole prompt (thousands of tokens) | one token |
| Bottleneck | compute | memory bandwidth |
| Parallelism | high — can fill the GPU | tiny — most time is spent reading KV |
| Optimisation | operator fusion, FlashAttention | bigger batch, KV quantisation |

**Decode is memory-bandwidth-bound:** every token generated reads the entire KV cache once. So KV cache is not just "memory that fits"; it directly determines generation speed.

That is also why **continuous batching** matters so much. Since single-sequence decode underutilises the GPU, you pack many requests together.

On the site, pay attention to the "overall speedup" number: it includes the non-cached prefill, so it does not equal the pure-decode speedup. Drag the generation length `m` up and watch the speedup climb — because the cacheable decode portion dominates.

## Common misconceptions

**"KV cache is an optional optimisation."** — In modern inference frameworks it is on by default. Without it, long-text generation is unbearably slow. What is genuinely optional is *how you manage* the cache (paging, quantisation, sliding window).

**"Memory is just the model size."** — Under serving workloads, KV cache is often the dominant memory consumer, especially at high concurrency + long context.

**"Quantisation only affects model weights."** — KV cache often benefits from quantisation even more than weights: it is bigger and is read more frequently.

## How to play with this on the site

1. Use the Qwen2.5-7B preset, read off the current configuration's memory.
2. Drag KV heads from 4 up to 28 (becomes MHA) and watch memory explode.
3. Drag batch from 1 up to 32 and watch memory scale linearly.
4. Drag generation length from 128 up to 1024 — "overall speedup" climbs.
5. Switch precision from FP16 to INT4 and watch memory divide by 4.
6. Inspect the memory curve: it is **linear**, but the slope is amplified by batch.

## Going further

- vLLM / PagedAttention: [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- GQA: [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- StreamingLLM: [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)
- FlashAttention: [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
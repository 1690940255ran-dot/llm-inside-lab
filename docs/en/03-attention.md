# 03 · Multi-Head Attention: where is the model actually looking?

> Module on the site: sidebar **③ Multi-Head Attention** — the site's flagship module

## In one sentence

Every position carries a question (Query), compares it against every position's nameplate (Key), and weighted-averages everyone's content (Value) according to the match. **Multiple heads do this in parallel; each head looks from a different angle.**

## The full formula

```
Q = X · W_Q      [n, d_model] → [n, d_model]
K = X · W_K
V = X · W_V

scores  = Q · Kᵀ / √d_k       every query gives every key a raw score
scores += mask (future → -∞)  a causal LM cannot peek ahead
weights = softmax(scores / T)  normalised into "how do I split my attention budget"
output  = weights · V          pull information back according to the weights
```

The multi-head version slices `d_model` into `h` chunks of `d_k = d_model / h` each, runs each chunk independently, then concatenates and linearly projects.

## Why divide by √d_k

This is the detail that gets asked about in every interview, and for good reason.

Assume each component of `q` and `k` is an i.i.d. zero-mean unit-variance random variable. Then `q·k` is the sum of `d_k` such products, so its variance is `d_k` and its standard deviation is `√d_k`.

**Higher dimension → larger-magnitude dot products.** At `d_k = 512`, a typical dot product is around ±23.

Those numbers then go into softmax. Softmax is extremely sensitive to scale — once the inputs are large, the output gets pushed to 0/1 corners, gradients vanish, and **training stalls**.

Dividing by `√d_k` pulls the variance back down to 1 and lets softmax operate in its comfortable zone.

On the site, drag the "temperature T" slider down to 0.3 and then up to 2.0. You are watching what happens *without* the `√d_k` scaling — the distribution gets pushed to extremes and almost everything collapses onto one or two cells.

## Why multiple heads

A single-head attention has one fundamental limitation: **it can only express one kind of relationship.**

An `n × n` attention matrix is "averaging-ist": each row has to normalise to sum 1, so if a position needs to attend to "the previous word" and "a semantically related word" at the same time, a single head can only split the weight between them and end up with a blurry compromise.

Once you have multiple heads, each one only has to specialise:

- some heads watch only the previous token (bigram-like)
- some heads stare at the first token (the attention sink, see below)
- some heads latch onto punctuation — those positions often summarise the whole sentence
- some heads do content matching (common in deeper layers, used for coreference)

**In the site's "all heads of this layer" overview, you will see several thumbnails with completely different patterns.** That is what multiple heads is for; no text explanation beats seeing it.

## Attention sink: counter-intuitive but critical

Real large models exhibit a robust phenomenon: **a large fraction of attention weight is dumped onto the first token (usually `<BOS>`), even when that token is semantically meaningless.**

You can derive it directly from softmax's properties: the output of softmax has to sum to 1, which means **every row has to spend its entire attention budget.** If a particular position has "nothing worth looking at", the model still has to spend the budget somewhere, and the first token is the universal fallback — it is visible to every position under the causal mask.

The site surfaces a "first-token sink ratio" stat. Switch between heads and watch it change.

This phenomenon has huge engineering value: **StreamingLLM** is built on top of it to enable infinite-length context. As long as you permanently keep the first token's KV, plus a sliding window of recent tokens, the model can stably process sequences far longer than it was trained on — no retraining needed.

## Are head behaviours learned or designed?

Learned from data, but **what gets learned is highly reproducible**. The literature keeps reporting the same canonical patterns:

| Pattern | What it looks like |
| --- | --- |
| Previous token | only looks at the previous word — equivalent to a bigram count |
| First token / sink | piles weight on the sentence's first token |
| Delimiter | watches commas/periods — those positions often aggregate whole-sentence summaries |
| Positional / diagonal | diagonal band — pays attention to itself and its neighbours |
| Content matching | semantic or spelling similarity (common in deeper layers) |
| Sparse induction | fires only for very specific patterns — looks like keyword retrieval |

The site's simulation injects these known patterns as "head-behaviour priors", so what you see is not random noise — it is a **reproduction of patterns repeatedly reported in the literature**.

## Why the causal mask is not optional

We train on "predict the next token". If position `i` is allowed to see position `i+1`, the model can simply copy the answer, the training objective collapses, and whatever it learns is useless at inference.

So the causal mask (upper triangle set to -∞) is not "a constraint bolted on for autoregressive decoding". **It is the mathematical consequence of the training objective itself.**

On the site, toggle the causal mask off and on and watch the top-right triangle of the heatmap: off, it lights up; on, it becomes hatched (masked).

## Complexity: attention's original sin

```
time:  O(n² · d)      every position compares against every position
space: O(n²)          the n×n attention matrix has to be materialised
```

Double the sequence, quadruple the compute and the memory. That is the core bottleneck of long context, and it is the entire motivation behind FlashAttention, sparse attention, linear attention and the rest.

To get a feel for the scale: at a 128K context length, the attention matrix alone has `128000² ≈ 1.64 × 10¹⁰` elements. In fp16 that is 32 GB — **a single layer does not fit.**

## Common misconceptions

**"Attention weight = importance."** — Not quite. Attention only represents *where information flows from*, not *what is important*. A high attention weight does not mean that token contributes most to the prediction; this is an open debate in interpretability.

**"More heads is always better."** — Clear diminishing returns. With too many heads, many of them learn redundant patterns; and each head's dimension `d_k` shrinks, which actually reduces per-head expressivity.

**"Attention is where the model 'understands' language."** — More precisely: attention handles **information routing** (who pulls from whom); the FFN handles **information processing** (what to do with what was pulled). Two different jobs.

## How to play with this on the site

1. Open "all heads of this layer" first, find the two heads with the most different patterns, click into each.
2. Slide through the layers: shallow layers are positional/local (diagonals, previous-token), deep layers are semantic/global.
4. Drag temperature to 0.3 and then 2.0 — feel how sharpness of the distribution changes.
4. Toggle off the causal mask and watch the top-right corner "come alive".
5. Drag distance decay down to 0 — attention falls apart into a featureless smear.
6. Click on any row and inspect this token's full attention distribution and top-8 targets.

## Going further

- Original paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Attention sink: [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)
- Head-behaviour analysis: [A Multiscale Visualization of Attention in the Transformer](https://arxiv.org/abs/1906.05714)
- Visualisation toolkit: [BertViz](https://github.com/jessevig/bertviz)
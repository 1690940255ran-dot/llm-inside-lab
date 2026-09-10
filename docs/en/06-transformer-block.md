# 06 · The Transformer Block: the residual highway and the FFN

> Module on the site: sidebar **⑥ Block Data Flow**

## In one sentence

The shape game inside one block is hilariously monotonous: **input `[n, d]`, output `[n, d]`, everything in between is `[n, d]`**. What makes it powerful is the residual highway that runs all the way through — each layer only *adds* a small delta on top of it, it does not *replace* it.

## The pre-norm structure

The modern mainstream form (LLaMA / Qwen / Mistral):

```
h ← h + Attention( LayerNorm(h) )
h ← h + FFN( LayerNorm(h) )
```

The original Transformer used post-norm:

```
h ← LayerNorm( h + Attention(h) )
```

Same bracket position, completely different behaviour. **In pre-norm the residual path is "clean"** — there is an unbroken straight line from input to output that touches no normalisation at all, so gradients flow back along it undamaged. That is why pre-norm models stack comfortably into the hundreds of layers, while post-norm gets twitchy at depth and demands careful warmup.

Nobody uses post-norm in production any more.

## The six sub-steps of one block

The site's animation splits one layer into six steps. Each one is worth looking at on its own:

1. **LayerNorm** — pull each row back to mean 0, variance 1. Stabilises numbers and keeps the attention score scale predictable.
2. **Attention** — the only place in the block where tokens actually exchange information with each other.
3. **+ residual** — the attention output is *added* to the original representation (not substituted!).
4. **LayerNorm** — renormalise once more before entering the FFN.
5. **FFN** — expand to 4× width, pass through GELU, shrink back. Each position computes independently; tokens do not talk here.
6. **+ residual** — add again; one block is done.

**The shape never changes.** Only inside the FFN does it briefly become `[n, 4d]`. That detail matters: because input and output have the same shape, blocks can be stacked without limit, and depth becomes a plain hyperparameter.

## The residual-stream view

If you draw the residual connection as a straight line, you get a clearer picture than any textbook block diagram:

```
h₀ ──────────────────────────────────────────▶ h_L
   │                │                │
   └─+ Attn₁(h₀)    └─+ Attn₂(h₁)    └─+ ...
```

**Information is not "passed" between layers; it flows along a highway, and each layer stacks a small extra thing on top.**

That view explains a lot:

- why deep models train: the gradient travels along the highway undamaged; it does not have to wind through any transformation
- why layer pruning works: if one layer's delta is small, removing it does not cost much
- why "logit lens" works: projecting a middle layer's representation straight to the vocabulary already produces meaningful predictions, because the information has been continuously usable all the way down

In the site's "residual contribution" chart you will see that both branches are **far smaller than 1**. That is the quantitative evidence for "only adding a little".

## FFN is the parameter elephant

Count a single block's parameters (ignoring biases and LayerNorm):

```
attention: W_Q, W_K, W_V, W_O each d×d →  4d²
FFN:       W₁ is d×4d, W₂ is 4d×d    →  8d²
total                                 →  12d²
```

**The FFN is two thirds of the parameters.** With GQA shrinking the KV projections, that ratio goes even higher.

A widespread view in the literature:

- **attention handles information routing** (who pulls from whom)
- **FFN handles knowledge storage and processing** (what to do with what was pulled)

Evidence includes: knowledge-editing methods (ROME, MEMIT) all locate themselves in FFN layers; FFN neurons exhibit clearly interpretable features ("this neuron is sensitive to parentheses in code").

In the site's residual-contribution bars, FFN's magnitude usually clearly dominates the attention branch — most of what each layer "shoves into the bag" comes from the FFN.

## Why FFN expands to 4× first

Attention is essentially a linear weighted average (post-softmax times V); its expressive power is limited. The FFN provides **non-linear transform capacity**:

```
FFN(x) = GELU(x·W₁) · W₂      [n, d] → [n, 4d] → [n, d]
```

Expanding is like lifting the representation into a higher-dimensional space, doing a non-linear "lookup" there, and projecting back. The higher the dimension, the richer the set of feature combinations you can express.

The 4× is an empirical sweet spot. Modern variants use SwiGLU, which introduces a third matrix and typically chooses an intermediate dimension of `8/3 × d` to keep the parameter count comparable.

## Inter-layer similarity: deep layers are fine-tuning

The site's "inter-layer representation similarity" matrix has a striking structure: **the bottom-right corner is almost completely white** (similarity close to 1).

That means later layers' representations look very much like each other. The deep layers are making small adjustments, not rebuilding from scratch.

Implications:

- **layer pruning has theoretical backing**: if layer 30 and layer 31 have similarity 0.99, dropping one is cheap
- **early exit works**: easy samples do not need every layer
- **tuning deep layers is more effective**: many LoRA experiments find that just the last few layers deliver most of the gain

You will also notice: the embedding layer (row 0) has relatively low similarity to every other layer — it has not been processed by anything yet.

## Common misconceptions

**"More layers = stronger model."** — Clear diminishing returns. The inter-layer similarity chart already shows why: deeper layers contribute less. With the same parameter budget, deepening vs widening is a trade-off that needs to be tested.

**"Residual connections are there to fight vanishing gradients."** — That is the consequence, not the essence. The real point is that **each layer only has to learn a residual delta**, turning "fit a mapping from scratch" into "fit a near-zero correction", which is dramatically easier.

**"LayerNorm just stabilises numbers."** — In pre-norm it does one more thing: it keeps the residual-branch inputs at a stable scale, so deep stacking does not blow up the representation norm.

## How to play with this on the site

1. Click "Play" and let the animation walk through one full block's six sub-steps.
2. Pause on the `Attention` step, observe the representation heatmap; step forward to `+ residual` and compare.
3. Pause on `FFN` and notice the shape label changes to `[n, 4d]`.
4. Look at the "residual contribution" bars and compare the attention-branch vs FFN-branch magnitudes.
5. Look at the "inter-layer similarity" matrix's bottom-right corner — how white is it?
6. Drop the layer count to 8 and see whether the block structure in the similarity matrix becomes more obvious.

## Going further

- Original Transformer: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- pre-norm analysis: [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- Residual-stream view: [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- FFN as knowledge storage: [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)
- SwiGLU: [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
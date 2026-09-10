# 07 · Mixture of Experts: why experts starve, and why the two remedies are not the same

> Module on the site: sidebar **⑦ Mixture of Experts**

## In one sentence

MoE trades "huge total parameter count, tiny per-token activation" for cheap compute, and the price is that **router load skews by nature**. The two ways of fixing it (aux-loss gradient vs load-feedback bias) look like they do the same job but differ enormously — because the former's gradient with respect to a dead expert is **exactly zero**.

## The parameter ledger: what MoE actually saves

Let `dense` be the dense part (token embeddings + per-layer attention + LayerNorm) and `E` the FFN parameters of one expert:

```
total  = dense + (nExperts + nShared) · E · nLayers
active = dense + (topK     + nShared) · E · nLayers
```

**Memory is sized by `total`; compute is sized by `active`.** These two numbers differ by an order of magnitude, and that gap is the entire business case for MoE.

You can check the formula against two published models on the site (within 5%):

| Model | Config | Published total/active | Computed here |
| --- | --- | --- | --- |
| Mixtral 8x7B | d=4096, L=32, FFN=14336, top-2 of 8 | 46.7B / 12.9B | 47.38B / 13.55B |
| Qwen3-30B-A3B | d=2048, L=48, FFN=768, top-8 of 128 | 30.5B / 3.3B | 30.11B / 2.93B |

Note how **fine-grained** Qwen3's experts are (FFN width 768, where a dense model of that size would use 4d). More, thinner experts means a larger combinatorial space for routing — but load balancing gets harder in exact proportion. The two are inseparable.

## What routing is

It is a tiny attention:

```
p(i,e) = softmax_e( cos(x_i, router_e) / T + b_e )
assign_i = TopK_e p(i,e)
y_i = Σ_{e ∈ assign_i} w(i,e) · Expert_e(x_i)      w = gate renormalised over the picked experts
```

`router_e` is each expert's router vector (in a real MoE, row e of the router linear layer); `b_e` is a logit bias. **These two groups of parameters are deliberately separate**: in a real MoE the router is its own small linear layer and the expert weights are a separate set of FFN parameters. Merge them into one and the balancing force collides head-on with the specialisation force, producing load curves that are noisy and irreproducible — a trap this project fell into in an earlier version.

## Why load skews: three layers of cause

### Layer 1: the data is already skewed (nothing to do with the router)

Topic frequency in real text follows Zipf: topic t takes a share ∝ `1/t`. The site uses 48 long-tail topics over 256 tokens, and the hottest topic alone takes **22%** — the top five take more than half.

**This layer is a prior of the data. A perfectly trained router cannot remove it.** To verify: set the balancing mode to "untreated" and you get a skew of 2.03× with **zero dead experts** — because there are far more topics (48) than experts (8), so every expert has work.

### Layer 2: routing reinforces itself

The specialisation update is a k-means-style centroid update:

```
router_e ← router_e + lr · w · ( x_i − cos(x_i, router_e) · router_e )
                                        └──────── tangential component ────────┘
```

The more tokens of a topic select an expert, the closer `router_e` is dragged to that topic, and the more likely it is to be selected again. That is how the division of labour grows on its own — the heatmap is block-diagonal once sorted by topic, and **nobody ever told it which expert should own which topic**.

### Layer 3: more experts than effective clusters ⇒ structurally dead experts

Drag "latent topics" down to 3 while keeping 8 experts and you immediately get **3 dead experts and 3.56× skew**.

The problem has now changed character: it is no longer "the split is uneven" but **"there is simply not enough work for 8 experts"**. You can watch the hot seat rotate among a few experts (whoever gets pushed up eats the whole stream) while the dead ones stay at exactly zero.

**Balancing can only do so much here**: it can raise a dead expert's bias and put it back on duty, but it cannot invent a division of labour that does not exist. This is exactly how large expert counts degrade when data diversity is limited.

## Two remedies, acting on the same parameter

Modern MoE gives every expert a logit bias `b_e`, and balancing **only touches that scalar**. The difference is the signal.

### Route A: the aux-loss analytic gradient (Switch Transformer, 2021)

The auxiliary loss is

```
L = n_E · Σ_e f_e · P_e        f_e = share of tokens dispatched to e, P_e = mean gate probability
```

The floor `L = 1` is attained at perfect balance (the site prints the aux loss next to that floor).

Following Switch, treat `f_e` as a constant (stop-gradient) and differentiate with respect to the bias:

```
∂L/∂b_e = n_E · f_e · (1/n) Σ_i p_i,e (1 − p_i,e)
```

**Look at `f_e`. A dead expert has `f_e = 0`, so ∂L/∂b_e is exactly zero.**

The site has a dedicated bar chart as evidence: switch to the "⑥ more experts than clusters" scenario, and the left chart is load while the right one is this gradient — a dead expert's bar prints as `0.0000`. This is not "a very small gradient", it is "the gradient is zero". The aux loss exerts **no direct force whatsoever** on a dead expert.

Two consequences, both observable on the site:

1. **The curve is noisy and non-monotonic.** Measured (strength → skew): `0 → 2.03`, `0.5 → 1.69`, `1 → 1.16`, `2 → 1.22`, `4 → 1.78`, `8 → 1.19`, `16 → 1.81`. Strength 1 is best; 4 and 16 are worse than 1.
2. **Over-driving it creates brand-new dead experts.** At strength 4 the bias is pushed to `|b| = 8.85`, which kills one of the eight experts outright (dead=1) and drops mean specialisation purity from 0.545 to 0.470.

The root cause is that this gradient **does not vanish at the optimum**: `f_e` is held constant, so the loss has no idea that balance has already been achieved. It keeps pushing, washing out the division of labour the router vectors had learned.

### Route B: the load-feedback bias (DeepSeek-V3, 2024)

Drop the gradient and feed the measured load deviation back instead:

```
dev_e = load_e / mean_load − 1
b_e  ← b_e − gain · dev_e / n_E
```

- An over-worked expert has `dev_e > 0` → its bias drops, it receives fewer tokens
- An idle expert has `dev_e < 0` → its bias rises, it receives more
- **A dead expert has `dev_e = −1` (the most negative) → its bias is driven to the top and it comes back on duty automatically**

This is a plain negative-feedback loop: **once load is even the deviation goes to zero and it stops by itself.**

Measured on the site (same x-axis, same configuration):

| Strength | 0 | 0.5 | 1 | 2 | 4 | 8 |
| --- | --- | --- | --- | --- | --- | --- |
| aux skew | 2.03 | 1.69 | **1.16** | 1.22 | 1.78 | 1.19 |
| feedback skew | 2.03 | 1.44 | 1.13 | **1.06** | 1.03 | 2.00 |
| \|b\| needed by aux | 0 | 1.10 | 2.21 | 4.41 | 8.85 | 17.5 |
| \|b\| needed by feedback | 0 | 0.20 | 0.25 | **0.30** | 0.25 | 0.15 |

Two numbers worth remembering:

- **Load feedback needs `|b| ≈ 0.30`; the aux loss needs `2.2` just to come close.** Bias lives in logit units (`cos/T`, roughly magnitude 3 at T=0.35), so 0.30 is "nudging the sign on the door" while 2.2 is "ripping the sign off".
- **The harder the aux loss pushes the bias, the more it wrecks the division of labour**: at strength 4, mean purity falls from 0.545 to 0.470 and a dead expert appears at the same time.

This is why the DeepSeek-V3 report says the aux loss damages model quality and that its balancing power is limited by the gradient signal, and switched to bias feedback. **It did not turn the aux weight down — it changed the type of signal.**

## In fairness: feedback overshoots too

At strength 8, feedback overshoots as well: skew bounces back to 2.00 with one dead expert. The gain is large enough that it over-reacts to instantaneous load fluctuation and swings the bias around.

So the honest conclusion is not "B beats A":

> **Using load itself as the signal fits the stated goal ("I want loads equal") better than using a loss gradient does** — because the error definition and the objective definition are the same object, while the aux loss merely optimises a surrogate. But under either route the signal strength has to be tuned, and both overshoot.

## Hard capacity: a third route, and what it really costs

`capacity factor` is a different mechanism: give each expert a quota `cap = ceil(n·topK/n_E · factor)` and **drop** tokens that exceed it.

- `factor = 1.25`: skew drops from 2.03 to 1.38 immediately, with zero dead experts
- **The price is 24 tokens fully dropped** (and another 24 missing an expert)

This is not simply "a little less computation": a dropped token **contributes nothing to the residual stream** in that layer — its representation skips the block entirely. And the dropped tokens are always the hot topic's tokens (they are the ones piling up on the throttled experts), so **what gets sacrificed is the part of the signal that matters most**.

That is why modern large models (DeepSeek-V3, Qwen3) prefer "bias feedback + mild capacity" over hard capacity limits.

## Common misconceptions

**"MoE experts specialise into interpretable semantic experts."** — The heatmap on the site is indeed block-diagonal, but that is **under this toy setup** (topics are orthogonal Gaussian clusters). Real MoE specialisation is far blurrier and varies with depth: early layers lean syntactic/positional, semantic behaviour only shows up deeper.

**"Adding an aux loss eliminates dead experts."** — The opposite: measured at strength 4, the aux loss **creates** a dead expert. Because with `f_e = 0` its gradient is zero, it cannot reach dead experts; meanwhile it is busy wrecking the live ones' division of labour.

**"Larger top-k is always better."** — Going from k=1 to k=2 drops skew from 2.03 to 1.44, but mean purity falls from 0.545 to 0.426. **Experts become less "expert" and more like a dense model** — you are trading sparsity for evenness. Mixtral's top-2 of 8 and Qwen3's top-8 of 128 are both points on that fine-grained trade-off curve.

**"Load balancing is a training trick that does not affect inference."** — It affects it enormously. Uneven load means some experts' weights get read and written repeatedly while others are barely touched, which directly determines expert-parallel communication volume and memory hot spots. It is the central scheduling problem in MoE inference engines.

## How to play with this on the site

1. Start with "① untreated: imbalanced" and look at the block-diagonal heatmap plus the load bars, to see that **the skew comes from the data itself**.
2. Switch to "⑥ more experts than clusters", then look at the "a dead expert's aux gradient is exactly zero" chart — several bars on the right are `0.0000`.
3. Set the balancing mode to `aux gradient`, strength 2, then `load feedback`, strength 2. Compare the current-config skew and `|b|` across the two.
4. Click "④ aux over-driven" and watch `dead experts` and `mean specialisation purity` degrade together.
5. Go back to "the two schools have completely different curves", drag the x-axis from 0 to 16, and watch the purple line (aux) turn around while the teal line (feedback) keeps falling.
6. Drag `top-k` from 1 to 2 to 4 and watch purity collapse.

## Going further

- Switch Transformer (first large-scale MoE with top-1 + aux loss): [Switch Transformers](https://arxiv.org/abs/2101.03961)
- GShard (origin of the capacity factor and capacity dropping): [GShard](https://arxiv.org/abs/2006.16668)
- Mixtral (the populariser of top-2 of 8): [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- DeepSeek-V3 (aux-loss-free load balancing): [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- Empirical study of routing collapse: [StableMoE](https://arxiv.org/abs/2302.14376)
- Scaling laws for fine-grained experts: [Scaling Laws for Fine-Grained Mixture of Experts](https://arxiv.org/abs/2402.07871)

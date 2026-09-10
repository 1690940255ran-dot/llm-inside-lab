# 04 · Sampling: what temperature / top-k / top-p each really change

> Module on the site: sidebar **④ Token-by-token Generation**

## In one sentence

Every step the model does exactly two things: **compute a distribution over the vocabulary, then draw one token from it**, append the token to the input, and continue. All three knobs operate on the step right before the draw.

## The full chain

```
logits ──(÷ temperature T)──▶ softmax ──(top-k cut)──▶ ──(top-p nucleus)──▶ renormalise ──▶ draw
```

Note the order: **temperature acts on the logits**; top-k / top-p act on the **probabilities**. That distinction matters, and we will come back to it.

## Temperature

```
p_i = exp(logit_i / T) / Σ_j exp(logit_j / T)
```

- **T → 0**: distribution collapses onto a one-hot — equivalent to greedy decoding
- **T = 1**: the model's native distribution
- **T > 1**: the long tail gets lifted up — output is more diffuse

Intuitively, temperature controls "how flat the distribution is". **It changes the shape of the whole distribution, continuously** — every candidate's probability moves, just by different amounts.

A common misunderstanding is "temperature controls randomness". More precisely: **temperature controls how confident the model feels about its own judgement.** At low T the model insists on its top choice; at high T it is willing to gamble.

## top-k

Keep only the `k` largest probabilities, zero out the rest, renormalise.

- `k = 1`: greedy
- small `k`: safe but dull, prone to repetition
- large `k`: diverse but prone to drift

**The problem with top-k:** `k` is fixed; it does not look at what the distribution actually looks like.

When the model is sure (e.g. "the capital of China is ___", where "北京" sits at 95%), `k = 50` drags in 49 garbage candidates. When the model is genuinely torn (open-ended continuation), `k = 5` chops off the reasonable long tail.

## top-p (nucleus sampling)

Sort probabilities in descending order and keep the smallest set whose cumulative probability **just exceeds `p`**:

```
sort: p₁ ≥ p₂ ≥ ... ≥ p_V
find the smallest m such that Σ(i=1..m) pᵢ ≥ p
keep the first m
```

**The candidate count adapts.** When the model is sure, only 2 candidates survive; when it is torn, dozens do. That is what makes top-p more elegant than top-k.

One implementation detail: the "candidate that crosses the threshold" must also be kept, otherwise the cumulative total falls below `p`. The site follows this rule.

## Temperature vs top-p: the key difference

This deserves to be called out on its own, because mixing them up is the most common sampling mistake:

| | Temperature | top-p / top-k |
| --- | --- | --- |
| What it operates on | logits (before softmax) | probabilities (after softmax) |
| How it changes things | continuously rescales the whole distribution | chops the support set off entirely |
| Effect | every candidate's probability moves; the long tail is lifted or depressed relatively | the long tail is zeroed out; what remains is renormalised |

**In the site's candidate table, the difference is obvious at a glance:** candidates chopped by top-k / top-p go grey in the entire row (probability shown as `—`); candidates merely depressed by temperature just have a smaller number but still appear.

So:

- want "more conservative but not too rigid" → lower the temperature
- want "chop off the obviously implausible tail" → lower top-p
- crank both at once → equivalent to greedy; the model spirals into repetition fast

## Common production configurations

| Scenario | Recommended |
| --- | --- |
| Code generation | T = 0.2, top-p = 0.95 |
| Factoid Q&A | T = 0 (greedy) or T = 0.1 |
| Creative writing | T = 0.8–1.0, top-p = 0.95 |
| Need diversity | T = 1.0+, top-p = 0.98 |

Generally **do not combine a strong top-k with a low temperature** — both push toward greedy, and the result is more rigid than either alone.

## Why low temperature leads to repetition

This is the failure mode most people notice first. The mechanism is straightforward:

1. At very low T every step almost surely picks the top-probability token.
2. Once the model steps into a familiar phrase, every subsequent top pick is highly determined.
3. The model walks down that "optimal path" all the way, and eventually loops.

Cranking the temperature up breaks that deterministic path, but at the cost of drifting off-topic. **This is why top-p is so popular**: it introduces randomness only where the model is genuinely uncertain, and stays decisive where the model is sure.

## On reproducibility

The site exposes a "random seed" slider. Same seed + same parameters = exactly the same output.

This matters in production:

- when iterating on a prompt, fix the seed so comparisons are fair
- when debugging a production bug, you need a seed to reproduce
- caveat: **even with a fixed seed, results can differ across batch sizes / hardware** because floating-point accumulation order changes

## Common misconceptions

**"temperature = 0 is greedy."** — Mathematically T = 0 is undefined (division by zero). The limit as T → 0 does equal greedy, so APIs usually accept 0 and special-case it internally.

**"Pick one of top-p or top-k."** — They can be stacked, top-k first then top-p. Usually unnecessary though.

**"Sampling only affects style."** — On reasoning tasks (math, code) the sampling strategy materially changes correctness. That is why benchmarks usually use greedy or very low temperature.

## How to play with this on the site

1. Set temperature to 0.1, click "resample 5 times" — all five samples come out near-identical (parrot mode).
2. Set temperature to 2.0 and resample — all five differ, but the text starts to babble.
3. Hold temperature at 1.0 and drag top-p from 1.0 down to 0.3 — watch the grey rows in the candidate table multiply.
4. Watch the "candidates kept" metric: it drops with smaller top-p but is **independent of temperature** (because temperature doesn't change the support set).
5. Compare the "raw p" and "after sampling p" columns to see what renormalisation did.

## Going further

- top-p paper: [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
- Reference implementations in Hugging Face `transformers`: `TemperatureLogitsWarper` / `TopKLogitsWarper` / `TopPLogitsWarper`
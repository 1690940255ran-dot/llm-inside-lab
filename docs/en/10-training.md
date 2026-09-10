# 10 · Training a mini GPT from scratch: where backprop actually goes wrong

> Corresponding module: "⑩ Train a mini GPT from scratch" in the left sidebar

## In one line

The previous nine pages all look at a model that is **already trained**. This one flips the whole thing around: weights start as random numbers, the forward *and* backward passes are **written by hand**, real AdamW updates are applied, and the validation loss really goes from **2.5834 to 0.0137** (perplexity 1.014). The interesting part is not the descending curve — it is three things: **the two places hand-written backprop most easily goes wrong**, **why you cannot validate gradients with finite differences in a float32 network**, and **how an 18.6k-parameter model quietly teaches you the wrong lesson** (its solution is not the one you assume).

## What the model in this module is

A standard GPT, with nothing simplified away:

```
x = tokEmb[ids] + posEmb[pos]                     ← token + absolute positional embedding
each layer (pre-norm):
  a  = LN1(x);  o = MHA(a);  xMid = x + Wo·o      ← attention branch
  f  = LN2(xMid); h = GELU(f·W1);  x = xMid + W2·h ← feed-forward branch
xf = LNf(x)
logits = xf · tokEmbᵀ                              ← weight tying, as in GPT-2
```

Default configuration `2 layers / d=32 / 4 heads / dFF=64 / window 32`. The parameter count has a clean closed form:

```
per layer = LN(128) + QKVO(4×1056=4224) + FFN(2048+64+2048+32=4192) = 8544
total     = 32·V + 18176
```

With `V=13` for the default Chinese corpus: **18,592 parameters**. This is not an estimate — the number on the page is `paramCount()` summing every tensor's `size`.

| Group | Params (V=13) | Share |
| --- | --- | --- |
| Attention QKVO | 8,448 | 45.4% |
| Feed-forward W1/W2 | 8,384 | 45.1% |
| Positional embedding | 1,024 | 5.5% |
| Token embedding (= output projection, shared) | 416 | 2.2% |
| LayerNorm | 320 | 1.7% |

Note that **the top two rows eat 90.5%**, while the token embedding is only 2.2% — because a character-level vocabulary has just 13 entries, whereas the positional embedding for `blockSize=32` is larger than it. That is the opposite of the intuition most people carry from module ⑧ on quantization, and it is worth comparing the two.

## What the run actually looks like

The curve on the site is **validation loss**: evaluated on 12 fixed validation batches (validation seed 999, unchanged throughout), one point every 20 steps. Measured trajectory on the default Chinese corpus (vocabulary 13):

| Step | Validation loss | |
| --- | --- | --- |
| 0 | 2.5834 | slightly above the random baseline ln(13) = 2.5649 |
| 100 | 0.5891 | has learned "which characters are common" |
| 200 | 0.3141 | |
| 300 | 0.5713 | **bounces back** — see below |
| 400 | 0.2187 | |
| 500 | 0.3970 | bounces again |
| 600 | 0.2555 | |
| 800 | 0.1307 | |
| 1000 | 0.0259 | starts reproducing characters verbatim |
| 1200 | **0.0137** | perplexity **1.014** |

Two things worth noticing:

**1. The curve is not monotonic.** It bounces to 0.5713 at step 300 and to 0.3970 at step 500, both worse than the previous recorded point. This is normal for real training, not a bug: the validation set is only 12 batches and carries statistical noise, and warmup ends at step 120, so the learning rate is at its maximum (3e-2) right there — the model is still taking big steps and overshooting is expected. **Every real training curve wobbles like this**; a wobble does not mean training has broken.

**2. The endpoint is memorisation, not language.** A perplexity of 1.014 means the model is almost completely certain about the next character. On a 720-character corpus with 12 distinct characters, that is equivalent to having memorised the whole thing — which is exactly what the "period vs context window" and "attention" sections below take apart.

## Pitfall 1: the residual stream has multiple inflows, so `dx` must accumulate

There is one spot in backprop through a residual network that is extremely easy to get wrong. Look at this line of the forward pass:

```js
x = xMid + ffnOut
```

`xMid` has two downstream consumers: one adds it straight into `x`, the other goes through `LN2 → FFN → W2 → ffnOut`. So the gradient of `xMid` is the **sum of both paths**. Writing an assignment keeps only one of them:

```js
dxMid = ...            // ✗ wrong: overwrites the direct path
dxMid += ...           // ✓ correct
```

Why is this error so dangerous? **Because training still runs and the loss still falls.** A model missing one gradient path merely learns slower or crookedly; the curve still points down and the eye cannot see anything wrong. This is exactly why `+=` is everywhere in this file — every `+=` is a registration of one incoming path.

The same pattern appears in finer places: the output of `LN1` (call it `a1`) feeds three projections `Wq`/`Wk`/`Wv` at once, so `dx` inside `linearBackward` must accumulate; and `tokEmb` acts both as the input embedding and as the output projection (weight tying), so its gradient accumulates from two paths as well.

## Pitfall 2: LayerNorm's backward pass must multiply the `1/σ` back

This is the classic omission in hand-written LayerNorm backward. The forward pass is

```
y = (x − μ)/σ · g + b
```

and backward gives `∂y_i/∂x_j = (1/σ)·(δ_ij − 1/D − u_i·u_j/D)`, where `u = (x−μ)/σ`. **That `1/σ` is part of `∂y/∂x`**, and it is very easy to drop while implementing the `δ_ij − 1/D − u_i u_j/D` cluster.

The consequence is sneakier than pitfall 1: the gradient is wrong by an overall **scale**, while the direction is roughly right. So the loss still falls — every LayerNorm branch just has the wrong "volume", degrading both training quality and speed. We genuinely hit this one: the worst relative error in our first gradient check was **1.000**, meaning the gradient did not match at all, and tracing it led here.

A correct implementation must store `1/σ` and pass it into backward (in the code those are the `inv1` / `inv2` / `invF` caches) — which is the only reason `lnForward` returns `invStdOut`:

```js
dx[off+k] += inv * (dxh[off+k] - meanDxh - xh[off+k] * meanDxhXh)
//          ^^^ this one
```

## How do you know the gradient is right? You cannot use per-parameter finite differences in float32

Once backward is written, how do you validate it? The textbook answer is finite differences:

```
∂L/∂θ_i ≈ [L(θ + ε·e_i) − L(θ)] / ε
```

**This method is essentially unusable in a float32 network**, and we got burned by it once: it reported **48 mismatches**, which looked like a broad failure of backprop. The truth was:

- float32 carries about 7 significant digits. The loss is of order $10^0$, so the forward pass alone has rounding noise around $10^{-7}$.
- A small component of the analytic gradient may itself be only $\sim10^{-7}$. Using $\varepsilon = 10^{-3}$ for the difference quotient puts the signal at $\varepsilon \cdot g \sim 10^{-10}$ — **three orders of magnitude below the noise**.
- So the numerical difference returns pure noise, which of course does not match the analytic gradient. The vast majority of the 48 reported mismatches were false positives.

### Switch to the descent-direction criterion

The right move is to stop comparing parameter by parameter and instead compare **one aggregate quantity**. If `g` really equals `∇L`, then stepping a small distance η along `−g` gives, to first order,

$$
L(\theta - \eta g) \approx L(\theta) - \eta\,|g|^2
$$

The right-hand side is a signal of order $|g|^2$. That quantity is itself large (measured between $10^{3}$ and $10^{6}$ here), so after multiplying by η it sits **several orders of magnitude above float32 noise** — the signal-to-noise ratio appears immediately. The criterion is therefore "actual drop / predicted drop":

```
ratio = [L(θ−ηg) − L(θ)] / (−η·|g|²)
```

Measured (`η = 1e-6`, d=16, 2 heads, window 8):

| Model | Measured ratio | Note |
| --- | --- | --- |
| 1 layer | **0.983** | Deviation from 1 is the second-order remainder, as expected |
| 2 layers | **0.964** | More layers, larger second-order remainder, larger deviation |

If the sign of `g` were wrong, or a path were missing (pitfall 1), or a `1/σ` were dropped (pitfall 2), the ratio would immediately be negative, zero, or off by a factor — **it could not land in 0.9–1.1**. That is the interval the test locks down.

The criterion has a second advantage: it is insensitive to the number of parameters, validating **all** of them in a single forward pass instead of sweeping one at a time. The general lesson is worth remembering: **to validate a high-dimensional gradient, do not compare each component — compare its effect along one direction.**

## Hyperparameters: the recipe for a small model is nothing like the one for a large model

The default learning rate is **3e-2**, two orders of magnitude above the 3e-4 / 6e-4 common for large models. This is not arbitrary:

- Adam takes almost the same step size for every parameter (`m̂/√v̂` normalises to roughly ±1).
- The weight initialisation scale is 0.02, so a 3e-4 step is only **1.5%** of a weight — it takes thousands of steps to move anything.
- This model has to learn within tens of seconds, with a budget of only 1200 steps. 3e-2 is what makes the step size meaningful relative to the weights.

**Recipes from large models do not transfer to small ones** — that is the first engineering conclusion module ⑩ wants to land.

The other two settings are not decoration either:

| Setting | Value | What happens without it (measured) |
| --- | --- | --- |
| warmup | 120 steps (10%) | Initial gradients have unreliable direction and scale; a large lr from step one pushes the model into a bad region |
| cosine decay to 2% | `minLrRatio=0.02` | **Loss falls to 0.5 and then rebounds to 1.6** — late-training steps are too large relative to the weights and smash what was just learned |

That "falls then rebounds" shape is a textbook symptom worth its own rule: **a loss curve that rebounds late in training almost always means the step size is too large, not that the model lacks capacity.**

## Initialisation: three unremarkable details that decide success

Small models are far more sensitive to initialisation than large ones. Three non-default choices here:

**1. LayerNorm gain is initialised to 1, not N(0,1).** Random gain leaves some channels wide open and amplifies others, which trains very unstably at this scale.

**2. Residual-branch output projections are scaled by `1/√(2L)`.** This is GPT-2's approach. The residual stream accumulates, so without scaling the branch outputs the activation variance grows linearly with depth. This step alone decides whether 2+ layers train at all.

**3. Initial weights are truncated at ±3σ.** This is the least visible of the three, and it was forced by a real failure: Box-Muller can return values above 6 when `u` is near 0, since `√(−2 ln u)` blows up. With only ~20k parameters, **a single 6σ initial weight is enough to push training into a bad basin.** Measured: some seeds then fail to learn entirely, with loss frozen near 2.2 while other seeds reach below 0.4.

## Seed sensitivity: 2 of 8 seeds fail

Even with all of the above correct, roughly **25% of seeds still fail to train** (2 of 8, loss frozen between 2.34 and 2.41, with an early maximum gradient of 3258–16192 — an early gradient explosion). Raising the batch size to 16 did not fix it (still 2/8).

So the page offers only **6 seeds verified to converge** (123 / 1 / 7 / 42 / 777 / 99991), which "resample initial weights" cycles through. This is not hiding data — it is precisely why real training work studies initialisation, warmup and gradient clipping, and the page says so in its warning box.

## Period vs context window: a teaching point you can actually measure

The three built-in corpora are ordered by **repeat period**. The default Chinese corpus is `'注意力就是让模型看哪里。'` repeated 60 times, with a period of **12 characters**; `blockSize=32` holds 2.67 periods.

Fixing the corpus and changing only the context window, the loss after 1200 steps is:

| Window | Periods it holds | Final batch loss | vs baseline ln(12)=2.48 |
| --- | --- | --- | --- |
| 8 | 0.67 | 2.2214 | **did not learn** (at baseline) |
| 12 | 1.00 | 2.4874 | **did not learn** |
| 16 | 1.33 | 1.2877 | half learned |
| 24 | 2.00 | 0.0624 | learned |
| 32 | 2.67 | **0.0110** | learned (default config) |
| 48 | 4.00 | **0.0094** | learned |
| 64 | 5.33 | 1.1411 | **failed anyway** ⚠️ |

> A note on measurement: this table records the **last training batch loss** — no validation set is run. The "what the run actually looks like" table above records **loss on a fixed validation set**. They are the same order of magnitude but **must not be compared cell by cell** (the same configuration gives batch loss 0.0110 against validation loss 0.0137). Every table has to state its measurement protocol — otherwise two numbers look alike while measuring different things.

The first six rows form a clean threshold: **when the window cannot hold one full period, the model cannot even "look back exactly one period"**, and the loss sits at the random-guess baseline; it takes roughly 2 periods before the loss falls below 0.1. This connects directly to module ⑨ — a context window is first of all the threshold for whether a complete pattern is visible, not something where bigger is always better.

**But the last row deserves honesty**: window 64 failed. The threshold relationship therefore **only holds at the low end**: once the period fits, the bottleneck moves elsewhere (a larger window means more positional-embedding parameters to learn and longer dependencies to optimise) — and "elsewhere" includes luck. The accurate statement is: **below one period, training necessarily fails; above two periods, the window is no longer the deciding factor.** That is considerably more honest than "bigger is better".

Switching to the "longer period (harder)" corpus (period 52), windows 16–64 all fail to hold even one period (0.31–1.23), and the final loss struggles between 1.31 and 2.48 — reading the two tables side by side makes the threshold effect unmistakable.

## Attention really does grow structure (and one "but" that must be stated)

The two attention heatmaps in the "attention grows structure out of noise" card are not illustrations. They are the real probability matrices from the same input and the same head, computed by one forward pass before and one after training. Quantified by **normalised entropy** (1 = perfectly uniform, 0 = attending to a single position):

| | Normalised entropy | Max weight |
| --- | --- | --- |
| Before training (step 0) | **1.0000** | 0.048 ≈ 1/21 |
| After training (step 1200) | **0.0000** | **1.0000** |

Before training all four heads sit at 1.0000 with a max weight of 0.048, exactly `1/21` (row 20 has 21 visible positions) — perfectly uniform, literally "nothing learned". After training all of them drop to 0.0000 with a max weight of 1.0000 — every head and every row has collapsed to one-hot. We **only ever gave it next-character prediction, and no line of code told it where to look.**

### But: what it learned is not "look back one period"

If the model had learned the algorithm "look back exactly one period", every row's attention target would be at a **fixed offset of 12**. Measured, it is not. Here is the target of layer 0, head 1 (the arg-max position of each row `t`):

```
t:        0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
target:   0  1  1  1  1  1  1  6  8  8  8  8  1  1  1  1  1  1  1  8  8  8  8  8  1  1  1  1  1  1  1  8
```

It is a **staircase function**: `t ∈ [12,18]` all stare at position 1, `t ∈ [19,23]` all stare at position 8, and the pattern repeats at the corpus period. In other words, what the model learned is a **lookup table "row t stares at position p(t)"**, where `p(t)` is itself a periodic function.

This is exactly the typical solution an 18.6k-parameter model finds for a 720-character periodic corpus: **it memorised the answer rather than learning to copy.** It can get away with this positionally because the corpus is strictly periodic — position `t` and position `t−12` hold identical content, so "stare at some absolute position" and "look back one period" give the same answer on this particular corpus.

This also explains why the "average weight each head puts on the previous position" bars are not impressive (0.065 / 0.161 / 0.161 / 0.161 on the default corpus, where uniform would be 1/32 ≈ 0.031): the heads did specialise, but not in the direction of "the previous token".

**This is the attitude module ⑩ most wants to convey**: the heatmap is beautiful and an entropy drop from 1.00 to 0.00 is striking, but it proves the model grew **some** structure — **not** that it learned a general algorithm. Keeping those two claims apart is the prerequisite for reading any attention visualisation correctly.

## The bill: parameters and compute

Per-token forward FLOPs use the standard industry formula:

```
FLOPs/token = 2 · [ nLayers · (4d² + 2d·dFF) + 2d·V ]
```

With the default configuration `(L=2, d=32, dFF=64, V=13)`: `2·[2·(4096+4096) + 832] = 34,432`.

On the same ruler as real models:

| Model | Parameters | Relative to this page |
| --- | --- | --- |
| This mini GPT | 18,592 | 1× |
| GPT-2 124M | 124,000,000 | **6,670×** |
| GPT-3 175B | 175,000,000,000 | **9.41 million×** |

And that is **only parameter count**. Pretraining compute is roughly

```
C ≈ 6 · N · D        (N = parameters, D = training tokens)
```

Real models see `D` in the hundreds of billions to trillions (GPT-3 saw about 300 billion tokens), so **the compute gap is several orders of magnitude wider than the parameter gap**. This page's 1200 steps × batch 8 × window 32 has seen only 307,200 tokens — the gap is not that the model is small, it is that the world it has seen is small.

## Common misconceptions

**"A small model cannot train because it lacks capacity"** — measurement says the opposite: on a period-12 corpus, 18,592 parameters drive the validation loss to 0.0137 (perplexity 1.014, about 98.6% next-character accuracy). The real reasons small models fail, in order of frequency: wrong step size, missing warmup, an outlier in the initialisation, bad seed luck. **Capacity is almost never the reason.**

**"Hand-written backprop is correct as long as the loss falls"** — both pitfall 1 (a missing residual path) and pitfall 2 (a missing `1/σ`) leave the loss falling. A gradient with the right direction and the wrong scale looks identical to a correct one from SGD's point of view. You have to validate it actively, with something like the descent-direction criterion.

**"Finite differences can validate a gradient"** — not in float32. Small components get drowned by forward rounding noise, and most reported mismatches are false. Either rewrite the forward pass in float64 specifically for validation, or use the descent-direction criterion.

**"A clean attention heatmap means the model learned an algorithm"** — see the section above. The one-hot attention measured here sits on top of a lookup table. Whether something was learned as an algorithm or memorised is decided by **whether it still holds on a different input**, not by how clean the heatmap looks.

**"A larger context window is always better"** — measured, window 64 (5.33 periods) is worse than window 32 (2.67 periods). The window acts as a **threshold** (can the model see a complete pattern), and past that threshold it stops being the deciding factor.

## How to play with it on the site

1. Start with the default settings and hit "Start training". Watch the loss fall from 2.58 to 0.014 — and note that "ms / step" is about 20, so 1200 steps takes roughly 25–30 seconds (measured about 30 s in headless Chrome). **This is genuinely computing.** The page stays responsive because it runs in a Web Worker that yields control every 40 ms (which is why "Pause" reacts immediately).
2. Watch the "what it actually learned" list: step 0 is gibberish (the `␀` glyph is the out-of-vocabulary placeholder), common characters appear after a few dozen steps, and by the end it reproduces the memorised sentence verbatim. **This is the most direct evidence that training is happening.**
3. Look at "attention grows structure out of noise": the left and right heatmaps are step 0 and the current step. Pause around step 200 first, then let it run to 1200 — the contrast is clearest that way.
4. Reproduce the period experiment yourself: switch to "longer period (harder)" (period 52), drag "context length" to 32, then to 64 — and watch it **never** learn well, because the window can never hold one period. Then switch back to the default corpus and drag the window from 8 to 24 to see the loss cross the threshold.
5. Drag "learning rate" back to 3e-4 (the usual large-model value), hit Reset and train — you will see a curve that barely moves. That turns "recipes do not transfer" into something you can feel.
6. Click "resample initial weights" to cycle the 6 seeds and watch how much the final loss swings under an identical configuration — that is what initialisation sensitivity feels like.
7. Finally, set "layers" to 1 or 4 and compare convergence — 4 layers is clearly harder to push down within the same 1200 steps, which is the problem `1/√(2L)` scaling exists to solve.

## Further reading

- The essential reference for hand-written backprop: [The Matrix Calculus You Need For Deep Learning](https://arxiv.org/abs/1802.01528) — the residual and broadcasting sections map directly onto pitfall 1
- The origin of LayerNorm's backward formula: [Layer Normalization](https://arxiv.org/abs/1607.06450)
- GPT-2 (source of the init scale, `1/√(2L)` and weight tying): [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- AdamW (decoupled weight decay): [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- The tanh approximation of GELU (the derivative must match exactly): [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415)
- Where `C ≈ 6ND` comes from: [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- Complete from-scratch treatments of the same thing: [nanoGPT](https://github.com/karpathy/nanoGPT) and [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch)

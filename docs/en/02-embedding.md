# 02 · Embeddings & Positional Encoding: a coordinate and a seat number for every token

> Module on the site: sidebar **② Embeddings & Positional Encoding**

## In one sentence

Embeddings turn discrete IDs into continuous vectors so that "similar" can be computed. Positional encoding puts word order back, because **self-attention is permutation-equivariant — shuffle the sentence and it cannot tell.**

## Embeddings: from "ID" to "coordinate"

After tokenization each token is just an integer. The problem is that **integers have no computable relationship**. ID 5 and ID 6 are not "closer" than ID 5 and ID 5000; that information simply isn't there.

An embedding layer is a lookup table:

```
X = Embedding[token_id]     shape [n, d_model]
```

Every token gets a `d_model`-dimensional vector. These vectors are **trained**, and the training objective nudges vectors whose meanings or usages are close to each other, so they end up close in vector space. The textbook example is `king - man + woman ≈ queen`.

The thing people often miss: **the embedding matrix is usually one of the single largest parameter blocks in the model.**

```
params = vocab_size × d_model
LLaMA 3: 128256 × 4096 ≈ 525M (embedding layer alone)
```

And the output layer needs an equally large unembedding matrix (some models tie the two together). That is exactly why vocab size is a hyperparameter you have to pick carefully.

## Why position has to be added explicitly

Look at the core attention formula:

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

`Q`, `K` and `V` all come from the same input `X`. Permute the rows of `X`, and the rows of `Q`, `K` and `V` get permuted in the same way; `QKᵀ` gets permuted the same way; **the final output gets permuted the same way.**

In other words: attention treats every position as "an element in a set", not as "the i-th word in a sequence." It cannot tell "the cat chased the mouse" apart from "the mouse chased the cat" — unless you tell it the order.

So positional encoding is not a nice-to-have. **It is mandatory.**

## Approach one: sinusoidal positional encoding

This is the original Transformer and it needs zero training:

```
PE[pos, 2i]   = sin(pos / 10000^(2i/d))
PE[pos, 2i+1] = cos(pos / 10000^(2i/d))
```

The design has one elegant property: **different dimensions correspond to different frequencies.**

- Small `i` → denominator close to 1 → low frequency → long wavelength → covers the whole sequence
- Large `i` → denominator huge → high frequency → adjacent positions already look completely different

A good analogy is a battery of clocks with different periods: one turns once a day (can tell morning from afternoon), one turns once a year (can tell the season). Combine them and you can pinpoint a moment uniquely.

Another key property: **`PE[pos+k]` can be expressed as a linear function of `PE[pos]`** (via a rotation matrix). That means the model can learn "relative offset k" rather than only "absolute position pos".

In the site's "positional similarity" heatmap you will see distinct **diagonal stripes** — similarity depends only on the distance between two positions, not on which positions they actually are. That is the source of the relative-position signal.

**The drawback:** sinusoidal is an *additive* absolute-position signal. The model has to learn to derive relative distance from it on its own. And it extrapolates poorly: trained on 512 positions, fed 4096 at inference, the whole thing collapses.

## Approach two: RoPE (rotary positional encoding)

LLaMA, Qwen and Mistral all use it. The idea is fundamentally different — **instead of adding a vector, you rotate the vector itself.**

Pair up the `d` dimensions two-by-two. For pair `j` at position `pos` the rotation angle is:

```
θ = pos / 10000^(2j/d)
```

Concretely it is applied to the query and the key:

```
q̃ = R(θ_pos) · q
k̃ = R(θ_pos') · k
```

The magic shows up in the dot product:

```
q̃ᵀk̃ = qᵀ R(θ_pos)ᵀ R(θ_pos') k = qᵀ R(θ_pos' - θ_pos) k
```

**The result depends only on the relative offset `pos' - pos`.** Relative position is not "learned" — it is mathematically guaranteed.

That is the root cause of RoPE's better extrapolation: the model has never seen "relative offset 5000", but as long as it has seen "rotating by this much angle", it can handle it.

In the site's RoPE demo, drag the "dimension pair j" slider:

- `j = 0`: the ray barely turns (low frequency) → it can express long distances
- larger `j`: the ray whirls → adjacent positions are already far apart → it can only distinguish nearby ones

## Why long-context extrapolation is so hard

The reason falls straight out of RoPE. During training the model only sees rotations for `pos ∈ [0, 4096]`. At inference you suddenly hand it `pos = 32768`. The angles that the low-frequency dimensions have to rotate through were never seen in training — the model has no idea what to do with them.

Hence the family of extrapolation tricks:

- **PI (Position Interpolation):** scale `pos` down so it stays in the trained range
- **NTK-aware:** scale different frequencies by different amounts (high-frequency scaled less, low-frequency scaled more)
- **YaRN:** layer an attention-temperature correction on top

Every one of these is answering the same question: **how do you make "unseen angles" land in a region the model can understand?**

## Common misconceptions

**"Positional encoding is a vector added to the embedding, so it's vector-plus-vector."** — That is true for sinusoidal. It is *not* true for RoPE. RoPE does not change the embedding itself; it changes the **rotational state of `q` and `k` when they compute attention**.

**"Relative position is always better than absolute."** — Usually yes, but absolute position has its uses (e.g. "which paragraph is this?"). Some models use both.

**"Positional encoding lets the model know word order."** — More precisely: it lets attention **have the chance** to use word order. The information is injected; whether the model actually uses it, and how, is up to training.

## How to play with this on the site

1. In the "embedding vector" heatmap, look at each token's row — note that tokens sharing the first character or the same broad category share similar color patterns.
2. In the "sinusoidal encoding" heatmap, scan horizontally: the leftmost columns barely change (low frequency), the rightmost oscillate wildly (high frequency).
3. In the "position similarity" chart find the diagonal stripes — they prove "similarity depends only on distance".
4. In the RoPE demo, drag the dimension pair and watch the ray's rotation speed change.
5. Look at the PCA scatter to get a feel for how much information you lose when squeezing high-dimensional semantics into 2D.

## Going further

- The original Transformer: [Attention Is All You Need](https://arxiv.org/abs/1706.03762), §3.5
- RoPE paper: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- For an extrapolation survey, the YaRN paper is a great starting point: [YaRN: Efficient Context Window Extension](https://arxiv.org/abs/2309.00071)
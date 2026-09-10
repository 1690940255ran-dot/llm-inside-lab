# 08 · Quantization: where the cost of 16 bits → 4 bits actually lands

> Module on the site: sidebar **⑧ Quantization**

## In one sentence

The hard part of quantization is not "how to turn floats into integers" — that is one rounding operation. The hard part is **how many weights a single scale has to govern**. The more it governs, the more damage an outlier can do. Understand that and you understand the whole line of work from NF4 to group-wise to per-channel.

## The two basic encodings

**Symmetric** (slice `[-maxabs, maxabs]` into `2^b` even steps):

```
scale = maxabs / qmax          qmax = 2^(b−1) − 1
code  = round(w / scale)       clamped to [−qmax−1, qmax]
ŵ     = code · scale
```

**Asymmetric** (map `[min, max]` onto `[0, 2^b−1]`, storing one extra zero point):

```
scale = (max − min) / (2^b − 1)
zero  = round(−min / scale)
code  = round(w / scale) + zero
ŵ     = (code − zero) · scale
```

Asymmetric costs a zero point per group but handles distributions that are shifted overall (activations after ReLU, say). Weights are near zero-mean and symmetric, so symmetric is usually enough.

## Granularity is the real protagonist

"How many weights one scale governs" is the **granularity**:

| Granularity | One scale per | Typical use |
| --- | --- | --- |
| per-tensor | whole tensor | cheapest, coarsest |
| per-channel | one row (one output channel) | the default for weights |
| group-wise | a 32/64/128 slice inside a row | standard for GPTQ / AWQ / NF4 |

Measured on the site (4 bit, 256 weights). **The same set of schemes leads to completely different conclusions depending on whether outliers are present**:

**With two 40σ outliers (the site default):**

| Scheme | SQNR | Effective levels | Bytes/weight |
| --- | --- | --- | --- |
| INT4 per-tensor | **11.25 dB** | 5 / 16 | 0.500 |
| INT4 per-channel(8) | 17.98 dB | 15 / 16 | 0.531 |
| INT4 group=32 | 17.98 dB | 15 / 16 | 0.625 |
| NF4 group=32 | 18.93 dB | 16 / 16 | 0.625 |
| INT4 asymmetric group=32 | **19.38 dB** | 16 / 16 | 0.625 |

**Without outliers (outlierCount = 0):**

| Scheme | SQNR | Effective levels |
| --- | --- | --- |
| INT4 per-tensor | 17.80 dB | 15 / 16 |
| INT4 per-channel(8) | 19.70 dB | 15 / 16 |
| INT4 group=32 | 19.70 dB | 15 / 16 |

**Reading the two tables side by side is the most important lesson in quantization:**

- With no outliers, per-tensor to group-wise is worth only **1.9 dB** — granularity is a nice-to-have
- Drop in two 40σ outliers and the same comparison becomes **6.7 dB** (4.7× less noise energy), with per-tensor's effective levels collapsing from 15 to 5

**So the value of grouping is not better average precision — it is confining an outlier's damage to one group.** It is a firebreak, not a more precise scale. That is also why in real workflows "deal with the outliers first" (AWQ, SmoothQuant, LLM.int8()) usually pays better than "make the groups smaller".

**About "per-channel and group=32 give identical numbers":** that is not a coincidence and not laziness. The weights on this page are a 1D sequence, so `per-channel` slices it into 8 chunks of 32, exactly the same cut as `group=32`. Real weights are 2D matrices: per-channel cuts by output row (one scale per row, and a row typically has thousands of weights) while group-wise cuts those rows again into small pieces — **one to two orders of magnitude finer**. That is the real origin of "group-wise generally beats per-channel". The site chooses to keep this equivalence visible and state it, rather than pretend the two granularities differ here.

## Outliers: the true enemy of 4 bit

Push 2 weights out to 40σ while leaving the other 254 alone, then watch the effective level count:

| Outlier strength | Effective levels | SQNR | MSE |
| --- | --- | --- | --- |
| 1σ | 15 / 16 | 17.78 dB | 1.76e−2 |
| 10σ | 7 / 16 | 10.44 dB | 1.65e−1 |
| 40σ | **5 / 16** | 11.25 dB | **1.02e+0** |
| 100σ | **3 / 16** | 18.75 dB | 1.06e+0 |

Look at columns three and four: **MSE gets monotonically worse (1.76e−2 → 1.02e+0, 58× worse), while SQNR first falls and then rises again (17.78 → 10.44 → 11.25 → 18.75 dB).**

This is a very important trap in quantization:

> **SQNR lies. MSE and the effective level count do not.**

The reason: SQNR is defined as signal energy over noise energy, and that 40σ outlier inflates the denominator (the signal energy) all by itself. The more extreme the outlier, the "better" SNR looks.

**Which is why you must judge quantization quality by MSE or perplexity, never by SNR.** The site shows both and pulls out "effective levels" separately — it is the most intuitive of the three: there are 16 levels to begin with, a 40σ outlier squeezes them so that only 5 levels still carry any weight, meaning **the other 11 levels are completely wasted (no weight lands on them)** and all 256 weights effectively take only 5 distinct values.

### Why group-wise rescues it

Once you group, an outlier only destroys **its own group**. The two tables above quantify it: if one of the 32 weights in a group is a 40σ monster, that group's scale is blown up but only 32 weights suffer; the other 7 groups are untouched. Per-tensor means the whole tensor goes down with it.

NF4 (the QLoRA scheme) goes further: instead of even levels it takes 16 **non-uniform levels placed at quantiles of a normal distribution** — because quantized weights are approximately zero-mean Gaussian, and non-uniform levels follow that curvature better. Measured on the site, NF4 (18.93 dB) does beat symmetric INT4 at the same granularity (17.98 dB). Asymmetric (19.38 dB) is a notch better still, at the cost of a zero point.

## The memory ledger: do not compute bits/8

This is the easiest thing to get wrong in practice. Group-wise quantization **must store each group's scale and zero point** (one fp16 each, 4 bytes total):

```
group=128  →  4/8 + 4/128 = 0.531 bytes/weight
group=32   →  4/8 + 4/32  = 0.625 bytes/weight   (18% more)
```

Measured on the site (7B and 70B):

| Precision | Bytes/weight | 7B | 70B |
| --- | --- | --- | --- |
| FP16 | 2.000 | 13.04 GB | 130.39 GB |
| INT8 | 1.000 | 6.52 GB | 65.19 GB |
| INT4 group=128 | 0.531 | 3.46 GB | 34.63 GB |
| NF4 group=64 | 0.563 | 3.67 GB | 36.67 GB |
| INT3 group=128 | 0.406 | 2.65 GB | 26.48 GB |
| INT2 group=64 | 0.313 | 2.04 GB | 20.37 GB |

A few numbers worth memorising:

- **7B in fp16 is 13.04 GB** (= 7e9 × 2 / 1024³). This is the number that explains why consumer cards cannot hold a 7B model — the weights alone exceed a 12 GB card.
- **7B INT4 group=128 is 3.46 GB**, not `7e9 × 0.5 = 3.26 GB`. The 0.2 GB difference is metadata.
- **Below 4 bit the returns decay fast**: INT3 saves 23% over INT4, INT2 saves another 23%, but precision loss falls off a cliff (only 4 effective levels left). This is why the community settled on 4 bit.

Note this is **weights only**. A real deployment adds the KV cache (which can exceed the weights at long context), activations, and the temporary buffer for dequantising back to fp16. That is why "a 7B INT4 model fits on a 12 GB card" is a claim that needs arithmetic: 3.46 GB of weights + KV cache at 8K context + framework overhead tends to sit right at the 12 GB edge.

## Common misconceptions

**"4-bit quantization just loses a little precision."** — 4 bit gives you 16 levels. Once an outlier appears, the effective count can fall to 5 or even 3, at which point **the quantization error is no longer "a little"** (measured MSE grew 58×). Which is why real workflows always pair quantization with calibration or a small fine-tune — QLoRA's LoRA patch exists precisely for this.

**"Use SNR to evaluate quantization quality."** — The site's own measurements show SNR is non-monotonic. Use MSE, perplexity, or just run a downstream task.

**"Smaller group size is always better."** — Precision does improve, but metadata costs `4/groupSize` per weight. group=32 costs 18% more memory than group=128; group=8 gives `0.5 + 0.5 = 1.0` byte/weight, which is **identical to INT8** — the whole point of quantizing for memory vanishes. That is a hard floor.

**"The benefit of quantization is memory."** — It is speed too. 4-bit weights consume a quarter of the memory bandwidth of fp16, and LLM decoding is **almost purely memory-bandwidth bound**, so quantization often yields near-linear decode speedups. The price is the compute spent dequantising.

## How to play with this on the site

1. Start with the default (4 bit / symmetric / per-tensor / 40σ outlier) and note the vertical lines on the histogram marking level positions — most of them have no weights anywhere near them.
2. Drag `outlier strength` from 0 to 100 while watching **effective levels** and **MSE**: MSE climbs the whole way, SQNR dips and then rises.
3. Switch `granularity` to `group-wise`, set groupSize to 32, and watch SQNR jump.
4. Switch `encoding` to `NF4` and compare against symmetric INT4 at the same granularity.
5. Drag `bits` from 2 to 8, watch SQNR rise and bytes/weight fall, and find the point where compressing further stops paying off.
6. Finally, look at the memory table and set groupSize to 8 to see metadata eat the entire benefit of quantization.

## Going further

- GPTQ (popularised group-wise, second-order, column-by-column): [GPTQ](https://arxiv.org/abs/2210.17323)
- AWQ (protect salient channels by activation importance): [AWQ](https://arxiv.org/abs/2306.00978)
- QLoRA and NF4 (non-uniform 4-bit levels): [QLoRA](https://arxiv.org/abs/2305.14314)
- LLM.int8() (finding and handling outlier feature dimensions): [LLM.int8()](https://arxiv.org/abs/2208.07339)
- SmoothQuant (migrating activation difficulty onto weights): [SmoothQuant](https://arxiv.org/abs/2211.10438)
- Quantization noise and scaling laws: [The case for 4-bit precision](https://arxiv.org/abs/2212.09720)

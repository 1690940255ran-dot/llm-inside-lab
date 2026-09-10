# 09 · Long-context Extension: why 4K training can serve 32K, and what the four methods really differ in

> Module on the site: sidebar **⑨ Long-context Extension**

## In one sentence

RoPE encodes position with **angular frequencies**, and angular frequencies can be edited. Every extension method (PI / NTK / YaRN) answers the same question: **at position 32K, which angular frequency should be used to rotate?** They differ only in "which dimensions are left alone" — and that single choice decides whether you sacrifice local precision or long-range resolution.

## RoPE's angular frequencies

For dimension pair j (there are `d/2` of them):

```
θ_j = base^(−2j/d)          base = 10000 by default
λ_j = 2π / θ_j              wavelength
```

At `d=128, base=10000`:

| Dimension j | θ_j | Wavelength λ_j | Turns within 4096 |
| --- | --- | --- | --- |
| 0 | 1.00e+0 | 6.3 | **651.90** |
| 8 | 3.16e−1 | 19.9 | 206.15 |
| 16 | 1.00e−1 | 62.8 | 65.19 |
| 32 | 1.00e−2 | 628.3 | 6.52 |
| 48 | 1.00e−3 | 6283.2 | 0.65 |
| 63 | 1.15e−4 | **54410.1** | **0.08** |

**Note the enormous span**: the highest-frequency dimension completes 651 turns within the training length (it carries fine position), while the lowest-frequency one does not finish a tenth of a turn (it carries macroscopic position). That geometric progression is how RoPE covers every scale from 1 token to tens of thousands.

You can count them on the site: **28.1% of dimensions complete less than one full turn within the training length.** Those low-frequency dimensions are the only carriers of long-range information.

## Why 4K serving 32K collapses: the geometric horizon

There is exactly one mechanism behind extrapolation failure: **phase aliasing**.

Dimension j can distinguish a distance Δ only while `Δ·θ_j < π` (the phase difference has not yet reached half a period). Beyond that it starts folding back, and `Δ=32768` and `Δ=32768−6.3` produce the same phase in that dimension — **the model can no longer tell them apart**.

Put all dimensions together and you get a hard bound:

```
geometric horizon = λ_max / 2 = 54410.1 / 2 = 27205
```

**27205 < 32768.** That is the complete mathematical explanation for "4K training feeding 32K must break" — not "the model is not smart enough", but **the longest wavelength folds back before it gets there**.

Measured surviving dimension counts on the site (`Δ·θ_j < π`, out of 64):

| Target length | Untreated | linear (PI) | NTK-aware | YaRN |
| --- | --- | --- | --- | --- |
| 4096 (training length) | 14 | 14 | 14 | 14 |
| 8192 | 9 | 14 | 13 | 14 |
| 16384 | 4 | 14 | 12 | 14 |
| 32768 | **0** | 14 | 11 | 14 |
| 65536 | **0** | 14 | 11 | 14 |
| 131072 | **0** | 14 | 10 | 14 |

The "untreated" column walks from 14 down to 0 — that is the collapse, start to finish. All three fixes push the horizon from 27205 to **217641 (×8)**, exactly the ratio of target to training length.

## Fix one: linear position interpolation (PI)

```
position t  →  t / s         equivalently  θ_j  →  θ_j / s
```

An 8× extension divides every θ by 8. The simplest possible approach, with the most explicit price — **it compresses the entire spectrum proportionally, including the high-frequency dimensions that were working fine**:

| Dimension j | θ untreated | θ under PI | λ under PI | Stretch |
| --- | --- | --- | --- | --- |
| 0 | 1.00e+0 | 1.25e−1 | 50.3 | 8.00 |
| 8 | 3.16e−1 | 3.95e−2 | 159.0 | 8.00 |
| 16 | 1.00e−1 | 1.25e−2 | 502.7 | 8.00 |
| 32 | 1.00e−2 | 1.25e−3 | 5026.5 | 8.00 |
| 63 | 1.15e−4 | 1.44e−5 | 435281.1 | 8.00 |

**Every dimension's stretch is 8.00, identical.**

The critical consequence is here:

```
minimum resolvable gap minGap = π / θ_max
untreated:  π / 1.00e+0  = 3.142
PI:         π / 1.25e−1  = 25.133      ← 8× larger
```

**PI widens the smallest distinguishable position difference from 3.1 tokens to 25.1 tokens.** Two positions 25 apart look practically identical to the model — **long-range reach bought with an 8× loss of local precision**.

That trade-off is PI's fundamental flaw, and the entire reason the next two methods exist.

## Fix two: NTK-aware (squeeze the low frequencies, leave the high ones)

The idea comes from an observation: **neural networks are far more sensitive to high-frequency features than to low-frequency ones** (NTK theory). So do not compress the whole spectrum proportionally — **squeeze only the low-frequency end and leave the high end untouched**, by enlarging `base`:

```
base' = base · s^(d / (d − 2))        (scale base by s)
```

`base` changes but the *form* of `θ_0` does not, so **high-frequency dimensions barely move while low-frequency ones are stretched a great deal**:

| Dimension j | θ untreated | θ under NTK | Stretch |
| --- | --- | --- | --- |
| 0 | 1.00e+0 | **1.00e+0** (untouched) | **1.00** |
| 8 | 3.16e−1 | 2.43e−1 | 1.30 |
| 16 | 1.00e−1 | 5.90e−2 | 1.70 |
| 32 | 1.00e−2 | 3.48e−3 | 2.88 |
| 48 | 1.00e−3 | 2.05e−4 | 4.88 |
| 63 | 1.15e−4 | 1.44e−5 | 8.00 |

The stretch profile is a **smooth curve** (dimensions 0→63):

```
1.00  1.30  1.70  2.21  2.88  3.74  4.88  6.35  8.00
```

Three numbers matter most:

- **θ_0 is untouched (1.00e+0)** ⇒ minGap stays at **3.142**; local precision is fully preserved
- Dimension 63 is fully stretched 8× ⇒ long-range reach also reaches 217641
- Dimensions in between transition smoothly along a geometric progression

**This is NTK's core advantage: you get both.** Unlike PI, which spends one 8× factor on everything, it allocates that 8× budget by frequency.

## Fix three: YaRN (banded by wavelength)

YaRN refines the NTK idea further: it draws two explicit wavelength thresholds `β_fast` and `β_slow` and treats the spectrum in three bands:

```
λ_j < L / β_fast    →  ramp = 0, no scaling at all (high frequency, untouched)
λ_j > L / β_slow    →  ramp = 1, fully scaled by s (low frequency, all in)
in between          →  linear transition
```

Measured ramp profile on the site (`β_fast=32, β_slow=1`):

```
dim    0    8    16   24   32   40   48   56   63
ramp 0.00 0.00 0.00 0.02 0.13 0.47 1.00 1.00 1.00
```

**Note the "flat — slope — flat" shape**: the first 16 dimensions do not move at all (ramp=0), 24→48 is the transition band, and beyond 48 everything is scaled (ramp=1).

The corresponding θ rewrites (the first three dimensions are untouched, which is the most visible difference from NTK):

| Dimension j | θ untreated | θ under YaRN | Stretch | ramp |
| --- | --- | --- | --- | --- |
| 0 | 1.00e+0 | 1.00e+0 | 1.00 | 0.00 |
| 8 | 3.16e−1 | 3.16e−1 | 1.00 | 0.00 |
| 16 | 1.00e−1 | 1.00e−1 | 1.00 | 0.00 |
| 32 | 1.00e−2 | 7.69e−3 | 1.30 | 0.13 |
| 48 | 1.00e−3 | 1.25e−4 | 8.00 | 1.00 |
| 63 | 1.15e−4 | 1.44e−5 | 8.00 | 1.00 |

The three stretch profiles side by side:

| Method | dim 0 | 8 | 16 | 24 | 32 | 40 | 48 | 56 | 63 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PI | 8.00 | 8.00 | 8.00 | 8.00 | 8.00 | 8.00 | 8.00 | 8.00 | 8.00 |
| NTK | 1.00 | 1.30 | 1.70 | 2.21 | 2.88 | 3.74 | 4.88 | 6.35 | 8.00 |
| YaRN | 1.00 | 1.00 | 1.00 | 1.04 | 1.30 | 2.65 | 8.00 | 8.00 | 8.00 |

**The difference is visible at a glance**: PI is a horizontal line (brute-force proportional), NTK is a smooth diagonal (continuous transition), YaRN is a polyline (explicit bands). YaRN's `β_fast`/`β_slow` are simply the endpoints of that slope — move it earlier and more dimensions get scaled (better long range, worse local precision), move it later and the reverse.

YaRN also carries an **attention temperature compensation** `1/t = 1/(0.1·ln s + 1)`: after extension the attention distribution flattens (softmax entropy rises), and the temperature squashes it back. The site lets you toggle it and watch the expected-score curve get scaled as a whole.

## A contradiction that has to be stated: the two metrics disagree

The site reports two metrics. **They do not always agree**, and keeping that visible is deliberate:

| Method | Surviving dims at 32K | Score at train length | Score at 32K | Retention |
| --- | --- | --- | --- | --- |
| Untreated | 0 / 64 | −0.053 | 0.056 | — |
| PI | 14 / 64 | 0.219 | −0.053 | negative |
| NTK | 11 / 64 | 0.222 | **0.093** | **42%** |
| YaRN | **14 / 64** | 0.289 | −0.005 | ≈0 |

Ranked by "surviving dimensions", YaRN wins (14) and NTK loses (11). Ranked by "expected attention score retention", **NTK wins (42%) and YaRN goes to zero**.

This is not a bug. The two metrics simply do not measure the same thing:

- **Surviving dimensions** just counts how many dimensions have not aliased. It is a **geometric count**.
- **Expected attention score** is `(1/n)Σ cos(Δ·θ_j)`; it asks whether those dimensions *together* still separate positions. It is a **geometric average**, so dimensions can cancel each other out.

And both are only **proxies**. In reality the geometric analysis puts the horizon at 217641 (217K), yet real models start degrading noticeably within a few thousand tokens of their trained length. At least four causes are invisible to the geometric analysis:

1. The model **has never seen** these new phase combinations; the extrapolated distribution is out of distribution.
2. Once attention scores are distorted, **softmax entropy changes**, and every downstream layer was trained under the old entropy.
3. Beyond position there is content — **long-range dependencies must exist in the training data** (needle-in-a-haystack requires the model to have seen needles).
4. The YaRN paper itself concedes that different layers have different frequency sensitivity; one global setting is a compromise.

So the only real evaluation is a long-context benchmark (RULER, Needle-in-a-Haystack, ∞Bench). **The value of the geometric analysis is explaining *why* it breaks and *which way* to fix it — not predicting the score afterwards.** Every number on the site is labelled as such.

## Common misconceptions

**"Extension means interpolating positions."** — Interpolation (PI) is only the simplest option and it has an explicit cost: minGap goes from 3.142 to 25.133, an 8× loss of local precision. Avoiding that cost is the entire reason NTK and YaRN exist.

**"NTK-aware is strictly a better PI."** — On protecting high frequencies, yes; but the price is that low-frequency dimensions are stretched harder (dimension 63 is also ×8, and the transition shape differs). NTK's base scaling is exponential in form, so at very large extension ratios (32×, say) the position distribution across the transition band becomes unnatural. YaRN turns this into a controllable knob with explicit bands and `β` parameters.

**"Just make base bigger."** — Changing base reshapes the entire θ distribution; it redefines which dimensions count as high frequency. But **the model was trained under the old base**: after the change, phases within the training length are all different too, and in-distribution performance collapses with everything else. That is why every method is careful to leave the training range approximately unchanged — PI via `/s` (which leaves positions alone), NTK by leaving `θ_0` fixed, YaRN by letting the first bands sit at ramp=0.

**"Which of the two metrics should I trust?"** — Neither is sufficient. They explain the mechanism, they do not predict the result. Judging an extension scheme properly requires long-context benchmarks.

## How to play with this on the site

1. Start with the "phase wrapping" chart: each sawtooth is one dimension's phase folding at `±π`. Push `target length` to 32768 and notice the high-frequency dimensions (densest sawtooth) fold back first.
2. Select "untreated" and watch the "aliasing and surviving dimensions" curve fall off a cliff near 27205 — matching the geometric horizon exactly.
3. Switch through `linear` / `NTK-aware` / `YaRN` and watch the "minimum resolvable gap" on the right: **PI jumps to 25.133, the other two stay at 3.142**.
4. Look at the "wavelength spectrum" chart to compare the three stretch profiles: a horizontal line, a diagonal, a polyline.
5. Drag `β_fast` from 32 down to 8 and watch YaRN's transition band move earlier and the surviving count change.
6. Turn on `YaRN temperature compensation` and watch the expected-score curve flatten.
7. Finally, sit with the "surviving dimensions vs target length" table and change `base` yourself — go from 10000 to 500000 and see how badly the in-training-length score is wrecked along the way.

## Going further

- RoPE original: [RoFormer](https://arxiv.org/abs/2104.09864)
- Linear position interpolation: [Extending Context Window of LLMs via Position Interpolation](https://arxiv.org/abs/2306.15595)
- NTK-aware (a community result rather than a formal paper; it started as an analysis post on Reddit): [NTK-Aware Scaled RoPE](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)
- YaRN (banding + temperature compensation; adopted by Qwen and Mistral): [YaRN](https://arxiv.org/abs/2309.00071)
- Where NTK theory came from: [Neural Tangent Kernel](https://arxiv.org/abs/1806.07572)
- Long-context benchmark: [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654)

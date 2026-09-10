# 01 · Tokenization: why the first hurdle is "how do you cut the text?"

> Module on the site: sidebar **① Tokenization**

## In one sentence

A model doesn't know characters — it only knows integers. The tokenizer's job is to chop any input into **subword units** and look each one up in a table. **How well it chops decides how expensive the model is, whether Chinese reads fluently, and whether arithmetic even works.**

## Why not character-level, why not word-level

Start by ruling out both extremes.

**Character-level** (one Chinese character per token): the vocabulary can be tiny (a few thousand entries is enough) but the sequence explodes. Attention is O(n²), so doubling the sequence quadruples the cost. And because the two halves of "模型" (literally "model") are now separate tokens, the model has to learn on its own that they always appear together — a complete waste of capacity.

**Word-level** (whitespace + a lexicon, as in English): the sequence is short, but the vocabulary explodes. English verbs alone are an infinite party, and worse: there is the **out-of-vocabulary (OOV)** problem. Any token the model never saw during training leaves it helpless, and it has to fall back to a single `<UNK>`.

Subword tokenization is the compromise: **keep high-frequency words whole, decompose low-frequency ones into meaningful fragments**. Then:

- The vocabulary stays manageable (GPT-2 ~50k, LLaMA 3 ~128k)
- Any new word can be assembled from known fragments, so OOV stops existing
- Fragments usually carry semantic weight (`un-`, `-ing`, `-tion`), which is free linguistic language knowledge the model gets to inherit

## What BPE actually does

Byte Pair Encoding is so straightforward it is almost embarrassing — four steps on loop:

```
1. pre-tokenize: split the text into non-overlapping units
2. initialize:  each unit is split into its individual characters
3. count:       find the most-frequent adjacent symbol pair (a, b) in the corpus
4. merge:       glue every (a, b) into a new symbol ab, log it, go back to step 3
```

Run that `N` times and you get a **priority-ranked merge table**. To encode new text, just walk the table from top to bottom and glue characters back together.

The hidden lever is step 1 — **BPE merges strictly inside a pre-tokenized unit; it never crosses one**. So:

- A run of CJK characters is one unit → Chinese merges are confined to within that run
- A run of ASCII letters and digits is one unit
- Every punctuation mark stands alone

This is why **Chinese almost never learns multi-character tokens**. In real large-vocabulary models it does happen (because the corpus is huge enough), but it costs you: a giant pile of Chinese entries now squeezes the budget for other languages. In small-vocabulary models, Chinese is essentially one token per character.

## A working example you can verify by hand

In an English corpus `the` appears 50 times; `th` appears 80 times. In round one, `th` wins on count → merged into `th`. In round two, `the` (now `th` + `e`) has a high count → merged into `the`. The end result: `the` is a single token.

In a Chinese corpus, "模型" appears 30 times, but "型" alone appears 100 times and "模" alone appears 90 times. If those happen to live in different pre-tokenized units, the pair never co-occurs in the count. **Whether they become a single token depends entirely on how pre-tokenization slices the text and how frequent their unit is.**

## Why Chinese is more "expensive" in tokens

This is a trap a lot of people fall into. Unpack it layer by layer:

1. **Information density differs.** Chinese expresses the same idea in many fewer characters than English. By raw count, Chinese looks cheaper.
2. **But the tokenizer doesn't slice by meaning — it slices by corpus frequency.** High-frequency English combinations merge generously: `information` can fit in 2 tokens. Chinese "信息" (information) is either two tokens or needs a dedicated slot in the vocab.
3. **The result:** in real workloads, the Chinese token count for the same content is routinely 1.5–2× the English one.

Practical consequences:

- **API cost**: billing is per token, so Chinese costs more.
- **Context budget**: in the same 128K window, Chinese holds less content.
- **Generation speed**: one Chinese character still takes one generation step.

In the site's tokenizer module, type the same content in Chinese and English and watch the token count and average-token-length stats — the gap is striking.

## The "numbers get sliced" trap

Another high-frequency landmine: GPT-2-era BPE would chop `1234` into `12` `34` or `1` `234` depending on the corpus. That single choice is one of the reasons early GPTs were terrible at arithmetic — the model sees randomly-sliced digit fragments and has no idea where each digit sits.

Since LLaMA 3 the mainstream approach has been **digit-by-digit splitting** (`1234` → `1` `2` `3` `4`), and arithmetic quality jumped noticeably. So if your task involves numbers, the tokenizer choice itself is a variable you have to check.

In the site, type a sentence with numbers in it (for example `GPT-4 has 1.8T parameters, 128K context.`) and watch how the digits get sliced.

## Compression ratio: a practical health metric

The site's "average token length" is just a compression ratio. It has one very practical use: **telling whether a tokenizer is a good fit for your domain text**.

How: take a representative chunk of your real corpus and run the tokenizer over it. Look at how many characters per token you get on average.

- English around 4 characters/token is normal
- Chinese around 1–1.5 characters/token is common
- If your domain text (medical jargon, code, chemical formulas, legal language) sits clearly below those baselines, this tokenizer is wasting your money in that scenario — the cost will be much higher than you expected

## Common misconceptions

**"A token is a word."** — No. A token is a statistical artifact. It can be a word, a stem, a character, punctuation, even a space.

**"The same word always tokenizes the same way."** — Not necessarily. BPE merges happen inside pre-tokenized units, so "hello" at the start of a sentence (with a leading space) and "hello" in the middle can differ. Many tokenizers encode the leading space separately, which is why `hello` and ` hello` are two different tokens.

**"Bigger vocabulary is always better."** — No. A bigger vocabulary means shorter sequences and less compute, but the embedding matrix and the final unembedding layer grow with it; the parameter count and VRAM go up; and rare tokens never get enough training signal to learn a good representation. This is a real trade-off.

## How to play with this on the site

1. Type a mixed Chinese/English sentence and compare the two tokenization granularities side by side.
2. Drag the "merge count" slider from 0 up to 400 and watch the token count drop while the vocabulary grows.
3. Open "merge process animation" and watch one unit get glued together, step by step.
4. Swap in a custom corpus (paste a paragraph from your own field) and see what merge rules come out.

## Going further

- Original paper: [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- Canonical implementation: [OpenAI GPT-2's `encoder.py`](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
- To understand why byte-level BPE wipes out OOV completely, read GPT-2's paper §2.2.
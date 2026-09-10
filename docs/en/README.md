# Companion essays — English

The site is for *seeing*. These essays are for *explaining*. Every article maps to one module, and the suggested loop is the same as on the Chinese side:

1. Play with the corresponding module first — build the intuition.
2. Come back here and write it down as something you can actually say.
3. Return to the site, twist a knob, and verify what you just read.

| # | Article | Module |
| --- | --- | --- |
| 01 | [Tokenization: why the first hurdle is "how do you cut the text?"](./01-tokenizer.md) | ① Tokenization |
| 02 | [Embeddings & Positional Encoding: a coordinate and a seat number for every token](./02-embedding.md) | ② Embeddings & Positional Encoding |
| 03 | [Multi-Head Attention: where is the model actually looking?](./03-attention.md) | ③ Multi-Head Attention |
| 04 | [Sampling: what temperature / top-k / top-p each really change](./04-sampling.md) | ④ Token-by-token Generation |
| 05 | [KV Cache: trading memory for compute](./05-kvcache.md) | ⑤ KV Cache |
| 06 | [The Transformer Block: the residual highway and the FFN](./06-transformer-block.md) | ⑥ Block Data Flow |

中文版的在同一 [../README.md](../README.md)；章节编号与中文版一一对应，可以中英对着读。

## One through-line

A large model really only does one thing: **predict the next token given what came before**.

If you follow that line forward, the six modules fall out naturally:

```
to predict the next token
  └─ you need the text as numbers first       → ① Tokenization
      └─ the numbers need to live in a space, with an order → ② Embeddings & Positional Encoding
          └─ tokens need to talk to each other → ③ Attention
              └─ one layer isn't enough — stack many → ⑥ Transformer Block
                  └─ finally a distribution comes out → ④ Sampling (picking which one)
                      └─ going one token at a time is too slow → ⑤ KV Cache
```

## A note on "simulation"

Every number on the site is a **deterministic simulation**, not a forward pass through a trained model. Each article repeats this reminder up front.

What is worth being explicit about: **the only thing simulated is the "learned weight matrices". The algorithms are real.**

- BPE merges are counted from real corpus frequencies;
- Attention runs the full Q/K/V projection, the `1/√d_k` scaling, the causal mask and softmax — the whole correct pipeline;
- The three sampling knobs follow the same logic as Hugging Face's `TemperatureLogitsWarper` / `TopKLogitsWarper` / `TopPLogitsWarper`;
- The KV-cache complexity analysis is a closed-form derivation that lines up with what a profiler will report.

So it is safe to use this site to **build intuition**. It is not safe to use it to **quote specific numbers**.
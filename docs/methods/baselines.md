# Baselines

Six methods without any of the new ideas. The two that matter for the master
table are `cot` (Baseline 1) and `selar` (Baseline 2).

| Method | Implementation | Knobs | Per-step statistics |
|---|---|---|---|
| `cot` | shared loop, latent steps off | none | entropy only |
| `cot_greedy` | shared loop, `do_sample=False` | none | entropy only |
| `selar` | shared loop | `SELAR_TOPK`, `SELAR_TAU`, `CONTRASTIVE_W`, `SOFT_MASK` | all |
| `selar_upstream` | `generate_selar()` verbatim | `SELAR_TOPK`, `SELAR_TAU` | none |
| `cot_upstream` | `generate_cot()` verbatim | none | none |
| `swir` | `generate_swir()` verbatim | `SWIR_ALPHA`, `SWIR_MAX_SWITCH` | none |

## `cot`: standard chain of thought

Temperature 0.6, top-k 20, top-p 0.95, min-p 0, thinking mode. The shared loop
with every latent mechanism off. It still computes the normalised top-3
entropy of every step, without acting on it. Its entropy histogram is
therefore directly comparable with SeLaR's, and the share of steps above τ
shows how often SeLaR *would* have intervened.

## `cot_greedy`

The same loop with `argmax` instead of sampling. The paper's "CoT (Greedy)".
Its run directory uses the decoding tag `greedy_L<max tokens>`.

## `selar`: vanilla SeLaR

Algorithm 1 of the paper inside the shared loop:

1. top-k (k = `SELAR_TOPK`) of the filtered distribution, renormalised;
   normalised entropy \(\bar H_t\);
2. exploratory if \(\bar H_t \ge \tau\) and the sampled token is no math symbol;
3. soft embedding \(\sum \hat p_v E[v]\), pushed by
   `CONTRASTIVE_W` · \(\bar H_t\) · (mixture − top-1 embedding);
4. on exploratory steps the next input is that embedding, otherwise the
   sampled token.

`SOFT_MASK` decides what happens to the *other* rows of the batch:

=== "per_row (default)"

    Each row gets its own decision: soft if *it* is exploratory, its sampled
    token otherwise. This is Algorithm 1, and the results do not depend on the
    batch size.

=== "batch_any"

    As soon as any row is exploratory, every row of the batch gets its soft
    embedding (rows below τ included). This reproduces the upstream code's
    behaviour, see [the batching quirk](../background/upstream-code.md#the-batching-quirk-of-generate_selar).

The base gate (k, τ) comes from `SELAR_PRESET`: by default the published
per-dataset setting ([presets](../guide/parameters.md#paper-presets)).

## `selar_upstream`

The authors' `generate_selar()` function, called with `selar_topk` = k and
`entropy_threshold` = τ. It includes the batching quirk and the upstream
stopping rule, and it records no statistics. Use it to show how far the
published code is from per-row SeLaR at the batch sizes used here.

## `cot_upstream`

The authors' `generate_cot()`. The test suite shows token equality with `cot`
(see [guarantees](index.md#guarantees)). On the GPU (smoke test, A100, bf16),
its four outputs were identical to `cot`'s, text for text. It is mainly a
sanity baseline.

## `swir`: SwiReasoning

The authors' copy of SwiReasoning (Shi et al., 2025): it switches between
explicit and latent mode based on whether the entropy rises or falls between
steps, with a window of 512 steps and optional switch counting.

| Knob | Flag | Default | Meaning |
|---|---|---|---|
| `SWIR_ALPHA` | `--alpha` | 1.0 | initial weight of the soft embedding against the `<think>` embedding when switching to latent mode |
| `SWIR_MAX_SWITCH` | `--max_switch_count` | `none` | after this many switches, inject `</think>` and later force the answer |

The authors used α = 0.9 (AIME) and 1.0 (MATH500, GPQA) for Qwen3-8B, and
α = 0.3 for Qwen3-1.7B on AIME, according to their result file names.

## Shared-loop details that apply to every method above

- **Stopping**: a row stops at any EOS id of the tokenizer or of
  `generation_config` (Qwen3: `<|im_end|>` 151645 and `<|endoftext|>` 151643).
  The three upstream functions stop only at `tokenizer.eos_token_id`.
- **Prefill**: `logits_to_keep=1`, so only the last position's logits are
  computed. `--no_logits_to_keep` restores the upstream prefill (bit-exact
  first step).
- **Finished rows** leave the batch, together with their KV cache rows, as
  soon as they emit EOS.

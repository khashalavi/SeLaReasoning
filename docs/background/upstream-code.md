# The upstream code

What the fork started from (commit `dc71537`), and the behaviours of that code
the experiments have to account for.

## Files

| File | Content |
|---|---|
| `src/generation_utils.py` | `set_seed`, `apply_sampling_filter` (top-k → top-p → min-p), `get_math_symbols_ids`, and three hand-written decoding loops: `generate_cot`, `generate_selar`, `generate_swir` |
| `src/grader.py` | answer extraction and checking with `math_verify` (boxed answer first), one grader per benchmark |
| `scripts/run.py` | evaluation entry point, launched with `torchrun`, one process per GPU |
| `scripts/merge.py` | merges the per-rank result files |
| `datasets/` | the five benchmarks as Hugging Face `load_from_disk` folders |
| `results/` | the authors' own result files (COT, SeLaR, SwiR; Qwen3-1.7B / 8B / 32B) |

SeLaR-X leaves all of these untouched. Its own code is in `src/selarx.py` and
`scripts/sela_*.py` ([file map](../reference/index.md)).

## Datasets

| Name | Folder | Problems | Question / answer columns |
|---|---|---:|---|
| `gsm8k` | `gsm8k_test` | 1,319 | `question` / `answer` (gold after `####`) |
| `math500` | `math_500_test` | 500 | `problem` / `answer` |
| `aime_2024` | `aime_2024_train` | 30 | `problem` / `answer` |
| `aime_2025` | `aime_2025` | 30 | `problem` / `answer` |
| `gpqa_diamond` | `gpqa_diamond_mc_test` | 198 | `problem` (options inside) / `solution` (e.g. `\boxed{D}`) |

## What `run.py` does

1. Splits the dataset into equal contiguous chunks, one per rank.
2. Wraps every question in the prompt
   `{q}\nPlease reason step by step, and make sure put your final answer within \boxed{}.`,
   applies the chat template with `enable_thinking=True`, left-pads the batch.
3. Calls one of the decoding loops; every loop samples with temperature, top-k,
   top-p and min-p, and stops a sequence at `tokenizer.eos_token_id`.
4. Splits the output at the last `</think>` (id 151668 for Qwen, 128014 for
   the Llama-based distill). The text after it is graded.
5. Counts tokens by re-encoding the decoded output, and writes
   `logs/<model>_<dataset>_<method>_…_rank<r>.json`.

SeLaR-X keeps steps 2 to 5 unchanged, so its numbers stay comparable with
`results/`.

## The batching quirk of `generate_selar`

!!! warning "In a batch, one uncertain row makes every row soft"
    ```python
    # step t: computed for ALL rows of the batch whenever any row is uncertain
    if is_high_entropy.any():
        soft_embeds = base_soft_embed + contrastive_strength * diff_unit * diff_norm
    ...
    use_soft_input = is_high_entropy

    # step t+1
    if use_soft_input.any():
        model_inputs = {"inputs_embeds": soft_embeds.unsqueeze(1), ...}   # every row
    else:
        model_inputs = {"input_ids": next_tokens.unsqueeze(1), ...}
    ```
    The choice is made for the **whole batch**. Rows whose own entropy is below
    τ, including rows blocked by the math guard, also get their soft mixture
    instead of their sampled token.

Consequences:

- With the README's `--batch_size 256` and about 10% exploratory steps per
  row, some row is uncertain at almost every step: \(1 - 0.9^{256} \approx 1\).
  The published SeLaR runs were therefore close to "soft everywhere, with an
  entropy-scaled push". That is not the selective method of Algorithm 1.
- The results **depend on the batch size**. At batch size 1 the code does
  exactly what the paper describes.

How SeLaR-X handles it:

| | Rows below τ get | Use it for |
|---|---|---|
| `selar` (default `SOFT_MASK=per_row`) | their sampled token, as in Algorithm 1 | the paper's method, independent of the batch size |
| `selar` with `SOFT_MASK=batch_any` | their soft embedding, like upstream | reproducing the quirk inside the shared loop |
| `selar_upstream` | upstream behaviour, verbatim function | comparing with the original code |

The tests show that `selar` with `batch_any` reproduces `generate_selar` token
for token on a batch, and that `selar` with `per_row` does so at batch size 1
([Findings](../findings/index.md#equivalence-with-the-upstream-code)). The
report counts both **gate decisions** and **soft inputs actually fed**, so the
quirk is visible in the diagnostics (`Soft inputs` > `Latent steps`).

## Other upstream behaviours kept on purpose

- **Filtered probabilities everywhere.** The entropy and the soft mixture use
  the distribution *after* temperature, top-k = 20 and top-p = 0.95. Steps
  where top-p leaves a single token have entropy exactly 0.
- **bf16 arithmetic.** Logits, probabilities, entropy and mixture are bf16 for
  bf16 models (Transformers 4.51 does not upcast logits). Entropy values are
  therefore quantised; the paper's maximum observed entropy is 0.9921875.
- **The sampled token is written** even when the next input is soft.
- **Math-symbol guard** on the sampled token.

## Upstream details SeLaR-X does differently

| Upstream | SeLaR-X | Why |
|---|---|---|
| stops only at `tokenizer.eos_token_id` | stops at every EOS id of the tokenizer **and** `generation_config` (Qwen3: 151645 and 151643) | a stray end-of-text token (151643) would otherwise run on to the token budget; applies to every method of the shared loop alike |
| copies the whole KV cache every step (`batch_select_indices` even when no row finished) | filters the cache only when a row finishes | same values, faster (CoT: 33.5 vs 34.8 ms/step in the benchmark) |
| prefill computes logits for every prompt position | `logits_to_keep=1` | less memory at large batch; `--no_logits_to_keep` restores the old prefill |
| `torchrun`, one rank per GPU, one big job | one Slurm job per shard, resumable | survives time limits and node failures |
| `requirements.txt` without the grader's dependencies | `requirements-sela.txt` with `math-verify` and `latex2sympy2_extended` | `src/grader.py` imports them |

## The authors' result files

`results/<COT|SeLaR|Swir>/<model>/<model>_<dataset>_<method>_<settings>_<max tokens>_<merged|rank0>.json`.
The settings in the file names record what produced each published number,
e.g. `selar_k3_t0.4_38912` for Qwen3-8B AIME24.

!!! info "Two inconsistencies, neither affecting a published number"
    - `merge.py` globs `…_<method>_<max tokens>_rank*.json`, which does not
      match SeLaR's own log names (`…_selar_k3_t0.5_…`). Several published
      SeLaR results are therefore single `rank0` files.
    - `Qwen3-1.7B_math500_selar_32768_merged.json` stores `"correct": 460`
      next to `"accuracy": 0.926`. Its 500 per-problem flags give 463 =
      92.6%, which is the published number. The report always recomputes from
      the per-problem flags.

The report lists these files as a reference next to the pipeline's own
CoT / SwiR / SeLaR runs ([Reporting](../guide/reporting.md#7-reference)).

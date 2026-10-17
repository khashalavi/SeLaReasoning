# sela_run.py

`scripts/sela_run.py` · evaluates **one shard** of one run on one GPU

The heart beat runs it through `Sela_beat.sh --worker`, which also activates
the environment and passes `--time_left` from Slurm. It can also be run
directly ([manual runs](../guide/manual-runs.md)).

## Flags

### Run identity

| Flag | Default | Meaning |
|---|---|---|
| `--method` | *required* | `cot`, `cot_greedy`, `selar`, an idea (`layer` `sem` `ltr` `fhg` `dynk`), ideas joined by `+`, `cot_upstream`, `selar_upstream`, `swir` |
| `--model_name` | *required* | Hugging Face id; used in the result files and for model quirks |
| `--model_path` | `--model_name` | local weights directory (the heart beat passes `$SELA_ROOT/models/<id>`) |
| `--dataset_name` | *required* | `gsm8k`, `math500`, `aime_2024`, `aime_2025`, `gpqa_diamond` |
| `--datasets_dir` | `<repo>/datasets` | where the `load_from_disk` folders are |
| `--n_samples` | all | use the first N problems |
| `--num_shards` | 1 | the run is split into this many contiguous shards |
| `--shard_index` | 0 | which one (0-based) |
| `--run_dir` | *required* | the run directory; this shard writes to `<run_dir>/shards/` |
| `--seed` | 41 | run seed (each batch is re-seeded from it, see below) |

### Decoding and execution

| Flag | Default | Meaning |
|---|---|---|
| `--batch_size` | 32 | problems per generation batch |
| `--max_new_tokens` | 32768 | generation budget |
| `--temperature`, `--top_p`, `--top_k`, `--min_p` | 0.6, 0.95, 20, 0.0 | sampling filter, as upstream |
| `--device` | `cuda` | `cpu` for tests on the login node |
| `--time_left` | none | Slurm's remaining time (`[d-]hh:mm:ss`); enables the time budget |
| `--no_logits_to_keep` | off | compute prefill logits at every position (upstream behaviour) |

### SwiR (`--method swir`)

| Flag | Default | Meaning |
|---|---|---|
| `--alpha` | 1.0 | SwiR's α₀ |
| `--max_switch_count` | none | SwiR's switch limit |

### SeLaR-X knobs

Generated from the fields of `SelarXConfig`. A flag that is not given keeps
the dataclass default ([parameters](../guide/parameters.md)).

| Flags | Type |
|---|---|
| `--topk`, `--tau`, `--contrastive_weight`, `--soft_mask` | int, float, float, `per_row`/`batch_any` |
| `--math_guard` / `--no-math_guard` | bool |
| `--layer_frac`, `--layer_tau`, `--layer_alpha`, `--layer_topk`, `--layer_direction`, `--layer_rel_scale`, `--layer_gate` | float, float, float, int, word, float, word |
| `--sem_topk`, `--sem_sim`, `--sem_window`, `--sem_gamma`, `--sem_floor` | int, float, int, float, float |
| `--ltr_max`, `--ltr_exit`, `--ltr_decay`, `--ltr_commit` | int, float, float, word |
| `--fhg_theta`, `--fhg_mode`, `--fhg_layers` | float, word, word |
| `--dynk_mass`, `--dynk_min`, `--dynk_max` | float, int, int |

An invalid combination (unknown idea, `dynk_min > dynk_max`, a word that is
not allowed, …) stops with a `ValueError` before the model is loaded.

## Method → generation function

| `--method` | Called |
|---|---|
| `cot`, `cot_greedy` | `generate_selarx` with latent steps off (`cot_greedy`: `do_sample=False`) |
| `selar`, ideas, combinations | `generate_selarx` with `SelarXConfig.for_method(method, **knobs)` |
| `cot_upstream` | `generate_cot` |
| `selar_upstream` | `generate_selar(selar_topk=topk, entropy_threshold=tau)` |
| `swir` | `generate_swir(alpha_0=alpha, max_switch_count=…)`, convergence words `</think>` for Qwen |

## What a shard does

1. **Problems.** Loads the dataset, keeps the first `n_samples`, and takes the
   shard's contiguous range: \(\lceil N / \text{num\_shards} \rceil\) problems
   per shard.
2. **Resume.** Reads `shard_<i>of<n>.partial.jsonl`, keeping every complete
   line. A half-written last line from a killed job is ignored. If
   `shard_<i>of<n>.json` already exists, it exits immediately.
3. **Config.** Writes `<run_dir>/config.json` if it does not exist yet.
4. **Model.** Tokenizer with left padding (pad = EOS if the tokenizer has
   none), model in its own dtype on GPU 0. The EOS set is the tokenizer's EOS
   plus `generation_config.eos_token_id`; `</think>` is looked up in the
   vocabulary.
5. **Batches.** For each batch of the remaining problems:
    - **time budget**: stop if 1.25 × the slowest batch so far would end
      within 5 minutes of the Slurm limit;
    - **prompt**: question + the upstream instruction, chat template with
      `enable_thinking=True`;
    - **seed**: `(seed · 1000003 + first index · 7919) mod 2³¹`, so that a
      resumed batch draws the same numbers;
    - **generation**, retried as two halves on CUDA out-of-memory
      (recursively; a single problem that still does not fit is recorded with
      `"error": "oom"`);
    - **grading**: split at the last `</think>`, `answer_match` on the rest;
      a grader exception counts as wrong;
    - the batch's records are appended to the `.partial.jsonl` and fsynced.
6. **Finish.** When every problem of the range is present, writes
   `shard_<i>of<n>.json` atomically (temporary file + rename) and deletes the
   partial file.

Exit codes: 0 for a complete shard and for a clean stop at the time budget
(the heart beat resubmits); non-zero for errors.

## Example output

```text
run_dir=/lustre/.../runs/Qwen3-0.6B/gsm8k/n4/selar_k3_t0.6/T0.6_p0.95_k20_mp0_L512/seed41
shard 1/1: problems [0, 4) of 4; 0 already done, 4 to go
model loaded in 2s on NVIDIA A100-SXM4-80GB
[01:24:42] batch 1/1: 4 problems in 18s, shard accuracy so far 1/4 = 25.00%
shard complete: 1/4 correct -> /lustre/.../shards/shard_000of001.json
```

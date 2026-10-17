# File formats

All files are UTF-8. JSON files are written atomically (temporary file,
fsync, rename), so a reader never sees half a file.

## `plan.tsv`

`$SELA_ROOT/plan.tsv`, rewritten every pass. One line per (phase, run). A run
in both phases appears twice. `-` stands for an empty field.

| Column | Example |
|---|---|
| `phase` | `main` |
| `method` | `sem+dynk` |
| `config` | `sem+dynk_k3_t0.6_sk5_ss0.82_sw15_sg1_sf0.3_dm0.9_dn2_dx8` |
| `knobs` | `SELAR_TOPK=3 SELAR_TAU=0.6 CONTRASTIVE_W=1 SOFT_MASK=per_row SEM_TOPK=5 …` |
| `model` | `Qwen/Qwen3-8B` |
| `dataset` | `gsm8k` |
| `n` | `1319` |
| `seed` | `41` |
| `run_dir` | `/lustre/…/runs/Qwen3-8B/gsm8k/n1319/<config>/T0.6_p0.95_k20_mp0_L32768/seed41` |
| `num_shards` | `4` |
| `is_default` | `1` if the configuration is the method's default |

`./Sela_beat.sh --plan-tsv` prints the same without writing it.

## `journal.txt`

```text
<YYYY-MM-DD HH:MM:SS> | <EVENT> | <key> | job=<id or ->
```

Keys and events: [The journal](../guide/heart-beat.md#the-journal).

## `*.attempts`

One line per job start: `<unix time> <slurm job id or "manual"> <problems already done>`.

```text
1792192597 28518075 0
```

Shards: `runs/…/shards/shard_<i>of<n>.attempts`. Environment / downloads:
`state/env.attempts`, `state/download_<model tag>.attempts`.

## `config.json`

Written once per run, by the first worker:

| Field | Content |
|---|---|
| `method`, `model`, `dataset`, `seed` | the run's identity |
| `n_samples`, `num_shards`, `batch_size`, `max_new_tokens` | its size |
| `decoding` | `temperature`, `top_p`, `top_k`, `min_p`, `do_sample` |
| `selarx` | every `SelarXConfig` field (null for `cot_upstream` and `swir`) |
| `swir` | `alpha`, `max_switch_count` (only for `swir`) |
| `logits_to_keep` | prefill mode |
| `code_version` | `<short commit>` or `<short commit>+dirty` |

## Shard files

`shards/shard_<i>of<n>.partial.jsonl`: one problem record per line, appended
after every batch.

`shards/shard_<i>of<n>.json`, once the shard is complete:

| Field | Content |
|---|---|
| `format` | `selarx-shard-v1` |
| `shard_index`, `num_shards`, `start`, `end` | the problem range [start, end) |
| `total`, `correct` | counts for this shard (kept adjacent at the top of the file, for cheap reading) |
| `run` | the same content as `config.json` |
| `env` | `torch`, `transformers`, `gpu`, `slurm_job_id` |
| `details` | the problem records, sorted by index |

### Problem record

| Field | Type | Content |
|---|---|---|
| `index` | int | position in the dataset (0-based) |
| `question`, `gold` | str | the problem and its reference answer |
| `prediction` | str | what `math_verify` extracted (`"No extracted answer"` if nothing) |
| `correct` | bool | graded result |
| `thinking`, `answer_content` | str | the text up to and after the last `</think>` |
| `tokens` | int | tokens of the decoded output (upstream's measure) |
| `gen_tokens` | int | generated ids including EOS |
| `hit_max` | bool | no EOS within the budget |
| `batch` | object | `size`, `seconds` (wall time of the batch), `seed` |
| `stats` | object | the per-step counters and `h_hist` (absent for upstream methods) |
| `error` | str | only `"oom"`, for a problem that did not fit even alone |

## `merged.json`

```text
{"summary": { …as summary.json… }, "details": [ …every problem record, sorted by index… ]}
```

## `summary.json`

| Field | Content |
|---|---|
| `format` | `selarx-summary-v1` |
| `run` | the run information plus `code_versions` and `gpus` of all shards |
| `total`, `correct`, `accuracy` | recomputed from the records |
| `length_stats` | `max_new_tokens`, `avg_total_token_len`, `correct_avg_total_token_len`, `wrong_avg_total_token_len` (names as in upstream `merge.py`) |
| `tpca` | average tokens ÷ accuracy (null at accuracy 0) |
| `hit_max_rate`, `errors`, `gpu_seconds` | truncation share, OOM records, generation GPU time |
| `per_index_correct` | `"0110…"`, one character per problem in index order (for paired tests) |
| `first_index` | index of the first problem (pairing check) |
| `activation` | `total_steps`, `latent_steps`, `ratio`, `avg_per_sample_ratio`, `soft_input_ratio`, `gate_open_ratio`, `guarded_ratio`, `vetoed_ratio`, `rollouts_per_sample`, `avg_rollout_len`, `commits`, `avg_k`, `avg_clusters`, `avg_ratio`, `avg_dynamic_threshold` |
| `entropy` | `mean`, `std`, `median`, `token_entropy_mean`, `share_ge_0.5`, `hist` (20 bins) |
| `merged_at` | timestamp |

`activation` and `entropy` are absent for the upstream methods. Ratios with
an empty denominator are null.

## `status.out`

A fixed-width table, rewritten every pass: `PHASE METHOD CONFIG MODEL DATASET N
SEED SHARDS JOBS SAMPLES ACC PROGRESS STATE`, followed by a summary per phase.
States: [Following progress](../guide/heart-beat.md#following-progress).

## `reports/all_runs.csv`

One row per merged run on disk; columns in
[The CSV](../guide/reporting.md#the-csv).

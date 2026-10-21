# Resources and timing

The RESOURCES block of `Sela_beat.sh` decides where jobs run and how the work
is cut into jobs.

## Jobs and partitions

| Job | Partition | Time | Memory | GPU | Variables |
|---|---|---|---|---|---|
| heart beat (`sela_beat`) | `sgpu_medium` | 24 h | 2G | none | `#SBATCH` header |
| shard (`sx_…`) | `sgpu_short` | 8 h | 64G | 1 | `GPU_PARTITION`, `GPU_TIME`, `GPU_MEM`, `GPU_GRES`, `GPU_EXTRA_SBATCH` |
| environment (`sx_setup_env`) | `sgpu_devel` | 1 h | 32G | 1 | `ENV_PARTITION` |
| download (`sx_dl_…`) | `sgpu_short` | 4 h | 32G, 4 CPUs | none | `DOWNLOAD_PARTITION` |
| smoke-test shards | `sgpu_devel` | 30 min | 64G | 1 | set by `--smoke` |

The marvin GPU partitions:

| Partition family | GPUs per node | `_devel` | `_short` | `_medium` | `_long` |
|---|---|---|---|---|---|
| `sgpu_*` | 4 × A100-SXM4 **80 GB** | 1 h | 8 h | 24 h | 7 d |
| `mlgpu_*` | 8 × A40 48 GB | 1 h | 8 h | 24 h | 7 d |

Every GPU node has AMD EPYC 7713 CPUs. A GPU job gets 32 CPU cores by default
(`DefCpuPerGPU=32`). The `medium` and `long` QoS caps a user at 48 GPUs.
`sgpu_devel` ran two of the smoke test's jobs at a time.

To use the A40 nodes: `GPU_PARTITION=mlgpu_short` and `GPU_SCALE=0.5` (halves
`BATCH_SIZE` and `SHARD_SIZE`). These tables are marvin's; for bender's
partitions see [Running on bender](bender.md).

## Shards and batches

```bash
declare -A SHARD_SIZE=([gsm8k]=330 [math500]=125 [gpqa_diamond]=99 [aime_2024]=30 [aime_2025]=30)
declare -A BATCH_SIZE=([gsm8k]=48 [math500]=32 [gpqa_diamond]=24 [aime_2024]=15 [aime_2025]=15)
```

A **shard** is the unit of scheduling: one job, a contiguous range of
problems. A **batch** is the unit of generation and of saving: the worker
appends each finished batch to the shard's `.partial.jsonl`.

The batch sizes follow from the KV cache, which dominates memory for long
reasoning traces. Qwen3-8B stores 36 layers × 8 KV heads × 128 dims × 2
(K, V) × 2 bytes ≈ **144 KiB per token**. The model takes 16 GB of the
80 GB. The remaining ~64 GB hold at most about 430,000 cached tokens per GPU,
in practice somewhat fewer next to the activations:

| Dataset | Avg tokens (Qwen3-8B CoT, authors' files) | Batch | Typical peak cache |
|---|---:|---:|---|
| GSM8K | 2,359 | 48 | 48 × ~2.6k tokens ≈ 18 GB |
| MATH500 | 5,313 | 32 | ~25 GB early; a few runaway traces to 32k later |
| GPQA | 9,608 | 24 | 20 × 10k ≈ 29 GB |
| AIME | 15,500–19,100 | 15 | 12 × 15k ≈ 26 GB |

Finished sequences leave the batch immediately, so the cache shrinks over a
batch. If a batch still runs out of memory, the worker frees it and retries it
as two halves, recursively. A single problem that does not fit is recorded as
wrong with `"error": "oom"`; this has not happened with Qwen3-8B on A100-80GB
at these sizes.

## How long things take

Per-step cost, measured on one A100-80GB (Qwen3-0.6B, batch 16, 256 steps):

| Method | ms / step | | Method | ms / step |
|---|---:|---|---|---:|
| `cot` | 33.53 | | `sem` | 34.13 |
| `cot_upstream` | 34.81 | | `ltr` | 34.16 |
| `selar` | 34.16 | | `fhg` | 35.89 |
| `selar_upstream` | 36.42 | | `dynk` | 34.50 |
| `layer` | 35.34 | | `sem+dynk` | 34.26 |
| | | | `layer+ltr` | 35.44 |

At these sizes the loop is launch-bound, not compute-bound. Every variant
costs at most 7% more than CoT. An 8B model costs more per step; expect
roughly 40–60 ms.

A batch takes as long as its **longest** trace: up to 32,768 steps (38,912
for AIME). Rough job durations for Qwen3-8B:

| Dataset | Per batch | Per shard | Shards |
|---|---|---|---:|
| GSM8K | 5–15 min | ~1–1.5 h (7 batches) | 4 |
| MATH500 | 10–25 min | ~1–1.6 h (4 batches) | 4 |
| GPQA | 15–25 min | ~1.5–2 h (5 batches) | 2 |
| AIME | 20–30 min | ~1 h (2 batches) | 1 |

That gives on the order of 100–150 GPU hours for the main phase (84 jobs),
and about the same again for the default sweep (89 jobs, mostly MATH500-100
and AIME24). With dozens of GPUs in parallel, a phase finishes within a day.

!!! note "Shards that hit the time limit"
    A shard should fit in `GPU_TIME`. If it does not, nothing is lost but the
    batch in progress. The worker reads Slurm's remaining time at start-up,
    and before each batch it checks whether 1.25 × its slowest batch so far
    still fits before the limit minus 5 minutes. If not, it stops cleanly.
    The heart beat resubmits, and the next job continues with the next batch.
    A first batch longer than the whole time limit can never finish. Stall
    protection then stops the shard after three attempts without progress;
    lower `BATCH_SIZE` for that dataset.

## Submission limits

| Variable | Default | Purpose |
|---|---|---|
| `MAX_SUBMIT_PER_PASS` | 150 | new GPU jobs per pass; the rest follow at the next pass |
| `SUBMIT_PAUSE` | 1 s | gap between `sbatch` calls; bursts trip the controller's rate limit |
| `SUBMIT_TIMEOUT` | 300 s | give up on one hanging `sbatch`; a job that got in is adopted next pass |
| `MAX_STALLED` | 3 | attempts without progress before a shard is `GAVE_UP` |
| `CHECK_INTERVAL` | 3600 s | time between passes |
| `PREREQ_INTERVAL` | 600 s | time between passes while the environment or a model is pending |

## Storage

| What | Size |
|---|---|
| `env/` | ~19 GB as reported by `du` on Lustre (torch with the CUDA 12.8 libraries) |
| `cache/pip` | ~4 GB |
| `models/Qwen/Qwen3-8B` | ~16 GB |
| one run's `merged.json` | MB to tens of MB: every problem's full reasoning trace |

The workspace expires: [check and extend it](setup.md#the-lustre-workspace).

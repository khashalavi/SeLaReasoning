# What is saved where

Everything the pipeline produces goes to Lustre. The repository receives only
the final report.

## Overview

```text
/lustre/scratch/data/s76kalav_hpc-memory/selareasoning/      = $SELA_ROOT
├── env/                       Python venv (.selar_env_complete when built)
├── models/<org>/<name>/       model weights (.download_complete when complete)
├── hf_home/                   Hugging Face cache
├── cache/                     pip, pycache, triton, inductor, xdg, nv, mpl
├── state/                     attempts files of the environment and download jobs
├── logs/
│   ├── sela_beat.log          every pass of the heart beat (sela_beat_smoke.log for --smoke)
│   ├── reporting.log          every run of reporting.sh
│   └── jobs/<job name>_<job id>.out   one file per Slurm job (stdout and stderr)
├── journal.txt                job history, one line per event
├── plan.tsv                   the expanded SELECTION of the last pass
├── status.out                 progress table of the last pass
├── reports/
│   ├── all_runs.csv           one row per merged run on disk
│   ├── upstream_cache.json    parsed copy of the authors' results/
│   └── history/               earlier versions of the report
├── runs/                      the results, see below
└── smoke/                     the same layout for --smoke: runs/, journal.txt, plan.tsv,
                               status.out, reports/, smoke_report.md
```

```text
~/SeLaReasoning/
└── result/selar_report.md     the final report (reporting.sh)
```

`result/` is singular. It is distinct from the upstream `results/` folder,
which holds the authors' published runs.

## Run directories

```text
runs/<model>/<dataset>/n<N>/<config tag>/<decoding tag>/seed<S>/
```

| Segment | Example | From |
|---|---|---|
| `<model>` | `Qwen3-8B` | the part of the Hugging Face id after `/` |
| `<dataset>` | `aime_2025` | `DATASETS` |
| `n<N>` | `n30`, `n100`, `n1319` | the number of problems actually used |
| `<config tag>` | `sem_k3_t0.5_sk5_ss0.82_sw15_sg1_sf0.3` | method + every knob ([tags](parameters.md#which-method-receives-which-knobs)) |
| `<decoding tag>` | `T0.6_p0.95_k20_mp0_L38912` | temperature, top-p, top-k, min-p, max new tokens (`greedy_L…` for `cot_greedy`) |
| `seed<S>` | `seed41` | `SEEDS` |

The path encodes the full configuration, which has three consequences:

- a finished run is never repeated, and two configurations never share a
  directory;
- changing a default creates new runs next to the old ones;
- runs of the same configuration from the main and the sweep phase are the
  same directory.

!!! warning "The path does not encode the code version"
    After changing the code of a method, its finished runs are **not** recomputed
    automatically. Delete the affected run directories, or the whole
    `runs/<model>/<dataset>/n<N>/<tag>`, and the next pass runs them again.
    Every run records the commit it ran with (`code_version`, `+dirty` when
    the working tree had uncommitted changes), in `config.json`, in each shard
    and in `summary.json`.

## Inside one run directory

| File | Written by | When | Content |
|---|---|---|---|
| `config.json` | the first worker of the run | at its start | method, model, dataset, problems, seed, shards, batch size, decoding, the full method configuration, code version |
| `shards/shard_<i>of<n>.partial.jsonl` | the worker | after every batch | one line per finished problem; deleted when the shard is complete |
| `shards/shard_<i>of<n>.attempts` | the worker | at every job start | `<epoch> <job id> <problems done>` (stall protection) |
| `shards/shard_<i>of<n>.json` | the worker | when the shard is complete | every problem of the shard, plus run and environment info |
| `merged.json` | the heart beat / `reporting.sh` | all shards complete | every problem of the run, plus the summary |
| `summary.json` | the heart beat / `reporting.sh` | same | the metrics, without the per-problem texts |

Field by field: [File formats](../reference/formats.md).

## What one problem record holds

An illustrative record of a vanilla `selar` run (texts shortened):

```text
{
  "index": 7,
  "question": "Find the number of ...",
  "gold": "204",
  "prediction": "[204, '204']",
  "correct": true,
  "thinking": "<think>\nOkay, so ...\n</think>",
  "answer_content": "\n\n... the answer is \\boxed{204}.",
  "tokens": 13412,
  "gen_tokens": 13409,
  "hit_max": false,
  "batch": {"size": 15, "seconds": 1311.4, "seed": 41000123},
  "stats": {"steps": 13409, "latent": 1430, "soft_inputs": 1430, "gate_open": 1502,
            "guarded": 72, "vetoed": 0, "rollouts": 0, "commits": 0, "k_sum": 0,
            "clusters_sum": 0, "ratio_sum": 0, "ratio_n": 0, "h_sum": 1563.2,
            "h_sq": 912.7, "h_n": 13409, "htok_sum": 1563.2, "thr_sum": 0, "thr_n": 0,
            "h_hist": [10551, 402, 311, 274, …  20 bins in total]}
}
```

`tokens` counts like the upstream code (re-encoding the decoded text);
`gen_tokens` is the number of generated ids including EOS; `hit_max` means no
EOS within the budget. The `stats` counters are described under
[Methods](../methods/index.md#what-every-step-records). The upstream
baselines have no `stats`.

## Cleaning up

| To | Delete |
|---|---|
| re-run one configuration | its run directory |
| re-run one shard | `shards/shard_<i>of<n>.json` (and `summary.json`, `merged.json`) |
| forget a stall | the shard's `.attempts` file, or `--retry-failed` |
| start the smoke test over | `smoke/runs` |
| rebuild the environment | `env/.selar_env_complete` |
| start everything over | stop the heart beat and its jobs, then delete `runs/`, `journal.txt`, `plan.tsv`, `status.out` (keep `env/` and `models/`) |

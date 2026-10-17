# Guide

How to get from an empty Lustre folder to the tables of a paper. Every step
below is an edit of the SELECTION block at the top of `Sela_beat.sh` and, at
most, one `sbatch`. The heart beat does the rest.

## The workflow

```mermaid
flowchart LR
    A[smoke test] --> B[main phase:<br/>baselines + 5 variants]
    B --> C[sweep phase:<br/>ablation grids]
    C --> D[set the chosen<br/>defaults]
    D --> E[Variant 6:<br/>COMBOS]
    E --> F[second model,<br/>more seeds]
    F --> G[report → paper]
```

| Step | Edit in `Sela_beat.sh` | GPU jobs (approx.) | Read in the report |
|---|---|---:|---|
| 1. Smoke test | nothing, `sbatch Sela_beat.sh --smoke` | 13 tiny | everything runs; `smoke_report.md` |
| 2. Main phase | defaults: `PHASES=(main)`, Qwen3-8B, five datasets | 84 | §2 master table, §3 ranking |
| 3. Sweeps | `PHASES=(main sweep)` | + 89 | §6 sweep tables |
| 4. Chosen defaults | set each variant's best values as `X=` | + 12 per changed variant | §2 with the tuned variants |
| 5. Combinations | `COMBOS=("a+b")` from §3 | + 12 per combo | §2 Variant 6 rows |
| 6. Robustness | add `"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"`, `SEEDS=(41 42 43)` | × models × seeds | §2 per model, ± std |

A run that has already finished is never repeated: run directories are named
after their full configuration ([outputs](outputs.md#run-directories)), so
changing a default creates *new* runs and leaves the old ones in place.

## Pages

| Page | Content |
|---|---|
| [Environment setup](setup.md) | the venv, model downloads, caches, a local CPU test env |
| [Running Sela_beat](heart-beat.md) | what one pass does, flags, monitoring, the journal, stall protection |
| [Selection options](selection.md) | every option of the SELECTION block, and how the plan is expanded |
| [Hyper-parameters](parameters.md) | every knob: default, sweep values, flag, run tag |
| [Resources and timing](resources.md) | partitions, batch and shard sizes, time estimates, rate limits |
| [Smoke test](smoke-test.md) | the 10-minute end-to-end check |
| [Manual runs and tests](manual-runs.md) | one shard by hand, CPU tests, merging and reporting by hand |
| [What is saved where](outputs.md) | the Lustre tree, run directories, the files of a run |
| [Reporting](reporting.md) | `reporting.sh`, every section of the report, metric definitions |
| [Troubleshooting](troubleshooting.md) | GAVE_UP, rate limits, OOM, stopping, starting over |

# Smoke test

A ten-minute end-to-end check of the whole pipeline on the real cluster:
environment, model download, every method on a GPU, merging, status and the
report. Run it after changing code, before spending GPU days.

## What `--smoke` runs

The flag replaces the SELECTION with a tiny one and keeps everything else:

| | Smoke setting |
|---|---|
| Methods | `cot cot_greedy selar selar_upstream cot_upstream swir layer sem ltr fhg dynk` |
| Combinations | `sem+dynk`, `layer+ltr` |
| Model | `Qwen/Qwen3-0.6B` (953 MB) |
| Data | the first 4 GSM8K problems, seed 41 |
| Generation | 512 new tokens, one batch of 4, one shard per run |
| Jobs | 13 GPU jobs on `sgpu_devel`, 30 min limit each |
| Tree | `$SELA_ROOT/smoke/`: own runs, journal, plan, status |
| Pace | a pass every 5 minutes; exits once all 13 runs are merged |
| Report | not automatic; `./reporting.sh --smoke` writes `$SELA_ROOT/smoke/smoke_report.md` |

The environment, the model store and the logs are shared with the main tree.
A model or environment built by the smoke test is reused by the real runs.

## Running it

```bash
cd ~/SeLaReasoning
sbatch Sela_beat.sh --smoke        # hands-off: passes every 5 min, exits when done
./reporting.sh --smoke             # afterwards
```

or step by step from the login node, one pass at a time:

```bash
./Sela_beat.sh --smoke --once      # 1st: submits the environment build (if missing)
./Sela_beat.sh --smoke --once      # 2nd: submits the model download (if missing)
./Sela_beat.sh --smoke --once      # 3rd: submits the 13 worker jobs
./Sela_beat.sh --smoke --once      # 4th: merges; repeat until all 13 are MERGED
```

!!! info "A finished smoke test is not repeated"
    Its runs are identified by their configuration. If all 13 are already
    merged, the smoke heart beat exits at its first pass. To repeat it after a
    code change, delete the smoke runs first:

    ```bash
    rm -r /lustre/scratch/data/s76kalav_hpc-memory/selareasoning/smoke/runs
    ```

## What to check

```bash
S=/lustre/scratch/data/s76kalav_hpc-memory/selareasoning
sacct -u $USER -S today -o JobName%32,State,Elapsed | grep sx_   # every job COMPLETED
grep -l Traceback $S/logs/jobs/sx_*                              # prints nothing
cat $S/smoke/status.out                                          # 13 rows, all MERGED
```

In `smoke_report.md`, section 5, the statistics should be plausible: vanilla
SeLaR around 10–15% latent steps, and `fhg` not at 0%. Accuracy means
nothing here (a 0.6B model, 512 tokens, 4 problems).

## The smoke test of 2026-10-17

All 13 jobs completed in 34 s to 2 min 40 s each; the differences came from
node load, not from the methods (the [same-GPU benchmark](resources.md#how-long-things-take)
shows ≤ 7% spread). The gate statistics led to two changes of the defaults
([Findings](../findings/index.md#gate-statistics)):

- `fhg` with the absolute threshold vetoed every opening → `FHG_MODE=relative`,
- the `layer` lens gate steered 61% of the steps → `LAYER_GATE=final` option,
  `LAYER_TAU` sweep up to 0.9.

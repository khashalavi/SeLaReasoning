

## Status

- The **smoke test has run** (`sbatch Sela_beat.sh --smoke`, 2026-10-17 01:31–01:39) on Qwen3-0.6B with GSM8K. All 13 shards finished, none gave up, and `smoke/journal.txt` has no failures. Output is in `$SELA_ROOT/smoke/`.
- The **main run has not started**. There is no `$SELA_ROOT/runs/` and no top-level `journal.txt`. The running `heart_beat` job (ID 28550558) belongs to Disentangling-Reasoning, not SeLaReasoning.
- Done: the venv (`env/`) and Qwen3-0.6B. **Not yet done:** Qwen3-8B, which the main phase needs. The heart beat downloads it automatically.
- `Sela_beat.sh` was edited at 01:56, after the smoke test (01:39). A fresh smoke run is recommended but optional.

## Next steps

```bash
cd ~/SeLaReasoning

# 1. Optional: preview the plan (writes nothing)
./Sela_beat.sh --plan

# 2. Recommended: re-run the smoke test on the current script
sbatch Sela_beat.sh --smoke

# 3. Required: start the main run (runs ~24 h, resubmits itself)
sbatch Sela_beat.sh

# 4. Optional: refresh the report by hand
./reporting.sh
```

## Where to look while it runs

```bash
squeue -u $USER                                        # sela_beat should be listed
S=/lustre/scratch/data/s76kalav_hpc-memory/selareasoning
tail -f $S/logs/sela_beat.log                          # heart beat log
cat $S/status.out                                      # progress table
tail $S/journal.txt                                    # job lifecycle
```

The main plan is 35 runs and 84 GPU jobs (about 14.5k problem generations) on Qwen3-8B. It uses the `sgpu_short` partition (A100 80 GB) with `SELAR_PRESET=paper`.

## Repository note

`git status` shows `next_step.md` as deleted and `next_steps.md` as untracked. Commit or restore them so the file is not lost.

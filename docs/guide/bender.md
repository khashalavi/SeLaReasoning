# Running on bender

The same experiments run on the bender cluster through two small entry
points. They have bender's `#SBATCH` headers and run the shared scripts with
`SELA_SERVER=bender`:

| bender | runs | marvin equivalent |
|---|---|---|
| `Sela_beat_bender.sh` | `Sela_beat.sh` (sourced) | `Sela_beat.sh` |
| `reporting_bender.sh` | `reporting.sh` | `reporting.sh` |

Both clusters share the SELECTION block, every flag, the journal format and
the job logic. They differ only in the `bender` profile of the RESOURCES
block in `Sela_beat.sh`.

```bash
./Sela_beat_bender.sh --plan          # what the SELECTION expands to on bender
sbatch Sela_beat_bender.sh --smoke    # every method end-to-end on Qwen3-0.6B (A40devel)
sbatch Sela_beat_bender.sh            # the hourly heart beat
./Sela_beat_bender.sh --status        # progress from the journal and the live queue
./reporting_bender.sh                 # -> result/selar_report_bender.md
```

## What differs from marvin

| | marvin | bender |
|---|---|---|
| `$SELA_ROOT` | `/lustre/scratch/data/s76kalav_hpc-memory/selareasoning` | `<repo>/sela_data` (git-ignored; bender has no Lustre) |
| heart beat | `sgpu_medium`, 24 h | `A40medium`, 24 h |
| shards | `sgpu_short` (A100 80 GB), 64G | `A40short` (A40 48 GB), 48G |
| environment build | `sgpu_devel` | `A40devel` |
| downloads | `sgpu_short` | `A40short` |
| `--smoke` shards | `sgpu_devel` | `A40devel` |
| `BATCH_SIZE`, `SHARD_SIZE` | as listed (`GPU_SCALE=1`) | halved (`GPU_SCALE=0.5`) |
| venv | `env/` (AMD EasyBuild tree) | `env_a40/` (default tree, Intel nodes) |
| torch | 2.7.0, CUDA 12.8 build | 2.7.0, CUDA 12.6 build |
| helper python | `/usr/bin/python3` | `python3`; below 3.8 the Python module |
| report | `result/selar_report.md` | `result/selar_report_bender.md` |

Each cluster keeps its own tree, journal and report. Runs on one cluster are
never resumed or merged on the other.

!!! note "Comparing numbers across clusters"
    `selar` and the variants gate per row, so the batch size does not change
    their gating for a problem. `selar_upstream` and `SOFT_MASK=batch_any`
    gate per batch, so their bender results are not comparable with
    marvin's. Different GPUs and batch sizes also change floating-point
    results and random draws. Keep each table to one cluster.

## The A100 nodes

`BENDER_GPU=A100` in the bender profile moves the environment build to
`A100devel` and the downloads and shards to `A100short`. It also keeps the
full batch sizes and builds a separate venv, `env_a100/`. The A100 nodes have
AMD EPYC CPUs, so that venv uses Python from
`/software/easybuild-AMD_A100/modules/all`, with MKL and oneDNN limited to
AVX2. This is the same fix as Disentangling-Reasoning's `new_env_a100`. The
two venvs exist side by side, so you can switch back and forth.

The environment job prints the GPU name and its memory (`cuda matmul ok: …
GiB`) in `sela_data/logs/jobs/sx_setup_env_<id>.out`. If bender's A100s have
40 GB rather than 80 GB, also set `GPU_SCALE=0.5` for them.

## Changing the SELECTION

Edit `Sela_beat.sh`, the same file as on marvin. A running bender beat
watches both files: the Slurm copy of `Sela_beat_bender.sh` and the
`Sela_beat.sh` it read at start-up. If either changes, it validates the new
version with `--plan` and hands over to a fresh job.

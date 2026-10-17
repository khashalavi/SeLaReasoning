# Troubleshooting

## Where to look first

```bash
S=/lustre/scratch/data/s76kalav_hpc-memory/selareasoning
tail -n 50 $S/logs/sela_beat.log            # what the heart beat did
cat $S/status.out                           # state of every run
ls -t $S/logs/jobs/ | head                  # newest job logs
grep -F '<part of a run tag>' $S/journal.txt | tail   # history of one run's jobs
```

## Shards stay `WAITING`

`WAITING` means that nothing is queued for the run. Common causes:

| Log line | Cause | Fix |
|---|---|---|
| `waiting for env::setup (job …, PENDING)` | the environment build is queued or running | wait; passes come every 10 min meanwhile |
| `waiting for download::<model> (…)` | the model download is queued or running | wait |
| `GAVE_UP env::setup` / `GAVE_UP download::…` | the build / download failed three times | read `logs/jobs/sx_setup_env_*.out` or `sx_dl_*`, fix, `--retry-failed` |
| none, and no heart beat in `squeue` | the heart beat is not running | `sbatch Sela_beat.sh` |

## `GAVE_UP`

A shard failed three times in a row without finishing a single additional
problem.

1. Find the job: `grep GAVE_UP $S/logs/sela_beat.log | tail`, then its log in
   `logs/jobs/` (the job name is in the journal's `SUBMITTED` lines).
2. Typical causes: a Python error (look for `Traceback`), an environment that
   no longer imports, a first batch that is longer than the time limit (lower
   `BATCH_SIZE`).
3. Fix, then reset the counters:
   [Stall protection](heart-beat.md#stall-protection).

## "Slurm temporarily unable to accept job, sleeping and retrying"

The cluster's controller rate-limits bursts of submissions. `sbatch` retries
on its own and the jobs get in. The heart beat paces its submissions
(`SUBMIT_PAUSE`) and caps one hanging call (`SUBMIT_TIMEOUT`). You see the
message in `sela_beat.log`; no action is needed.

## Out of memory

`OOM at batch size 32 -> retrying as 16 + 16` in a job log is the worker's
own recovery, and the batch's problems are generated anyway. Frequent halving
wastes time: lower `BATCH_SIZE` for that dataset (or use the A100 rather than
the A40 partition). `OOM even at batch size 1` records the problem as wrong
with `"error": "oom"`. Their number per run is the field `errors` of
`summary.json`.

## A job hit its time limit

Nothing to do. The shard resumes from its `.partial.jsonl` in the next job.
Worker logs that end with `TIME BUDGET: … stopping` are the clean version of
the same: the worker saw that the next batch would not fit.

## I changed the code; old results stay

Run directories encode the configuration, not the code version. Delete the
affected run directories ([cleaning up](outputs.md#cleaning-up)). The
`code_version` in `config.json` / `summary.json` tells which commit produced a
run (`+dirty`: with uncommitted changes).

## I edited `Sela_beat.sh`; nothing changed

The running heart beat notices the edit at its next pass, at most an hour
later. If the log says `Sela_beat.sh was edited but does not validate`, the
edit has an error. `./Sela_beat.sh --plan` shows which one; fix it and the
next pass hands over. To apply an edit immediately, cancel and resubmit:

```bash
scancel -u $USER -n sela_beat && sbatch Sela_beat.sh
```

## The report is old or missing

`AUTO_REPORT=1` runs `reporting.sh` after every pass; its errors go to
`logs/reporting.log`. `./reporting.sh` regenerates the report at any time.
The report covers what is in `plan.tsv`. Runs outside the current plan appear
only in the CSV.

## `python` fails on the login node with "libpython3.11.so.1.0: cannot open shared object file"

The cluster venv is built from the AMD module tree for the GPU nodes and does
not run on the (Intel) login node. Use it inside a job or an `srun` session
([manual runs](manual-runs.md#interactively-on-a-gpu-node)). On the login
node use the CPU environment from [setup](setup.md#a-cpu-environment-for-the-tests-optional).
The stdlib-only tools (`reporting.sh`, `sela_tools.py`) need no venv.

## Stopping and starting over

```bash
scancel -u $USER -n sela_beat
squeue -u $USER -h -o '%i %j' | awk '$2 ~ /^sx_/ {print $1}' | xargs -r scancel
```

To start from scratch, also delete `runs/`, `journal.txt`, `plan.tsv` and
`status.out`, keeping `env/` and `models/`.

## The workspace is about to expire

`sela_beat.log` warns 10 days ahead (`WS_WARN_DAYS`):
`ws_extend memory 30`. See [setup](setup.md#the-lustre-workspace).

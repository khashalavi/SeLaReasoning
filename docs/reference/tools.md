# sela_tools.py and sela_report.py

Both use only the Python standard library and run with the system
`python3` (3.9 on marvin) anywhere: in the CPU-only heart-beat job, on the
login node, in a report job.

## `scripts/sela_tools.py`

```bash
python3 -B scripts/sela_tools.py merge  --run_dir DIR --num_shards N
python3 -B scripts/sela_tools.py status --plan STATUS.tsv --out status.out [--interval MIN]
```

### `merge`

Reads all `shards/shard_<i>of<n>.json` of a run, checks that no problem index
appears twice, sorts the records, and writes `merged.json` and
`summary.json` ([formats](formats.md#summaryjson)). It exits with an error if
a shard is missing. The heart beat calls it for every run whose shards are
complete, and again when a shard is newer than the summary.

| Function | Use |
|---|---|
| `merge_run(run_dir, num_shards)` | the merge; returns the summary |
| `needs_merge(run_dir, num_shards)` | all shards present and the summary missing or older |
| `summarize(details, run)` | the metrics of a list of problem records |
| `shard_counts(path)` | (total, correct) from the first 4 KB of a shard file, without parsing the details |
| `partial_counts(path)` | (lines, correct) of a `.partial.jsonl` |

### `status`

Input: one tab-separated line per run, written by the heart beat:
`phase method tag model dataset n seed run_dir num_shards running pending stuck`.
Output: the status table, with done/total shards, problems done, accuracy so
far (merged summaries, finished shards, partial files) and a 20-character
progress bar.

## `scripts/sela_report.py`

```bash
python3 -B scripts/sela_report.py --root $SELA_ROOT --repo . --out result/selar_report.md \
    [--csv $SELA_ROOT/reports/all_runs.csv] [--cache $SELA_ROOT/reports/upstream_cache.json]
```

| Step | Function |
|---|---|
| read the plan | `load_plan(root)` |
| merge newly finished runs of the plan | `needs_merge` / `merge_run` |
| read every `summary.json` below `runs/` (the `shards/` folders are skipped) | `scan_runs`, `run_record` |
| read the authors' results, with a cache keyed by path, size and modification time | `load_upstream` |
| aggregate seeds per (model, method, dataset) | `Cell`: mean, std, completeness, progress |
| paired tests | `paired_p` → `mcnemar` |
| sections 1–7 | `section_coverage`, `section_master`, `section_decision`, `section_details`, `section_diagnostics`, `section_sweeps`, `section_reference` |
| the CSV | `write_csv` |

The section contents are described in [Reporting](../guide/reporting.md).
`--root` can point at the smoke tree (`$SELA_ROOT/smoke`); `reporting.sh
--smoke` does that.

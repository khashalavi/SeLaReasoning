# Reference

## File map

```text
SeLaReasoning/
├── Sela_beat.sh             heart beat: orchestrator, and the worker / setup / download jobs
├── reporting.sh             writes result/selar_report.md
├── requirements-sela.txt    the cluster environment (torch is installed separately)
├── zensical.toml, docs/     this documentation
├── src/
│   ├── selarx.py            SeLaR-X: the shared decoding loop, all variants
│   ├── generation_utils.py  upstream: sampling filter, math ids, generate_cot / _selar / _swir
│   └── grader.py            upstream: math_verify answer checking
├── scripts/
│   ├── sela_run.py          one resumable shard of one run (the GPU job)
│   ├── sela_tools.py        merge shards, write the status table (stdlib only)
│   ├── sela_report.py       the report (stdlib only)
│   ├── run.py, merge.py     upstream evaluation scripts (unchanged)
├── tests/test_selarx.py     CPU tests of src/selarx.py
├── datasets/                the five benchmarks (upstream)
├── results/                 the authors' published runs (upstream)
├── result/                  the final report (created by reporting.sh)
├── improvement.md           the research blueprint
└── paper/                   the SeLaR paper
```

| File | Runs where | Python | Written to |
|---|---|---|---|
| `Sela_beat.sh` (orchestrator) | CPU job on `sgpu_medium`, or the login node | system `python3` for merge/status | `$SELA_ROOT` |
| `Sela_beat.sh --worker` → `sela_run.py` | GPU job | cluster venv | the run directory |
| `Sela_beat.sh --setup-env` / `--download` | GPU node / any compute node | AMD module Python, then the venv | `env/`, `models/` |
| `reporting.sh` → `sela_report.py` | anywhere | system `python3` (3.9) | `result/`, `$SELA_ROOT/reports/` |
| `tests/test_selarx.py` | login node | a CPU venv | nothing |

## Pages

| Page | Content |
|---|---|
| [sela_run.py](sela-run.md) | every command-line flag of the worker, what it does step by step |
| [selarx.py API](selarx-api.md) | `SelarXConfig`, `generate_selarx`, the building blocks |
| [File formats](formats.md) | `plan.tsv`, journal, attempts, `config.json`, shard files, `summary.json`, the CSV |
| [sela_tools.py and sela_report.py](tools.md) | merging, status table, report generator |

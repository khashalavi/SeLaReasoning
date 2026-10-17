# Selection options

The SELECTION block at the top of `Sela_beat.sh` decides **what** runs. The
hyper-parameters below it decide **how** ([Hyper-parameters](parameters.md)),
and RESOURCES decide **where** ([Resources](resources.md)). After every edit:

```bash
./Sela_beat.sh --plan     # the expanded matrix with totals; validates the edit
```

## The options

### `PHASES`

```bash
PHASES=(main)             # main | sweep | both: (main sweep)
```

| Phase | Runs | Hyper-parameters | Feeds |
|---|---|---|---|
| `main` | every entry of `METHODS` and `COMBOS` × `MODELS` × `DATASETS` × `SEEDS`, `N_SAMPLES` problems | the defaults (`X=`) | report §2–5: master table, ranking, details, diagnostics |
| `sweep` | every entry of `SWEEP_METHODS` × `SWEEP_MODELS` × `SWEEP_DATASETS` × `SWEEP_SEEDS`, `SWEEP_N_SAMPLES` problems | the `X_SWEEP` arrays, see `SWEEP_MODE` | report §6: sweep tables |

### `METHODS`

```bash
METHODS=(cot selar layer sem ltr fhg dynk)
```

Any of `cot`, `cot_greedy`, `cot_upstream`, `selar`, `selar_upstream`, `swir`,
`layer`, `sem`, `ltr`, `fhg`, `dynk` ([Methods](../methods/index.md)). The
master table shows Baseline 1 (`cot`), Baseline 2 (`selar`) and Variants 1–5
in the order of `improvement.md`, followed by the extra baselines.

### `COMBOS`

```bash
COMBOS=()                          # Variant 6
COMBOS=("sem+dynk" "ltr+dynk")     # two combinations
```

Ideas joined with `+`, at least two, each at most once
([Combinations](../methods/combinations.md)).

### `MODELS`

```bash
MODELS=(
    "Qwen/Qwen3-8B"
    # "Qwen/Qwen3-1.7B"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)
```

Hugging Face ids. Each one is downloaded once into `$SELA_ROOT/models/`. The
run directories use the part after the `/`. The worker uses a single GPU.
Qwen3-32B (about 65 GB of weights) fits on one A100-80GB, but leaves only
about 14 GB for the KV cache, i.e. a batch of one or two long traces. It
would need much smaller `BATCH_SIZE` values and would run slowly.

### `DATASETS`

```bash
DATASETS=(gsm8k math500 aime_2024 aime_2025 gpqa_diamond)
```

| Name | Problems | Max new tokens | Shard size | Batch size | Shards |
|---|---:|---:|---:|---:|---:|
| `gsm8k` | 1,319 | 32,768 | 330 | 48 | 4 |
| `math500` | 500 | 32,768 | 125 | 32 | 4 |
| `aime_2024` | 30 | 38,912 | 30 | 15 | 1 |
| `aime_2025` | 30 | 38,912 | 30 | 15 | 1 |
| `gpqa_diamond` | 198 | 32,768 | 99 | 24 | 2 |

### `SEEDS`

```bash
SEEDS=(41)                # (41 42 43 44) -> mean ± std per cell
```

Each seed is a separate run. Within a run every batch is re-seeded
deterministically from the seed and its first problem index, so a resumed
shard draws the same random numbers as an uninterrupted one.

### `N_SAMPLES`

```bash
N_SAMPLES=all             # or a number: the first N problems of every dataset
```

`N` above a dataset's size means the whole dataset (AIME with `100` is all 30
problems). The problem count is part of the run directory (`n1319`, `n100`),
so subsets never mix with full runs.

### The sweep phase

```bash
SWEEP_METHODS=(layer sem ltr fhg dynk)   # also: selar (sweeps SELAR_TOPK x SELAR_TAU), swir
SWEEP_MODELS=("Qwen/Qwen3-8B")
SWEEP_DATASETS=(math500 aime_2024)       # tune on these, report on the others
SWEEP_N_SAMPLES=100
SWEEP_SEEDS=(41)
SWEEP_MODE=oat                           # oat | grid
```

| `SWEEP_MODE` | Configurations per method |
|---|---|
| `oat` (one at a time) | the default, plus every sweep value of one knob with all other knobs at their defaults |
| `grid` | the full cartesian product of all the method's `X_SWEEP` arrays |

With the shipped sweep arrays, `oat` gives per model and dataset:

| Method | Configurations | Varied knobs |
|---|---:|---|
| `layer` | 17 | `LAYER_FRAC` 3, `LAYER_TAU` 6, `LAYER_GATE` 1, `LAYER_ALPHA` 3, `LAYER_TOPK` 2, `LAYER_DIRECTION` 1 (+ default) |
| `sem` | 10 | `SEM_SIM` 4, `SEM_WINDOW` 3, `SEM_GAMMA` 2 (+ default) |
| `ltr` | 8 | `LTR_MAX` 3, `LTR_EXIT` 2, `LTR_DECAY` 2 (+ default) |
| `fhg` | 6 | `FHG_THETA` 3, `FHG_MODE` 1, `FHG_LAYERS` 1 (+ default) |
| `dynk` | 6 | `DYNK_MASS` 3, `DYNK_MAX` 2 (+ default) |

`grid` for `layer` would be 4 · 7 · 2 · 4 · 3 · 2 · 1 = 1,344 configurations
per dataset; use it only with a few short sweep arrays.

## How the plan is expanded

For each phase, each method (combinations included), model and dataset:

1. **Base gate.** With `SELAR_PRESET=paper`, `SELAR_TOPK` and `SELAR_TAU` take
   the published values for this model and dataset
   ([presets](parameters.md#paper-presets)); with `fixed`, the variables as
   set.
2. **Configurations.** The method's sweepable knobs give one configuration
   (main phase) or many (sweep phase, by `SWEEP_MODE`). Fixed knobs are added:
   the base gate for every input-level variant, `CONTRASTIVE_W` and
   `SOFT_MASK` for `selar` and the input-level variants.
3. **Runs.** × seeds. Each run gets a directory named after its configuration
   ([run directories](outputs.md#run-directories)).
4. **Shards.** ⌈N / shard size⌉ shards per run; one GPU job each.

A run that appears in both phases is planned once and listed in both. Example:
the sweep's default configuration on AIME24 (30 problems) is the same run as
the main phase's.

| Default selection | Runs | GPU jobs | Problem generations |
|---|---:|---:|---:|
| `PHASES=(main)` | 35 | 84 | 14,539 |
| `PHASES=(main sweep)` | 124 | 173 | 20,499 |

### Knob values are canonical

Numbers are rewritten with `%g` before they are compared or used in names:
`0.50` → `0.5`, `1.0` → `1`, `0.90` → `0.9`. A sweep value that equals the
default therefore never creates a duplicate run.

### Validation

`--plan`, every pass, and the [edit hand-over](heart-beat.md#editing-while-it-runs)
check the selection before anything is submitted:

```text
ERROR: COMBOS 'sem+dynx': unknown idea 'dynx'
ERROR: LAYER_DIRECTION: 'embedd' is not one of: embed_delta lens_contrast
ERROR: SEM_SIM: '0,82' is not a number
```

Checked: phases, method names, combinations (≥ 2 known ideas, no repeats),
dataset names, `N_SAMPLES` (`all` or a positive integer), seeds (integers),
`SWEEP_MODE`, `SELAR_PRESET`, every knob's default and sweep values (numbers,
or one of the allowed words).

## Recipes

=== "Baselines only, quickly"

    ```bash
    PHASES=(main)
    METHODS=(cot selar)
    DATASETS=(aime_2024 aime_2025)
    ```

=== "Quick look at all variants"

    ```bash
    PHASES=(main)
    N_SAMPLES=100          # first 100 problems; AIME stays complete
    ```

=== "Full blueprint"

    ```bash
    PHASES=(main sweep)
    COMBOS=("sem+dynk")    # after the ranking of the first round
    ```

=== "Second model, more seeds"

    ```bash
    MODELS=("Qwen/Qwen3-8B" "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    SEEDS=(41 42 43)
    ```

=== "Reproduce the upstream code"

    ```bash
    METHODS=(cot selar selar_upstream swir)
    SOFT_MASK=batch_any    # selar then reproduces the batching quirk in the shared loop
    ```

# Hyper-parameters

Every knob lives in the HYPER-PARAMETERS block of `Sela_beat.sh` as a pair:

```bash
SEM_SIM=0.82;            SEM_SIM_SWEEP=(0.75 0.80 0.85 0.90)
#  ^ default (main phase)  ^ values the sweep phase tries
```

The heart beat passes each knob to `scripts/sela_run.py` as a flag and puts a
short code of it into the run's directory name (the *tag*). Defaults follow
`improvement.md`; deviations are marked ✱ and explained on the method pages.

## Decoding

The same for every method (paper §4.1). Part of the run directory's decoding
segment, e.g. `T0.6_p0.95_k20_mp0_L32768`.

| Variable | Default | Flag | Meaning |
|---|---|---|---|
| `TEMPERATURE` | 0.6 | `--temperature` | softmax temperature |
| `TOP_P` | 0.95 | `--top_p` | nucleus filter |
| `TOP_K` | 20 | `--top_k` | top-k filter (applied before top-p) |
| `MIN_P` | 0.0 | `--min_p` | min-p filter |
| `MAX_NEW_TOKENS[dataset]` | 32768; AIME 38912 | `--max_new_tokens` | generation budget |

`cot_greedy` ignores the sampling values; its decoding segment is
`greedy_L<max tokens>`.

## Base SeLaR gate

Used by `selar`, `selar_upstream` and every input-level variant (`sem`,
`ltr`, `fhg`, `dynk` and combinations without `layer`).

| Variable | Default | Sweep | Flag | Tag | Meaning |
|---|---|---|---|---|---|
| `SELAR_PRESET` | `paper` | | | | `paper`: k and τ per (model, dataset), see below; `fixed`: the two values below everywhere |
| `SELAR_TOPK` | 3 | 2 3 5 | `--topk` | `k` | k of the top-k entropy and of the soft mixture |
| `SELAR_TAU` | 0.5 | 0.3 0.4 0.5 0.6 0.7 | `--tau` | `t` | threshold τ on the normalised entropy |
| `CONTRASTIVE_W` | 1.0 | | `--contrastive_weight` | `w` (omitted at 1) | weight of the contrastive push (paper: 1) |
| `SOFT_MASK` | `per_row` | | `--soft_mask` | `mask` (omitted at `per_row`) | `per_row` (paper) or `batch_any` (upstream quirk) |

Only the `selar` sweep varies `SELAR_TOPK` and `SELAR_TAU`. The variants keep
the base gate fixed and sweep their own knobs.

### Paper presets

With `SELAR_PRESET=paper`, the base gate is taken from the paper and the
authors' result files:

| Model | GSM8K | MATH500 | GPQA | AIME24 | AIME25 | Source |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-8B | k3 τ0.6 | k3 τ0.5 | **k5 τ0.5** | k3 τ0.4 | k3 τ0.5 | App. B; GPQA from the result file name |
| Qwen/Qwen3-1.7B | k3 τ0.5 | k3 τ0.5 | k3 τ0.5 | k3 τ0.5 | k3 τ0.5 | result file names |
| Qwen/Qwen3-32B | k3 τ0.6 | k3 τ0.5 | k3 τ0.7 | k3 τ0.4 | **k3 τ0.6** | App. B; AIME25 from the result file name |
| other `Qwen/Qwen3-*` | k3 τ0.6 | k3 τ0.5 | k3 τ0.7 | k3 τ0.4 | k3 τ0.5 | App. B |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | k3 τ0.5 | k3 τ0.6 | k3 τ0.8 | k3 τ0.5 | k3 τ0.6 | Fig. 7 |
| any other model | `SELAR_TOPK` / `SELAR_TAU` | | | | | |

The table lives in `selar_preset()` in `Sela_beat.sh`. For why GPQA is
k = 5, see [Fair comparisons](../background/fair-comparison.md#2-the-published-gpqa-setting-is-not-the-one-the-text-states).

## Variant 1 · `layer`

| Variable | Default | Sweep | Flag | Tag |
|---|---|---|---|---|
| `LAYER_FRAC` | 0.5 | 0.25 0.33 0.5 0.67 | `--layer_frac` | `lf` |
| `LAYER_TAU` | 0.5 | 0.3 0.4 0.5 0.6 0.7 0.8✱ 0.9✱ | `--layer_tau` | `lt` |
| `LAYER_GATE` ✱ | `lens` | lens final | `--layer_gate` | `lg` (omitted at `lens`) |
| `LAYER_ALPHA` | 1.0 | 0.5 1.0 1.5 2.0 | `--layer_alpha` | `la` |
| `LAYER_TOPK` | 3 | 2 3 5 | `--layer_topk` | `lk` |
| `LAYER_DIRECTION` ✱ | `embed_delta` | embed_delta lens_contrast | `--layer_direction` | `ld` |
| `LAYER_REL_SCALE` ✱ | 0.1 | 0.1 | `--layer_rel_scale` | `lr` (only with `lens_contrast`) |

[:octicons-arrow-right-24: Layer steering](../methods/layer.md)

## Variant 2 · `sem`

| Variable | Default | Sweep | Flag | Tag |
|---|---|---|---|---|
| `SEM_TOPK` | 5 | 5 | `--sem_topk` | `sk` |
| `SEM_SIM` | 0.82 | 0.75 0.80 0.85 0.90 | `--sem_sim` | `ss` |
| `SEM_WINDOW` | 15 | 5 10 15 20 | `--sem_window` | `sw` |
| `SEM_GAMMA` | 1.0 | 0.5 1.0 1.5 | `--sem_gamma` | `sg` |
| `SEM_FLOOR` ✱ | 0.3 | 0.3 | `--sem_floor` | `sf` |

`SEM_SIM=2` disables merging, `SEM_WINDOW=0` the running threshold.
[:octicons-arrow-right-24: Semantic gating](../methods/sem.md)

## Variant 3 · `ltr`

| Variable | Default | Sweep | Flag | Tag |
|---|---|---|---|---|
| `LTR_MAX` | 3 | 2 3 4 5 | `--ltr_max` | `rm` |
| `LTR_EXIT` | 0.2 | 0.15 0.20 0.25 | `--ltr_exit` | `rx` |
| `LTR_DECAY` | 0.9 | 0.8 0.9 1.0 | `--ltr_decay` | `rd` |
| `LTR_COMMIT` | `sample` | sample | `--ltr_commit` | `rc` (omitted at `sample`) |

[:octicons-arrow-right-24: Latent rollouts](../methods/ltr.md)

## Variant 4 · `fhg`

| Variable | Default | Sweep | Flag | Tag |
|---|---|---|---|---|
| `FHG_THETA` | 1.0 | 0.8 1.0 1.2 1.4 | `--fhg_theta` | `ft` |
| `FHG_MODE` ✱ | `relative` | relative absolute | `--fhg_mode` | `fm` |
| `FHG_LAYERS` | `upper` | upper all | `--fhg_layers` | `fl` |

[:octicons-arrow-right-24: Functional gating](../methods/fhg.md)

## Variant 5 · `dynk`

| Variable | Default | Sweep | Flag | Tag |
|---|---|---|---|---|
| `DYNK_MASS` | 0.9 | 0.80 0.85 0.90 0.95 | `--dynk_mass` | `dm` |
| `DYNK_MIN` | 2 | 2 | `--dynk_min` | `dn` |
| `DYNK_MAX` | 8 | 5 8 10 | `--dynk_max` | `dx` |

[:octicons-arrow-right-24: Dynamic k](../methods/dynk.md)

## SwiR

| Variable | Default | Sweep | Flag | Tag |
|---|---|---|---|---|
| `SWIR_ALPHA` | 1.0 | 0.6 0.9 1.0 | `--alpha` | `a` |
| `SWIR_MAX_SWITCH` | `none` | none | `--max_switch_count` (left out at `none`) | `s` |

## Which method receives which knobs

| Method | Swept knobs | Fixed knobs (always passed, never swept) |
|---|---|---|
| `cot`, `cot_greedy`, `cot_upstream` | none | none |
| `selar` | `SELAR_TOPK`, `SELAR_TAU` | `CONTRASTIVE_W`, `SOFT_MASK` |
| `selar_upstream` | `SELAR_TOPK`, `SELAR_TAU` | none |
| `swir` | `SWIR_ALPHA`, `SWIR_MAX_SWITCH` | none |
| `sem`, `ltr`, `fhg`, `dynk` and their combinations | the ideas' knobs | `SELAR_TOPK`, `SELAR_TAU`, `CONTRASTIVE_W`, `SOFT_MASK` |
| any method containing `layer` | the ideas' knobs | none (the input is never soft) |

Example tags:

| Method | Tag |
|---|---|
| `selar` (Qwen3-8B, GSM8K) | `selar_k3_t0.6` |
| `sem` | `sem_k3_t0.6_sk5_ss0.82_sw15_sg1_sf0.3` |
| `fhg` | `fhg_k3_t0.6_ft1_fmrelative_flupper` |
| `layer` | `layer_lf0.5_lt0.5_la1_lk3_ldembed_delta` |
| `layer`, final gate, lens_contrast | `layer_lf0.5_lt0.5_la1_lk3_ldlens_contrast_lr0.1_lgfinal` |
| `sem+dynk` | `sem+dynk_k3_t0.6_sk5_ss0.82_sw15_sg1_sf0.3_dm0.9_dn2_dx8` |

## Defaults outside the heart beat

`scripts/sela_run.py` and `SelarXConfig` have the same defaults as the table
above (e.g. `fhg_mode="relative"`), except the base gate: `topk=3`, `tau=0.5`,
since presets are a heart-beat feature. A flag that is not given keeps the
Python default ([sela_run.py](../reference/sela-run.md)).

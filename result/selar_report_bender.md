# SeLaR-X — Experiment Report

Generated 2026-10-22 13:27 from `/home/s76kalav/SeLaReasoning/sela_data` by `reporting.sh`. Plan: `plan.tsv` of the last Sela_beat pass; every number below is recomputed from the per-problem results on Lustre (all runs as CSV: `/home/s76kalav/SeLaReasoning/sela_data/reports/all_runs.csv`).

**How to read.** Accuracy in %. `±` = std over seeds. `*` = significantly different from CoT, `†` = from vanilla SeLaR (exact McNemar test on the same problems, p < 0.05; seeds pooled). TPCA = (α·T_c + (1−α)·T_w)/α, tokens per correct answer; *TPCA Improvement* > 0 means fewer tokens per correct answer than CoT. *Activation Ratio* = share of decoding steps with a latent intervention (soft input, or hidden-state steering for Variant 1). `… x%` = still running. AIME has 30 problems: one problem = 3.33 points.

## 1. Coverage

- planned runs: **35** (main 35) — merged **0**, in progress **0**, not started **35**
- runs on disk: **0** (0 not in the current plan, listed in the CSV only)
- GPU time of merged runs: **0.0 h** (generation only, from the per-batch timers)

## 2. Master comparison (improvement.md §7)

### Qwen/Qwen3.5-2B

| Method | GSM8K Acc (%) | MATH500 Acc (%) | AIME24 Acc (%) | AIME25 Acc (%) | GPQA Acc (%) | Avg Acc (%) | Δ Avg vs SeLaR | TPCA Improvement vs CoT | Activation Ratio |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline 1: Standard CoT | … 0% | … 0% | … 0% | … 0% | … 0% | – | – | 0.0% | 0.0% |
| Baseline 2: Vanilla SeLaR | … 0% | … 0% | … 0% | … 0% | … 0% | – | 0.00 | – | – |
| Variant 1: Layer-Wise Steering (Hidden-State) | … 0% | … 0% | … 0% | … 0% | … 0% | – | – | – | – |
| Variant 2: Semantic Adaptive Gating (Sem-SeLaR) | … 0% | … 0% | … 0% | … 0% | … 0% | – | – | – | – |
| Variant 3: Multi-Step Rollouts (LTR-SeLaR) | … 0% | … 0% | … 0% | … 0% | … 0% | – | – | – | – |
| Variant 4: Functional Head Gating (Fact/Logic) | … 0% | … 0% | … 0% | … 0% | … 0% | – | – | – | – |
| Variant 5: Nucleus Dynamic-K (DynK-SeLaR) | … 0% | … 0% | … 0% | … 0% | … 0% | – | – | – | – |


## 3. Which variant wins? (improvement.md §7–8)

No variant has complete results next to vanilla SeLaR yet.

## 4. Per-dataset details

#### Qwen/Qwen3.5-2B — GSM8K

| Method | Acc (%) | Correct | Avg tokens | T_c | T_w | TPCA | TPCA vs CoT | Truncated | Activation | p vs CoT | p vs SeLaR | Config |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Baseline 1: Standard CoT | … 0% | – | – | – | – | – | – | – | – | – | – | cot |
| Baseline 2: Vanilla SeLaR | … 0% | – | – | – | – | – | – | – | – | – | – | selar_k3_t0.5 |
| Variant 1: Layer-Wise Steering (Hidden-State) | … 0% | – | – | – | – | – | – | – | – | – | – | layer_lf0.5_lt0.5_la1_lk3_ldembed_delta |
| Variant 2: Semantic Adaptive Gating (Sem-SeLaR) | … 0% | – | – | – | – | – | – | – | – | – | – | sem_k3_t0.5_sk5_ss0.82_sw15_sg1_sf0.3 |
| Variant 3: Multi-Step Rollouts (LTR-SeLaR) | … 0% | – | – | – | – | – | – | – | – | – | – | ltr_k3_t0.5_rm3_rx0.2_rd0.9 |
| Variant 4: Functional Head Gating (Fact/Logic) | … 0% | – | – | – | – | – | – | – | – | – | – | fhg_k3_t0.5_ft1_fmrelative_flupper |
| Variant 5: Nucleus Dynamic-K (DynK-SeLaR) | … 0% | – | – | – | – | – | – | – | – | – | – | dynk_k3_t0.5_dm0.9_dn2_dx8 |

#### Qwen/Qwen3.5-2B — MATH500

| Method | Acc (%) | Correct | Avg tokens | T_c | T_w | TPCA | TPCA vs CoT | Truncated | Activation | p vs CoT | p vs SeLaR | Config |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Baseline 1: Standard CoT | … 0% | – | – | – | – | – | – | – | – | – | – | cot |
| Baseline 2: Vanilla SeLaR | … 0% | – | – | – | – | – | – | – | – | – | – | selar_k3_t0.5 |
| Variant 1: Layer-Wise Steering (Hidden-State) | … 0% | – | – | – | – | – | – | – | – | – | – | layer_lf0.5_lt0.5_la1_lk3_ldembed_delta |
| Variant 2: Semantic Adaptive Gating (Sem-SeLaR) | … 0% | – | – | – | – | – | – | – | – | – | – | sem_k3_t0.5_sk5_ss0.82_sw15_sg1_sf0.3 |
| Variant 3: Multi-Step Rollouts (LTR-SeLaR) | … 0% | – | – | – | – | – | – | – | – | – | – | ltr_k3_t0.5_rm3_rx0.2_rd0.9 |
| Variant 4: Functional Head Gating (Fact/Logic) | … 0% | – | – | – | – | – | – | – | – | – | – | fhg_k3_t0.5_ft1_fmrelative_flupper |
| Variant 5: Nucleus Dynamic-K (DynK-SeLaR) | … 0% | – | – | – | – | – | – | – | – | – | – | dynk_k3_t0.5_dm0.9_dn2_dx8 |

#### Qwen/Qwen3.5-2B — AIME24

| Method | Acc (%) | Correct | Avg tokens | T_c | T_w | TPCA | TPCA vs CoT | Truncated | Activation | p vs CoT | p vs SeLaR | Config |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Baseline 1: Standard CoT | … 0% | – | – | – | – | – | – | – | – | – | – | cot |
| Baseline 2: Vanilla SeLaR | … 0% | – | – | – | – | – | – | – | – | – | – | selar_k3_t0.5 |
| Variant 1: Layer-Wise Steering (Hidden-State) | … 0% | – | – | – | – | – | – | – | – | – | – | layer_lf0.5_lt0.5_la1_lk3_ldembed_delta |
| Variant 2: Semantic Adaptive Gating (Sem-SeLaR) | … 0% | – | – | – | – | – | – | – | – | – | – | sem_k3_t0.5_sk5_ss0.82_sw15_sg1_sf0.3 |
| Variant 3: Multi-Step Rollouts (LTR-SeLaR) | … 0% | – | – | – | – | – | – | – | – | – | – | ltr_k3_t0.5_rm3_rx0.2_rd0.9 |
| Variant 4: Functional Head Gating (Fact/Logic) | … 0% | – | – | – | – | – | – | – | – | – | – | fhg_k3_t0.5_ft1_fmrelative_flupper |
| Variant 5: Nucleus Dynamic-K (DynK-SeLaR) | … 0% | – | – | – | – | – | – | – | – | – | – | dynk_k3_t0.5_dm0.9_dn2_dx8 |

#### Qwen/Qwen3.5-2B — AIME25

| Method | Acc (%) | Correct | Avg tokens | T_c | T_w | TPCA | TPCA vs CoT | Truncated | Activation | p vs CoT | p vs SeLaR | Config |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Baseline 1: Standard CoT | … 0% | – | – | – | – | – | – | – | – | – | – | cot |
| Baseline 2: Vanilla SeLaR | … 0% | – | – | – | – | – | – | – | – | – | – | selar_k3_t0.5 |
| Variant 1: Layer-Wise Steering (Hidden-State) | … 0% | – | – | – | – | – | – | – | – | – | – | layer_lf0.5_lt0.5_la1_lk3_ldembed_delta |
| Variant 2: Semantic Adaptive Gating (Sem-SeLaR) | … 0% | – | – | – | – | – | – | – | – | – | – | sem_k3_t0.5_sk5_ss0.82_sw15_sg1_sf0.3 |
| Variant 3: Multi-Step Rollouts (LTR-SeLaR) | … 0% | – | – | – | – | – | – | – | – | – | – | ltr_k3_t0.5_rm3_rx0.2_rd0.9 |
| Variant 4: Functional Head Gating (Fact/Logic) | … 0% | – | – | – | – | – | – | – | – | – | – | fhg_k3_t0.5_ft1_fmrelative_flupper |
| Variant 5: Nucleus Dynamic-K (DynK-SeLaR) | … 0% | – | – | – | – | – | – | – | – | – | – | dynk_k3_t0.5_dm0.9_dn2_dx8 |

#### Qwen/Qwen3.5-2B — GPQA

| Method | Acc (%) | Correct | Avg tokens | T_c | T_w | TPCA | TPCA vs CoT | Truncated | Activation | p vs CoT | p vs SeLaR | Config |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Baseline 1: Standard CoT | … 0% | – | – | – | – | – | – | – | – | – | – | cot |
| Baseline 2: Vanilla SeLaR | … 0% | – | – | – | – | – | – | – | – | – | – | selar_k3_t0.5 |
| Variant 1: Layer-Wise Steering (Hidden-State) | … 0% | – | – | – | – | – | – | – | – | – | – | layer_lf0.5_lt0.5_la1_lk3_ldembed_delta |
| Variant 2: Semantic Adaptive Gating (Sem-SeLaR) | … 0% | – | – | – | – | – | – | – | – | – | – | sem_k3_t0.5_sk5_ss0.82_sw15_sg1_sf0.3 |
| Variant 3: Multi-Step Rollouts (LTR-SeLaR) | … 0% | – | – | – | – | – | – | – | – | – | – | ltr_k3_t0.5_rm3_rx0.2_rd0.9 |
| Variant 4: Functional Head Gating (Fact/Logic) | … 0% | – | – | – | – | – | – | – | – | – | – | fhg_k3_t0.5_ft1_fmrelative_flupper |
| Variant 5: Nucleus Dynamic-K (DynK-SeLaR) | … 0% | – | – | – | – | – | – | – | – | – | – | dynk_k3_t0.5_dm0.9_dn2_dx8 |


## 5. Latent-step diagnostics

Gate statistics per run: *Gate open* = entropy above the (static or dynamic) threshold; *Math-guarded* / *FHG vetoed* = share of those openings blocked by the math-symbol guard / the attention-FFN gate; *avg clusters* (sem), *avg k_t* (dynk), *rollouts* (ltr), *avg R_act* (fhg) and *avg τ_t* (dynamic threshold) only for the variants that use them. Entropy columns describe the gate's entropy signal (logit lens at l* for Variant 1).

–

## 6. Hyper-parameter sweeps (sweep phase)

One table per variant; **bold** = best average. These problems are part of the test sets, so pick the configuration here and report it on the other datasets / the full sets.

No sweep-phase runs in the plan (add `sweep` to PHASES).

## 7. Reference: the authors' published runs (`results/`)

### Qwen3-1.7B

| Method | GSM8K | MATH500 | AIME24 | AIME25 | GPQA | Avg | Settings / note |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Standard CoT (published) | 90.07 | 92.00 | 50.00 | 33.33 | 39.39 | 60.96 |  |
| SwiR (published) | 89.84 | 92.00 | 46.67 | 23.33 | 37.88 | 57.94 | AIME24: a0.3_sNone, AIME25: a0.3_s100 |
| Vanilla SeLaR (published) | 90.60 | 92.60 | 53.33 | 36.67 | 35.35 | 61.71 | GSM8K: k3_t0.5, AIME24: k3_t0.5, AIME25: k3_t0.5, GPQA: k3_t0.5 |

### Qwen3-32B

| Method | GSM8K | MATH500 | AIME24 | AIME25 | GPQA | Avg | Settings / note |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Vanilla SeLaR (published) | 96.06 | 97.60 | 83.33 | 80.00 | 67.17 | 84.83 | AIME24: k3_t0.4, AIME25: k3_t0.6 |

### Qwen3-8B

| Method | GSM8K | MATH500 | AIME24 | AIME25 | GPQA | Avg | Settings / note |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Standard CoT (published) | 95.45 | 98.00 | 76.67 | 66.67 | 61.62 | 79.68 |  |
| SwiR (published) | 95.68 | 97.00 | 60.00 | 66.67 | 62.63 | 76.39 | MATH500: a1.0_sNone, AIME24: a0.9_sNone, AIME25: a0.9_sNone, GPQA: a1.0_sNone |
| Vanilla SeLaR (published) | 95.83 | 97.00 | 83.33 | 80.00 | 61.62 | 83.56 | GSM8K: k3_t0.6, MATH500: k3_t0.5, AIME24: k3_t0.4, AIME25: k3_t0.5, GPQA: k5_t0.5 |

Accuracies recomputed from the per-problem `correct` flags of the files in `results/` (single runs; the published avg uses all five datasets).

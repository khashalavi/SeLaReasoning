# Fair comparisons

What to keep in mind before a number from this pipeline goes into a paper.

## 1. SeLaR's thresholds were tuned on the test sets

Appendix B: *"we fix k = 3 across all experiments and select dataset-specific
thresholds that maximize individual benchmark performance"*. Table 1 is the
per-benchmark best row of the sensitivity sweep in Table 3:

| Qwen3-8B | GSM8K | MATH500 | GPQA | AIME24 | AIME25 | Avg |
|---|---:|---:|---:|---:|---:|---:|
| CoT (sampling) | 95.45 | 98.00 | 61.62 | 76.67 | 66.67 | 79.68 |
| SeLaR, best setting per benchmark (Table 1) | 95.83 | 97.00 | 61.62 | 83.33 | 80.00 | **83.56** |
| SeLaR, k = 3, τ = 0.5 everywhere (Table 3) | 95.60 | 97.00 | 55.05 | 76.67 | 80.00 | **80.86** |

With one setting for all benchmarks the gain over CoT is +1.18, not +3.88, and
GPQA drops 6.6 points below CoT.

`Sela_beat.sh` therefore offers both:

| `SELAR_PRESET` | Base gate (k, τ) | Use for |
|---|---|---|
| `paper` (default) | per (model, dataset) as published, see [presets](../guide/parameters.md#paper-presets) | the strongest SeLaR baseline; matches Table 1 |
| `fixed` | `SELAR_TOPK`, `SELAR_TAU` everywhere | a tuning-free comparison |

Every input-level variant (`sem`, `ltr`, `fhg`, `dynk` and their combinations)
inherits the same base gate. A variant is therefore always compared with
vanilla SeLaR under identical (k, τ), whichever preset is chosen.

## 2. The published GPQA setting is not the one the text states

Appendix B states k = 3, τ = 0.7 for GPQA. Table 3 gives 60.10 for that
setting. The published 61.62 matches the row k = 5, τ = 0.5, and the authors'
file is `results/SeLaR/Qwen3-8B/Qwen3-8B_gpqa_diamond_selar_k5_t0.5_32768_rank0.json`.
The `paper` preset follows the file (k = 5, τ = 0.5), since that is what
produced the number.

## 3. AIME is 30 problems

One problem moves accuracy by 3.33 points. With a single sample per problem
the binomial standard error at 75% accuracy is
\(\sqrt{0.75 \cdot 0.25 / 30} \approx 7.9\) points. Most of SeLaR's published
average gain on Qwen3-8B comes from AIME (+2 and +4 problems).

- Use several seeds for AIME (`SEEDS=(41 42 43 44)`), so the report shows
  mean ± std.
- Read the per-dataset p-values, not only the averages.

## 4. Sweeps run on test problems

There are no development splits. The sweep phase tunes on
`SWEEP_DATASETS=(math500 aime_2024)`, by default on the first 100 MATH500
problems and all 30 of AIME24. A defensible protocol:

1. choose each variant's configuration on the sweep datasets only,
2. set it as the default in `Sela_beat.sh`,
3. report on **GSM8K, AIME25 and GPQA**, which the choice never saw, and on
   the full sets,
4. state in the paper that MATH500 and AIME24 were used for selection.

## 5. The batching quirk

The official code's results depend on the batch size
([details](upstream-code.md#the-batching-quirk-of-generate_selar)).
`selar` in SeLaR-X follows Algorithm 1 per row. If reviewers compare with the
published numbers, run `selar_upstream` next to it.

## 6. Significance

The report tests every method against CoT and against vanilla SeLaR with an
**exact McNemar test** on the same problems: are the problems one method
solves and the other does not balanced? Seeds are pooled pair by pair. Treat
the p-values as indicative: pooled seeds of the same problem are not
independent.

## 7. Cost: report TPCA, not token counts

Average tokens on correct answers favour methods that only solve easy
problems (survivorship bias, paper Finding 4). TPCA, the tokens per correct
answer, does not. The report gives TPCA, the change of TPCA against CoT, and
\(T_c\) / \(T_w\) separately.

## Checklist

- [ ] same decoding settings and token budget for every method (enforced by the pipeline)
- [ ] variant and vanilla SeLaR use the same base gate (enforced)
- [ ] `SELAR_PRESET` stated; ideally both `paper` and `fixed` reported
- [ ] hyper-parameters chosen on the sweep datasets only
- [ ] ≥ 3 seeds on AIME; mean ± std reported
- [ ] per-dataset p-values against CoT and SeLaR
- [ ] TPCA next to accuracy
- [ ] `selar_upstream` next to `selar` if published numbers are quoted

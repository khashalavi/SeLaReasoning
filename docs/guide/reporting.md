# Reporting

`reporting.sh` writes **one** report, `result/selar_report.md`, from
everything on Lustre. It needs only the system `python3`, takes seconds, and
runs after every heart-beat pass (`AUTO_REPORT=1`).

```bash
./reporting.sh            # login node
sbatch reporting.sh       # the same as a small CPU job (sgpu_short, 15 min, 4G)
./reporting.sh --smoke    # the smoke tree -> $SELA_ROOT/smoke/smoke_report.md
```

What it does:

1. copies the previous report to `$SELA_ROOT/reports/history/` (only if it
   changed),
2. merges every run of `plan.tsv` whose shards are all finished but not yet
   merged,
3. reads every `summary.json` below `runs/` and the authors' files in
   `results/` (parsed once, cached in `reports/upstream_cache.json`),
4. writes the report and `reports/all_runs.csv`,
5. appends its output to `logs/reporting.log`.

## Sections of the report

### 1. Coverage

Planned runs (main / sweep), how many are merged, in progress or not started,
and how many runs on disk are not part of the current plan. It also gives the
GPU time of the merged runs, from the per-batch timers, and lists the runs in
progress with their completion.

### 2. Master comparison

The table of `improvement.md` §7, one per model, from the **main phase**:

| Column | Meaning |
|---|---|
| `<dataset> Acc (%)` | accuracy; `±` = standard deviation over seeds; `… 37%` = still running; `(1/2 seeds)` = incomplete |
| `*` | significantly different from CoT (exact McNemar, p < 0.05) |
| `†` | significantly different from vanilla SeLaR (variants only) |
| Avg Acc (%) | mean over the complete datasets; `(4/5)` when some are missing |
| Δ Avg vs SeLaR | mean of (method − SeLaR) over the datasets where both are complete |
| TPCA Improvement vs CoT | mean over datasets of (TPCA_CoT − TPCA) / TPCA_CoT; positive = fewer tokens per correct answer |
| Activation Ratio | mean over datasets of the share of latent steps; `–` for methods without statistics |

Rows: Baseline 1 `cot`, Baseline 2 `selar`, Variants 1–5, the combinations
(Variant 6), then the extra baselines.

### 3. Which variant wins?

All variants and combinations ranked by Δ Avg Acc against vanilla SeLaR,
averaged over models, with:

- the TPCA change against CoT,
- the number of datasets where the variant is significantly better or worse
  than SeLaR,
- the basis (models and number of datasets compared).

Below the ranking:

- **Variant 6 suggestion**: if at least two *single* ideas are ahead of
  SeLaR, the line `COMBOS=("a+b")` for the best two;
- **Paper framing**: for the leading single idea, the title and pitch from
  `improvement.md` §8 (Variants 1–4; `dynk` has none there).

### 4. Per-dataset details

One table per model and dataset:

| Column | Meaning |
|---|---|
| Correct | correct / total, summed over seeds |
| Avg tokens, \(T_c\), \(T_w\) | average tokens over all, correct, wrong answers |
| TPCA, TPCA vs CoT | tokens per correct answer, and its relative change against CoT |
| Truncated | share of problems that hit the token budget without EOS |
| Activation | share of latent steps |
| p vs CoT, p vs SeLaR | exact McNemar p-values |
| Config | the run tag (the full configuration) |

### 5. Latent-step diagnostics

Per method and dataset, from the per-step counters:

| Column | Definition |
|---|---|
| Latent steps | latent ÷ steps |
| Soft inputs | soft inputs fed ÷ steps (`batch_any` makes it exceed latent steps) |
| Gate open | steps whose entropy passed the (static or dynamic) threshold ÷ steps |
| Math-guarded | openings blocked by the math-symbol guard ÷ openings |
| FHG vetoed | openings blocked by fhg ÷ openings |
| Rollouts/sample, Rollout len | ltr: rollouts per problem; latent steps ÷ rollouts |
| avg k_t | dynk: mean candidate-set size on latent steps |
| avg clusters | sem: mean number of semantic clusters per step |
| avg R_act | fhg: mean attention/FFN norm ratio |
| avg τ_t | sem: mean dynamic threshold |
| H mean, H median, P(H ≥ 0.5) | the gate's entropy signal: mean, median (20-bin histogram), share ≥ 0.5 |

For `layer` with the lens gate, the entropy columns describe the logit lens at
\(l^\ast\), not the output.

### 6. Hyper-parameter sweeps

One table per variant and model, from the **sweep phase**. Rows are
configurations, labelled by the knobs that differ from the default (`◆` marks
the default). Columns: accuracy per sweep dataset, the average (**bold** =
best), Δ against the default, and activation. Rows are sorted by the average.

The base gate (k, τ) of the variants is left out of the labels: it follows
`SELAR_PRESET` and can differ per dataset. For the `selar` sweep, k and τ
*are* the swept knobs.

### 7. Reference

The authors' published runs in `results/` (CoT, SwiR, SeLaR for
Qwen3-1.7B / 8B / 32B), recomputed from their per-problem flags, with the
settings from their file names. Where the pipeline has the same model and
method, a "this pipeline" row follows, with the difference to the published
number in brackets.

## Metric definitions

Accuracy
:   correct ÷ total, recomputed from the per-problem `correct` flags. Grading
    is `src/grader.py` (`math_verify`) on the text after the last `</think>`,
    exactly as in the upstream `run.py`.

Tokens
:   per problem, `len(tokenizer.encode(decoded output))`, as upstream. The
    generated-id count is kept separately (`gen_tokens`).

TPCA
:   \(\mathrm{TPCA} = \dfrac{\alpha T_c + (1-\alpha) T_w}{\alpha} = \dfrac{\overline{T}}{\alpha}\):
    average tokens per problem divided by accuracy. Undefined at accuracy 0.

Activation ratio
:   latent steps ÷ decoding steps, pooled over all problems of a run (each
    problem's own ratio is in `avg_per_sample_ratio` of `summary.json`).

McNemar test
:   for two methods on the same problems, count \(b\) = solved only by the
    first and \(c\) = solved only by the second;
    \(p = \min\big(1,\, 2 \sum_{i=0}^{\min(b,c)} \binom{b+c}{i} 2^{-(b+c)}\big)\).
    Seeds are paired seed by seed and pooled, which treats them as independent.
    Read the result as indicative.

GPU time
:   the sum, over problems, of their batch's wall time divided by the batch
    size. Model loading and queueing are not included.

## The CSV

`$SELA_ROOT/reports/all_runs.csv` has one row per merged run on disk,
including runs outside the current plan (manual runs, old defaults):

| Columns | |
|---|---|
| identity | `model`, `dataset`, `n`, `method`, `config`, `seed`, `run_dir`, `code` (code versions), `max_new_tokens` |
| accuracy | `acc`, `correct` |
| tokens | `tokens`, `tc`, `tw`, `tpca`, `hit_max` |
| activation | `act`, `soft_in`, `gate_open`, `guarded`, `vetoed`, `rollouts`, `roll_len`, `avg_k`, `clusters`, `ratio`, `thr` |
| entropy | `h_mean`, `h_median`, `h_tok` (token-level mean, differs from `h_mean` for sem and layer), `h_ge` (share ≥ 0.5) |
| cost | `gpu_h` |

Load it with pandas for plots: `pd.read_csv(".../reports/all_runs.csv")`.

# Findings

What has been measured so far, as of 2026-10-17. The main experiments
(Qwen3-8B, five benchmarks) have not run yet. This page collects what the
tests, the smoke test and a benchmark established, and the decisions taken
because of them.

## Equivalence with the upstream code

`tests/test_selarx.py`, 36 tests, all passing (CPU, torch 2.7.0,
Transformers 4.51.0):

| Claim | Result |
|---|---|
| `cot` reproduces `generate_cot` | token-identical, batch of 4 left-padded prompts of different lengths |
| `selar` with `batch_any` reproduces `generate_selar` | token-identical on a batch; soft inputs > gate decisions (the quirk) |
| per-row `selar` reproduces `generate_selar` at batch size 1 | token-identical for three different prompts |
| every idea reduces to its parent when switched off | token-identical: layer α = 0, fhg θ = ±∞, ltr without horizon, dynk with fixed k |

So differences between methods in the experiments come from the methods, not
from the loop.

The same holds on the GPU. In the smoke test (A100, bf16, batch of 4), `cot`
and `cot_upstream` produced **identical texts for all 4 problems**. Per-row
`selar` and `selar_upstream` produced **different texts for all 4**: even at
batch size 4, the batching quirk changes the trajectories.

## The upstream code

| Finding | Consequence |
|---|---|
| `generate_selar` gives every row of a batch its soft embedding as soon as one row is uncertain | published SeLaR results depend on the batch size; at batch 256 nearly every step is soft for every row. `selar` gates per row; `selar_upstream` keeps the original |
| SeLaR's τ was chosen per benchmark on the test sets | with one setting for all benchmarks the average gain over CoT on Qwen3-8B is +1.18 instead of +3.88 → `SELAR_PRESET=fixed` as an alternative |
| published Qwen3-8B GPQA = k 5, τ 0.5, text says k 3, τ 0.7 | the `paper` preset follows the result file |
| one result file has a stale `correct` field (460 vs 463 flags) | the report recomputes from per-problem flags; the published 92.60 is right |
| upstream copies the whole KV cache every step | SeLaR-X filters only when a row finishes; CoT 33.5 vs 34.8 ms/step |

## Environment and cluster

| | |
|---|---|
| GPU nodes | NVIDIA A100-SXM4-80GB (sgpu), AMD EPYC 7713, 32 cores per GPU by default |
| environment build | 2 min 53 s; torch 2.7.0+cu128 runs on the nodes |
| model download | Qwen3-0.6B 953 MB in 24 s |
| `sgpu_devel` | two concurrent jobs per user |
| submissions | bursts of `sbatch` are rate-limited → submissions paced at 1 s |

## Speed

Same GPU, Qwen3-0.6B, batch 16, 256 steps; full table under
[Resources](../guide/resources.md#how-long-things-take). CoT 33.5 ms/step;
every variant within +7% (the slowest is fhg at 35.9 ms with its 28 forward
hooks). The decoding loop is launch-bound at this size, so the extra tensor
operations of the variants hardly matter.

## Gate statistics

Smoke test: Qwen3-0.6B, the first 4 GSM8K problems, 512 new tokens, seed 41,
paper preset (k = 3, τ = 0.6). Small numbers, but the patterns are large.

| Method | Latent steps | Gate open | Guarded | Vetoed | Other | H mean | H median | P(H ≥ 0.5) |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| `cot` | 0% | – | – | – | | 0.187 | 0.039 | 19.6% |
| `selar` | 12.5% | 12.7% | 1.9% | – | | 0.175 | 0.038 | 17.3% |
| `layer` (lens gate, τ 0.5) | **61.5%** | 72.0% | 14.6% | – | | 0.643 | 0.739 | 72.2% |
| `layer` (final gate) | 18.4% | 18.6% | | – | | | | |
| `sem` | 20.4% | 21.0% | 2.8% | – | 1.64 clusters, τ_t 0.555 | 0.233 | 0.039 | 25.8% |
| `ltr` | 16.6% | 14.7% | 1.0% | – | 52.8 rollouts, length 1.61 | 0.183 | 0.038 | 18.3% |
| `fhg` (absolute θ 1) | **0%** | 14.8% | 0% | **100%** | R_act 0.582 | 0.187 | 0.039 | 19.6% |
| `fhg` (relative θ 1) | 9.1% | 14.4% | | 35.7% | R_act 0.596 | | | |
| `dynk` | 15.3% | 15.5% | 1.3% | – | k_t 2.86 | 0.198 | 0.040 | 20.1% |
| `sem+dynk` | 20.3% | 20.7% | 1.7% | – | 1.55 clusters, k_t 2.42 | 0.227 | 0.038 | 24.8% |
| `layer+ltr` | 57.0% | 69.8% | 13.4% | – | 119 rollouts, length 2.34 | 0.633 | 0.724 | 69.9% |

Empty cells were not recorded: the `layer` final-gate and `fhg` relative rows
come from single check jobs, summarised by hand.

What this says:

- **Vanilla SeLaR behaves as in the paper.** 12.5% latent steps; the paper
  reports 6–14% for Qwen3-8B.
- **fhg as written never fires.** The upper layers' MLP outputs are larger
  than the attention outputs (R ≈ 0.58). With an absolute θ of 0.8–1.4, every
  opening is vetoed and the method equals CoT. → **`FHG_MODE=relative`** is
  now the default; it vetoes about a third of the openings.
- **The logit lens at L/2 is not a selective gate.** Its entropy is high
  almost everywhere (median 0.74), so τ = 0.5 steers 61% of the steps. →
  **`LAYER_GATE=final`** added (SeLaR's own trigger, steering inside the
  network: 18%), and the `LAYER_TAU` sweep extended to 0.9.
- **sem fires more often than vanilla**, not less. Normalising by the number
  of live clusters (log M) raises the entropy of steps where top-p left only
  two candidates. Whether this helps or hurts is for the main runs to show.
- **Rollouts are short.** On average 1.6 steps; the exit threshold, not the
  horizon M = 3, ends most of them.

## Open questions for the main runs

- Does any variant beat vanilla SeLaR **beyond noise**, especially outside
  AIME? (report §3, McNemar)
- Does sem reduce over-triggering on the less calibrated
  DeepSeek-R1-Distill-Llama-8B, the case it was designed for?
- Is \(R_{\text{act}}\) lower on GPQA than on the math sets, as the fhg
  hypothesis predicts? (report §5, `avg R_act`)
- With the same trigger (`layer_gate=final`), is the hidden-state locus better
  or worse than the input embedding?

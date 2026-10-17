# Variant 2 · Semantic gating

`--method sem` · Idea 2 of `improvement.md` · gate signal + threshold

Token entropy counts `" 3"`, `"3"` and `"three"` as three competing
candidates, even though they mean the same. That inflates the entropy and
fires the gate on surface-form noise. The paper's own diagnosis points the
same way: the less calibrated DeepSeek-R1 distill triggers more exploratory
steps and gains less. Sem-SeLaR changes **when** latent reasoning starts, and
nothing else.

## Formulation

### A. Semantic clusters

Take the top `SEM_TOPK` candidates \(v_1 \ldots v_k\) of the filtered
distribution and their renormalised probabilities \(p_i\). Two candidates are
linked if the cosine similarity of their **input embeddings** exceeds
\(\theta_{\text{sim}}\):

$$
S_{ij} = \frac{E(v_i) \cdot E(v_j)}{\lVert E(v_i) \rVert\, \lVert E(v_j) \rVert}, \qquad
i \sim j \iff S_{ij} > \theta_{\text{sim}}
$$

Clusters \(\mathcal C_m\) are the connected components (transitively: if a ~ b
and b ~ c, then a, b, c form one cluster). Their mass is
\(P(\mathcal C_m) = \sum_{v_i \in \mathcal C_m} p_i\).

### B. Semantic entropy

$$
H^{\text{sem}}_t = -\sum_{m=1}^{M} P(\mathcal C_m) \log P(\mathcal C_m), \qquad
\bar H^{\text{sem}}_t = \frac{H^{\text{sem}}_t}{\log M} \quad (M \ge 2), \qquad
\bar H^{\text{sem}}_t = 0 \quad (M = 1)
$$

\(M\) counts the clusters that carry probability mass. Candidates removed by
top-p filtering have \(p = 0\) and do not count.

!!! note "Normalising by log M is not the same as by log k"
    Vanilla SeLaR divides by \(\log k\) even when top-p has zeroed some of the
    top-k candidates. Sem-SeLaR divides by \(\log M\). A step with candidates
    (0.85, 0.15, 0) has vanilla entropy 0.42 / log 3 = 0.39, below τ = 0.5,
    and semantic entropy 0.42 / log 2 = 0.61, above it. The test
    `test_sem_gate_normalises_by_active_clusters` pins this down. It is one
    reason why Sem-SeLaR can fire *more* often than vanilla.

### C. Running-window threshold

Over the last \(W\) steps of the same sequence (population statistics):

$$
\mu_t = \frac1W \sum_{i=1}^{W} \bar H^{\text{sem}}_{t-i}, \qquad
\sigma_t = \sqrt{\frac1W \sum_{i=1}^{W} \big(\bar H^{\text{sem}}_{t-i} - \mu_t\big)^2}, \qquad
\tau_t = \max\big(\text{floor},\; \mu_t + \gamma\, \sigma_t\big)
$$

A step is exploratory if \(\bar H^{\text{sem}}_t > \tau_t\). Until \(W\) steps
exist, the static τ of the base gate is used (\(\ge \tau\), as in vanilla).

!!! warning "Why there is a floor"
    `improvement.md` defines \(\tau_t = \mu_t + \gamma\sigma_t\) without a
    lower bound. During a run of fully confident steps, where top-p leaves one
    token and the entropy is exactly 0, \(\mu = \sigma = 0\) and every
    following wobble above 0 would fire. `SEM_FLOOR` (default 0.3) bounds the
    threshold from below. It is the one addition to the formula, and is
    sweepable.

## What stays vanilla

The gate decides *whether* a step is exploratory. The soft embedding is still
SeLaR's: the top `SELAR_TOPK` candidates, the token-level entropy as the
contrastive strength, and the same math guard. Sem-SeLaR inherits the base
gate (k, τ) from `SELAR_PRESET`, like vanilla, and τ is used before the
window is full.

## Implementation

| Detail | Choice |
|---|---|
| embeddings | `model.get_input_embeddings().weight`, gathered and normalised per step (no precomputed copy of the matrix) |
| similarity | one batched matrix product per step, \(k \times k\) per row |
| connected components | transitive closure by repeated boolean matrix squaring (\(\lceil \log_2 (k-1) \rceil\) products), cluster id = first member |
| history | a per-row ring buffer of length W; rows advance in lockstep, so one shared write position |

Ablations without new code:

| Setting | Isolates |
|---|---|
| `SEM_SIM=2` | no merging: only the running threshold and the log M normalisation remain |
| `SEM_WINDOW=0` | static threshold: only the semantic entropy remains |

## Knobs

| Variable | Flag | Default | Sweep | Meaning |
|---|---|---|---|---|
| `SEM_TOPK` | `--sem_topk` | 5 | 5 | candidates that are clustered |
| `SEM_SIM` | `--sem_sim` | 0.82 | 0.75 0.80 0.85 0.90 | cosine similarity that links two candidates |
| `SEM_WINDOW` | `--sem_window` | 15 | 5 10 15 20 | history length W; 0 = static τ |
| `SEM_GAMMA` | `--sem_gamma` | 1.0 | 0.5 1.0 1.5 | γ: standard deviations above the running mean |
| `SEM_FLOOR` | `--sem_floor` | 0.3 | 0.3 | lower bound of \(\tau_t\) |

## What the smoke test showed

Qwen3-0.6B, 4 GSM8K problems, 512 tokens: the gate fired on **20.4%** of the
steps, against 12.5% for vanilla SeLaR. There were on average **1.64**
clusters per step. That is mostly because confident steps have a single live
candidate, not because of merging. The dynamic threshold averaged
**0.555**. The report's diagnostics give these numbers for every run
(`avg clusters`, `avg τ_t`).

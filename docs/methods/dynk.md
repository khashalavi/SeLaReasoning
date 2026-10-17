# Variant 5 · Dynamic k

`--method dynk` · Idea 5 of `improvement.md` · candidate set

SeLaR mixes a fixed top-3 into its soft embedding. Some exploratory steps
have two strong candidates (50/50), others five or six plausible ones. A fixed
k either drops real alternatives or mixes in noise, and the paper's Table 3
shows that larger fixed k hurts on average. DynK-SeLaR sizes the candidate set
per step by **cumulative probability mass**.

## Formulation

Sort the filtered distribution, \(p_{(1)} \ge p_{(2)} \ge \dots\), and take the
smallest nucleus that holds \(p_{\text{mass}}\):

$$
F(m) = \sum_{j=1}^{m} p_{(j)}, \qquad
k_t = \operatorname{clamp}\Big(\min\{\, m : F(m) \ge p_{\text{mass}} \,\},\; k_{\min},\; k_{\max}\Big)
$$

Renormalise over the \(k_t\) candidates and build SeLaR's regularised soft
embedding from them. The contrastive strength uses that set's own entropy,
normalised by \(\log k_t\):

$$
\hat p_i = \frac{p_{(i)}}{\sum_{j \le k_t} p_{(j)}}, \qquad
e_t = \sum_{i \le k_t} \hat p_i\, E[v_{(i)}], \qquad
\bar H^{(k_t)}_t = \frac{-\sum_{i \le k_t} \hat p_i \log \hat p_i}{\log k_t}
$$

## What stays vanilla

The **gate** is unchanged: the step is exploratory if the normalised entropy
of the fixed top-`SELAR_TOPK` set reaches τ. DynK-SeLaR changes only *what*
is mixed once a step is exploratory. This isolates the construction from the
trigger. With `sem+dynk` the trigger becomes semantic as well.

!!! info "Vanilla is a special case"
    `DYNK_MIN = DYNK_MAX = 3` gives \(k_t = 3\) always, and the test
    `test_dynk_with_fixed_k_is_selar` checks token equality with vanilla.

## Implementation

`torch.topk(probs, k_max)` on the filtered distribution, a cumulative sum,
and a mask of the first \(k_t\) entries. Because top-p = 0.95 has already
removed the tail, \(F(m)\) reaches 1 within the surviving tokens. If more
than \(k_{\max}\) tokens are needed for \(p_{\text{mass}}\), \(k_t = k_{\max}\).
The report gives the average \(k_t\) over the latent steps (`avg k_t`).

## Knobs

| Variable | Flag | Default | Sweep | Meaning |
|---|---|---|---|---|
| `DYNK_MASS` | `--dynk_mass` | 0.9 | 0.80 0.85 0.90 0.95 | nucleus mass \(p_{\text{mass}}\) |
| `DYNK_MIN` | `--dynk_min` | 2 | 2 | lower bound of \(k_t\) (≥ 2) |
| `DYNK_MAX` | `--dynk_max` | 8 | 5 8 10 | upper bound of \(k_t\) |

The gate uses the base gate (k, τ) from `SELAR_PRESET`.

## What the smoke test showed

On exploratory steps \(k_t\) averaged **2.86**, close to SeLaR's fixed 3, with
15.3% latent steps. The interesting cases are the tails: two-way forks
(\(k_t\) = 2) and broad steps (\(k_t\) = 8). Look at them in the per-problem
statistics of `merged.json` if the averages are similar.

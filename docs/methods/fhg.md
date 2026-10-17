# Variant 4 · Functional gating

`--method fhg` · Idea 4 of `improvement.md` · extra gate

SeLaR helps on multi-step math but is flat on GPQA, which depends on factual
recall. `improvement.md` proposes to tell the two kinds of uncertainty apart
by **where the network does its work**. Attention-dominant processing would
indicate procedural reasoning, where exploration helps. FFN-dominant
processing would indicate parametric recall, where a soft mixture blurs the
fact.

!!! note "A hypothesis, not a known mechanism"
    The mapping "attention = logic, FFN = facts" is the idea's premise, not an
    established result. Every run records the average ratio, so the report can
    check whether it differs between GPQA and the math sets at all.

## Formulation

At step t, over the monitored layers \(\mathcal L\) (the upper half
\(L/2 \ldots L-1\), or all), at the current position:

$$
R_{\text{act}} = \frac{\sum_{l \in \mathcal L} \lVert a^{l}_{\text{attn}} \rVert_2}{\sum_{l \in \mathcal L} \lVert a^{l}_{\text{FFN}} \rVert_2}
$$

\(a_{\text{attn}}\) is the attention block's output (after `o_proj`) and
\(a_{\text{FFN}}\) the MLP's output, both before they are added to the residual
stream. The step is exploratory only if **both** hold:

$$
\bar H_t \ge \tau \quad\text{and}\quad R_{\text{eff}} > \theta
$$

## Two modes: `FHG_MODE`

=== "relative (default)"

    \(R_{\text{eff}} = R_{\text{act}} / \overline{R}\), where \(\overline{R}\)
    is the mean of the sequence's previous ratios (1 on the first step). With
    θ = 1 the gate passes steps that are *more attention-dominant than this
    sequence usually is*. This is scale-free across models and layers.

=== "absolute (as written)"

    \(R_{\text{eff}} = R_{\text{act}}\), compared with θ directly.

    !!! warning "With θ around 1 this vetoes everything"
        The MLP outputs of the upper layers are larger than the attention
        outputs. In the smoke test (Qwen3-0.6B) \(R_{\text{act}} \approx 0.58\)
        on average, so θ = 1 vetoed **100%** of the gate openings, and `fhg`
        silently became plain CoT. Every value of the blueprint's sweep
        (0.8–1.4) lies above 0.58. That is why `relative` is the default and
        `absolute` appears in the sweep only as a comparison.

| Smoke test, θ = 1 | Gate open | Vetoed | Latent steps | avg \(R_{\text{act}}\) |
|---|---:|---:|---:|---:|
| `absolute` | 14.8% | 100% | 0% | 0.582 |
| `relative` | 14.4% | 35.7% | 9.1% | 0.596 |

## Implementation

Forward hooks on `layer.self_attn` and `layer.mlp` of every monitored layer
add up the L2 norms of the last position (in float32) during each forward
pass. The ratio is available right after the forward pass:

| Locus | Ratio used for step t |
|---|---|
| input (fhg alone, or with sem / ltr / dynk) | step t's own forward pass |
| `layer`, lens gate | step t − 1's: the hook at \(l^\ast\) fires before the upper layers of step t have run |
| `layer`, final gate | step t's own forward pass |

The report shows the share of openings that fhg vetoed (`FHG vetoed`) and
the average ratio (`avg R_act`).

## Knobs

| Variable | Flag | Default | Sweep | Meaning |
|---|---|---|---|---|
| `FHG_THETA` | `--fhg_theta` | 1.0 | 0.8 1.0 1.2 1.4 | threshold θ on \(R_{\text{eff}}\) |
| `FHG_MODE` | `--fhg_mode` | `relative` | relative absolute | see above |
| `FHG_LAYERS` | `--fhg_layers` | `upper` | upper all | monitored layers: L/2 … L − 1, or all |

The entropy part uses the base gate (k, τ) from `SELAR_PRESET`.

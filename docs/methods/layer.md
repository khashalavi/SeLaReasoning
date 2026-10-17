# Variant 1 · Layer-wise steering

`--method layer` · Idea 1 of `improvement.md` · hidden-state locus

SeLaR changes the **input embedding**, so the intervention has to pass through
every shallow layer before it reaches the middle of the network. Variant 1
intervenes **inside** the network instead: it modifies the hidden state at a
middle layer \(l^\ast\), and keeps the input discrete.

## Formulation (as written in `improvement.md`)

At decoding step \(t\), with \(h_t^{l^\ast}\) the residual stream after layer
\(l^\ast = \lfloor \text{frac} \cdot L \rfloor\):

1. **Logit lens.** \(p_t^{l^\ast} = \operatorname{softmax}\big(W_U \operatorname{RMSNorm}(h_t^{l^\ast}) / T\big)\),
   filtered like the output distribution (top-k 20, top-p 0.95).
2. **Lens entropy.** \(\bar H_t^{l^\ast}\) over the top-k lens candidates,
   normalised by \(\log k\).
3. **Steer if uncertain.** If \(\bar H_t^{l^\ast} \ge \tau\):

$$
\Delta_t = h_t^{l^\ast} - E\big(v^{l^\ast}_{t,\text{top1}}\big), \qquad
\tilde h_t^{l^\ast} = h_t^{l^\ast} + \alpha\, \bar H_t^{l^\ast}\, \frac{\Delta_t}{\lVert\Delta_t\rVert + \epsilon}\, \lVert\Delta_t\rVert
$$

4. Layers \(l^\ast + 1 \ldots L\) continue from \(\tilde h_t^{l^\ast}\). The next
   token is sampled from the final distribution as usual.

The steered position's keys and values in the later layers are computed from
\(\tilde h\), so later steps attend to the steered representation.

## Implementation

A forward hook on `model.model.layers[l*]`, active only on decoding steps
(sequence length 1), never on the prefill:

| Detail | Choice |
|---|---|
| \(l^\ast\) | `int(layer_frac · L)`, clamped to [0, L − 2]. Qwen3-8B (36 layers) → 18, Qwen3-1.7B / 0.6B (28) → 14, R1-Distill-Llama-8B (32) → 16 |
| lens | `lm_head(model.model.norm(h)) / T`, then `apply_sampling_filter`, softmax |
| gate | the same `_Gate` class as the input locus, with `LAYER_TOPK` / `LAYER_TAU` |
| math guard | on the **current input token**: a math symbol entering the context keeps an unsteered representation |
| arithmetic | the step is computed in float32 and cast back to the model dtype |
| hooks | registered per batch and always removed afterwards (`try/finally`) |

## Two gates: `LAYER_GATE`

=== "lens (as written)"

    The gate reads the logit-lens distribution **at layer l\* of the same
    step** and steers that same step.

    ```text
    step t:  layers 0..l*  →  lens entropy  →  steer h_t  →  layers l*+1..L  →  sample x_t+1
    ```

    Mid-layer lens distributions are much less certain than the output. In
    the smoke test (Qwen3-0.6B, GSM8K) the lens entropy at L/2 had median
    0.74. At τ = 0.5 the gate opened on **72%** of the steps and steered
    **61.5%**, against 12.5% exploratory steps for vanilla SeLaR. The
    variant is then no longer selective. The sweep therefore extends
    `LAYER_TAU` to 0.8 and 0.9.

=== "final"

    The gate is **SeLaR's own gate** on the output distribution of step t (same
    k, τ, guard on the sampled token). If it fires, the hidden state is steered
    at step t + 1, the step at which SeLaR would have fed a soft embedding.

    ```text
    step t:    … → output distribution → SeLaR gate fires
    step t+1:  layers 0..l*  →  steer h_t+1 (direction from the lens)  →  layers l*+1..L
    ```

    Trigger and timing are identical to vanilla SeLaR; only the **locus**
    differs. This isolates the question "is steering inside the network better
    than changing the input?". In the smoke test it steered 18.4% of the steps.
    The strength uses the gate's entropy of the output distribution; the
    direction uses the lens candidates.

## Two directions: `LAYER_DIRECTION`

=== "embed_delta (as written)"

    \(\Delta = h - E(v_{\text{top1}})\), step \(= \alpha \bar H \Delta\).

    !!! warning "This is mostly a rescaling of the residual stream"
        At middle layers \(\lVert h \rVert\) is far larger than the norm of an
        input embedding, so \(\Delta \approx h\) and the step is close to
        \(h \leftarrow (1 + \alpha\bar H)\, h\). With pre-norm (RMSNorm) blocks
        the later layers see the same normalised input. Their additive updates
        are, however, relatively smaller against the enlarged residual. The
        effect is that later layers count less at uncertain positions, rather
        than that the state moves towards the alternatives. It is implemented
        as written; whether it helps is an empirical question for the sweep.

=== "lens_contrast (alternative)"

    Not in `improvement.md`. The direction from the lens top-1 towards the
    mixture of the lens top-k, in unembedding space, scaled relative to the
    residual norm:

    $$
    d = \sum_{i \le k} \hat p_i\, U_{v_i} - U_{v_1}, \qquad
    \tilde h = h + \alpha\, \bar H\, \beta\, \lVert h \rVert\, \frac{d}{\lVert d \rVert}
    $$

    \(U\) = `lm_head.weight`, \(\beta\) = `LAYER_REL_SCALE` (default 0.1:
    at most \(\alpha\beta\) = 10% of the residual norm). This is the hidden-state
    analogue of SeLaR's push *away from the dominant token*.

## Knobs

| Variable | Flag | Default | Sweep | Meaning |
|---|---|---|---|---|
| `LAYER_FRAC` | `--layer_frac` | 0.5 | 0.25 0.33 0.5 0.67 | \(l^\ast = \lfloor \text{frac} \cdot L \rfloor\) |
| `LAYER_TAU` | `--layer_tau` | 0.5 | 0.3 … 0.9 | gate threshold on the lens (or, with `final`, the output) entropy |
| `LAYER_GATE` | `--layer_gate` | `lens` | lens final | see above |
| `LAYER_ALPHA` | `--layer_alpha` | 1.0 | 0.5 1.0 1.5 2.0 | steering factor α |
| `LAYER_TOPK` | `--layer_topk` | 3 | 2 3 5 | candidates for the entropy and the direction |
| `LAYER_DIRECTION` | `--layer_direction` | `embed_delta` | embed_delta lens_contrast | see above |
| `LAYER_REL_SCALE` | `--layer_rel_scale` | 0.1 | 0.1 | β, only used by `lens_contrast` (left out of the run tag otherwise) |

The base SeLaR gate (`SELAR_TOPK`, `SELAR_TAU`) and `CONTRASTIVE_W` do not
apply: with `layer`, the input is never soft.

## In combinations

`layer` switches the locus; the other ideas keep their meaning on it.

| Combination | Effect on the hidden-state locus |
|---|---|
| `layer+sem` | the gate uses the semantic entropy of the lens (or output) candidates and the running threshold |
| `layer+ltr` | steering continues for up to M consecutive steps while the entropy stays ≥ \(\tau_{\text{exit}}\); α decays by `LTR_DECAY` per step |
| `layer+fhg` | with the `lens` gate, the attention/FFN ratio of the *previous* step is used (the later layers of the current step have not run yet when the hook fires); with `final`, the current step's |
| `layer+dynk` | the candidate set for the direction and the strength's entropy become the nucleus of size \(k_t\) |

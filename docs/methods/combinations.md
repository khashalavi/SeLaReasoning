# Variant 6 · Combinations

`--method a+b[+c…]` · Variant 6 of `improvement.md`

The five ideas act on different parts of a decoding step (see
[Anatomy of one step](index.md#anatomy-of-one-step)), so any set of distinct
ideas forms a well-defined method:

| Idea | Slot | Composes with the others by |
|---|---|---|
| `sem` | gate signal and threshold | replacing the entropy the gate compares, and the threshold |
| `fhg` | extra gate | an additional AND condition on the gate |
| `ltr` | horizon | wrapping the gate decision in the rollout state machine |
| `dynk` | candidate set | changing what is mixed once a step is exploratory |
| `layer` | locus | moving the whole intervention from the input to layer \(l^\ast\) |

## Naming

Ideas joined with `+`, each at most once, in any order:
`sem+dynk`, `ltr+fhg`, `layer+sem+ltr+fhg+dynk`. With five ideas there are 26
combinations of two or more. `cot` and `selar` cannot be combined: they *are*
the absence of ideas.

In `Sela_beat.sh`:

```bash
COMBOS=("sem+dynk" "ltr+dynk")
```

A combination receives the union of its ideas' knobs, all at their default
values (main phase) or swept together (sweep phase, with
`SWEEP_METHODS=("sem+dynk")`). Its run tag lists them all, e.g.
`sem+dynk_k3_t0.6_sk5_ss0.82_sw15_sg1_sf0.3_dm0.9_dn2_dx8`.

## How the pieces interact

| Combination | Behaviour |
|---|---|
| `sem+dynk` | semantic entropy with a running threshold decides; the nucleus of mass 0.9 is mixed |
| `sem+ltr` | a rollout starts when the semantic entropy passes \(\tau_t\), and continues while the semantic entropy stays ≥ \(\tau_{\text{exit}}\) |
| `ltr+fhg` | fhg gates only the **start** of a rollout; continuation depends on the entropy alone |
| `ltr+dynk` | each soft step of a rollout uses its own \(k_t\), with the decayed contrastive weight |
| `layer+…` | the same gates and candidates, applied to the lens (or output) distribution, steering the hidden state instead of the input; see [layer in combinations](layer.md#in-combinations) |

All combinations are covered by the tests (consistency of the per-step
counters, hooks removed afterwards). Two of them ran in the smoke test on the
GPU: `sem+dynk` and `layer+ltr`.

## Choosing the top-2

`improvement.md` asks for "Best 1 + Best 2". The report's section 3 ranks
every single idea by its average accuracy difference to vanilla SeLaR over the
datasets where both are complete. When at least two ideas are ahead, it
suggests the line to paste:

```text
Variant 6 suggestion (top-2 single ideas that beat vanilla SeLaR):
set COMBOS=("sem+dynk") in Sela_beat.sh.
```

Paste it into `Sela_beat.sh`. The running heart beat picks up the edit at its
next pass and submits the new runs.

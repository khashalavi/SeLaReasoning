"""
CPU tests for src/selarx.py on a tiny random Qwen3 model (no download needed):

    python -m pytest -q tests/test_selarx.py

They pin down the two properties the extension experiments rely on:
  1. the shared loop reproduces the upstream baselines token for token, and
  2. every idea degenerates to its parent method when switched "off" by its
     own hyper-parameters, so a difference in the results can only come from
     the idea itself.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from src.generation_utils import set_seed, generate_cot, generate_selar  # noqa: E402
from src.selarx import (IDEAS, SelarXConfig, _Gate, generate_selarx, nucleus_k,  # noqa: E402
                        semantic_entropy)

VOCAB, PAD, EOS = 96, 0, 13
GEN = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, max_new_tokens=48, do_sample=True)
MATH_IDS = torch.arange(40, 52)


class StubTokenizer:
    eos_token_id = EOS
    pad_token_id = PAD


TOK = StubTokenizer()


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=256, tie_word_embeddings=False,
                      use_sliding_window=False, sliding_window=None)
    m = Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():
        m.lm_head.weight.mul_(16.0)      # ~50% of the steps above tau=0.5
    return m


def batch(n=4, seed=0):
    """Left-padded prompts of different lengths."""
    g = torch.Generator().manual_seed(seed)
    lengths = [6, 9, 4, 9, 7, 5][:n]
    width = max(lengths)
    ids = torch.full((n, width), PAD, dtype=torch.long)
    mask = torch.zeros((n, width), dtype=torch.long)
    for i, length in enumerate(lengths):
        ids[i, width - length:] = torch.randint(14, VOCAB, (length,), generator=g)
        mask[i, width - length:] = 1
    return ids, mask


def run_x(model, method, ids, mask, seed=1, trace=None, **knobs):
    set_seed(seed)
    cfg = SelarXConfig.for_method(method, **knobs)
    return generate_selarx(model, TOK, ids, mask, cfg, math_ids_tensor=MATH_IDS,
                           logits_to_keep=False, trace=trace, **GEN)


# ---------------------------------------------------------------------------
# 1. Upstream equivalence
# ---------------------------------------------------------------------------

def test_cot_matches_upstream(model):
    ids, mask = batch()
    set_seed(1)
    ref = generate_cot(model, TOK, input_ids=ids, attention_mask=mask, **GEN)
    out, stats = run_x(model, "cot", ids, mask)
    assert torch.equal(ref, out)
    assert sum(stats["latent"]) == 0


def test_selar_batch_any_matches_upstream(model):
    ids, mask = batch()
    set_seed(1)
    ref = generate_selar(model, TOK, input_ids=ids, attention_mask=mask, selar_topk=3,
                         entropy_threshold=0.5, math_ids_tensor=MATH_IDS, **GEN)
    out, stats = run_x(model, "selar", ids, mask, soft_mask="batch_any")
    assert torch.equal(ref, out)
    # the quirk: whenever one row is uncertain, every row gets a soft input
    assert sum(stats["soft_inputs"]) > sum(stats["latent"])


@pytest.mark.parametrize("row", [0, 1, 2])
def test_selar_per_row_matches_upstream_at_batch_size_one(model, row):
    ids, mask = batch()
    ids, mask = ids[row:row + 1], mask[row:row + 1]
    set_seed(1)
    ref = generate_selar(model, TOK, input_ids=ids, attention_mask=mask, selar_topk=3,
                         entropy_threshold=0.5, math_ids_tensor=MATH_IDS, **GEN)
    out, _ = run_x(model, "selar", ids, mask)
    assert torch.equal(ref, out)


def test_per_row_soft_inputs_equal_gate_decisions(model):
    ids, mask = batch()
    _, stats = run_x(model, "selar", ids, mask)
    # the last decision of a row that hits EOS is never fed back
    assert sum(stats["soft_inputs"]) <= sum(stats["latent"])
    assert 0 < sum(stats["latent"]) < sum(stats["steps"])


# ---------------------------------------------------------------------------
# 2. Every idea reduces to its parent method when switched off
# ---------------------------------------------------------------------------

def test_layer_with_zero_alpha_is_cot(model):
    ids, mask = batch()
    ref, _ = run_x(model, "cot", ids, mask)
    out, stats = run_x(model, "layer", ids, mask, layer_alpha=0.0)
    assert torch.equal(ref, out)
    assert sum(stats["latent"]) > 0          # the gate fired, the step was just zero


def test_layer_steering_changes_the_trajectory(model):
    ids, mask = batch()
    ref, _ = run_x(model, "cot", ids, mask)
    for gate in ("lens", "final"):
        for direction in ("embed_delta", "lens_contrast"):
            out, _ = run_x(model, "layer", ids, mask, layer_gate=gate, layer_direction=direction,
                           layer_alpha=2.0, layer_rel_scale=0.5)
            assert not torch.equal(ref, out), (gate, direction)


def test_layer_final_gate_with_zero_alpha_is_cot(model):
    ids, mask = batch()
    ref, _ = run_x(model, "cot", ids, mask)
    out, stats = run_x(model, "layer", ids, mask, layer_gate="final", layer_alpha=0.0)
    assert torch.equal(ref, out)
    assert sum(stats["latent"]) > 0


def test_layer_final_gate_triggers_like_selar(model):
    """Same prefill distribution and sampled token -> same first decision as SeLaR."""
    ids, mask = batch()
    t_sel, t_lay = [], []
    run_x(model, "selar", ids, mask, trace=t_sel)
    run_x(model, "layer", ids, mask, trace=t_lay, layer_gate="final", layer_topk=3, layer_tau=0.5)
    assert t_sel[0] == t_lay[0]


def test_fhg_that_never_opens_is_cot(model):
    ids, mask = batch()
    ref, _ = run_x(model, "cot", ids, mask)
    out, stats = run_x(model, "fhg", ids, mask, fhg_theta=1e9)
    assert torch.equal(ref, out)
    assert sum(stats["latent"]) == 0 and sum(stats["vetoed"]) > 0


def test_fhg_that_always_opens_is_selar(model):
    ids, mask = batch()
    ref, _ = run_x(model, "selar", ids, mask)
    out, stats = run_x(model, "fhg", ids, mask, fhg_theta=-1.0)
    assert torch.equal(ref, out)
    assert sum(stats["vetoed"]) == 0


def test_ltr_without_horizon_or_exit_is_selar(model):
    ids, mask = batch()
    ref, _ = run_x(model, "selar", ids, mask)
    out, _ = run_x(model, "ltr", ids, mask, ltr_max=10**6, ltr_exit=0.5, ltr_decay=1.0)
    assert torch.equal(ref, out)


def test_dynk_with_fixed_k_is_selar(model):
    ids, mask = batch()
    ref, _ = run_x(model, "selar", ids, mask)
    out, _ = run_x(model, "dynk", ids, mask, dynk_min=3, dynk_max=3)
    assert torch.equal(ref, out)


def test_sem_gate_normalises_by_active_clusters():
    """improvement.md 3.2: H_sem / log(M) with M = #clusters that carry mass.

    Without merging it equals the vanilla gate while all top-k candidates are
    alive; once top-p filtering zeroes one, M < k and the semantic entropy is
    larger than the vanilla (log k normalised) one.
    """
    E = torch.eye(8)
    probs = torch.zeros(2, 8)
    probs[0, :3] = torch.tensor([0.5, 0.3, 0.2])   # three live candidates
    probs[1, :2] = torch.tensor([0.85, 0.15])      # third candidate filtered to p=0
    vanilla = _Gate(SelarXConfig.for_method("selar"), 3, 0.5, 2, "cpu", E)
    sem = _Gate(SelarXConfig.for_method("sem", sem_topk=3, sem_sim=2.0, sem_window=0), 3, 0.5, 2, "cpu", E)
    dv = vanilla.decide(probs, None, None, None)
    ds = sem.decide(probs, None, None, None)
    assert torch.allclose(dv.h_tok[0], ds.h_gate[0], atol=1e-5)
    assert dv.fire.tolist() == [True, False]       # 0.42 / log 3 = 0.39 < 0.5
    assert ds.fire.tolist() == [True, True]        # 0.42 / log 2 = 0.61


# ---------------------------------------------------------------------------
# 3. Idea-specific behaviour
# ---------------------------------------------------------------------------

def max_consecutive(trace):
    """Longest run of consecutive fired steps per original row."""
    run, best = {}, {}
    for rows, fire in trace:
        if fire is None:
            continue
        for r, f in zip(rows, fire):
            run[r] = run.get(r, 0) + 1 if f else 0
            best[r] = max(best.get(r, 0), run[r])
    return best


def test_ltr_respects_the_horizon(model):
    ids, mask = batch()
    trace = []
    _, stats = run_x(model, "ltr", ids, mask, trace=trace, ltr_max=2, ltr_exit=0.0)
    assert max(max_consecutive(trace).values()) <= 2
    assert sum(stats["commits"]) > 0 and sum(stats["rollouts"]) > 0


def test_ltr_hysteresis_extends_rollouts(model):
    ids, mask = batch()
    _, loose = run_x(model, "ltr", ids, mask, ltr_max=50, ltr_exit=0.0)
    _, strict = run_x(model, "ltr", ids, mask, ltr_max=1, ltr_exit=0.5)
    assert sum(loose["latent"]) / sum(loose["steps"]) > sum(strict["latent"]) / sum(strict["steps"])


def test_sem_dynamic_threshold_is_recorded(model):
    ids, mask = batch()
    _, stats = run_x(model, "sem", ids, mask, sem_window=5, sem_floor=0.1)
    assert sum(stats["thr_n"]) > 0
    assert all(c >= n for c, n in zip(stats["clusters_sum"], stats["steps"]))  # >= 1 cluster/step


@pytest.mark.parametrize("method,gate", [(m, "lens") for m in IDEAS] + [
    ("sem+dynk", "lens"), ("ltr+fhg", "lens"), ("layer+sem+ltr+fhg+dynk", "lens"),
    ("sem+ltr+fhg+dynk", "lens"), ("layer", "final"), ("layer+ltr", "final"),
    ("layer+sem+ltr+fhg+dynk", "final")])
def test_every_method_runs_and_stats_are_consistent(model, method, gate):
    ids, mask = batch(n=5)
    trace = []
    out, stats = run_x(model, method, ids, mask, trace=trace, fhg_mode="relative", fhg_theta=1.0,
                       layer_gate=gate)
    prompt = ids.shape[1]
    for i in range(ids.shape[0]):
        gen = out[i, prompt:].tolist()
        n_gen = gen.index(EOS) + 1 if EOS in gen else len(gen)
        assert stats["steps"][i] == n_gen
        assert 0 <= stats["latent"][i] <= stats["steps"][i]
        assert sum(stats["h_hist"][i]) == stats["h_n"][i]
    for i, layer in enumerate(model.model.layers):
        assert not layer._forward_hooks and not layer.self_attn._forward_hooks, i


def test_logits_to_keep_runs(model):
    ids, mask = batch()
    set_seed(1)
    out, _ = generate_selarx(model, TOK, ids, mask, SelarXConfig.for_method("sem+dynk"),
                             math_ids_tensor=MATH_IDS, logits_to_keep=True, **GEN)
    assert out.shape[0] == ids.shape[0]


def test_bad_methods_are_rejected():
    for bad in ("sem+sem", "foo", "selar+sem", "cot+layer"):
        with pytest.raises(ValueError):
            SelarXConfig.for_method(bad)
    with pytest.raises(ValueError):
        SelarXConfig.for_method("dynk", dynk_min=5, dynk_max=3)


# ---------------------------------------------------------------------------
# 4. Building blocks
# ---------------------------------------------------------------------------

def test_semantic_entropy_without_merges_is_token_entropy():
    E = torch.eye(8)
    idx = torch.tensor([[0, 1, 2]])
    p = torch.tensor([[0.5, 0.3, 0.2]])
    h, m = semantic_entropy(E, idx, p, sim=0.5)
    ref = -(p * p.log()).sum() / torch.log(torch.tensor(3.0))
    assert m.item() == 3 and torch.allclose(h, ref.reshape(1), atol=1e-6)


def test_semantic_entropy_merges_synonyms():
    E = torch.eye(8)
    E[1] = E[0] + 0.01                        # token 1 ~ token 0
    idx = torch.tensor([[0, 1, 2]])
    p = torch.tensor([[0.5, 0.3, 0.2]])
    h, m = semantic_entropy(E, idx, p, sim=0.9)
    mass = torch.tensor([0.8, 0.2])
    ref = -(mass * mass.log()).sum() / torch.log(torch.tensor(2.0))
    assert m.item() == 2 and torch.allclose(h, ref.reshape(1), atol=1e-6)


def test_semantic_entropy_single_cluster_is_zero_and_transitive():
    E = torch.zeros(4, 2)
    E[0] = torch.tensor([1.0, 0.0])
    E[1] = torch.tensor([0.9, 0.44])          # cos(0,1) ~ 0.90
    E[2] = torch.tensor([0.62, 0.78])         # cos(1,2) ~ 0.90, cos(0,2) ~ 0.62
    idx = torch.tensor([[0, 1, 2]])
    p = torch.tensor([[0.4, 0.35, 0.25]])
    h, m = semantic_entropy(E, idx, p, sim=0.85)
    assert m.item() == 1 and h.item() == 0.0


def test_nucleus_k():
    p = torch.tensor([[0.5, 0.3, 0.15, 0.05],
                      [0.97, 0.02, 0.01, 0.0]])
    assert nucleus_k(p, 0.9, 2, 8).tolist() == [3, 2]
    flat = torch.full((1, 10), 0.1)
    assert nucleus_k(flat, 0.9, 2, 8).tolist() == [8]

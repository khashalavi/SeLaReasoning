"""
SeLaR-X: vanilla SeLaR plus the training-free extensions planned in
improvement.md, behind ONE decoding loop so that every method shares the same
sampling, stopping and bookkeeping code.

    method     what changes (everything else is vanilla SeLaR)
    ---------  ------------------------------------------------------------------
    cot        no latent step at all: plain sampling through the same loop
    selar      vanilla SeLaR (Algorithm 1 of the paper, gated per row)
    layer      Idea 1  steer the hidden state at layer l* instead of the input
                       embedding, gated by the logit-lens entropy at l*
                       (layer_gate="final": by SeLaR's own gate instead)
    sem        Idea 2  semantic (embedding-clustered) entropy + running-window
                       threshold
    ltr        Idea 3  multi-step latent rollouts with an exit threshold
    fhg        Idea 4  extra gate on the attention/FFN activation-norm ratio
    dynk       Idea 5  nucleus (cumulative-mass) dynamic k for the soft embedding
    a+b(+c)    Variant 6: any combination of the five ideas, e.g. "sem+dynk".
               The ideas act on different parts of the step, so they compose:
               gate signal (sem), extra gate (fhg), candidate set (dynk),
               horizon (ltr) and locus (layer: input embedding -> hidden state).

Equivalences checked by tests/test_selarx.py:
  * cot                          == generate_cot   (same tokens, same seed)
  * selar, soft_mask=batch_any   == generate_selar (upstream code, any batch)
  * selar, soft_mask=per_row     == generate_selar at batch size 1
"""

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F

from src.generation_utils import apply_sampling_filter

IDEAS = ("layer", "sem", "ltr", "fhg", "dynk")
ENTROPY_BINS = 20


@dataclass
class SelarXConfig:
    latent: bool = True             # False: plain CoT through the same loop
    # ---- vanilla SeLaR ----
    topk: int = 3
    tau: float = 0.5
    contrastive_weight: float = 1.0
    # per_row:   rows below the threshold keep their discrete token (paper, Alg. 1)
    # batch_any: every row gets its soft embedding as soon as ANY row in the batch
    #            is above the threshold -- what the upstream generate_selar() does
    soft_mask: str = "per_row"
    math_guard: bool = True         # math-symbol tokens always stay discrete
    # ---- Idea 1: layer-wise hidden-state steering ----
    layer: bool = False
    layer_frac: float = 0.5         # l* = floor(layer_frac * L)
    layer_tau: float = 0.5
    layer_alpha: float = 1.0
    layer_topk: int = 3
    # embed_delta:   delta = h - E(v_top1), as written in improvement.md
    # lens_contrast: unembedding-space direction from the lens top-1 towards the
    #                top-k mixture, scaled to layer_rel_scale * ||h||
    layer_direction: str = "embed_delta"
    layer_rel_scale: float = 0.1
    # lens:  gate on the logit-lens entropy at l*, steer the same step (as written)
    # final: SeLaR's own gate (final distribution of step t) decides whether to
    #        steer at step t+1 -- same trigger as vanilla SeLaR, only the locus differs
    layer_gate: str = "lens"
    # ---- Idea 2: semantic entropy + dynamic threshold ----
    sem: bool = False
    sem_topk: int = 5
    sem_sim: float = 0.82           # cosine similarity that merges two candidates
    sem_window: int = 15            # 0 = static threshold (tau)
    sem_gamma: float = 1.0
    sem_floor: float = 0.3          # lower bound of the dynamic threshold
    # ---- Idea 3: latent trajectory rollouts ----
    ltr: bool = False
    ltr_max: int = 3                # max consecutive soft steps
    ltr_exit: float = 0.2           # stay latent while entropy >= ltr_exit
    ltr_decay: float = 0.9          # contrastive weight * decay^(m-1)
    ltr_commit: str = "sample"      # token emitted when the horizon forces a commit
    # ---- Idea 4: functional (attention vs. FFN) gating ----
    fhg: bool = False
    fhg_theta: float = 1.0
    # relative: R / mean(R so far) > theta | absolute: R > theta (as written; R ~ 0.6
    # on Qwen3, so theta ~ 1 vetoes every opening -- see the smoke test in the docs)
    fhg_mode: str = "relative"
    fhg_layers: str = "upper"       # upper: layers L/2..L-1 | all
    # ---- Idea 5: nucleus dynamic k ----
    dynk: bool = False
    dynk_mass: float = 0.9
    dynk_min: int = 2
    dynk_max: int = 8

    @classmethod
    def for_method(cls, method, **knobs):
        """'cot' | 'selar' | an idea | ideas joined by '+', plus knob overrides."""
        if method == "cot":
            cfg = cls(latent=False)
        elif method == "selar":
            cfg = cls()
        else:
            parts = method.split("+")
            unknown = [p for p in parts if p not in IDEAS]
            if unknown or len(set(parts)) != len(parts):
                raise ValueError(f"bad method '{method}' (ideas: {', '.join(IDEAS)})")
            cfg = cls(**{p: True for p in parts})
        for key, value in knobs.items():
            if value is None:
                continue
            if key not in cls.__dataclass_fields__ or key in IDEAS + ("latent",):
                raise ValueError(f"unknown SeLaR-X knob '{key}'")
            setattr(cfg, key, value)
        cfg.validate()
        return cfg

    @property
    def input_locus(self):
        return self.latent and not self.layer

    def validate(self):
        choices = {
            "soft_mask": ("per_row", "batch_any"),
            "layer_direction": ("embed_delta", "lens_contrast"),
            "layer_gate": ("lens", "final"),
            "ltr_commit": ("sample", "argmax"),
            "fhg_mode": ("absolute", "relative"),
            "fhg_layers": ("upper", "all"),
        }
        for key, allowed in choices.items():
            if getattr(self, key) not in allowed:
                raise ValueError(f"{key}={getattr(self, key)!r}, expected one of {allowed}")
        if min(self.topk, self.layer_topk, self.sem_topk) < 2:
            raise ValueError("topk, layer_topk and sem_topk must be >= 2")
        if not 2 <= self.dynk_min <= self.dynk_max:
            raise ValueError(f"need 2 <= dynk_min <= dynk_max (got {self.dynk_min}, {self.dynk_max})")
        if self.ltr_max < 1 or self.sem_window < 0:
            raise ValueError("ltr_max must be >= 1 and sem_window >= 0")
        if not 0.0 < self.layer_frac < 1.0:
            raise ValueError(f"layer_frac must lie in (0, 1) (got {self.layer_frac})")

    def to_dict(self):
        return asdict(self)


# ============================================================================
# Building blocks
# ============================================================================

def semantic_entropy(E, idx, probs, sim):
    """Entropy over clusters of near-synonymous candidates.

    Candidates whose input embeddings have cosine similarity > sim are linked;
    clusters are the connected components. Returns (entropy normalised by
    log(#clusters) -- 0 when everything falls into one cluster --, #clusters).
    """
    p = probs.float()
    p = p / (p.sum(dim=-1, keepdim=True) + 1e-10)
    emb = F.normalize(E[idx].float(), dim=-1)                       # [b, k, d]
    k = idx.shape[-1]
    reach = (emb @ emb.transpose(1, 2)) > sim
    reach = reach | torch.eye(k, dtype=torch.bool, device=idx.device)
    hops = 1
    while hops < k - 1:                                               # transitive closure
        reach = (reach.float() @ reach.float()) > 0
        hops *= 2
    rep = reach.float().argmax(dim=-1)                                # first member = cluster id
    mass = torch.zeros_like(p).scatter_add_(1, rep, p)
    present = mass > 0
    n_clusters = present.sum(dim=-1)
    plogp = torch.where(present, mass * torch.log(mass.clamp_min(1e-12)), torch.zeros_like(mass))
    ent = -plogp.sum(dim=-1)
    norm = torch.log(n_clusters.clamp(min=2).float())
    h = torch.where(n_clusters >= 2, ent / norm, torch.zeros_like(ent))
    return h.clamp(0.0, 1.0), n_clusters


def nucleus_k(sorted_probs, mass, k_min, k_max):
    """Smallest m with F(m) = sum_{j<=m} p_(j) >= mass, clamped to [k_min, k_max]."""
    cum = torch.cumsum(sorted_probs.float(), dim=-1)
    k = (cum < mass).sum(dim=-1) + 1
    return k.clamp(min=k_min, max=k_max)


def _candidate_mixture(d, cfg):
    """Weights, token ids and entropy of the soft-embedding candidate set.

    Vanilla: the top-k renormalised distribution and its gate entropy.
    dynk:    the top-k_t nucleus, entropy normalised by log(k_t).
    """
    if not cfg.dynk:
        return d.p_norm, d.top_i, d.h_tok, None
    p = d.ext_p[:, :cfg.dynk_max]
    k_t = nucleus_k(p, cfg.dynk_mass, cfg.dynk_min, min(cfg.dynk_max, p.shape[-1]))
    keep = torch.arange(p.shape[-1], device=p.device).unsqueeze(0) < k_t.unsqueeze(-1)
    p = p * keep
    p_norm = p / (p.sum(dim=-1, keepdim=True) + 1e-10)
    ent = -torch.sum(p_norm * torch.log(p_norm + 1e-10), dim=-1)
    h = torch.clamp(ent / torch.log(k_t.to(ent.dtype)), 0.0, 1.0)
    return p_norm, d.ext_i[:, :cfg.dynk_max], h, k_t


def soft_embedding(E, d, cfg):
    """Regularised soft embedding (Eq. 8-9 of the paper), same op order as upstream."""
    p_norm, idx, h, k_t = _candidate_mixture(d, cfg)
    topk_embeddings = E[idx]
    base_soft_embed = torch.sum(p_norm.unsqueeze(-1) * topk_embeddings, dim=1)
    top1_embed = topk_embeddings[:, 0, :]
    diff_direction = base_soft_embed - top1_embed
    diff_norm = torch.norm(diff_direction, dim=-1, keepdim=True) + 1e-10
    diff_unit = diff_direction / diff_norm
    if cfg.ltr:
        decay = cfg.ltr_decay ** (d.roll_pos.clamp(min=1) - 1).to(torch.float32)
        contrastive_strength = (cfg.contrastive_weight * decay.to(h.dtype) * h).unsqueeze(-1)
    else:
        contrastive_strength = cfg.contrastive_weight * h.unsqueeze(-1)
    return base_soft_embed + contrastive_strength * diff_unit * diff_norm, k_t


class _Decision:
    """Everything the gate computed for one step (fields default to None)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, name):
        return None

    def select(self, keep):
        """Keep only the rows still decoding (every tensor field is per row)."""
        for name, value in list(self.__dict__.items()):
            if torch.is_tensor(value) and value.dim() > 0:
                self.__dict__[name] = value[keep]
        return self


def _candidates(probs, k, max_entropy, cfg):
    """Top-k candidates and their normalised entropy exactly as in
    generate_selar, plus the longer list sem / dynk need. Stateless."""
    topk_probs, topk_indices = torch.topk(probs, k=k, dim=-1)
    probs_sum = topk_probs.sum(dim=-1, keepdim=True)
    probs_normalized = topk_probs / (probs_sum + 1e-10)
    log_probs = torch.log(probs_normalized + 1e-10)
    entropy = -torch.sum(probs_normalized * log_probs, dim=-1)
    h_tok = torch.clamp(entropy / max_entropy, 0.0, 1.0)
    d = _Decision(top_i=topk_indices, p_norm=probs_normalized, h_tok=h_tok)
    k_ext = max(cfg.sem_topk if cfg.sem else 0, cfg.dynk_max if cfg.dynk else 0)
    if k_ext:
        d.ext_p, d.ext_i = torch.topk(probs, k=min(k_ext, probs.shape[-1]), dim=-1)
    return d


class _Gate:
    """Decides, per row, whether a step is exploratory.

    Shared by both loci: the input locus feeds it the final-layer distribution
    (decision = is the NEXT input soft?), the hidden locus the logit-lens
    distribution at layer l* (decision = steer the CURRENT hidden state?).
    Rows advance in lockstep and are only ever removed, so the step counter
    and the history ring buffer can share one write position.
    """

    def __init__(self, cfg, k, tau, n_rows, device, E):
        self.cfg, self.k, self.tau, self.E = cfg, k, tau, E
        self.max_entropy = torch.log(torch.tensor(float(k), device=device))
        self.steps = 0
        self.window = cfg.sem_window if cfg.sem else 0
        self.hist = torch.zeros(n_rows, self.window, device=device) if self.window else None
        self.roll = torch.zeros(n_rows, dtype=torch.long, device=device)
        self.ratio_sum = torch.zeros(n_rows, device=device)
        self.ratio_n = 0

    def select(self, keep):
        if self.hist is not None:
            self.hist = self.hist[keep]
        self.roll = self.roll[keep]
        self.ratio_sum = self.ratio_sum[keep]

    def decide(self, probs, guard_tokens, math_ids, ratio):
        cfg = self.cfg
        d = _candidates(probs, self.k, self.max_entropy, cfg)
        h_tok = d.h_tok

        # ---- gate signal (Idea 2: semantic entropy) ----
        if cfg.sem:
            h_gate, d.n_clusters = semantic_entropy(self.E, d.ext_i[:, :cfg.sem_topk],
                                                    d.ext_p[:, :cfg.sem_topk], cfg.sem_sim)
        else:
            h_gate = h_tok
        d.h_gate = h_gate

        # ---- threshold (Idea 2: running window, static tau until it is full) ----
        if self.window and self.steps >= self.window:
            mu = self.hist.mean(dim=-1)
            sigma = self.hist.std(dim=-1, correction=0)
            d.thr = torch.clamp(mu + cfg.sem_gamma * sigma, min=cfg.sem_floor)
            enter = h_gate.float() > d.thr
        else:
            enter = h_gate >= self.tau
        if self.window:
            self.hist[:, self.steps % self.window] = h_gate.float()
        self.steps += 1
        d.gate_open = enter

        # ---- Idea 4: attention-dominant (procedural) steps only ----
        if cfg.fhg and ratio is not None:
            r = ratio
            if cfg.fhg_mode == "relative":
                if self.ratio_n:
                    r = ratio / (self.ratio_sum / self.ratio_n).clamp_min(1e-12)
                else:
                    r = torch.ones_like(ratio)
                self.ratio_sum = self.ratio_sum + ratio
                self.ratio_n += 1
            procedural = r > cfg.fhg_theta
            d.vetoed = enter & ~procedural
            enter = enter & procedural

        # ---- math symbols stay discrete ----
        if cfg.math_guard and math_ids is not None and guard_tokens is not None:
            guard = (guard_tokens.unsqueeze(-1) == math_ids).any(dim=-1)
        else:
            guard = torch.zeros_like(enter)
        d.guarded = enter & guard

        # ---- Idea 3: rollout state machine ----
        if cfg.ltr:
            in_roll = self.roll > 0
            uncertain = h_gate >= cfg.ltr_exit
            cont = in_roll & ~guard & uncertain & (self.roll < cfg.ltr_max)
            start = ~in_roll & enter & ~guard
            d.commit = in_roll & ~guard & uncertain & (self.roll >= cfg.ltr_max)
            fire = cont | start
            ones, zeros = torch.ones_like(self.roll), torch.zeros_like(self.roll)
            self.roll = torch.where(cont, self.roll + 1, torch.where(start, ones, zeros))
            d.roll_pos, d.started = self.roll.clone(), start
        else:
            fire = enter & ~guard
        d.fire = fire
        return d


class _Hooks:
    """Forward hooks for Idea 1 (steering at layer l*) and Idea 4 (activation norms)."""

    def __init__(self, model, cfg, gate, sampling, math_ids):
        self.cfg, self.gate, self.sampling, self.math_ids = cfg, gate, sampling, math_ids
        self.E = model.get_input_embeddings().weight
        self.norm = model.model.norm
        self.lm_head = model.lm_head
        layers = model.model.layers
        n_layers = len(layers)
        self.handles = []
        self.decode = False
        self.current_tokens = None
        self.prev_ratio = None          # ratio of the previous forward (hidden locus)
        self.decision = None            # lens gate: decision made inside this forward
        self.pending = None             # final gate: decision of the previous step
        self.steered = None             # rows steered in this forward
        self.k_t = None
        self.attn_acc = self.mlp_acc = None
        if cfg.fhg:
            first = 0 if cfg.fhg_layers == "all" else n_layers // 2
            for layer in layers[first:]:
                self.handles.append(layer.self_attn.register_forward_hook(self._attn_hook))
                self.handles.append(layer.mlp.register_forward_hook(self._mlp_hook))
        if cfg.layer:
            self.l_star = min(max(int(cfg.layer_frac * n_layers), 0), n_layers - 2)
            self.handles.append(layers[self.l_star].register_forward_hook(self._steer_hook))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def before_forward(self, decode, current_tokens):
        self.decode, self.current_tokens = decode, current_tokens
        self.attn_acc = self.mlp_acc = None
        self.decision = self.k_t = self.steered = None

    def ratio(self):
        """Attention/FFN output-norm ratio of the last position, this forward."""
        return self.attn_acc / (self.mlp_acc + 1e-6)

    def select(self, keep):
        if self.prev_ratio is not None:
            self.prev_ratio = self.prev_ratio[keep]
        if self.pending is not None:
            self.pending.select(keep)

    @staticmethod
    def _last_norm(output):
        out = output[0] if isinstance(output, tuple) else output
        return torch.linalg.vector_norm(out[:, -1, :].float(), dim=-1)

    def _attn_hook(self, module, args, output):
        n = self._last_norm(output)
        self.attn_acc = n if self.attn_acc is None else self.attn_acc + n

    def _mlp_hook(self, module, args, output):
        n = self._last_norm(output)
        self.mlp_acc = n if self.mlp_acc is None else self.mlp_acc + n

    def _steer_hook(self, module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        if not self.decode or hs.shape[1] != 1:
            return output
        cfg = self.cfg
        if cfg.layer_gate == "final" and (self.pending is None or not self.pending.fire.any()):
            return output
        h = hs[:, -1, :]
        # logit lens at l*, filtered exactly like the final distribution
        lens = self.lm_head(self.norm(h)) / self.sampling["temperature"]
        lens = apply_sampling_filter(lens, top_k=self.sampling["top_k"],
                                     top_p=self.sampling["top_p"], min_p=self.sampling["min_p"])
        probs = F.softmax(lens, dim=-1)
        if cfg.layer_gate == "final":
            # trigger and strength from SeLaR's gate on the previous step's final
            # distribution; the direction still comes from the lens candidates
            d = self.pending
            h_c = _candidate_mixture(d, cfg)[2]
            p_norm, idx, _, self.k_t = _candidate_mixture(
                _candidates(probs, self.gate.k, self.gate.max_entropy, cfg), cfg)
        else:
            d = self.gate.decide(probs, self.current_tokens, self.math_ids,
                                 self.prev_ratio if cfg.fhg else None)
            self.decision = d
            if not d.fire.any():
                return output
            p_norm, idx, h_c, self.k_t = _candidate_mixture(d, cfg)
        self.steered = d.fire
        alpha = torch.full_like(h_c, cfg.layer_alpha, dtype=torch.float32)
        if cfg.ltr:
            alpha = alpha * cfg.ltr_decay ** (d.roll_pos.clamp(min=1) - 1).to(torch.float32)
        scale = (alpha * h_c.float()).unsqueeze(-1)
        hf = h.float()
        if cfg.layer_direction == "embed_delta":
            # improvement.md 2.2: h~ = h + H * delta/(||delta||+eps) * ||delta||
            delta = hf - self.E[idx[:, 0]].float()
            delta_norm = torch.norm(delta, dim=-1, keepdim=True)
            step = scale * (delta / (delta_norm + 1e-10)) * delta_norm
        else:
            U = self.lm_head.weight[idx].float()                          # [b, k, d]
            direction = torch.sum(p_norm.float().unsqueeze(-1) * U, dim=1) - U[:, 0, :]
            unit = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-10)
            step = scale * cfg.layer_rel_scale * torch.norm(hf, dim=-1, keepdim=True) * unit
        steered = torch.where(d.fire.unsqueeze(-1), (hf + step).to(h.dtype), h)
        hs = steered.unsqueeze(1)
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs


class _Recorder:
    """Per-sample counters, indexed by the ORIGINAL batch row."""

    KEYS = ("steps", "latent", "soft_inputs", "gate_open", "guarded", "vetoed",
            "rollouts", "commits", "k_sum", "clusters_sum", "ratio_sum", "ratio_n",
            "h_sum", "h_sq", "h_n", "htok_sum", "thr_sum", "thr_n")

    def __init__(self, n, device):
        self.c = {k: torch.zeros(n, dtype=torch.float64, device=device) for k in self.KEYS}
        self.hist = torch.zeros(n, ENTROPY_BINS, dtype=torch.float64, device=device)

    def add(self, key, rows, values):
        self.c[key].index_add_(0, rows, values.to(torch.float64))

    def entropy(self, rows, h, h_tok):
        h = h.float()
        self.add("h_sum", rows, h)
        self.add("h_sq", rows, h * h)
        self.add("h_n", rows, torch.ones_like(h))
        self.add("htok_sum", rows, h_tok.float())
        bins = (h * ENTROPY_BINS).long().clamp(0, ENTROPY_BINS - 1)
        self.hist.index_put_((rows, bins), torch.ones_like(h, dtype=torch.float64), accumulate=True)

    def decision(self, rows, d, k_t):
        self.entropy(rows, d.h_gate, d.h_tok)
        self.add("latent", rows, d.fire)
        self.add("gate_open", rows, d.gate_open)
        self.add("guarded", rows, d.guarded)
        if d.vetoed is not None:
            self.add("vetoed", rows, d.vetoed)
        if d.started is not None:
            self.add("rollouts", rows, d.started)
            self.add("commits", rows, d.commit)
        if d.n_clusters is not None:
            self.add("clusters_sum", rows, d.n_clusters)
        if d.thr is not None:
            self.add("thr_sum", rows, d.thr)
            self.add("thr_n", rows, torch.ones_like(d.thr))
        if k_t is not None:
            self.add("k_sum", rows, torch.where(d.fire, k_t, torch.zeros_like(k_t)))

    def export(self):
        out = {k: v.tolist() for k, v in self.c.items()}
        out["h_hist"] = self.hist.tolist()
        return out


# ============================================================================
# Generation
# ============================================================================

@torch.no_grad()
def generate_selarx(model, tokenizer, input_ids, attention_mask, cfg, *,
                    temperature=1.0, top_p=1.0, top_k=0, min_p=0.0,
                    max_new_tokens=32768, do_sample=True, math_ids_tensor=None,
                    eos_token_ids=None, logits_to_keep=True, trace=None):
    """Batched SeLaR-X decoding.

    Returns (ids, stats): ids is [batch, prompt+generated] padded with
    pad_token_id like the upstream generate_* functions; stats maps every
    _Recorder key to one value per sample (plus "h_hist", 20 bins each).
    trace: optional list; receives one (orig_rows, fire) pair per step (tests).
    """
    device = input_ids.device
    batch_size = input_ids.shape[0]
    E = model.get_input_embeddings().weight
    if eos_token_ids is None:
        eos_token_ids = [tokenizer.eos_token_id]
    eos = torch.tensor(sorted({int(t) for t in eos_token_ids if t is not None}), device=device)
    sampling = {"temperature": temperature, "top_k": top_k, "top_p": top_p, "min_p": min_p}

    rec = _Recorder(batch_size, device)
    all_generated = [input_ids[i].clone().tolist() for i in range(batch_size)]
    unfinished_idx = list(range(batch_size))
    rows = torch.arange(batch_size, device=device)       # original row of every live row

    attn_mask = attention_mask.clone()
    past_key_values = None
    next_tokens = None
    use_soft_input = torch.zeros(batch_size, dtype=torch.bool, device=device)
    soft_embeds = None

    gate = None
    if cfg.latent:
        k, tau = (cfg.layer_topk, cfg.layer_tau) if cfg.layer else (cfg.topk, cfg.tau)
        gate = _Gate(cfg, k, tau, batch_size, device, E)
    hooks = _Hooks(model, cfg, gate, sampling, math_ids_tensor) if (cfg.layer or cfg.fhg) else None
    max_entropy_cot = torch.log(torch.tensor(float(cfg.topk), device=device))
    extra = {"logits_to_keep": 1} if logits_to_keep else {}

    try:
        for step in range(max_new_tokens):
            cur_batch = rows.shape[0]
            if past_key_values is None:
                model_inputs = {"input_ids": input_ids, "attention_mask": attn_mask}
            else:
                attention_mask_new = torch.ones((cur_batch, 1), dtype=attn_mask.dtype, device=device)
                attn_mask = torch.cat([attn_mask, attention_mask_new], dim=1)
                if cfg.input_locus and use_soft_input.any():
                    if cfg.soft_mask == "batch_any":
                        embeds = soft_embeds
                        rec.add("soft_inputs", rows, torch.ones_like(use_soft_input))
                    else:
                        embeds = torch.where(use_soft_input.unsqueeze(-1), soft_embeds, E[next_tokens])
                        rec.add("soft_inputs", rows, use_soft_input)
                    model_inputs = {"inputs_embeds": embeds.unsqueeze(1), "attention_mask": attn_mask}
                else:
                    model_inputs = {"input_ids": next_tokens.unsqueeze(1), "attention_mask": attn_mask}

            if hooks is not None:
                hooks.before_forward(past_key_values is not None, next_tokens)
            outputs = model(**model_inputs, past_key_values=past_key_values,
                            use_cache=True, return_dict=True, **extra)
            past_key_values = outputs.past_key_values

            logits = outputs.logits[:, -1, :] / temperature
            logits_filtered = apply_sampling_filter(logits, top_k=top_k, top_p=top_p, min_p=min_p)
            probs = F.softmax(logits_filtered, dim=-1)
            if do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(probs, dim=-1)

            ratio = hooks.ratio() if (hooks is not None and cfg.fhg) else None
            fire = None
            if cfg.input_locus:
                d = gate.decide(probs, next_tokens, math_ids_tensor, ratio)
                if cfg.ltr and cfg.ltr_commit == "argmax" and d.commit.any():
                    next_tokens = torch.where(d.commit, torch.argmax(probs, dim=-1), next_tokens)
                k_t = None
                if d.fire.any():
                    soft_embeds, k_t = soft_embedding(E, d, cfg)
                use_soft_input = d.fire
                rec.decision(rows, d, k_t)
                fire = d.fire
            elif cfg.layer and cfg.layer_gate == "final":
                # decide now, steer during the next forward (like SeLaR's soft input)
                if hooks.steered is not None and hooks.k_t is not None:
                    rec.add("k_sum", rows, torch.where(hooks.steered, hooks.k_t, torch.zeros_like(hooks.k_t)))
                d = gate.decide(probs, next_tokens, math_ids_tensor, ratio)
                rec.decision(rows, d, None)
                hooks.pending = d
                fire = d.fire
            elif cfg.layer:
                if hooks.decision is not None:          # None on the prefill step
                    rec.decision(rows, hooks.decision, hooks.k_t)
                    fire = hooks.decision.fire
            else:
                # CoT: entropy statistics only, nothing changes the decoding
                p = torch.topk(probs, k=cfg.topk, dim=-1)[0]
                p = p / (p.sum(dim=-1, keepdim=True) + 1e-10)
                h = torch.clamp(-torch.sum(p * torch.log(p + 1e-10), dim=-1) / max_entropy_cot, 0.0, 1.0)
                rec.entropy(rows, h, h)
            if ratio is not None:
                rec.add("ratio_sum", rows, ratio)
                rec.add("ratio_n", rows, torch.ones_like(ratio))
                hooks.prev_ratio = ratio
            rec.add("steps", rows, torch.ones(cur_batch, device=device))
            if trace is not None:
                trace.append((rows.tolist(), None if fire is None else fire.tolist()))

            for bi, token in enumerate(next_tokens.tolist()):
                all_generated[unfinished_idx[bi]].append(token)

            cur_finished = torch.isin(next_tokens, eos)
            if cur_finished.any():
                keep_idx = (~cur_finished).nonzero(as_tuple=False).squeeze(-1)
                unfinished_idx = [unfinished_idx[i] for i in keep_idx.tolist()]
                if len(unfinished_idx) == 0:
                    break
                rows = rows[keep_idx]
                next_tokens = next_tokens[keep_idx]
                attn_mask = attn_mask[keep_idx]
                use_soft_input = use_soft_input[keep_idx]
                if soft_embeds is not None:
                    soft_embeds = soft_embeds[keep_idx]
                if gate is not None:
                    gate.select(keep_idx)
                if hooks is not None:
                    hooks.select(keep_idx)
                past_key_values.batch_select_indices(keep_idx)
    finally:
        if hooks is not None:
            hooks.remove()

    maxlen = max(len(g) for g in all_generated)
    out = torch.full((batch_size, maxlen), tokenizer.pad_token_id or 0, dtype=torch.long, device=device)
    for i, ids in enumerate(all_generated):
        out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return out, rec.export()

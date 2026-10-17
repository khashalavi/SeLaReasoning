# selarx.py API

`src/selarx.py` · import as `from src.selarx import SelarXConfig, generate_selarx`

## Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.generation_utils import set_seed, get_math_symbols_ids
from src.selarx import SelarXConfig, generate_selarx

path = "/lustre/scratch/data/s76kalav_hpc-memory/selareasoning/models/Qwen/Qwen3-8B"
tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto", device_map={"": 0})

text = tok.apply_chat_template([{"role": "user", "content": "What is 17 * 23? ..."}],
                               tokenize=False, add_generation_prompt=True, enable_thinking=True)
enc = tok([text], return_tensors="pt", padding=True).to(model.device)
math_ids = torch.tensor(sorted(get_math_symbols_ids(tok)), device=model.device)

cfg = SelarXConfig.for_method("sem+dynk", tau=0.5, sem_sim=0.85)
set_seed(41)
ids, stats = generate_selarx(model, tok, enc["input_ids"], enc["attention_mask"], cfg,
                             temperature=0.6, top_p=0.95, top_k=20, max_new_tokens=4096,
                             math_ids_tensor=math_ids, eos_token_ids=[151645, 151643])
print(tok.decode(ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
print(stats["latent"][0] / stats["steps"][0])     # share of latent steps
```

## `SelarXConfig`

A dataclass; every field is a knob. `for_method` builds it.

| Field | Default | Meaning |
|---|---|---|
| `latent` | True | False = plain CoT (set by `for_method("cot")`) |
| `topk`, `tau` | 3, 0.5 | the base gate |
| `contrastive_weight` | 1.0 | weight of the contrastive push |
| `soft_mask` | `per_row` | `per_row` or `batch_any` |
| `math_guard` | True | math symbols stay discrete |
| `layer`, `sem`, `ltr`, `fhg`, `dynk` | False | which ideas are on (set by `for_method`) |
| `layer_frac`, `layer_tau`, `layer_alpha`, `layer_topk` | 0.5, 0.5, 1.0, 3 | [Variant 1](../methods/layer.md) |
| `layer_direction`, `layer_rel_scale`, `layer_gate` | `embed_delta`, 0.1, `lens` | |
| `sem_topk`, `sem_sim`, `sem_window`, `sem_gamma`, `sem_floor` | 5, 0.82, 15, 1.0, 0.3 | [Variant 2](../methods/sem.md) |
| `ltr_max`, `ltr_exit`, `ltr_decay`, `ltr_commit` | 3, 0.2, 0.9, `sample` | [Variant 3](../methods/ltr.md) |
| `fhg_theta`, `fhg_mode`, `fhg_layers` | 1.0, `relative`, `upper` | [Variant 4](../methods/fhg.md) |
| `dynk_mass`, `dynk_min`, `dynk_max` | 0.9, 2, 8 | [Variant 5](../methods/dynk.md) |

| Member | |
|---|---|
| `SelarXConfig.for_method(method, **knobs)` | `"cot"`, `"selar"`, an idea, or ideas joined by `+`; knobs set fields (`None` = keep default). Raises `ValueError` for unknown ideas, repeated ideas, unknown knobs, or knobs that would switch ideas |
| `validate()` | checks the word-valued fields and the ranges (`topk`, `layer_topk`, `sem_topk` ≥ 2; 2 ≤ `dynk_min` ≤ `dynk_max`; `ltr_max` ≥ 1; `sem_window` ≥ 0; 0 < `layer_frac` < 1) |
| `input_locus` | True if the next input can be soft (latent and not `layer`) |
| `to_dict()` | all fields, as stored in `config.json` |

## `generate_selarx`

```python
generate_selarx(model, tokenizer, input_ids, attention_mask, cfg, *,
                temperature=1.0, top_p=1.0, top_k=0, min_p=0.0,
                max_new_tokens=32768, do_sample=True, math_ids_tensor=None,
                eos_token_ids=None, logits_to_keep=True, trace=None) -> (ids, stats)
```

| Argument | Meaning |
|---|---|
| `input_ids`, `attention_mask` | left-padded prompt batch |
| `cfg` | a `SelarXConfig` |
| sampling arguments | as in the upstream loops |
| `math_ids_tensor` | token ids of math symbols (`get_math_symbols_ids`), for the guard |
| `eos_token_ids` | ids that end a sequence; default `[tokenizer.eos_token_id]` |
| `logits_to_keep` | compute only the last position's logits at prefill |
| `trace` | a list that receives `(original rows, fire mask)` per step (tests) |

Returns:

- **`ids`**: `[batch, prompt + generated]`, padded with `pad_token_id` after
  each sequence's EOS, like the upstream functions.
- **`stats`**: `{counter: [value per problem]}` for every counter in
  [What every step records](../methods/index.md#what-every-step-records),
  plus `h_hist` (20 bins per problem).

Requirements on the model: a decoder with `model.model.layers`,
`model.model.norm`, `model.lm_head`, `layer.self_attn`, `layer.mlp`, and a
`DynamicCache` with `batch_select_indices` (Qwen3 and Llama in Transformers
4.51). One device: the soft embeddings and hooks assume everything on one
GPU.

## Building blocks

| Function / class | Purpose |
|---|---|
| `semantic_entropy(E, idx, probs, sim)` | clusters the candidates `idx` by cosine similarity of `E[idx]` > `sim`; returns (entropy normalised by log #clusters, #clusters) |
| `nucleus_k(sorted_probs, mass, k_min, k_max)` | smallest m with cumulative mass ≥ `mass`, clamped |
| `soft_embedding(E, d, cfg)` | regularised soft embedding (paper Eq. 8–9) from a gate decision, with the LTR decay and the dynk candidate set; same operation order as upstream |
| `_candidates(probs, k, max_entropy, cfg)` | top-k probabilities, renormalised; normalised entropy; the longer candidate list for sem / dynk. Stateless |
| `_Gate` | the per-row decision: gate signal, threshold, fhg, math guard, rollout state. One instance per batch; `decide()` advances its state exactly once per step; `select(keep)` drops finished rows |
| `_Hooks` | forward hooks: steering at \(l^\ast\) (`layer`), attention / MLP output norms (`fhg`); `remove()` is always called |
| `_Recorder` | per-problem float64 counters on the GPU, indexed by the *original* batch row so that finished rows keep their values |
| `_Decision` | everything one gate decision computed; missing fields read as `None`; `select(keep)` filters every per-row tensor |
| `IDEAS` | `("layer", "sem", "ltr", "fhg", "dynk")` |

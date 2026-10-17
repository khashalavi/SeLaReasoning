"""
Evaluate ONE shard of one SeLaR-X run -- the GPU job Sela_beat.sh submits.

A run is (method + hyper-parameters, model, dataset, #problems, seed); its
problems are split into contiguous shards so that each shard fits into one
Slurm job. Everything is written below --run_dir (on Lustre):

    <run_dir>/config.json                      run configuration (first job writes it)
    <run_dir>/shards/shard_<i>of<n>.partial.jsonl   one line per finished problem
    <run_dir>/shards/shard_<i>of<n>.json       final shard file (written once complete)

Resumable: every finished batch is appended to the .partial.jsonl file and a
restarted job only generates what is missing there. A batch that runs out of
GPU memory is retried as two half-size batches. With --time_left (Slurm's
remaining wall clock, passed by Sela_beat.sh) the job stops before a batch
that would not finish in time, instead of being killed half way through it.

Prompting, answer extraction, grading and token counting follow scripts/run.py
so that numbers stay comparable with the authors' results/ folder.
"""

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
from datasets import load_from_disk  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from src.generation_utils import (  # noqa: E402
    set_seed, get_math_symbols_ids, generate_cot, generate_selar, generate_swir,
)
from src.grader import answer_match  # noqa: E402
from src.selarx import IDEAS, SelarXConfig, generate_selarx  # noqa: E402

# dataset name -> (folder under datasets/, question column, answer column)
DATASETS = {
    "gsm8k": ("gsm8k_test", "question", "answer"),
    "math500": ("math_500_test", "problem", "answer"),
    "aime_2024": ("aime_2024_train", "problem", "answer"),
    "aime_2025": ("aime_2025", "problem", "answer"),
    "gpqa_diamond": ("gpqa_diamond_mc_test", "problem", "solution"),
}
UPSTREAM_METHODS = ("cot_upstream", "selar_upstream", "swir")
KNOBS = [f for f in SelarXConfig.__dataclass_fields__.values()
         if f.name not in IDEAS + ("latent",)]
PROMPT_SUFFIX = "\nPlease reason step by step, and make sure put your final answer within \\boxed{}."


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True,
                   help="cot | cot_greedy | selar | an idea (layer sem ltr fhg dynk) | ideas joined "
                        "by '+' | cot_upstream | selar_upstream | swir")
    p.add_argument("--model_name", required=True, help="HF id (names, chat template quirks)")
    p.add_argument("--model_path", default=None, help="local weights dir (default: --model_name)")
    p.add_argument("--dataset_name", required=True, choices=sorted(DATASETS))
    p.add_argument("--datasets_dir", default=os.path.join(REPO_ROOT, "datasets"))
    p.add_argument("--n_samples", type=int, default=None, help="first N problems (default: all)")
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=32768)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--min_p", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--device", default="cuda")
    p.add_argument("--time_left", default=None, help="Slurm time left ([d-]hh:mm:ss)")
    p.add_argument("--no_logits_to_keep", action="store_true",
                   help="compute prefill logits for every position (bit-exact upstream prefill)")
    # SwiR (upstream generate_swir)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--max_switch_count", type=int, default=None)
    # SeLaR-X knobs, all optional (None = SelarXConfig default)
    for f in KNOBS:
        if f.type in (bool, "bool"):
            p.add_argument(f"--{f.name}", default=None, action=argparse.BooleanOptionalAction)
        else:
            caster = {"int": int, "float": float, "str": str}.get(f.type, f.type)
            p.add_argument(f"--{f.name}", type=caster, default=None)
    return p.parse_args()


def selarx_config(args):
    knobs = {f.name: getattr(args, f.name) for f in KNOBS}
    if args.method in ("cot", "cot_greedy", "cot_upstream", "swir"):
        return SelarXConfig.for_method("cot", **knobs)
    if args.method == "selar_upstream":
        return SelarXConfig.for_method("selar", **knobs)
    return SelarXConfig.for_method(args.method, **knobs)


def parse_time_left(text):
    """Slurm %L ('1-02:03:04', '02:03:04', '03:04') -> seconds, None if unknown."""
    if not text or not text[0].isdigit():
        return None
    days, _, clock = text.rpartition("-")
    parts = [int(x) for x in clock.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return int(days or 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def git_version():
    """'<commit>' or '<commit>+dirty'. Read-only: GIT_OPTIONAL_LOCKS=0 stops
    `git status` from rewriting .git/index in the (home-directory) repo."""
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    try:
        head = subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "--short", "HEAD"], env=env,
                              capture_output=True, text=True, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "-C", REPO_ROOT, "status", "--porcelain", "--untracked-files=no"],
                               env=env, capture_output=True, text=True, timeout=20).stdout.strip()
        return (head + ("+dirty" if dirty else "")) or None
    except Exception:
        return None


def compact(value):
    """Counters come back as floats; keep integers as ints in the json."""
    return int(value) if float(value).is_integer() else round(value, 6)


def write_json_atomic(path, obj, indent=None):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_partial(path):
    """index -> record for every complete line (a killed job may leave half a line)."""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                done[rec["index"]] = rec
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return done


def load_problems(args):
    folder, q_col, a_col = DATASETS[args.dataset_name]
    ds = load_from_disk(os.path.join(args.datasets_dir, folder))
    n_total = len(ds) if args.n_samples is None else min(args.n_samples, len(ds))
    per_shard = math.ceil(n_total / args.num_shards)
    start = args.shard_index * per_shard
    end = min(start + per_shard, n_total)
    rows = ds.select(range(start, end)) if end > start else []
    questions = list(rows[q_col]) if end > start else []
    answers = list(rows[a_col]) if end > start else []
    if args.dataset_name == "gsm8k":
        golds = [str(a).split("####")[-1].strip() for a in answers]
    else:
        golds = [str(a).strip() for a in answers]
    return n_total, start, end, {start + i: (q, g) for i, (q, g) in enumerate(zip(questions, golds))}


class Evaluator:
    def __init__(self, args, cfg):
        self.args, self.cfg = args, cfg
        path = args.model_path or args.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        device_map = {"": 0} if args.device == "cuda" else {"": args.device}
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto", device_map=device_map)
        self.model.eval()
        self.device = self.model.device
        self.math_ids = torch.tensor(sorted(get_math_symbols_ids(self.tokenizer)), device=self.device)
        eos = {self.tokenizer.eos_token_id}
        gen_eos = getattr(self.model.generation_config, "eos_token_id", None)
        eos.update(gen_eos if isinstance(gen_eos, (list, tuple)) else [gen_eos])
        self.eos_ids = sorted(t for t in eos if t is not None)
        think_end = self.tokenizer.convert_tokens_to_ids("</think>")
        if think_end is None or think_end == self.tokenizer.unk_token_id:
            think_end = 128014 if "Llama" in args.model_name else 151668   # scripts/run.py
        self.think_end = think_end

    def generate(self, model_inputs):
        a, cfg = self.args, self.cfg
        gen_kwargs = dict(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k, min_p=a.min_p,
                          max_new_tokens=a.max_new_tokens, do_sample=a.method != "cot_greedy")
        if a.method == "cot_upstream":
            return generate_cot(self.model, self.tokenizer, **model_inputs, **gen_kwargs), None
        if a.method == "selar_upstream":
            return generate_selar(self.model, self.tokenizer, **model_inputs, selar_topk=cfg.topk,
                                  entropy_threshold=cfg.tau, math_ids_tensor=self.math_ids,
                                  **gen_kwargs), None
        if a.method == "swir":
            conv = "</think>" if "Qwen" in a.model_name else "\n\n</think>\n\n"
            return generate_swir(self.model, self.tokenizer, **model_inputs, alpha_0=a.alpha,
                                 max_switch_count=a.max_switch_count, math_ids_tensor=self.math_ids,
                                 convergence_words=conv, **gen_kwargs), None
        return generate_selarx(self.model, self.tokenizer, model_inputs["input_ids"],
                               model_inputs["attention_mask"], cfg, math_ids_tensor=self.math_ids,
                               eos_token_ids=self.eos_ids, logits_to_keep=not a.no_logits_to_keep,
                               **gen_kwargs)

    def run_batch(self, indices, problems, seed):
        tok = self.tokenizer
        questions = [problems[i][0] for i in indices]
        texts = [tok.apply_chat_template([{"role": "user", "content": q + PROMPT_SUFFIX}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=True)
                 for q in questions]
        model_inputs = tok(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        set_seed(seed)
        t0 = time.time()
        with torch.no_grad():
            generated_ids, stats = self.generate(model_inputs)
        seconds = time.time() - t0
        prompt_len = model_inputs["input_ids"].shape[1]
        records = []
        for bi, idx in enumerate(indices):
            output_ids = generated_ids[bi][prompt_len:].tolist()
            end = next((j for j, t in enumerate(output_ids) if t in self.eos_ids), None)
            n_gen = end + 1 if end is not None else len(output_ids)
            pred = tok.decode(output_ids, skip_special_tokens=True)
            try:
                index = len(output_ids) - output_ids[::-1].index(self.think_end)
            except ValueError:
                index = 0
            thinking = tok.decode(output_ids[:index], skip_special_tokens=True).strip()
            answer_content = pred[len(thinking):]
            gold = problems[idx][1]
            try:
                correct, prediction = answer_match(self.args.dataset_name, answer_content, gold)
            except Exception as e:                       # grader must never kill a batch
                correct, prediction = False, f"grader error: {type(e).__name__}: {e}"
            rec = {
                "index": idx, "question": questions[bi], "gold": gold, "prediction": prediction,
                "correct": bool(correct), "thinking": thinking, "answer_content": answer_content,
                "tokens": len(tok.encode(pred, add_special_tokens=False)),
                "gen_tokens": n_gen, "hit_max": end is None,
                "batch": {"size": len(indices), "seconds": round(seconds, 2), "seed": seed},
            }
            if stats is not None:
                rec["stats"] = {k: compact(v[bi]) for k, v in stats.items() if k != "h_hist"}
                rec["stats"]["h_hist"] = [int(x) for x in stats["h_hist"][bi]]
            records.append(rec)
        return records

    def run_resilient(self, indices, problems, seed_base):
        """run_batch, halving the batch on CUDA OOM (outside the except block, so
        the failed batch's KV cache is really released before the retry)."""
        oom = False
        try:
            return self.run_batch(indices, problems, batch_seed(seed_base, indices[0]))
        except torch.cuda.OutOfMemoryError:
            oom = True
        if oom:
            gc.collect()
            torch.cuda.empty_cache()
            if len(indices) == 1:
                print(f"OOM even at batch size 1 for problem {indices[0]} -- recorded as wrong", flush=True)
                q, g = problems[indices[0]]
                return [{"index": indices[0], "question": q, "gold": g, "prediction": "OOM",
                         "correct": False, "thinking": "", "answer_content": "", "tokens": 0,
                         "gen_tokens": 0, "hit_max": False, "error": "oom",
                         "batch": {"size": 1, "seconds": 0.0, "seed": None}}]
            half = len(indices) // 2
            print(f"OOM at batch size {len(indices)} -> retrying as {half} + {len(indices) - half}", flush=True)
            return (self.run_resilient(indices[:half], problems, seed_base)
                    + self.run_resilient(indices[half:], problems, seed_base))


def batch_seed(seed, first_index):
    """Deterministic per-batch seed, so a resumed shard draws the same numbers."""
    return (seed * 1000003 + first_index * 7919) % (2 ** 31)


def main():
    args = parse_args()
    t_start = time.time()
    cfg = selarx_config(args)
    n_total, start, end, problems = load_problems(args)

    shard_dir = os.path.join(args.run_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    stem = os.path.join(shard_dir, f"shard_{args.shard_index:03d}of{args.num_shards:03d}")
    final_path, partial_path = stem + ".json", stem + ".partial.jsonl"
    if os.path.exists(final_path):
        print(f"shard already complete: {final_path}")
        return

    run_info = {
        "method": args.method, "model": args.model_name, "dataset": args.dataset_name,
        "n_samples": n_total, "seed": args.seed, "num_shards": args.num_shards,
        "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size,
        "decoding": {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                     "min_p": args.min_p, "do_sample": args.method != "cot_greedy"},
        "selarx": cfg.to_dict() if args.method not in ("cot_upstream", "swir") else None,
        "swir": {"alpha": args.alpha, "max_switch_count": args.max_switch_count} if args.method == "swir" else None,
        "logits_to_keep": not args.no_logits_to_keep, "code_version": git_version(),
    }
    config_path = os.path.join(args.run_dir, "config.json")
    if not os.path.exists(config_path):
        write_json_atomic(config_path, run_info, indent=2)

    done = load_partial(partial_path)
    todo = [i for i in range(start, end) if i not in done]
    print(f"run_dir={args.run_dir}\nshard {args.shard_index + 1}/{args.num_shards}: problems "
          f"[{start}, {end}) of {n_total}; {len(done)} already done, {len(todo)} to go", flush=True)

    budget = parse_time_left(args.time_left)
    deadline = t_start + budget - 300 if budget else None
    if todo:
        ev = Evaluator(args, cfg)
        print(f"model loaded in {time.time() - t_start:.0f}s on {torch.cuda.get_device_name(0) if args.device == 'cuda' else args.device}", flush=True)
        slowest = 0.0
        for b0 in range(0, len(todo), args.batch_size):
            batch = todo[b0:b0 + args.batch_size]
            if deadline and slowest and time.time() + 1.25 * slowest > deadline:
                print(f"TIME BUDGET: {deadline - time.time():.0f}s left < 1.25 x slowest batch "
                      f"({slowest:.0f}s) -- stopping; the next job resumes here.", flush=True)
                return
            t0 = time.time()
            records = ev.run_resilient(batch, problems, args.seed)
            slowest = max(slowest, time.time() - t0)
            with open(partial_path, "a", encoding="utf-8") as f:
                f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
                f.flush()
                os.fsync(f.fileno())
            for r in records:
                done[r["index"]] = r
            n_ok = sum(r["correct"] for r in done.values())
            print(f"[{time.strftime('%H:%M:%S')}] batch {b0 // args.batch_size + 1}/"
                  f"{math.ceil(len(todo) / args.batch_size)}: {len(records)} problems in {time.time() - t0:.0f}s, "
                  f"shard accuracy so far {n_ok}/{len(done)} = {n_ok / len(done):.2%}", flush=True)
            del records
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()

    details = [done[i] for i in sorted(done) if start <= i < end]
    missing = sorted(set(range(start, end)) - {d["index"] for d in details})
    if missing:
        raise RuntimeError(f"shard incomplete, missing problems {missing[:10]}...")
    shard = {
        "format": "selarx-shard-v1", "shard_index": args.shard_index, "num_shards": args.num_shards,
        "start": start, "end": end, "total": len(details),
        "correct": sum(d["correct"] for d in details), "run": run_info,
        "env": {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
                "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else args.device,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
        "details": details,
    }
    write_json_atomic(final_path, shard)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    print(f"shard complete: {shard['correct']}/{shard['total']} correct -> {final_path}", flush=True)


if __name__ == "__main__":
    main()

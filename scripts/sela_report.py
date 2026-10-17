"""
The final SeLaR-X report -- standard library only, so it runs with the system
python3 anywhere (login node, or the CPU-only Sela_beat job).

    python3 scripts/sela_report.py --root $SELA_ROOT --repo . \
        --out result/selar_report.md --csv $SELA_ROOT/reports/all_runs.csv

Reads <root>/plan.tsv (what Sela_beat.sh runs), every summary.json below
<root>/runs, and the authors' published results in <repo>/results/ (as a
reference), and writes ONE Markdown report. Called by reporting.sh.
"""

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import time
from collections import OrderedDict, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sela_tools import merge_run, needs_merge, partial_counts, shard_counts, shard_path  # noqa: E402

DATASETS = ["gsm8k", "math500", "aime_2024", "aime_2025", "gpqa_diamond"]
DLABEL = {"gsm8k": "GSM8K", "math500": "MATH500", "aime_2024": "AIME24",
          "aime_2025": "AIME25", "gpqa_diamond": "GPQA"}
IDEAS = ["layer", "sem", "ltr", "fhg", "dynk"]
# improvement.md section 7, in its order
ROWS = OrderedDict([
    ("cot", "Baseline 1: Standard CoT"),
    ("selar", "Baseline 2: Vanilla SeLaR"),
    ("layer", "Variant 1: Layer-Wise Steering (Hidden-State)"),
    ("sem", "Variant 2: Semantic Adaptive Gating (Sem-SeLaR)"),
    ("ltr", "Variant 3: Multi-Step Rollouts (LTR-SeLaR)"),
    ("fhg", "Variant 4: Functional Head Gating (Fact/Logic)"),
    ("dynk", "Variant 5: Nucleus Dynamic-K (DynK-SeLaR)"),
])
EXTRA = OrderedDict([
    ("cot_greedy", "Extra: CoT (greedy)"),
    ("cot_upstream", "Extra: CoT, upstream generate_cot"),
    ("selar_upstream", "Extra: SeLaR, upstream generate_selar (batch quirk)"),
    ("swir", "Extra: SwiR"),
])
# improvement.md section 8
FRAMING = {
    "layer": ("Steering LLM Thoughts from Within: Training-Free Intermediate Latent Reasoning "
              "in Transformer Hidden Spaces",
              "Input embedding modifications are lossy; hidden-state steering inside "
              "mid-transformer layers unlocks direct control over latent reasoning chains."),
    "sem": ("Calibrated Latent Exploration: Semantic Entropy and Adaptive Thresholding for "
            "Robust Training-Free LLM Reasoning",
            "Standard entropy over-triggers on token surface-forms; semantic clustering "
            "stabilizes latent reasoning across both calibrated and uncalibrated LLM families."),
    "ltr": ("Latent Trajectory Rollouts: Continuous Multi-Step Exploration for Complex "
            "Logical Deduction",
            "Single-step soft intervention is a local smoother; multi-step rollouts enable "
            "continuous lookahead search in continuous space without training."),
    "fhg": ("Knowing When to Speculate: Task-Adaptive Latent Reasoning via Attention-FFN "
            "Activation Routing",
            "Solves the core trade-off between procedural logic exploration and rigid factual recall."),
}
UPSTREAM_DIRS = {"COT": "cot", "SeLaR": "selar", "Swir": "swir"}
LATENT = set(IDEAS) | {"selar"}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, digits=2, scale=1.0, sign=False, suffix=""):
    if x is None:
        return "–"
    return ("%+.*f" if sign else "%.*f") % (digits, x * scale) + suffix


def fmt_int(x):
    return "–" if x is None else "{:,}".format(int(round(x)))


def table(headers, rows, align=None):
    align = align or ["l"] + ["r"] * (len(headers) - 1)
    sep = {"l": ":---", "r": "---:", "c": ":---:"}
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep[a] for a in align) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |")
    return "\n".join(out)


def mcnemar(a, b):
    """Exact two-sided McNemar test on paired 0/1 strings (same problems, same order)."""
    n01 = sum(1 for x, y in zip(a, b) if x == "1" and y == "0")
    n10 = sum(1 for x, y in zip(a, b) if x == "0" and y == "1")
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def method_label(method):
    if method in ROWS:
        return ROWS[method]
    if method in EXTRA:
        return EXTRA[method]
    if "+" in method:
        return "Variant 6: Combined (%s)" % " + ".join(method.split("+"))
    return method


def method_order(methods):
    known = [m for m in ROWS if m in methods]
    combos = sorted(m for m in methods if "+" in m)
    extra = [m for m in EXTRA if m in methods]
    rest = sorted(m for m in methods if m not in ROWS and m not in EXTRA and "+" not in m)
    return known + combos + extra + rest


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_plan(root):
    rows, path = [], os.path.join(root, "plan.tsv")
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 10:
                continue
            rows.append({"phase": p[0], "method": p[1], "config": p[2], "knobs": p[3], "model": p[4],
                         "dataset": p[5], "n": int(p[6]), "seed": int(p[7]), "run_dir": p[8],
                         "num_shards": int(p[9]), "is_default": len(p) > 10 and p[10] == "1"})
    return rows


def run_record(run_dir, s):
    """Flat metrics of one merged run."""
    run = s.get("run") or {}
    act = s.get("activation") or {}
    ent = s.get("entropy") or {}
    ls = s.get("length_stats") or {}
    parts = run_dir.rstrip("/").split("/")
    return {
        "run_dir": run_dir, "model": run.get("model"), "dataset": run.get("dataset"),
        "method": run.get("method"), "config": parts[-3] if len(parts) >= 3 else "",
        "seed": run.get("seed"), "n": s.get("total"), "correct": s.get("correct"),
        "acc": s.get("accuracy"), "tokens": ls.get("avg_total_token_len"),
        "tc": ls.get("correct_avg_total_token_len"), "tw": ls.get("wrong_avg_total_token_len"),
        "tpca": s.get("tpca"), "hit_max": s.get("hit_max_rate"), "errors": s.get("errors"),
        "act": act.get("ratio"), "soft_in": act.get("soft_input_ratio"), "gate_open": act.get("gate_open_ratio"),
        "guarded": act.get("guarded_ratio"), "vetoed": act.get("vetoed_ratio"),
        "rollouts": act.get("rollouts_per_sample"), "roll_len": act.get("avg_rollout_len"),
        "avg_k": act.get("avg_k"), "clusters": act.get("avg_clusters"), "ratio": act.get("avg_ratio"),
        "thr": act.get("avg_dynamic_threshold"), "h_mean": ent.get("mean"), "h_median": ent.get("median"),
        "h_tok": ent.get("token_entropy_mean"), "h_ge": ent.get("share_ge_0.5"),
        "gpu_h": (s.get("gpu_seconds") or 0.0) / 3600.0, "pic": s.get("per_index_correct"),
        "first": s.get("first_index"), "max_new_tokens": run.get("max_new_tokens"),
        "code": ",".join(run.get("code_versions") or [str(run.get("code_version"))]),
    }


def scan_runs(runs_dir):
    found = {}
    for dirpath, dirnames, filenames in os.walk(runs_dir):
        if "summary.json" in filenames:
            try:
                with open(os.path.join(dirpath, "summary.json"), encoding="utf-8") as f:
                    found[dirpath] = run_record(dirpath, json.load(f))
            except (ValueError, OSError):
                continue
        dirnames[:] = [d for d in dirnames if d != "shards"]
    return found


def progress(entry):
    done = 0
    for i in range(entry["num_shards"]):
        path = shard_path(entry["run_dir"], i, entry["num_shards"])
        if os.path.exists(path):
            done += shard_counts(path)[0]
        elif os.path.exists(path[:-5] + ".partial.jsonl"):
            done += partial_counts(path[:-5] + ".partial.jsonl")[0]
    return done / entry["n"] if entry["n"] else 0.0


def load_upstream(repo, cache_path):
    """The authors' published runs (results/<COT|SeLaR|Swir>/<model>/*.json)."""
    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                cache = json.load(f)
        except ValueError:
            cache = {}
    out, changed = [], False
    base = os.path.join(repo, "results")
    for mdir, method in UPSTREAM_DIRS.items():
        for model in sorted(os.listdir(os.path.join(base, mdir))) if os.path.isdir(os.path.join(base, mdir)) else []:
            folder = os.path.join(base, mdir, model)
            for name in sorted(os.listdir(folder)):
                if not name.endswith(".json"):
                    continue
                path = os.path.join(folder, name)
                st = os.stat(path)
                key = "v2|%s|%d|%d" % (os.path.relpath(path, repo), st.st_size, int(st.st_mtime))
                if key not in cache:
                    with open(path, encoding="utf-8") as f:
                        d = json.load(f)
                    rest = name[len(model) + 1:]
                    dataset = next((ds for ds in sorted(DATASETS, key=len, reverse=True) if rest.startswith(ds)), None)
                    correct = sum(1 for x in d.get("details", []) if x.get("correct"))
                    total = len(d.get("details", [])) or d.get("total", 0)
                    ls = d.get("length_stats", {})
                    acc = correct / total if total else None
                    # "selar_k3_t0.6_32768_merged.json" -> "k3_t0.6"
                    settings = re.sub(r"^(cot|selar|swir)_?", "", rest[len(dataset or "") + 1:-5])
                    settings = re.sub(r"_?\d+_(merged|rank\d+)$", "", settings)
                    cache[key] = {"model": model, "method": method, "dataset": dataset,
                                  "settings": settings, "acc": acc,
                                  "correct": correct, "total": total,
                                  "tokens": ls.get("avg_total_token_len"),
                                  "tpca": ls.get("avg_total_token_len") / acc if acc else None}
                    changed = True
                out.append(cache[key])
    if changed and cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            live = {k: v for k, v in cache.items() if k.startswith("v2|")}
            with open(cache_path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(live, f)
            os.replace(cache_path + ".tmp", cache_path)
        except OSError:
            pass
    return out


# ---------------------------------------------------------------------------
# aggregation over seeds
# ---------------------------------------------------------------------------

class Cell:
    """All seeds of one (model, method, config set, dataset) in one phase."""

    def __init__(self, entries, runs):
        self.entries = entries
        self.done = [runs[e["run_dir"]] for e in entries if e["run_dir"] in runs]
        self.planned = len(entries)
        self.complete = self.planned > 0 and len(self.done) == self.planned
        accs = [r["acc"] for r in self.done]
        self.acc = mean(accs)
        self.std = statistics.stdev(accs) if len(accs) > 1 else None
        self.tpca = mean(r["tpca"] for r in self.done)
        self.act = mean(r["act"] for r in self.done)
        self.progress = mean(progress(e) for e in entries) if not self.complete and entries else None

    def by_seed(self):
        return {r["seed"]: r for r in self.done}

    def text(self, marks=""):
        if not self.entries:
            return "–"
        if not self.done:
            return "… %.0f%%" % (100 * (self.progress or 0.0))
        s = fmt(self.acc, scale=100)
        if self.std is not None:
            s += "±%.2f" % (100 * self.std)
        if not self.complete:
            s += " (%d/%d seeds)" % (len(self.done), self.planned)
        return s + marks


def paired_p(a, b):
    """McNemar over all seeds both cells have (pairs pooled)."""
    if a is None or b is None:
        return None
    sa, sb = a.by_seed(), b.by_seed()
    xa = xb = ""
    for seed in sorted(set(sa) & set(sb)):
        ra, rb = sa[seed], sb[seed]
        if ra["pic"] and rb["pic"] and len(ra["pic"]) == len(rb["pic"]) and ra["first"] == rb["first"]:
            xa += ra["pic"]
            xb += rb["pic"]
    return mcnemar(xa, xb) if xa else None


def cells_for(plan, runs, phase, model):
    cells = defaultdict(dict)                           # method -> dataset -> Cell
    groups = defaultdict(list)
    for e in plan:
        if e["phase"] == phase and e["model"] == model:
            groups[(e["method"], e["dataset"])].append(e)
    for (method, dataset), entries in groups.items():
        cells[method][dataset] = Cell(entries, runs)
    return cells


# ---------------------------------------------------------------------------
# report sections
# ---------------------------------------------------------------------------

def section_master(plan, runs):
    out = []
    models = list(OrderedDict.fromkeys(e["model"] for e in plan if e["phase"] == "main"))
    ranking = {}
    for model in models:
        cells = cells_for(plan, runs, "main", model)
        datasets = [d for d in DATASETS if any(d in c for c in cells.values())]
        cot, sel = cells.get("cot", {}), cells.get("selar", {})
        rows = []
        for method in method_order(cells):
            c = cells[method]
            row, deltas, tp, acts, avg_parts = [method_label(method)], [], [], [], []
            for d in datasets:
                cell = c.get(d)
                if cell is None:
                    row.append("–")
                    continue
                marks = ""
                if method != "cot" and cell.complete:
                    p = paired_p(cell, cot.get(d))
                    marks += "*" if p is not None and p < 0.05 else ""
                if method not in ("cot", "selar") and cell.complete:
                    p = paired_p(cell, sel.get(d))
                    marks += "†" if p is not None and p < 0.05 else ""
                row.append(cell.text(marks))
                if cell.complete:
                    avg_parts.append(cell.acc)
                    if d in sel and sel[d].complete:
                        deltas.append(cell.acc - sel[d].acc)
                    if d in cot and cot[d].complete and cot[d].tpca and cell.tpca:
                        tp.append((cot[d].tpca - cell.tpca) / cot[d].tpca)
                    if cell.act is not None:
                        acts.append(cell.act)
            avg = mean(avg_parts)
            avg_txt = fmt(avg, scale=100)
            if avg is not None and len(avg_parts) < len(datasets):
                avg_txt += " (%d/%d)" % (len(avg_parts), len(datasets))
            row += [avg_txt,
                    fmt(mean(deltas), scale=100, sign=True) if method != "selar" else "0.00",
                    fmt(mean(tp), digits=1, scale=100, sign=True, suffix="%") if method != "cot" else "0.0%",
                    fmt(mean(acts), digits=1, scale=100, suffix="%") if method not in ("cot", "cot_greedy", "cot_upstream") else "0.0%"]
            rows.append(row)
            if method in IDEAS or "+" in method:
                ranking.setdefault(method, []).append({
                    "model": model, "delta": mean(deltas), "n": len(deltas), "tpca": mean(tp),
                    "better": sum(1 for d in datasets if d in c and c[d].complete and d in sel and sel[d].complete
                                  and c[d].acc > sel[d].acc and (paired_p(c[d], sel[d]) or 1) < 0.05),
                    "worse": sum(1 for d in datasets if d in c and c[d].complete and d in sel and sel[d].complete
                                 and c[d].acc < sel[d].acc and (paired_p(c[d], sel[d]) or 1) < 0.05)})
        headers = ["Method"] + ["%s Acc (%%)" % DLABEL[d] for d in datasets] + \
                  ["Avg Acc (%)", "Δ Avg vs SeLaR", "TPCA Improvement vs CoT", "Activation Ratio"]
        out.append("### %s\n" % model)
        out.append(table(headers, rows))
        out.append("")
    return out, ranking


def section_details(plan, runs):
    out = []
    models = list(OrderedDict.fromkeys(e["model"] for e in plan if e["phase"] == "main"))
    for model in models:
        cells = cells_for(plan, runs, "main", model)
        for d in DATASETS:
            present = [m for m in method_order(cells) if d in cells[m]]
            if not present:
                continue
            cot, sel = cells.get("cot", {}).get(d), cells.get("selar", {}).get(d)
            rows = []
            for m in present:
                c = cells[m][d]
                if not c.done:
                    rows.append([method_label(m), c.text()] + ["–"] * 10 + [c.entries[0]["config"]])
                    continue
                r = c.done[0] if len(c.done) == 1 else None
                p_cot = paired_p(c, cot) if m != "cot" else None
                p_sel = paired_p(c, sel) if m not in ("cot", "selar") else None
                rows.append([
                    method_label(m), c.text(),
                    "%d/%d" % (sum(x["correct"] for x in c.done), sum(x["n"] for x in c.done)),
                    fmt_int(mean(x["tokens"] for x in c.done)), fmt_int(mean(x["tc"] for x in c.done)),
                    fmt_int(mean(x["tw"] for x in c.done)), fmt_int(c.tpca),
                    fmt((cot.tpca - c.tpca) / cot.tpca if (cot and cot.tpca and c.tpca and m != "cot") else None,
                        digits=1, scale=100, sign=True, suffix="%"),
                    fmt(mean(x["hit_max"] for x in c.done), digits=1, scale=100, suffix="%"),
                    fmt(c.act, digits=1, scale=100, suffix="%"),
                    fmt(p_cot, digits=3), fmt(p_sel, digits=3),
                    r["config"] if r else c.entries[0]["config"],
                ])
            out.append("#### %s — %s\n" % (model, DLABEL[d]))
            out.append(table(["Method", "Acc (%)", "Correct", "Avg tokens", "T_c", "T_w", "TPCA",
                              "TPCA vs CoT", "Truncated", "Activation", "p vs CoT", "p vs SeLaR", "Config"],
                             rows, ["l"] + ["r"] * 11 + ["l"]))
            out.append("")
    return out


def section_diagnostics(plan, runs):
    out = []
    models = list(OrderedDict.fromkeys(e["model"] for e in plan if e["phase"] == "main"))
    for model in models:
        cells = cells_for(plan, runs, "main", model)
        rows = []
        for m in method_order(cells):
            for d in DATASETS:
                c = cells[m].get(d)
                if not c or not c.done or c.done[0]["h_mean"] is None:
                    continue
                g = lambda k: mean(x[k] for x in c.done)  # noqa: E731
                uses = lambda idea: idea in m.split("+")   # noqa: E731
                latent = m in LATENT or "+" in m
                rows.append([method_label(m).split(":")[0] + ("" if m in ROWS or "+" in m else " (%s)" % m), DLABEL[d],
                             fmt(g("act"), 1, 100, suffix="%"),
                             fmt(g("soft_in"), 1, 100, suffix="%") if latent and not uses("layer") else "–",
                             fmt(g("gate_open"), 1, 100, suffix="%") if latent else "–",
                             fmt(g("guarded"), 1, 100, suffix="%") if latent else "–",
                             fmt(g("vetoed"), 1, 100, suffix="%") if uses("fhg") else "–",
                             fmt(g("rollouts"), 1) if uses("ltr") else "–",
                             fmt(g("roll_len"), 2) if uses("ltr") else "–",
                             fmt(g("avg_k"), 2), fmt(g("clusters"), 2), fmt(g("ratio"), 3), fmt(g("thr"), 3),
                             fmt(g("h_mean"), 3), fmt(g("h_median"), 3), fmt(g("h_ge"), 1, 100, suffix="%")])
        if rows:
            out.append("### %s\n" % model)
            out.append(table(["Method", "Data", "Latent steps", "Soft inputs", "Gate open", "Math-guarded",
                              "FHG vetoed", "Rollouts/sample", "Rollout len", "avg k_t", "avg clusters",
                              "avg R_act", "avg τ_t", "H mean", "H median", "P(H≥0.5)"], rows))
            out.append("")
    return out


BASE_GATE = ("SELAR_TOPK", "SELAR_TAU", "CONTRASTIVE_W", "SOFT_MASK")


def parse_knobs(knobs):
    return OrderedDict(kv.split("=", 1) for kv in knobs.split() if "=" in kv) if knobs != "-" else OrderedDict()


def swept_knobs(method, knobs):
    """The knobs that identify a sweep config. For the variants the base gate is
    not swept -- it follows SELAR_PRESET and may differ per dataset."""
    kv = parse_knobs(knobs)
    if method not in ("selar", "selar_upstream"):
        kv = OrderedDict((k, v) for k, v in kv.items() if k not in BASE_GATE)
    return " ".join("%s=%s" % x for x in kv.items())


def section_sweeps(plan, runs):
    out = []
    groups = defaultdict(list)
    for e in plan:
        if e["phase"] == "sweep":
            groups[(e["method"], e["model"])].append(e)
    for (method, model), entries in sorted(groups.items()):
        datasets = [d for d in DATASETS if any(e["dataset"] == d for e in entries)]
        configs = OrderedDict()
        for e in entries:
            configs.setdefault(swept_knobs(method, e["knobs"]), []).append(e)
        values = defaultdict(set)
        for knobs in configs:
            for k, v in parse_knobs(knobs).items():
                values[k].add(v)
        varied = [k for k in values if len(values[k]) > 1]
        default_kv = next((parse_knobs(k) for k, es in configs.items() if any(e["is_default"] for e in es)), None)
        default_avg = None
        rows = []
        for knobs, es in configs.items():
            kv = parse_knobs(knobs)
            if default_kv is not None:            # show only what differs from the default
                diff = [k for k in varied if kv.get(k) != default_kv.get(k)]
                shown = diff or varied
            else:
                shown = varied
            per_d, accs, acts = [], [], []
            for d in datasets:
                c = Cell([e for e in es if e["dataset"] == d], runs)
                per_d.append(c.text())
                if c.complete:
                    accs.append(c.acc)
                    if c.act is not None:
                        acts.append(c.act)
            avg = mean(accs) if len(accs) == len(datasets) else None
            is_def = any(e["is_default"] for e in es)
            if is_def:
                default_avg = avg
            rows.append([avg, is_def, [(k, kv.get(k, "")) for k in shown], per_d, mean(acts)])
        rows.sort(key=lambda r: (r[0] is None, -(r[0] or 0)))
        best = rows[0][0] if rows and rows[0][0] is not None else None
        table_rows = []
        for avg, is_def, kvs, per_d, act in rows:
            name = ("◆ " if is_def else "") + (", ".join("%s=%s" % kv for kv in kvs) or "default")
            avg_txt = fmt(avg, scale=100)
            if avg is not None and avg == best:
                avg_txt = "**%s**" % avg_txt
            delta = fmt(avg - default_avg, scale=100, sign=True) if (avg is not None and default_avg is not None) else "–"
            table_rows.append([name] + per_d + [avg_txt, delta, fmt(act, 1, 100, suffix="%")])
        n_txt = ", ".join("%s n=%d" % (DLABEL[d], next(e["n"] for e in entries if e["dataset"] == d)) for d in datasets)
        out.append("### %s — %s (%s)\n" % (method_label(method), model, n_txt))
        out.append(table(["Config (◆ = default)"] + ["%s Acc (%%)" % DLABEL[d] for d in datasets] +
                         ["Avg", "Δ vs default", "Activation"], table_rows))
        out.append("")
    return out


def section_decision(ranking):
    out = []
    agg = []
    for method, items in ranking.items():
        deltas = [i["delta"] for i in items if i["delta"] is not None]
        if not deltas:
            continue
        agg.append((mean(deltas), method, items))
    if not agg:
        return ["No variant has complete results next to vanilla SeLaR yet."]
    agg.sort(key=lambda x: -x[0])
    rows = []
    for rank, (delta, method, items) in enumerate(agg, 1):
        rows.append([rank, method_label(method), fmt(delta, scale=100, sign=True),
                     fmt(mean(i["tpca"] for i in items), 1, 100, sign=True, suffix="%"),
                     sum(i["better"] for i in items), sum(i["worse"] for i in items),
                     ", ".join("%s (%d datasets)" % (i["model"].split("/")[-1], i["n"]) for i in items)])
    out.append(table(["Rank", "Variant", "Δ Avg Acc vs SeLaR", "TPCA vs CoT", "Datasets sig. better",
                      "Datasets sig. worse", "Basis"], rows, ["r", "l", "r", "r", "r", "r", "l"]))
    out.append("")
    singles = [(d, m) for d, m, _ in agg if m in IDEAS]
    positive = [m for d, m in singles if d > 0]
    if len(positive) >= 2:
        out.append("**Variant 6 suggestion** (top-2 single ideas that beat vanilla SeLaR): set "
                   "`COMBOS=(\"%s\")` in `Sela_beat.sh`." % "+".join(positive[:2]))
    elif positive:
        out.append("Only **%s** beats vanilla SeLaR on average so far — a top-2 combination "
                   "(Variant 6) has no second partner yet." % positive[0])
    else:
        out.append("No single idea beats vanilla SeLaR on average yet.")
    if singles and singles[0][0] > 0:
        winner = singles[0][1]
        out.append("")
        if winner in FRAMING:
            title, pitch = FRAMING[winner]
            out.append("**Paper framing for the current leader (%s, improvement.md §8):** *\"%s\"* — %s"
                       % (winner, title, pitch))
        else:
            out.append("The current leader (%s) has no framing in improvement.md §8 yet." % winner)
    out.append("")
    out.append("Ranking by the mean accuracy difference to vanilla SeLaR over the datasets where both "
               "runs are complete (averaged over models). \"Sig.\" = exact McNemar test, p < 0.05.")
    return out


def section_reference(upstream, plan, runs):
    out = []
    if not upstream:
        return ["No authors' result files found in results/."]
    by_model = defaultdict(lambda: defaultdict(dict))
    for u in upstream:
        if u["dataset"]:
            by_model[u["model"]][u["method"]][u["dataset"]] = u
    ours = {}
    for model in OrderedDict.fromkeys(e["model"] for e in plan if e["phase"] == "main"):
        ours[model.split("/")[-1]] = cells_for(plan, runs, "main", model)
    for model in sorted(by_model):
        rows = []
        for method in ("cot", "swir", "selar"):
            if method not in by_model[model]:
                continue
            m = by_model[model][method]
            accs = [m[d]["acc"] for d in DATASETS if d in m]
            rows.append(["%s (published)" % method_label(method).split(": ")[-1]] +
                        [fmt(m[d]["acc"], scale=100) if d in m else "–" for d in DATASETS] +
                        [fmt(mean(accs), scale=100), ", ".join("%s: %s" % (DLABEL[d], m[d]["settings"])
                                                               for d in DATASETS if d in m and m[d]["settings"])])
            cells = ours.get(model, {}).get(method)
            if cells:
                row = ["%s (this pipeline)" % method_label(method).split(": ")[-1]]
                accs_ours = []
                for d in DATASETS:
                    c = cells.get(d)
                    if c and c.complete:
                        accs_ours.append(c.acc)
                        delta = " (%+.2f)" % (100 * (c.acc - m[d]["acc"])) if d in m else ""
                        row.append(fmt(c.acc, scale=100) + delta)
                    else:
                        row.append(c.text() if c else "–")
                row += [fmt(mean(accs_ours), scale=100), "Δ to published in brackets"]
                rows.append(row)
        out.append("### %s\n" % model)
        out.append(table(["Method"] + [DLABEL[d] for d in DATASETS] + ["Avg", "Settings / note"], rows,
                         ["l"] + ["r"] * 6 + ["l"]))
        out.append("")
    out.append("Accuracies recomputed from the per-problem `correct` flags of the files in `results/` "
               "(single runs; the published avg uses all five datasets).")
    return out


def section_coverage(plan, runs, all_runs):
    unique = OrderedDict()
    for e in plan:
        unique.setdefault(e["run_dir"], e)
    merged = [e for e in unique.values() if e["run_dir"] in runs]
    pending = [e for e in unique.values() if e["run_dir"] not in runs]
    started = [(progress(e), e) for e in pending]
    in_prog = sorted([x for x in started if x[0] > 0], key=lambda x: (-x[0], x[1]["config"], x[1]["seed"]))
    gpu_h = sum(r["gpu_h"] for r in all_runs.values())
    lines = ["- planned runs: **%d** (%s) — merged **%d**, in progress **%d**, not started **%d**"
             % (len(unique), ", ".join("%s %d" % (p, sum(1 for e in unique.values() if e["phase"] == p))
                                     for p in ("main", "sweep") if any(e["phase"] == p for e in unique.values())),
                len(merged), len(in_prog), len(pending) - len(in_prog)),
             "- runs on disk: **%d** (%d not in the current plan, listed in the CSV only)"
             % (len(all_runs), len(set(all_runs) - set(unique))),
             "- GPU time of merged runs: **%.1f h** (generation only, from the per-batch timers)" % gpu_h]
    if in_prog:
        lines.append("")
        lines.append(table(["In progress", "Model", "Dataset", "Seed", "Done"],
                           [[e["config"], e["model"].split("/")[-1], DLABEL[e["dataset"]], e["seed"],
                             "%.0f%%" % (100 * p)] for p, e in in_prog[:20]]))
    return lines


def write_csv(path, all_runs):
    cols = ["model", "dataset", "n", "method", "config", "seed", "acc", "correct", "tokens", "tc", "tw",
            "tpca", "hit_max", "act", "soft_in", "gate_open", "guarded", "vetoed", "rollouts", "roll_len",
            "avg_k", "clusters", "ratio", "thr", "h_mean", "h_median", "h_tok", "h_ge", "gpu_h",
            "max_new_tokens", "code", "run_dir"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in sorted(all_runs.values(), key=lambda r: (str(r["model"]), str(r["dataset"]), str(r["method"]),
                                                          str(r["config"]), r["seed"] or 0)):
            w.writerow([r.get(c) for c in cols])
    os.replace(path + ".tmp", path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="SELA_ROOT (or its smoke/ tree)")
    p.add_argument("--repo", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--csv", default=None)
    p.add_argument("--cache", default=None, help="cache of the parsed results/ files")
    args = p.parse_args()

    plan = load_plan(args.root)
    # runs that finished since the last Sela_beat pass
    for run_dir, n_shards in OrderedDict((e["run_dir"], e["num_shards"]) for e in plan).items():
        if needs_merge(run_dir, n_shards):
            try:
                merge_run(run_dir, n_shards)
                print("merged %s" % run_dir)
            except (OSError, ValueError) as e:
                print("WARNING: could not merge %s: %s" % (run_dir, e))
    all_runs = scan_runs(os.path.join(args.root, "runs"))
    upstream = load_upstream(args.repo, args.cache)
    if args.csv:
        write_csv(args.csv, all_runs)

    master, ranking = section_master(plan, all_runs)
    doc = ["# SeLaR-X — Experiment Report", "",
           "Generated %s from `%s` by `reporting.sh`. Plan: `plan.tsv` of the last Sela_beat pass; "
           "every number below is recomputed from the per-problem results on Lustre%s."
           % (time.strftime("%Y-%m-%d %H:%M"), args.root,
              " (all runs as CSV: `%s`)" % args.csv if args.csv else ""),
           "",
           "**How to read.** Accuracy in %. `±` = std over seeds. `*` = significantly different from "
           "CoT, `†` = from vanilla SeLaR (exact McNemar test on the same problems, p < 0.05; seeds "
           "pooled). TPCA = (α·T_c + (1−α)·T_w)/α, tokens per correct answer; *TPCA Improvement* > 0 "
           "means fewer tokens per correct answer than CoT. *Activation Ratio* = share of decoding steps "
           "with a latent intervention (soft input, or hidden-state steering for Variant 1). `… x%` = "
           "still running. AIME has 30 problems: one problem = 3.33 points.",
           "", "## 1. Coverage", ""]
    doc += section_coverage(plan, all_runs, all_runs)
    doc += ["", "## 2. Master comparison (improvement.md §7)", ""]
    doc += master if master else ["No main-phase runs in the plan yet."]
    doc += ["", "## 3. Which variant wins? (improvement.md §7–8)", ""]
    doc += section_decision(ranking)
    doc += ["", "## 4. Per-dataset details", ""]
    doc += section_details(plan, all_runs) or ["–"]
    doc += ["", "## 5. Latent-step diagnostics", "",
            "Gate statistics per run: *Gate open* = entropy above the (static or dynamic) threshold; "
            "*Math-guarded* / *FHG vetoed* = share of those openings blocked by the math-symbol guard / "
            "the attention-FFN gate; *avg clusters* (sem), *avg k_t* (dynk), *rollouts* (ltr), "
            "*avg R_act* (fhg) and *avg τ_t* (dynamic threshold) only for the variants that use them. "
            "Entropy columns describe the gate's entropy signal (logit lens at l* for Variant 1).", ""]
    doc += section_diagnostics(plan, all_runs) or ["–"]
    doc += ["", "## 6. Hyper-parameter sweeps (sweep phase)", "",
            "One table per variant; **bold** = best average. These problems are part of the test sets, "
            "so pick the configuration here and report it on the other datasets / the full sets.", ""]
    doc += section_sweeps(plan, all_runs) or ["No sweep-phase runs in the plan (add `sweep` to PHASES)."]
    doc += ["", "## 7. Reference: the authors' published runs (`results/`)", ""]
    doc += section_reference(upstream, plan, all_runs)
    doc.append("")

    tmp = args.out + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(doc))
    os.replace(tmp, args.out)
    print("report -> %s (%d runs on disk, %d plan entries)" % (args.out, len(all_runs), len(plan)))


if __name__ == "__main__":
    main()

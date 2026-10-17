"""
Bookkeeping for Sela_beat.sh -- standard library only, so it runs with the
system python3 inside the CPU-only orchestrator job (no venv, no GPU).

    python3 scripts/sela_tools.py merge  --run_dir DIR --num_shards N
    python3 scripts/sela_tools.py status --plan PLAN.tsv --out STATUS.out [--interval MIN]

merge   all shards of a run -> <run_dir>/merged.json (every problem) and
        <run_dir>/summary.json (metrics only; what reporting.sh reads)
status  rewrites the human-readable progress table of the current plan
"""

import argparse
import json
import math
import os
import re
import sys
import time

SUMMARY_FORMAT = "selarx-summary-v1"
COUNTERS = ("steps", "latent", "soft_inputs", "gate_open", "guarded", "vetoed", "rollouts",
            "commits", "k_sum", "clusters_sum", "ratio_sum", "ratio_n", "h_sum", "h_sq", "h_n",
            "htok_sum", "thr_sum", "thr_n")


def write_json_atomic(path, obj, indent=None):
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def shard_path(run_dir, i, n):
    return os.path.join(run_dir, "shards", "shard_%03dof%03d.json" % (i, n))


def ratio(a, b):
    return a / b if b else None


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def hist_median(hist):
    """Median of a [0, 1] histogram with equal-width bins (linear inside the bin)."""
    total = sum(hist)
    if not total:
        return None
    width, acc = 1.0 / len(hist), 0.0
    for i, c in enumerate(hist):
        if acc + c >= total / 2.0 and c:
            return (i + (total / 2.0 - acc) / c) * width
        acc += c
    return 1.0


def summarize(details, run):
    """Metrics of one run. Accuracy and token statistics follow scripts/merge.py;
    TPCA = (a*T_c + (1-a)*T_w) / a (improvement.md, paper Eq. 10)."""
    n = len(details)
    correct = [d for d in details if d["correct"]]
    wrong = [d for d in details if not d["correct"]]
    acc = len(correct) / n if n else 0.0
    avg_tok = mean(d["tokens"] for d in details)
    out = {
        "format": SUMMARY_FORMAT,
        "run": run,
        "total": n,
        "correct": len(correct),
        "accuracy": acc,
        "length_stats": {
            "max_new_tokens": run.get("max_new_tokens"),
            "avg_total_token_len": avg_tok,
            "correct_avg_total_token_len": mean(d["tokens"] for d in correct),
            "wrong_avg_total_token_len": mean(d["tokens"] for d in wrong),
        },
        "tpca": avg_tok / acc if acc > 0 else None,
        "hit_max_rate": mean(1.0 if d.get("hit_max") else 0.0 for d in details),
        "errors": sum(1 for d in details if d.get("error")),
        "gpu_seconds": sum(d["batch"]["seconds"] / max(d["batch"]["size"], 1) for d in details),
        "per_index_correct": "".join("1" if d["correct"] else "0" for d in details),
        "first_index": details[0]["index"] if details else None,
    }
    with_stats = [d["stats"] for d in details if d.get("stats")]
    if with_stats:
        s = {k: sum(x.get(k, 0) for x in with_stats) for k in COUNTERS}
        hist = [sum(col) for col in zip(*[x["h_hist"] for x in with_stats])]
        h_mean = ratio(s["h_sum"], s["h_n"])
        h_var = ratio(s["h_sq"], s["h_n"])
        out["activation"] = {
            "total_steps": s["steps"],
            "latent_steps": s["latent"],
            "ratio": ratio(s["latent"], s["steps"]),
            "avg_per_sample_ratio": mean(x["latent"] / x["steps"] for x in with_stats if x["steps"]),
            "soft_input_ratio": ratio(s["soft_inputs"], s["steps"]),
            "gate_open_ratio": ratio(s["gate_open"], s["steps"]),
            "guarded_ratio": ratio(s["guarded"], s["gate_open"]),
            "vetoed_ratio": ratio(s["vetoed"], s["gate_open"]),
            "rollouts_per_sample": ratio(s["rollouts"], len(with_stats)),
            "avg_rollout_len": ratio(s["latent"], s["rollouts"]) if s["rollouts"] else None,
            "commits": s["commits"],
            "avg_k": ratio(s["k_sum"], s["latent"]) if s["k_sum"] else None,
            "avg_clusters": ratio(s["clusters_sum"], s["steps"]) if s["clusters_sum"] else None,
            "avg_ratio": ratio(s["ratio_sum"], s["ratio_n"]),
            "avg_dynamic_threshold": ratio(s["thr_sum"], s["thr_n"]),
        }
        out["entropy"] = {
            "mean": h_mean,
            "std": math.sqrt(max(h_var - h_mean ** 2, 0.0)) if h_mean is not None else None,
            "median": hist_median(hist),
            "token_entropy_mean": ratio(s["htok_sum"], s["h_n"]),
            "share_ge_0.5": ratio(sum(hist[len(hist) // 2:]), sum(hist)),
            "hist": hist,
        }
    return out


def needs_merge(run_dir, num_shards):
    """All shards present, and summary.json missing or older than one of them."""
    paths = [shard_path(run_dir, i, num_shards) for i in range(num_shards)]
    if not all(os.path.exists(p) for p in paths):
        return False
    summary = os.path.join(run_dir, "summary.json")
    return not os.path.exists(summary) or max(os.path.getmtime(p) for p in paths) > os.path.getmtime(summary)


def merge_run(run_dir, num_shards):
    shards = []
    for i in range(num_shards):
        path = shard_path(run_dir, i, num_shards)
        if not os.path.exists(path):
            raise FileNotFoundError("missing shard: %s" % path)
        with open(path, encoding="utf-8") as f:
            shards.append(json.load(f))
    details = sorted((d for s in shards for d in s["details"]), key=lambda d: d["index"])
    indices = [d["index"] for d in details]
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate problem indices across shards of %s" % run_dir)
    run = dict(shards[0]["run"])
    run["code_versions"] = sorted({str(s["run"].get("code_version")) for s in shards})
    run["gpus"] = sorted({str(s.get("env", {}).get("gpu")) for s in shards})
    summary = summarize(details, run)
    summary["merged_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json_atomic(os.path.join(run_dir, "merged.json"), {"summary": summary, "details": details})
    write_json_atomic(os.path.join(run_dir, "summary.json"), summary, indent=1)
    return summary


def cmd_merge(args):
    try:
        summary = merge_run(args.run_dir, args.num_shards)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))
    print("merged %d problems, accuracy %.2f%% -> %s" % (summary["total"], 100 * summary["accuracy"], args.run_dir))


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

SHARD_HEAD = re.compile(r'"total": (\d+), "correct": (\d+)')


def shard_counts(path):
    """(total, correct) from the head of a shard json without parsing the details."""
    with open(path, encoding="utf-8") as f:
        m = SHARD_HEAD.search(f.read(4096))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def partial_counts(path):
    total = correct = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            total += 1
            correct += bool(rec.get("correct"))
    return total, correct


def bar(pct):
    n = min(int(pct / 5 + 0.5), 20)
    return "[%s%s] %5.1f%%" % ("#" * n, "-" * (20 - n), pct)


def cmd_status(args):
    rows = []
    with open(args.plan, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            rows.append(line.rstrip("\n").split("\t"))
    fmt = "%-6s %-15s %-34s %-16s %-12s %5s %4s %7s %9s %11s %8s  %-28s %s\n"
    out = []
    out.append("=" * 168 + "\n")
    out.append(" SeLaR-X STATUS -- %s (refreshed every ~%s min by Sela_beat.sh)\n"
               % (time.strftime("%Y-%m-%d %H:%M:%S"), args.interval))
    out.append(" ACC = accuracy so far (final once MERGED); SHARDS = done/total; JOBS = running/pending\n")
    out.append("=" * 168 + "\n\n")
    out.append(fmt % ("PHASE", "METHOD", "CONFIG", "MODEL", "DATASET", "N", "SEED", "SHARDS", "JOBS",
                      "SAMPLES", "ACC", "PROGRESS", "STATE"))
    totals = {}
    for (phase, method, tag, model, dataset, n, seed, run_dir, n_shards, running, pending, stuck) in rows:
        n_shards, n_exp = int(n_shards), int(n)
        done_shards = done = correct = 0
        summary = os.path.join(run_dir, "summary.json")
        if os.path.exists(summary):
            with open(summary, encoding="utf-8") as f:
                s = json.load(f)
            done_shards, done, correct, state = n_shards, s["total"], s["correct"], "MERGED"
        else:
            for i in range(n_shards):
                path = shard_path(run_dir, i, n_shards)
                if os.path.exists(path):
                    t, c = shard_counts(path)
                    done_shards += 1
                elif os.path.exists(path[:-5] + ".partial.jsonl"):
                    t, c = partial_counts(path[:-5] + ".partial.jsonl")
                else:
                    t = c = 0
                done += t
                correct += c
            if int(stuck):
                state = "GAVE_UP(%s)" % stuck
            elif int(running):
                state = "RUNNING"
            elif int(pending):
                state = "PENDING"
            elif done_shards == n_shards:
                state = "TO_MERGE"
            else:
                state = "WAITING"
        acc = "%.2f%%" % (100.0 * correct / done) if done else "-"
        pct = 100.0 * done / n_exp if n_exp else 0.0
        tag_short = tag if len(tag) <= 34 else tag[:31] + "..."
        out.append(fmt % (phase, method, tag_short, model.split("/")[-1][:16], dataset, n, seed,
                          "%d/%d" % (done_shards, n_shards), "%s/%s" % (running, pending),
                          "%d/%d" % (done, n_exp), acc, bar(pct), state))
        t = totals.setdefault(phase, [0, 0, 0, 0])
        t[0] += 1
        t[1] += state == "MERGED"
        t[2] += done
        t[3] += n_exp
    out.append("\n summary:\n")
    for phase, (runs, merged, done, n_exp) in sorted(totals.items()):
        out.append("   %-6s runs merged %d/%d   problems done %d/%d (%.1f%%)\n"
                   % (phase, merged, runs, done, n_exp, 100.0 * done / n_exp if n_exp else 0.0))
    tmp = "%s.tmp.%d" % (args.out, os.getpid())
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(out)
    os.replace(tmp, args.out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--run_dir", required=True)
    m.add_argument("--num_shards", type=int, required=True)
    s = sub.add_parser("status")
    s.add_argument("--plan", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--interval", default="60")
    args = p.parse_args()
    {"merge": cmd_merge, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    main()

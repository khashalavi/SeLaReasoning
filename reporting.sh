#!/bin/bash
#SBATCH --partition=sgpu_short
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=sela_report
#SBATCH --mem=4G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# ============================================================================
# reporting.sh — writes THE final report of the SeLaR-X experiments:
#
#     ./result/selar_report.md            (marvin)
#     ./result/selar_report_bender.md     (bender: reporting_bender.sh, which
#                                          runs this file with SELA_SERVER=bender)
#
# from everything Sela_beat.sh produced in $SELA_ROOT (marvin: Lustre,
# bender: <repo>/sela_data): the current
# plan (plan.tsv), every merged run (runs/**/summary.json) -- runs whose
# shards are all finished but not merged yet are merged first -- and, as a
# reference, the authors' published results in ./results/. Contents:
# coverage, the master comparison table of improvement.md §7 (+ McNemar
# significance), the variant ranking with the Variant 6 / paper-framing
# suggestion (§7-8), per-dataset tables, latent-step diagnostics, the sweep
# tables, and the comparison with the published numbers.
#
# Only the report itself goes into the repo. In $SELA_ROOT:
#   $SELA_ROOT/reports/all_runs.csv     every run on disk, one row each
#   $SELA_ROOT/reports/history/         earlier versions of the report
#   $SELA_ROOT/logs/reporting.log       this script's log
#
# Usage (from anywhere in the repo; on bender: reporting_bender.sh):
#   ./reporting.sh            takes seconds, needs only a python3 >= 3.8 (stdlib;
#                             an older system python3 -> the Python module)
#   sbatch reporting.sh       the same as a small CPU-only job
#   ./reporting.sh --smoke    report of the `Sela_beat.sh --smoke` tree, written
#                             to $SELA_ROOT/smoke/smoke_report.md instead
# Sela_beat.sh also runs it after every pass (AUTO_REPORT=1).
# ============================================================================
set -euo pipefail

find_repo_root() {
    local dir="$1"
    while [ "$dir" != "/" ]; do
        [ -d "$dir/.git" ] && { printf '%s\n' "$dir"; return 0; }
        dir="$(dirname "$dir")"
    done
    return 1
}
# Slurm runs a copy of this script from a spool dir -- use SLURM_SUBMIT_DIR,
# else the path the job was submitted with (scontrol "Command="). A server
# wrapper (reporting_bender.sh) passes the repo it found as SELA_REPO.
START_DIR="${SELA_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
REPO_ROOT="$(find_repo_root "$START_DIR" || true)"
if [ -z "$REPO_ROOT" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    submitted="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -oE 'Command=[^ ]+' | cut -d= -f2 || true)"
    [ -n "$submitted" ] && REPO_ROOT="$(find_repo_root "$(dirname "$submitted")" || true)"
fi
[ -n "$REPO_ROOT" ] || { echo "ERROR: could not locate repo root (.git) above $START_DIR" >&2; exit 1; }

# Same per-server defaults as the RESOURCES profiles of Sela_beat.sh.
SELA_SERVER="${SELA_SERVER:-marvin}"
case "$SELA_SERVER" in
    marvin) SELA_ROOT="${SELA_ROOT:-/lustre/scratch/data/s76kalav_hpc-memory/selareasoning}"
            PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"; REPORT_FILE=selar_report.md ;;
    bender) SELA_ROOT="${SELA_ROOT:-$REPO_ROOT/sela_data}"
            PYTHON_BIN="${PYTHON_BIN:-python3}"; REPORT_FILE=selar_report_bender.md ;;
    *) echo "ERROR: unknown SELA_SERVER '$SELA_SERVER' (marvin | bender)" >&2; exit 1 ;;
esac

# sela_report.py needs python >= 3.8 (math.comb); an older one -> the Python module.
py_ok() { "$1" -c 'import sys; sys.exit(sys.version_info < (3, 8))' 2>/dev/null; }
if ! py_ok "$PYTHON_BIN"; then
    set +u
    type module >/dev/null 2>&1 || for f in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh; do
        # shellcheck disable=SC1090
        [ -f "$f" ] && { source "$f"; break; }
    done
    module load Python/3.11.3-GCCcore-12.3.0 >/dev/null 2>&1 || true
    set -u
    PYTHON_BIN=python3
    py_ok "$PYTHON_BIN" || { echo "ERROR: no python3 >= 3.8 found -- set PYTHON_BIN" >&2; exit 1; }
fi

TREE="$SELA_ROOT"
OUT="$REPO_ROOT/result/$REPORT_FILE"
if [ "${1:-}" = "--smoke" ]; then
    TREE="$SELA_ROOT/smoke"
    OUT="$TREE/smoke_report.md"
fi

[ -d "$TREE" ] || { echo "ERROR: $TREE does not exist -- run the heart beat first ($([ "$SELA_SERVER" = marvin ] && echo Sela_beat.sh || echo "Sela_beat_$SELA_SERVER.sh"))" >&2; exit 1; }
mkdir -p "$SELA_ROOT/logs" "$TREE/reports/history" "$(dirname "$OUT")"
LOG="$SELA_ROOT/logs/reporting.log"

{
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] reporting.sh job=${SLURM_JOB_ID:-manual} tree=$TREE"
    # keep the previous version in $SELA_ROOT, once per distinct content
    if [ -f "$OUT" ]; then
        last="$(ls -t "$TREE/reports/history/"*.md 2>/dev/null | head -n 1 || true)"
        if [ -z "$last" ] || ! cmp -s "$OUT" "$last"; then
            cp -p "$OUT" "$TREE/reports/history/$(basename "$OUT" .md)_$(date -r "$OUT" +%Y%m%d-%H%M%S).md"
        fi
    fi
    "$PYTHON_BIN" -B "$REPO_ROOT/scripts/sela_report.py" --root "$TREE" --repo "$REPO_ROOT" \
        --out "$OUT" --csv "$TREE/reports/all_runs.csv" --cache "$SELA_ROOT/reports/upstream_cache.json"
} 2>&1 | tee -a "$LOG"

#!/bin/bash
#SBATCH --partition=A40devel
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=sela_report
#SBATCH --mem=4G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# ============================================================================
# reporting_bender.sh — reporting.sh on the bender cluster: writes
#
#     ./result/selar_report_bender.md
#
# from everything Sela_beat_bender.sh produced in <repo>/sela_data/ (bender
# has no Lustre). Identical to reporting.sh except for the #SBATCH partition
# above and SELA_SERVER=bender (bender and marvin don't share partition names,
# and Slurm reads #SBATCH lines from the file at submission time).
#   <repo>/sela_data/reports/all_runs.csv   every run on disk, one row each
#   <repo>/sela_data/reports/history/       earlier versions of the report
#   <repo>/sela_data/logs/reporting.log     this script's log
#
# Usage (on bender, from anywhere inside the repo):
#   ./reporting_bender.sh            takes seconds, needs only a python3 >= 3.8
#   sbatch reporting_bender.sh       the same as a small CPU-only job
#   ./reporting_bender.sh --smoke    report of the `Sela_beat_bender.sh --smoke`
#                                    tree -> sela_data/smoke/smoke_report.md
# Sela_beat_bender.sh also runs it after every pass (AUTO_REPORT=1).
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
# else the path the job was submitted with (scontrol "Command=").
START_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_ROOT="$(find_repo_root "$START_DIR" || true)"
if [ ! -f "${REPO_ROOT:-/nonexistent}/reporting.sh" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    submitted="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -oE 'Command=[^ ]+' | cut -d= -f2 || true)"
    [ -n "$submitted" ] && REPO_ROOT="$(find_repo_root "$(dirname "$submitted")" || true)"
fi
[ -f "${REPO_ROOT:-/nonexistent}/reporting.sh" ] || {
    echo "ERROR: could not locate the SeLaReasoning repo (.git + reporting.sh) above $START_DIR" >&2; exit 1; }

SELA_SERVER=bender SELA_REPO="$REPO_ROOT" exec bash "$REPO_ROOT/reporting.sh" "$@"

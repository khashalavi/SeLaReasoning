#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=sela_beat
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --mem=2G
# NOTE: CPU-only orchestrator (no --gres), like Disentangling-Reasoning's
# heart_beat.sh on bender. --partition must allow the 24h --time above
# (A40medium does; A40devel caps at 1h); TIME_LIMIT_SECONDS in Sela_beat.sh
# assumes 24h.
# ============================================================================
# Sela_beat_bender.sh — the SeLaR-X heart beat on the bender cluster.
#
# Runs Sela_beat.sh -- the same SELECTION, flags, journal and job logic --
# with the bender profile of its RESOURCES block (SELA_SERVER=bender):
#   storage     bender has no Lustre: env, models, caches, runs, logs and the
#               journal all go to <repo>/sela_data/ (git-ignored)
#   GPUs        BENDER_GPU=A40 (default): shards on A40short (8h), env build
#               on A40devel, downloads on A40short, --smoke on A40devel;
#               A40 = 48 GB, so BATCH_SIZE and SHARD_SIZE are halved
#               (GPU_SCALE=0.5). BENDER_GPU=A100: the A100* partitions.
#   venv        sela_data/env_a40 (Python module from the default, Intel tree)
#               or sela_data/env_a100 (/software/easybuild-AMD_A100 tree, AVX2);
#               transformers 5 for TF5_MODELS (Qwen3.5) in env_tf5_a40/env_tf5_a100
#   report      ./result/selar_report_bender.md (reporting_bender.sh)
# Every job it submits (shards, env build, downloads, its own re-init after
# ~24h) runs this file again, so all of them stay on bender's partitions.
#
# HOW TO RUN (on bender, from anywhere inside the repo):
#     ./Sela_beat_bender.sh --plan          what the SELECTION expands to
#     sbatch Sela_beat_bender.sh --smoke    every method end-to-end on Qwen3-0.6B
#     sbatch Sela_beat_bender.sh            the hourly heart beat (keeps going)
#     ./Sela_beat_bender.sh --status        progress from journal + live queue
#     ./reporting_bender.sh                 -> result/selar_report_bender.md
#   All flags: ./Sela_beat_bender.sh --help (they are Sela_beat.sh's).
#   Edit the SELECTION (or the bender profile) in Sela_beat.sh: a running beat
#   notices at its next pass and hands over to a fresh job, as on marvin.
# ============================================================================

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
SELA_REPO="$(find_repo_root "$START_DIR")"
if [ ! -f "${SELA_REPO:-/nonexistent}/Sela_beat.sh" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    submitted="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -oE 'Command=[^ ]+' | cut -d= -f2)"
    [ -n "$submitted" ] && SELA_REPO="$(find_repo_root "$(dirname "$submitted")")"
fi
if [ ! -f "${SELA_REPO:-/nonexistent}/Sela_beat.sh" ]; then
    echo "ERROR: could not locate the SeLaReasoning repo (.git + Sela_beat.sh) above $START_DIR" >&2
    exit 1
fi

SELA_SERVER=bender
# shellcheck source=Sela_beat.sh
source "$SELA_REPO/Sela_beat.sh"

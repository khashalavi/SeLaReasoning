#!/bin/bash
#SBATCH --partition=sgpu_medium
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=sela_beat
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --mem=2G
# NOTE: CPU-only orchestrator (no --gres), submitted like Disentangling-
# Reasoning's heart_beat.sh on marvin. --partition must allow the 24h --time
# above (sgpu_medium does; *_devel caps at 1h). Keep TIME_LIMIT_SECONDS below
# in sync with --time.
# ============================================================================
# Sela_beat.sh — the heart beat of the SeLaR-X experiments (improvement.md):
# hourly (re-)submission of every (method, model, dataset, seed, shard) the
# SELECTION below asks for, with a journal so we always know exactly which
# Slurm job belongs to which piece of work.
#
# EVERYTHING IS WRITTEN TO LUSTRE ($SELA_ROOT), NEVER INTO THE REPO:
#   env/                 Python venv (torch + requirements-sela.txt), built once
#   models/<org>/<name>  model weights, downloaded once (.download_complete marker)
#   hf_home/ cache/      HF hub / pip / triton / pycache caches
#   runs/<model>/<dataset>/n<N>/<config>/<decoding>/seed<S>/
#        config.json, shards/shard_<i>of<n>.{json,partial.jsonl,attempts},
#        merged.json (every problem), summary.json (metrics)
#   logs/sela_beat.log   this script's log;  logs/jobs/<job name>_<job id>.out
#   journal.txt          job lifecycle journal (format below)
#   plan.tsv             the expanded SELECTION of the last pass (reporting.sh reads it)
#   status.out           progress table, rewritten every pass
#   reports/             report history + all_runs.csv (reporting.sh)
# The one file that lands in the repo is the final report ./result/selar_report.md,
# written by reporting.sh -- run after every pass when AUTO_REPORT=1.
#
# WHAT ONE PASS DOES (every CHECK_INTERVAL, default 1h):
#   0. prerequisites, each its own Slurm job, submitted once and tracked like
#      everything else:  env not built -> --setup-env job (GPU node: the venv
#      must be built from the AMD module tree, like setup-env/build_env_a100_
#      marvin.slurm);  model not in models/ -> --download job per model.
#      Shards of a model wait until both are ready.
#   1. expand SELECTION -> runs -> shards (= GPU jobs), write plan.tsv
#   2. for every shard:
#        FINISHED  if shards/shard_<i>of<n>.json exists
#        else look up the job id recorded for it and ask squeue:
#          RUNNING / PENDING -> in the pipeline, leave it alone
#          gone / failed     -> resubmit; the worker resumes from .partial.jsonl
#        GAVE_UP once MAX_STALLED attempts in a row made no progress (crash
#        loop) -- see logs/jobs/, fix, then --retry-failed
#   3. merge every run whose shards are all finished -> summary.json
#   4. rewrite status.out; run reporting.sh (AUTO_REPORT=1)
#
# JOURNAL FORMAT (append-only, one line per event):
#   <timestamp> | <EVENT> | <key> | job=<id>
#   key:   shard::<run dir below runs/>::<i>of<n> | run::<run dir below runs/>
#          | env::setup | download::<model>
#   EVENT: SUBMITTED RUNNING FINISHED DIED ADOPTED SUBMIT_FAILED GAVE_UP MERGED
#          RESET <slurm-state>
#
# HOW TO RUN -- submit it once:
#     sbatch Sela_beat.sh
#   It runs ~24h (#SBATCH --time), checking hourly, and ~10 min before the
#   wall clock it submits a FRESH Sela_beat job and exits, so it keeps going
#   indefinitely. Edit the SELECTION while it runs: at the next pass it
#   notices, validates the new version and hands over to a fresh job running
#   it (an invalid edit is logged and ignored). Before the first real run:
#     ./Sela_beat.sh --plan          what the SELECTION expands to
#     sbatch Sela_beat.sh --smoke    every method end-to-end on a tiny model
#
# USEFUL FLAGS:
#     --plan          print the expanded experiment matrix (runs, GPU jobs), exit
#     --plan-tsv      the same as tab-separated values (the plan.tsv format)
#     --dry-run       one pass showing what WOULD be submitted; writes nothing
#     --status        one pass that only reports (journal + live queue)
#     --once          a single real pass, no hourly loop
#     --smoke         tiny end-to-end check of every method (Qwen3-0.6B, 4 GSM8K
#                     problems, 512 tokens, sgpu_devel) in $SELA_ROOT/smoke/;
#                     checks every 5 min and exits when all of it is merged
#                     (report: ./reporting.sh --smoke)
#     --retry-failed  reset the stall counters of GAVE_UP shards / jobs
#   Internal -- the jobs this script submits run the script itself as:
#     --setup-env | --download MODEL | --worker ATTEMPTS_FILE -- <sela_run.py args>
# ============================================================================

set -uo pipefail

# ---------------------------------------------------------------------------
# SELECTION — edit these to focus on what you care about
# ---------------------------------------------------------------------------
# PHASES: which stages of the improvement.md blueprint to run (any combination)
#   main   every METHOD and COMBO with its DEFAULT hyper-parameters on DATASETS
#          -> the master comparison table (improvement.md section 7)
#   sweep  the ablation grids (the *_SWEEP arrays below) of SWEEP_METHODS on
#          SWEEP_DATASETS / SWEEP_N_SAMPLES (improvement.md sections 2.4-6.4)
PHASES=(main)

# Methods. One run per model x dataset x seed each.
#   cot             Baseline 1: standard CoT, sampling
#   selar           Baseline 2: vanilla SeLaR, gated per row (paper Algorithm 1)
#   layer           Variant 1: layer-wise hidden-state steering        (Idea 1)
#   sem             Variant 2: semantic entropy + dynamic threshold    (Idea 2)
#   ltr             Variant 3: multi-step latent trajectory rollouts   (Idea 3)
#   fhg             Variant 4: functional (attention/FFN) head gating  (Idea 4)
#   dynk            Variant 5: nucleus dynamic-k soft embedding        (Idea 5)
# extra baselines:
#   cot_greedy      CoT, greedy decoding
#   selar_upstream  the authors' generate_selar() verbatim -- in a batch EVERY
#                   row gets a soft input as soon as ANY row is uncertain, so
#                   its results depend on the batch size (selar does not)
#   cot_upstream    the authors' generate_cot() verbatim
#   swir            SwiReasoning, authors' implementation (SWIR_* below)
METHODS=(cot selar layer sem ltr fhg dynk)

# Variant 6 (improvement.md 7): combinations, "+"-joined ideas out of
# layer sem ltr fhg dynk, e.g. COMBOS=("sem+dynk" "ltr+dynk"). The report
# ranks the variants and suggests the top-2 combination.
COMBOS=()

MODELS=(
    "Qwen/Qwen3-8B"
    # "Qwen/Qwen3-1.7B"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"     # less calibrated (Idea 2)
)
DATASETS=(gsm8k math500 aime_2024 aime_2025 gpqa_diamond)
SEEDS=(41)              # several seeds -> mean +- std in the report (AIME: 30 problems,
                        # one problem = 3.33 points, a single seed is very noisy)
N_SAMPLES=all           # all | N  (the first N problems of every dataset)

# sweep phase
SWEEP_METHODS=(layer sem ltr fhg dynk)        # "selar" sweeps SELAR_TOPK x SELAR_TAU
SWEEP_MODELS=("Qwen/Qwen3-8B")
SWEEP_DATASETS=(math500 aime_2024)            # tune on these, report on the others
SWEEP_N_SAMPLES=100     # all | N
SWEEP_SEEDS=(41)
SWEEP_MODE=oat          # oat:  the default + one knob changed at a time
                        # grid: every combination of the *_SWEEP arrays

# ---------------------------------------------------------------------------
# HYPER-PARAMETERS — X is the default (main phase), X_SWEEP what the sweep
# phase tries. Defaults and grids follow improvement.md unless noted.
# ---------------------------------------------------------------------------
# Decoding: identical for every method (paper section 4.1)
TEMPERATURE=0.6; TOP_P=0.95; TOP_K=20; MIN_P=0.0
declare -A MAX_NEW_TOKENS=([gsm8k]=32768 [math500]=32768 [gpqa_diamond]=32768
                           [aime_2024]=38912 [aime_2025]=38912)

# Base SeLaR gate, shared by selar, selar_upstream and every input-level
# variant (sem ltr fhg dynk and their combos):
#   SELAR_PRESET=paper  k/tau per (model, dataset) as published (selar_preset()
#                       below). The paper picked tau per benchmark ON the test
#                       sets (App. B) -- the strongest possible SeLaR baseline.
#   SELAR_PRESET=fixed  SELAR_TOPK / SELAR_TAU everywhere; k=3 tau=0.5 is the
#                       best single setting of the paper's Table 3.
SELAR_PRESET=paper
SELAR_TOPK=3;            SELAR_TOPK_SWEEP=(2 3 5)
SELAR_TAU=0.5;           SELAR_TAU_SWEEP=(0.3 0.4 0.5 0.6 0.7)
CONTRASTIVE_W=1.0       # contrastive regularisation weight (paper: 1)
SOFT_MASK=per_row       # per_row (paper) | batch_any (the upstream batching quirk)

# Variant 1 -- layer (2.4).  l* = floor(LAYER_FRAC * #layers)
LAYER_FRAC=0.5;          LAYER_FRAC_SWEEP=(0.25 0.33 0.5 0.67)
# The logit lens is much less certain at mid layers than the output (smoke test,
# Qwen3-0.6B: lens entropy median 0.74 at L/2 -> tau=0.5 steered 61% of the steps,
# vanilla SeLaR 12.5%), hence 0.8 / 0.9 beyond the improvement.md grid.
LAYER_TAU=0.5;           LAYER_TAU_SWEEP=(0.3 0.4 0.5 0.6 0.7 0.8 0.9)
# lens  = gate on the lens entropy at l*, steer that same step (as written)
# final = SeLaR's own gate on the output distribution decides, the hidden state
#         is steered at the next step -> same trigger as vanilla SeLaR, only the
#         locus differs (isolates the effect of steering inside the network)
LAYER_GATE=lens;         LAYER_GATE_SWEEP=(lens final)
LAYER_ALPHA=1.0;         LAYER_ALPHA_SWEEP=(0.5 1.0 1.5 2.0)
LAYER_TOPK=3;            LAYER_TOPK_SWEEP=(2 3 5)
# embed_delta   = the formula as written: h += alpha*H*(h - E(v_top1)). At mid
#                 layers ||h|| >> ||E(v)||, so this is close to h *= 1+alpha*H.
# lens_contrast = (not in improvement.md) push h from the logit-lens top-1 towards
#                 the top-k mixture in unembedding space, by LAYER_REL_SCALE*||h||
LAYER_DIRECTION=embed_delta; LAYER_DIRECTION_SWEEP=(embed_delta lens_contrast)
LAYER_REL_SCALE=0.1;     LAYER_REL_SCALE_SWEEP=(0.1)          # e.g. (0.05 0.1 0.2)

# Variant 2 -- sem (3.4)
SEM_TOPK=5;              SEM_TOPK_SWEEP=(5)                    # e.g. (3 5 8)
SEM_SIM=0.82;            SEM_SIM_SWEEP=(0.75 0.80 0.85 0.90)   # 2 = never merge (ablation)
SEM_WINDOW=15;           SEM_WINDOW_SWEEP=(5 10 15 20)         # 0 = static tau (ablation)
SEM_GAMMA=1.0;           SEM_GAMMA_SWEEP=(0.5 1.0 1.5)
# not in improvement.md: without a floor, tau_t = mu + gamma*sigma drops to 0
# after any run of fully confident steps and every small wobble fires
SEM_FLOOR=0.3;           SEM_FLOOR_SWEEP=(0.3)                 # e.g. (0.2 0.3 0.4)

# Variant 3 -- ltr (4.4)
LTR_MAX=3;               LTR_MAX_SWEEP=(2 3 4 5)
LTR_EXIT=0.2;            LTR_EXIT_SWEEP=(0.15 0.20 0.25)
LTR_DECAY=0.9;           LTR_DECAY_SWEEP=(0.8 0.9 1.0)
LTR_COMMIT=sample;       LTR_COMMIT_SWEEP=(sample)             # e.g. (sample argmax)

# Variant 4 -- fhg (5.4). R = sum ||attn out|| / sum ||FFN out|| over the
# monitored layers. absolute: R > theta, as written -- but the smoke test
# measured R ~ 0.58 (Qwen3-0.6B), so theta in 0.8..1.4 vetoed 100% of the
# openings and fhg silently became CoT. relative (default): R divided by its
# running mean over the sequence > theta -- scale-free across models.
FHG_THETA=1.0;           FHG_THETA_SWEEP=(0.8 1.0 1.2 1.4)
FHG_MODE=relative;       FHG_MODE_SWEEP=(relative absolute)
FHG_LAYERS=upper;        FHG_LAYERS_SWEEP=(upper all)          # upper = layers L/2..L

# Variant 5 -- dynk (6.4)
DYNK_MASS=0.9;           DYNK_MASS_SWEEP=(0.80 0.85 0.90 0.95)
DYNK_MIN=2;              DYNK_MIN_SWEEP=(2)
DYNK_MAX=8;              DYNK_MAX_SWEEP=(5 8 10)

# SwiR baseline (METHODS=swir). Authors used alpha 0.9-1.0 for Qwen3-8B.
SWIR_ALPHA=1.0;          SWIR_ALPHA_SWEEP=(0.6 0.9 1.0)
SWIR_MAX_SWITCH=none;    SWIR_MAX_SWITCH_SWEEP=(none)          # none | N

# ---------------------------------------------------------------------------
# RESOURCES
# ---------------------------------------------------------------------------
SELA_ROOT="${SELA_ROOT:-/lustre/scratch/data/s76kalav_hpc-memory/selareasoning}"
GPU_PARTITION=sgpu_short        # A100 nodes, 8h. mlgpu_short = A40 48GB: halve BATCH_SIZE
GPU_TIME=08:00:00
GPU_MEM=64G
GPU_GRES=gpu:1
GPU_EXTRA_SBATCH=()             # e.g. (--qos=... --constraint=...)
ENV_PARTITION=sgpu_devel        # --setup-env (needs an AMD GPU node; ~3 min)
DOWNLOAD_PARTITION=sgpu_short   # --download (CPU only, 32G)
# Problems per GPU job (shard) and per forward batch. A shard should fit in
# GPU_TIME (a killed shard resumes, but loses its unfinished batch); a batch
# that runs out of memory is retried at half size by the worker.
declare -A SHARD_SIZE=([gsm8k]=330 [math500]=125 [gpqa_diamond]=99 [aime_2024]=30 [aime_2025]=30)
declare -A BATCH_SIZE=([gsm8k]=48 [math500]=32 [gpqa_diamond]=24 [aime_2024]=15 [aime_2025]=15)
MAX_STALLED=3                   # attempts in a row without progress -> GAVE_UP
MAX_SUBMIT_PER_PASS=150         # cap on new GPU jobs per pass (the rest wait an hour)
SUBMIT_PAUSE=1                  # seconds between sbatch calls: slurmctld rate-limits bursts
SUBMIT_TIMEOUT=300              # give up on one sbatch call after this long (retried next pass)
AUTO_REPORT=1                   # run reporting.sh after every pass
EXIT_WHEN_DONE=0                # 1: stop (no re-init) once every planned run is merged;
                                # 0: keep checking hourly, e.g. for SELECTION edits
WS_WARN_DAYS=10                 # warn when the Lustre workspace expires sooner

# Python on marvin's GPU nodes (AMD EPYC): the AMD EasyBuild tree, AVX2 only.
PYTHON_MODULE="Python/3.11.3-GCCcore-12.3.0"
AMD_MODULEPATH="/opt/software/easybuild-AMD/modules/all"
TORCH_VERSION=2.7.0
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
SYS_PYTHON=/usr/bin/python3     # stdlib-only helpers (merge, status) in this CPU job

# Slurm account whose queue we inspect (defaults to whoever runs the script).
SLURM_USER="${SLURM_USER:-$(id -un)}"

# How often to re-check (seconds). 3600 = every hour; PREREQ_INTERVAL while the
# env build / a model download is still pending, so the shards go out soon after.
CHECK_INTERVAL="${CHECK_INTERVAL:-3600}"
PREREQ_INTERVAL="${PREREQ_INTERVAL:-600}"

# The job runs for ~24h (see #SBATCH --time), then re-initializes itself a few
# minutes before the wall clock: under Slurm it submits a fresh Sela_beat job;
# standalone it re-exec's. Keep TIME_LIMIT_SECONDS in sync with #SBATCH --time.
TIME_LIMIT_SECONDS="${TIME_LIMIT_SECONDS:-$(( 24 * 3600 ))}"
REINIT_MARGIN="${REINIT_MARGIN:-$(( 10 * 60 ))}"   # reinit 10 min early => ~23:50

# --smoke: every method once, tiny, in its own tree (journal, runs, status)
apply_smoke() {
    PHASES=(main)
    METHODS=(cot cot_greedy selar selar_upstream cot_upstream swir layer sem ltr fhg dynk)
    COMBOS=("sem+dynk" "layer+ltr")
    MODELS=("Qwen/Qwen3-0.6B")
    DATASETS=(gsm8k)
    SEEDS=(41)
    N_SAMPLES=4
    local d
    for d in "${!MAX_NEW_TOKENS[@]}"; do
        MAX_NEW_TOKENS[$d]=512; SHARD_SIZE[$d]=4; BATCH_SIZE[$d]=4
    done
    GPU_PARTITION=sgpu_devel
    GPU_TIME=00:30:00
    AUTO_REPORT=0
    EXIT_WHEN_DONE=1
    CHECK_INTERVAL=300
}

# ---------------------------------------------------------------------------
# Internal tables (no need to edit)
# ---------------------------------------------------------------------------
IDEAS=(layer sem ltr fhg dynk)
BASE_METHODS=(cot cot_greedy cot_upstream selar selar_upstream swir)
declare -A DATASET_SIZE=([gsm8k]=1319 [math500]=500 [gpqa_diamond]=198 [aime_2024]=30 [aime_2025]=30)
declare -A IDEA_KNOBS=(
    [layer]="LAYER_FRAC LAYER_TAU LAYER_ALPHA LAYER_TOPK LAYER_DIRECTION LAYER_REL_SCALE LAYER_GATE"
    [sem]="SEM_TOPK SEM_SIM SEM_WINDOW SEM_GAMMA SEM_FLOOR"
    [ltr]="LTR_MAX LTR_EXIT LTR_DECAY LTR_COMMIT"
    [fhg]="FHG_THETA FHG_MODE FHG_LAYERS"
    [dynk]="DYNK_MASS DYNK_MIN DYNK_MAX"
)
# knob -> scripts/sela_run.py flag
declare -A KNOB_FLAG=(
    [SELAR_TOPK]=topk [SELAR_TAU]=tau [CONTRASTIVE_W]=contrastive_weight [SOFT_MASK]=soft_mask
    [LAYER_FRAC]=layer_frac [LAYER_TAU]=layer_tau [LAYER_ALPHA]=layer_alpha [LAYER_TOPK]=layer_topk
    [LAYER_DIRECTION]=layer_direction [LAYER_REL_SCALE]=layer_rel_scale [LAYER_GATE]=layer_gate
    [SEM_TOPK]=sem_topk [SEM_SIM]=sem_sim [SEM_WINDOW]=sem_window [SEM_GAMMA]=sem_gamma
    [SEM_FLOOR]=sem_floor
    [LTR_MAX]=ltr_max [LTR_EXIT]=ltr_exit [LTR_DECAY]=ltr_decay [LTR_COMMIT]=ltr_commit
    [FHG_THETA]=fhg_theta [FHG_MODE]=fhg_mode [FHG_LAYERS]=fhg_layers
    [DYNK_MASS]=dynk_mass [DYNK_MIN]=dynk_min [DYNK_MAX]=dynk_max
    [SWIR_ALPHA]=alpha [SWIR_MAX_SWITCH]=max_switch_count
)
# knob -> short name in run directory tags (and values left out of the tag)
declare -A KNOB_TAG=(
    [SELAR_TOPK]=k [SELAR_TAU]=t [CONTRASTIVE_W]=w [SOFT_MASK]=mask
    [LAYER_FRAC]=lf [LAYER_TAU]=lt [LAYER_ALPHA]=la [LAYER_TOPK]=lk [LAYER_DIRECTION]=ld
    [LAYER_REL_SCALE]=lr [LAYER_GATE]=lg
    [SEM_TOPK]=sk [SEM_SIM]=ss [SEM_WINDOW]=sw [SEM_GAMMA]=sg [SEM_FLOOR]=sf
    [LTR_MAX]=rm [LTR_EXIT]=rx [LTR_DECAY]=rd [LTR_COMMIT]=rc
    [FHG_THETA]=ft [FHG_MODE]=fm [FHG_LAYERS]=fl
    [DYNK_MASS]=dm [DYNK_MIN]=dn [DYNK_MAX]=dx
    [SWIR_ALPHA]=a [SWIR_MAX_SWITCH]=s
)
declare -A KNOB_TAG_DEFAULT=([CONTRASTIVE_W]=1 [SOFT_MASK]=per_row [LTR_COMMIT]=sample [LAYER_GATE]=lens)
declare -A KNOB_CHOICES=(
    [SOFT_MASK]="per_row batch_any" [LAYER_DIRECTION]="embed_delta lens_contrast" [LAYER_GATE]="lens final"
    [LTR_COMMIT]="sample argmax" [FHG_MODE]="absolute relative" [FHG_LAYERS]="upper all"
)

# k/tau the paper used per (model, dataset): App. B "final selection" for
# Qwen3, Fig. 7 for DeepSeek-R1-Distill-Llama-8B, and the authors' result file
# names in results/SeLaR/ where those are more specific. Qwen3-8B GPQA: the
# published 61.62 comes from k=5 tau=0.5 (results/SeLaR/Qwen3-8B/*gpqa*k5_t0.5*),
# although App. B states k=3 tau=0.7 for every model.
selar_preset() {   # $1 model $2 dataset -> "k tau" (empty: no preset)
    case "$1|$2" in
        "Qwen/Qwen3-8B|gpqa_diamond")     echo "5 0.5" ;;
        "Qwen/Qwen3-1.7B|"*)              echo "3 0.5" ;;
        "Qwen/Qwen3-32B|aime_2025")       echo "3 0.6" ;;
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B|gsm8k")        echo "3 0.5" ;;
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B|math500")      echo "3 0.6" ;;
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B|gpqa_diamond") echo "3 0.8" ;;
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B|aime_2024")    echo "3 0.5" ;;
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B|aime_2025")    echo "3 0.6" ;;
        "Qwen/Qwen3-"*"|gsm8k")           echo "3 0.6" ;;
        "Qwen/Qwen3-"*"|math500")         echo "3 0.5" ;;
        "Qwen/Qwen3-"*"|gpqa_diamond")    echo "3 0.7" ;;
        "Qwen/Qwen3-"*"|aime_2024")       echo "3 0.4" ;;
        "Qwen/Qwen3-"*"|aime_2025")       echo "3 0.5" ;;
        *) echo "" ;;
    esac
}

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
ORIG_ARGS=("$@")
MODE=beat; ONCE=0; DRY_RUN=0; STATUS_ONLY=0; PLAN_ONLY=0; SMOKE=0; RETRY_FAILED=0
DOWNLOAD_MODEL=""; WORKER_ATTEMPTS=""; WORKER_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --once)         ONCE=1 ;;
        --dry-run)      DRY_RUN=1; ONCE=1 ;;
        --status)       STATUS_ONLY=1; ONCE=1 ;;
        --plan)         PLAN_ONLY=1 ;;
        --plan-tsv)     PLAN_ONLY=2 ;;
        --smoke)        SMOKE=1 ;;
        --retry-failed) RETRY_FAILED=1 ;;
        --setup-env)    MODE=setup_env ;;
        --download)     MODE=download; DOWNLOAD_MODEL="${2:?--download needs a model id}"; shift ;;
        --worker)
            MODE=worker; WORKER_ATTEMPTS="${2:?--worker needs an attempts file}"; shift 2
            [ "${1:-}" = "--" ] && shift
            WORKER_ARGS=("$@"); break ;;
        -h|--help) sed -n '/^# =====/,/^# =====/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
    shift
done
[ "$SMOKE" -eq 1 ] && apply_smoke

# ---------------------------------------------------------------------------
# Repo root + storage (all on Lustre) + logging + journal
# ---------------------------------------------------------------------------
find_repo_root() {
    local dir="$1"
    while [ "$dir" != "/" ]; do
        [ -d "$dir/.git" ] && { printf '%s\n' "$dir"; return 0; }
        dir="$(dirname "$dir")"
    done
    return 1
}
# Slurm copies this script to a spool dir on the compute node, so BASH_SOURCE
# no longer points at the repo once submitted -- prefer SLURM_SUBMIT_DIR (the
# dir where `sbatch` was invoked), exactly like heart_beat.sh; if that is not
# inside the repo, the path the job was submitted with (scontrol "Command=").
START_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_ROOT="$(find_repo_root "$START_DIR")"
if [ -z "${REPO_ROOT:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    submitted="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -oE 'Command=[^ ]+' | cut -d= -f2)"
    [ -n "$submitted" ] && REPO_ROOT="$(find_repo_root "$(dirname "$submitted")")"
fi
if [ -z "${REPO_ROOT:-}" ] || [ ! -f "$REPO_ROOT/scripts/sela_run.py" ]; then
    echo "ERROR: could not locate the SeLaReasoning repo (.git + scripts/sela_run.py) above $START_DIR" >&2
    exit 1
fi
cd "$REPO_ROOT" || exit 1
SELF="$REPO_ROOT/Sela_beat.sh"

ENV_DIR="$SELA_ROOT/env"
ENV_FLAG=".selar_env_complete"
MODEL_STORE="$SELA_ROOT/models"
CACHE_DIR="$SELA_ROOT/cache"
HF_HOME_DIR="$SELA_ROOT/hf_home"
LOG_DIR="$SELA_ROOT/logs"
JOB_LOG_DIR="$LOG_DIR/jobs"
STATE_DIR="$SELA_ROOT/state"
TREE="$SELA_ROOT"                           # runs / journal / plan / status
[ "$SMOKE" -eq 1 ] && TREE="$SELA_ROOT/smoke"
RUNS_DIR="$TREE/runs"
JOURNAL="$TREE/journal.txt"
PLAN_FILE="$TREE/plan.tsv"
STATUS_FILE="$TREE/status.out"
BEAT_LOG="$LOG_DIR/sela_beat$([ "$SMOKE" -eq 1 ] && echo _smoke).log"

log() {
    local msg
    msg="[$(date +'%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    [ "$DRY_RUN" -eq 1 ] || [ "$MODE" != beat ] || echo "$msg" >> "$BEAT_LOG"
}

# ---------------------------------------------------------------------------
# Job modes -- what the Slurm jobs submitted by this script run
# ---------------------------------------------------------------------------
# Python on a marvin GPU node: the AMD EasyBuild tree must come first in
# MODULEPATH (Intel-tree modules die with SIGILL on AMD EPYC), and MKL/oneDNN
# are capped at AVX2 -- see Disentangling-Reasoning/script/model_resources.sh.
load_python_module() {
    type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh
    if grep -q AuthenticAMD /proc/cpuinfo 2>/dev/null; then
        case ":${MODULEPATH:-}:" in
            *":$AMD_MODULEPATH:"*) ;;
            *) export MODULEPATH="$AMD_MODULEPATH:${MODULEPATH:-}" ;;
        esac
    fi
    module load "$PYTHON_MODULE"
}

activate_env() {
    load_python_module || return 1
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate" || return 1
    export MKL_ENABLE_INSTRUCTIONS=AVX2 DNNL_MAX_CPU_ISA=AVX2
}

# Every cache onto Lustre, so nothing grows in the (small) home directory.
export_caches() {
    export HF_HOME="$HF_HOME_DIR" PIP_CACHE_DIR="$CACHE_DIR/pip" XDG_CACHE_HOME="$CACHE_DIR/xdg" \
           TRITON_CACHE_DIR="$CACHE_DIR/triton" TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/inductor" \
           CUDA_CACHE_PATH="$CACHE_DIR/nv" PYTHONPYCACHEPREFIX="$CACHE_DIR/pycache" \
           MPLCONFIGDIR="$CACHE_DIR/mpl" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
           WANDB_DISABLED=true
    mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$PYTHONPYCACHEPREFIX"
}

# One line per job start: "<epoch> <job id> <progress>". A shard counts as
# stalled when its last MAX_STALLED attempts all started from the same progress.
record_attempt() {
    mkdir -p "$(dirname "$1")"
    printf '%s %s %s\n' "$(date +%s)" "${SLURM_JOB_ID:-manual}" "$2" >> "$1"
}

stalled_attempts() {   # $1 attempts file, $2 current progress -> trailing attempts without progress
    [ -f "$1" ] || { echo 0; return; }
    awk -v p="$2" '{ n = ($3 == p) ? n + 1 : 0 } END { print n + 0 }' "$1"
}

job_banner() {
    echo "======================================================================"
    echo " Sela_beat.sh $1 -- job ${SLURM_JOB_ID:-manual} -- $(date +'%F %T')"
    echo "======================================================================"
}

setup_env_main() {
    job_banner "--setup-env"
    record_attempt "$STATE_DIR/env.attempts" 0
    export_caches
    load_python_module || { echo "ERROR: module load $PYTHON_MODULE failed"; return 1; }
    echo "python: $(command -v python3) ($(python3 --version 2>&1))"
    if [ -d "$ENV_DIR" ] && [ ! -f "$ENV_DIR/$ENV_FLAG" ]; then
        case "$ENV_DIR" in
            "$SELA_ROOT"/*) echo "removing incomplete env $ENV_DIR"; rm -rf "$ENV_DIR" ;;
            *) echo "ERROR: refusing to remove $ENV_DIR (outside $SELA_ROOT)"; return 1 ;;
        esac
    fi
    python3 -m venv "$ENV_DIR" || return 1
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate" || return 1
    export MKL_ENABLE_INSTRUCTIONS=AVX2 DNNL_MAX_CPU_ISA=AVX2
    python -m pip install --upgrade pip || return 1
    python -m pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_URL" || return 1
    python -m pip install -r "$REPO_ROOT/requirements-sela.txt" || return 1
    python - <<'PY' || return 1
import torch, transformers, datasets, math_verify, latex2sympy2_extended  # noqa: F401
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| gpu", torch.cuda.is_available())
print("transformers", transformers.__version__, "| datasets", datasets.__version__)
assert torch.cuda.is_available(), "no GPU visible -- --setup-env must run on a GPU node"
x = torch.randn(512, 512, device="cuda")
print("cuda matmul ok:", torch.isfinite(x @ x).all().item(), "on", torch.cuda.get_device_name(0))
PY
    date +'%F %T' > "$ENV_DIR/$ENV_FLAG"
    echo "env ready: $ENV_DIR"
}

# Same file selection as Disentangling-Reasoning/script/download_model.sh.
download_main() {
    local model="$1" dest="$MODEL_STORE/$1"
    job_banner "--download $model"
    record_attempt "$STATE_DIR/download_$(model_tag "$model").attempts" 0
    [ -f "$dest/.download_complete" ] && { echo "ready: $dest"; return 0; }
    export_caches
    activate_env || { echo "ERROR: could not activate $ENV_DIR"; return 1; }
    mkdir -p "$dest"
    MODEL="$model" DEST="$dest" python - <<'PY' || return 1
import os
from huggingface_hub import HfApi, snapshot_download
repo_id, dest = os.environ["MODEL"], os.environ["DEST"]
token = os.environ.get("HF_TOKEN") or None
files = HfApi().list_repo_files(repo_id, token=token)
ignore = ["*.h5", "*.msgpack", "*.onnx", "*.onnx_data", "onnx/*", "*.gguf",
          "*.pth", "original/*", "flax_model*", "tf_model*"]
if any(f.endswith(".safetensors") for f in files):
    ignore.append("*.bin")
snapshot_download(repo_id=repo_id, local_dir=dest, token=token, ignore_patterns=ignore, max_workers=4)
PY
    touch "$dest/.download_complete"
    echo "done: $dest ($(du -sh "$dest" | cut -f1))"
}

worker_main() {
    local partial="${WORKER_ATTEMPTS%.attempts}.partial.jsonl" done=0 left=""
    [ -f "$partial" ] && done=$(wc -l < "$partial")
    record_attempt "$WORKER_ATTEMPTS" "$done"     # first, so even a broken env counts
    job_banner "--worker"
    export_caches
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    activate_env || { echo "ERROR: could not activate $ENV_DIR"; return 1; }
    [ -n "${SLURM_JOB_ID:-}" ] && left="$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null)"
    echo "time left: ${left:-unknown}   resuming from $done finished problems"
    python -u "$REPO_ROOT/scripts/sela_run.py" "${WORKER_ARGS[@]}" ${left:+--time_left "$left"}
}

model_tag() {
    basename "$1" | tr -cd 'A-Za-z0-9' | tr 'A-Z' 'a-z' | cut -c1-12
}

case "$MODE" in
    setup_env) setup_env_main; exit $? ;;
    download)  download_main "$DOWNLOAD_MODEL"; exit $? ;;
    worker)    worker_main; exit $? ;;
esac

# ---------------------------------------------------------------------------
# Validation -- a typo must fail here, not as a crash loop of GPU jobs
# ---------------------------------------------------------------------------
die() { echo "ERROR: $*" >&2; exit 1; }
is_number() { [[ "$1" =~ ^[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?$ ]]; }
# canonical spelling of a knob value, so 0.50 / 0.5 and 1.0 / 1 are one config
canon() {
    if is_number "$1"; then awk -v x="$1" 'BEGIN { printf "%g", x }'; else printf '%s' "$1"; fi
}
in_list() { local x="$1"; shift; local y; for y in "$@"; do [ "$x" = "$y" ] && return 0; done; return 1; }

validate_selection() {
    local m part phase d k v vals=()
    for phase in "${PHASES[@]}"; do in_list "$phase" main sweep || die "unknown phase '$phase'"; done
    for m in "${METHODS[@]}" "${SWEEP_METHODS[@]}"; do
        in_list "$m" "${BASE_METHODS[@]}" "${IDEAS[@]}" || die "unknown method '$m'"
    done
    for m in "${COMBOS[@]}"; do
        [[ "$m" == *+* ]] || die "COMBOS entry '$m' must join >= 2 ideas with '+'"
        for part in ${m//+/ }; do in_list "$part" "${IDEAS[@]}" || die "COMBOS '$m': unknown idea '$part'"; done
        [ "$(tr '+' '\n' <<< "$m" | sort | uniq -d)" = "" ] || die "COMBOS '$m' repeats an idea"
    done
    for d in "${DATASETS[@]}" "${SWEEP_DATASETS[@]}"; do
        [ -n "${DATASET_SIZE[$d]:-}" ] || die "unknown dataset '$d' (known: ${!DATASET_SIZE[*]})"
    done
    [ "${#MODELS[@]}" -gt 0 ] || die "MODELS is empty"
    for v in "$N_SAMPLES" "$SWEEP_N_SAMPLES"; do
        [ "$v" = all ] || [[ "$v" =~ ^[1-9][0-9]*$ ]] || die "N_SAMPLES / SWEEP_N_SAMPLES must be 'all' or a positive integer (got '$v')"
    done
    for v in "${SEEDS[@]}" "${SWEEP_SEEDS[@]}"; do [[ "$v" =~ ^[0-9]+$ ]] || die "bad seed '$v'"; done
    in_list "$SWEEP_MODE" oat grid || die "SWEEP_MODE must be oat or grid"
    in_list "$SELAR_PRESET" paper fixed || die "SELAR_PRESET must be paper or fixed"
    for k in "${!KNOB_FLAG[@]}"; do
        vals=("${!k}")
        if declare -p "${k}_SWEEP" >/dev/null 2>&1; then
            eval "vals+=(\"\${${k}_SWEEP[@]}\")"
        fi
        for v in "${vals[@]}"; do
            if [ -n "${KNOB_CHOICES[$k]:-}" ]; then
                # shellcheck disable=SC2086
                in_list "$v" ${KNOB_CHOICES[$k]} || die "$k: '$v' is not one of: ${KNOB_CHOICES[$k]}"
            elif [ "$k" = SWIR_MAX_SWITCH ]; then
                [ "$v" = none ] || [[ "$v" =~ ^[0-9]+$ ]] || die "SWIR_MAX_SWITCH: '$v' (none | N)"
            else
                is_number "$v" || die "$k: '$v' is not a number"
            fi
        done
    done
}

canonicalize_knobs() {
    local k v vals out
    for k in "${!KNOB_FLAG[@]}"; do
        printf -v "$k" '%s' "$(canon "${!k}")"
        declare -p "${k}_SWEEP" >/dev/null 2>&1 || continue
        eval "vals=(\"\${${k}_SWEEP[@]}\")"
        out=()
        for v in "${vals[@]}"; do out+=("$(canon "$v")"); done
        eval "${k}_SWEEP=(\"\${out[@]}\")"
    done
    TEMPERATURE="$(canon "$TEMPERATURE")"; TOP_P="$(canon "$TOP_P")"; MIN_P="$(canon "$MIN_P")"
}

# ---------------------------------------------------------------------------
# Plan: SELECTION -> runs -> shards
#   PLAN entry: phase|method|tag|knobs|model|dataset|n|seed|run_dir|n_shards|
#               is_default|python args   (is_default: the method's default config)
# ---------------------------------------------------------------------------
# Knobs a method sweeps ("+" combos: the union of their ideas' knobs).
method_knobs() {
    local idea names=""
    case "$1" in
        cot|cot_greedy|cot_upstream) ;;
        selar|selar_upstream) names="SELAR_TOPK SELAR_TAU" ;;
        swir) names="SWIR_ALPHA SWIR_MAX_SWITCH" ;;
        *) for idea in ${1//+/ }; do names+="${IDEA_KNOBS[$idea]} "; done ;;
    esac
    echo "$names"
}

# Knobs a method passes at their current value but never sweeps: the base
# gate of every input-level variant (sem/ltr/fhg/dynk); a variant with
# "layer" gates on the logit lens instead (LAYER_TOPK / LAYER_TAU).
method_fixed_knobs() {
    case "$1" in
        selar) echo "CONTRASTIVE_W SOFT_MASK" ;;
        cot|cot_greedy|cot_upstream|selar_upstream|swir) echo "" ;;
        *layer*) echo "" ;;
        *) echo "SELAR_TOPK SELAR_TAU CONTRASTIVE_W SOFT_MASK" ;;
    esac
}

sweep_values() {   # values of KNOB_SWEEP, or just the default when there is no sweep array
    if declare -p "${1}_SWEEP" >/dev/null 2>&1; then
        local ref="${1}_SWEEP[@]"
        printf '%s\n' "${!ref}"
    else
        printf '%s\n' "${!1}"
    fi
}

# One "K=V K=V ..." line per configuration of the swept knobs.
configs_for() {   # $1 method, $2 default | oat | grid
    local method="$1" mode="$2" k kk v line p
    local -a knobs partial next
    read -r -a knobs <<< "$(method_knobs "$method")"
    if [ "${#knobs[@]}" -eq 0 ]; then echo ""; return; fi
    line=""; for k in "${knobs[@]}"; do line+="$k=${!k} "; done
    case "$mode" in
        default) echo "${line% }" ;;
        oat)
            echo "${line% }"
            for k in "${knobs[@]}"; do
                while IFS= read -r v; do
                    { [ -z "$v" ] || [ "$v" = "${!k}" ]; } && continue
                    line=""
                    for kk in "${knobs[@]}"; do
                        if [ "$kk" = "$k" ]; then line+="$kk=$v "; else line+="$kk=${!kk} "; fi
                    done
                    echo "${line% }"
                done < <(sweep_values "$k")
            done ;;
        grid)
            partial=("")
            for k in "${knobs[@]}"; do
                next=()
                while IFS= read -r v; do
                    [ -n "$v" ] || continue
                    for p in "${partial[@]}"; do next+=("${p:+$p }$k=$v"); done
                done < <(sweep_values "$k")
                partial=("${next[@]}")
            done
            printf '%s\n' "${partial[@]}" ;;
    esac
}

config_tag() {   # method, "K=V ..." -> e.g. sem_k3_t0.6_sk5_ss0.82_sw15_sg1_sf0.3
    local tag="$1" kv k v
    for kv in $2; do
        k="${kv%%=*}"; v="${kv#*=}"
        [ "${KNOB_TAG_DEFAULT[$k]:-}" = "$v" ] && continue
        [ "$k" = LAYER_REL_SCALE ] && [[ " $2 " == *" LAYER_DIRECTION=embed_delta "* ]] && continue
        tag+="_${KNOB_TAG[$k]}$v"
    done
    printf '%s' "$tag"
}

decode_tag() {   # $1 dataset, $2 method
    if [ "$2" = cot_greedy ]; then
        echo "greedy_L${MAX_NEW_TOKENS[$1]}"
    else
        echo "T${TEMPERATURE}_p${TOP_P}_k${TOP_K}_mp${MIN_P}_L${MAX_NEW_TOKENS[$1]}"
    fi
}

add_run() {   # phase method "K=V ..." model dataset n_samples seed is_default
    local phase="$1" method="$2" cfg="$3" model="$4" dataset="$5" nsamp="$6" seed="$7" isdef="$8"
    local size="${DATASET_SIZE[$5]}" n k kv v fixed="" tag run_dir n_shards pyargs
    n="$size"
    [ "$nsamp" != all ] && [ "$nsamp" -lt "$size" ] && n="$nsamp"
    for k in $(method_fixed_knobs "$method"); do fixed+="$k=${!k} "; done
    cfg="${fixed}${cfg}"; cfg="${cfg% }"
    tag="$(config_tag "$method" "$cfg")"
    run_dir="$RUNS_DIR/${model##*/}/$dataset/n$n/$tag/$(decode_tag "$dataset" "$method")/seed$seed"
    n_shards=$(( (n + ${SHARD_SIZE[$dataset]} - 1) / ${SHARD_SIZE[$dataset]} ))
    pyargs="--method $method --model_name $model --model_path $MODEL_STORE/$model"
    pyargs+=" --dataset_name $dataset --n_samples $n --num_shards $n_shards --run_dir $run_dir"
    pyargs+=" --batch_size ${BATCH_SIZE[$dataset]} --max_new_tokens ${MAX_NEW_TOKENS[$dataset]}"
    pyargs+=" --temperature $TEMPERATURE --top_p $TOP_P --top_k $TOP_K --min_p $MIN_P --seed $seed"
    for kv in $cfg; do
        k="${kv%%=*}"; v="${kv#*=}"
        [ "$k" = SWIR_MAX_SWITCH ] && [ "$v" = none ] && continue
        pyargs+=" --${KNOB_FLAG[$k]} $v"
    done
    PLAN+=("$phase|$method|$tag|${cfg:--}|$model|$dataset|$n|$seed|$run_dir|$n_shards|$isdef|$pyargs")
}

PLAN=()
build_plan() {
    PLAN=()
    local phase mode nsamp method model dataset seed cfg preset def_cfg isdef
    local -a methods models datasets seeds
    # base gate per (model, dataset): shadow the globals so configs_for sees them
    local base_k="$SELAR_TOPK" base_t="$SELAR_TAU"
    local SELAR_TOPK SELAR_TAU
    for phase in "${PHASES[@]}"; do
        if [ "$phase" = main ]; then
            mode=default; nsamp="$N_SAMPLES"
            methods=("${METHODS[@]}" "${COMBOS[@]}"); models=("${MODELS[@]}")
            datasets=("${DATASETS[@]}"); seeds=("${SEEDS[@]}")
        else
            mode="$SWEEP_MODE"; nsamp="$SWEEP_N_SAMPLES"
            methods=("${SWEEP_METHODS[@]}"); models=("${SWEEP_MODELS[@]}")
            datasets=("${SWEEP_DATASETS[@]}"); seeds=("${SWEEP_SEEDS[@]}")
        fi
        for method in "${methods[@]}"; do
          for model in "${models[@]}"; do
            for dataset in "${datasets[@]}"; do
              SELAR_TOPK="$base_k"; SELAR_TAU="$base_t"
              if [ "$SELAR_PRESET" = paper ]; then
                  preset="$(selar_preset "$model" "$dataset")"
                  [ -n "$preset" ] && { SELAR_TOPK="${preset% *}"; SELAR_TAU="${preset#* }"; }
              fi
              def_cfg="$(configs_for "$method" default)"
              while IFS= read -r cfg; do
                  isdef=0; [ "$cfg" = "$def_cfg" ] && isdef=1
                  for seed in "${seeds[@]}"; do
                      add_run "$phase" "$method" "$cfg" "$model" "$dataset" "$nsamp" "$seed" "$isdef"
                  done
              done < <(configs_for "$method" "$mode")
            done
          done
        done
    done
}

plan_tsv() {
    printf '# phase\tmethod\tconfig\tknobs\tmodel\tdataset\tn\tseed\trun_dir\tnum_shards\tis_default\n'
    local entry
    local -a f
    for entry in "${PLAN[@]}"; do
        IFS='|' read -r -a f <<< "$entry"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${f[@]:0:11}"
    done
}

write_plan_file() {
    mkdir -p "$TREE"
    plan_tsv > "$PLAN_FILE.tmp.$$" && mv -f "$PLAN_FILE.tmp.$$" "$PLAN_FILE"
}

print_plan() {
    local entry phase method tag knobs model dataset n seed run_dir n_shards isdef rest
    declare -A seen=()
    local runs=0 shards=0 problems=0
    printf '%-6s %-22s %-16s %-13s %5s %5s %7s  %s\n' PHASE METHOD MODEL DATASET N SEED SHARDS CONFIG
    for entry in "${PLAN[@]}"; do
        IFS='|' read -r phase method tag knobs model dataset n seed run_dir n_shards isdef rest <<< "$entry"
        printf '%-6s %-22s %-16s %-13s %5s %5s %7s  %s\n' "$phase" "$method" "${model##*/}" "$dataset" "$n" "$seed" "$n_shards" "$tag"
        [ -n "${seen[$run_dir]:-}" ] && continue
        seen[$run_dir]=1
        runs=$((runs + 1)); shards=$((shards + n_shards)); problems=$((problems + n))
    done
    echo
    echo "total: ${#PLAN[@]} plan entries -> $runs distinct runs, $shards GPU jobs (shards), $problems problem generations"
    echo "runs under: $RUNS_DIR"
}

# ---------------------------------------------------------------------------
# Journal (same format and helpers as heart_beat.sh)
# ---------------------------------------------------------------------------
journal_append() {   # event key [jobid]; no-op in dry-run / status modes
    local event="$1" key="$2" jobid="${3:--}"
    [ "$DRY_RUN" -eq 1 ] && return 0
    [ "$STATUS_ONLY" -eq 1 ] && return 0
    printf '%s | %s | %s | job=%s\n' "$(date +'%Y-%m-%d %H:%M:%S')" "$event" "$key" "$jobid" >> "$JOURNAL"
}
journal_last_line() { grep -F " | $1 | " "$JOURNAL" 2>/dev/null | tail -n 1; }
journal_last_event() {
    local line; line="$(journal_last_line "$1")"
    [ -z "$line" ] && return 0
    awk -F' \\| ' '{print $2}' <<< "$line"
}
journal_last_jobid() {
    grep -F " | $1 | " "$JOURNAL" 2>/dev/null | grep -oE 'job=[0-9]+' | tail -n 1 | cut -d= -f2
}

# ---------------------------------------------------------------------------
# Queue snapshot helpers.  QUEUE holds one line per job: "<id>|<state>|<name>"
# ---------------------------------------------------------------------------
snapshot_queue() {
    # QUEUE_OK distinguishes "command failed" from "queue is genuinely empty":
    # on a squeue hiccup we must NOT treat every running job as gone.
    if QUEUE="$(squeue -u "$SLURM_USER" -h -o '%i|%T|%j' 2>/dev/null)"; then
        QUEUE_OK=1
    else
        QUEUE_OK=0; QUEUE=""
    fi
}
queue_state_of() { awk -F'|' -v id="$1" '$1==id{print $2; exit}' <<< "$QUEUE"; }
queue_live_id_by_name() {
    awk -F'|' -v n="$1" \
        '$3==n && $2 ~ /^(RUNNING|PENDING|CONFIGURING|COMPLETING|RESIZING|REQUEUED|SUSPENDED)$/ {print $1; exit}' \
        <<< "$QUEUE"
}

# Live state of the job tracked under <key>: sets JOB_STATE to RUNNING,
# PENDING or NEED_SUBMIT and JOB_ID, journaling transitions on the way.
# A dead job is journaled once, not again on every pass it stays unresubmitted.
track_job() {
    local key="$1" name="$2" last jobid state live
    last="$(journal_last_event "$key")"
    jobid="$(journal_last_jobid "$key")"
    JOB_STATE=NEED_SUBMIT; JOB_ID="$jobid"
    if [ -n "$jobid" ]; then
        state="$(queue_state_of "$jobid")"
        case "$state" in
            RUNNING)
                JOB_STATE=RUNNING
                if [ "$last" != RUNNING ]; then
                    journal_append RUNNING "$key" "$jobid"
                    log "RUNNING   $key (job $jobid)"
                fi ;;
            PENDING|CONFIGURING|COMPLETING|RESIZING|REQUEUED|SUSPENDED)
                JOB_STATE=PENDING ;;
            "")
                if in_list "$last" SUBMITTED RUNNING ADOPTED; then
                    journal_append DIED "$key" "$jobid"
                    log "DIED      $key (job $jobid gone, not finished)"
                fi ;;
            *)
                if in_list "$last" SUBMITTED RUNNING ADOPTED; then
                    journal_append "$state" "$key" "$jobid"
                    log "ENDED     $key (job $jobid state=$state)"
                fi ;;
        esac
    fi
    # never (re)submit while a live job with this name is queued
    if [ "$JOB_STATE" = NEED_SUBMIT ]; then
        live="$(queue_live_id_by_name "$name")"
        if [ -n "$live" ]; then
            journal_append ADOPTED "$key" "$live"
            log "ADOPTED   $key (job $live already in queue)"
            JOB_ID="$live"
            [ "$(queue_state_of "$live")" = RUNNING ] && JOB_STATE=RUNNING || JOB_STATE=PENDING
        fi
    fi
}

# submit <key> <label> <sbatch args...>  -> SUBMITTED / SUBMIT_FAILED
submit() {
    local key="$1" label="$2" newid; shift 2
    if [ "$STATUS_ONLY" -eq 1 ]; then
        log "NEEDS RUN $label"
        return 1
    elif [ "$DRY_RUN" -eq 1 ]; then
        N_WOULD=$((N_WOULD + 1))
        log "WOULD SUBMIT $label -> sbatch $*"
        return 1
    fi
    # A burst of submissions trips slurmctld's rate limit ("temporarily unable
    # to accept job"), after which sbatch sleeps and retries on its own --
    # pace the calls, and cap a hung one (a job that still got in is ADOPTED
    # by name on the next pass).
    [ "$N_SUBMITTED" -gt 0 ] && sleep "$SUBMIT_PAUSE"
    newid="$(timeout "$SUBMIT_TIMEOUT" sbatch --parsable "$@" 2>>"$BEAT_LOG")"
    newid="${newid%%;*}"   # strip ';cluster' if present
    if [[ "$newid" =~ ^[0-9]+$ ]]; then
        journal_append SUBMITTED "$key" "$newid"
        N_SUBMITTED=$((N_SUBMITTED + 1))
        log "SUBMITTED $label -> job $newid"
        return 0
    fi
    journal_append SUBMIT_FAILED "$key" "-"
    log "ERROR: submit failed for $label (output: '$newid')"
    return 1
}

gave_up() {   # key label attempts-file
    if [ "$(journal_last_event "$1")" != GAVE_UP ]; then
        journal_append GAVE_UP "$1" "$(journal_last_jobid "$1")"
        log "GAVE_UP   $2 -- $MAX_STALLED attempts without progress; see $JOB_LOG_DIR/ and $3, then run --retry-failed"
    fi
}

# ensure_prereq <key> <job name> <ready 0|1> <attempts file> <sbatch args...>
# 0 = ready. Otherwise keeps exactly one job for it in the queue and returns 1.
ensure_prereq() {
    local key="$1" name="$2" ready="$3" attempts="$4"; shift 4
    if [ "$ready" -eq 1 ]; then
        if [ "$(journal_last_event "$key")" != FINISHED ] && [ -n "$(journal_last_line "$key")" ]; then
            journal_append FINISHED "$key" "$(journal_last_jobid "$key")"
            log "FINISHED  $key"
        fi
        return 0
    fi
    track_job "$key" "$name"
    case "$JOB_STATE" in
        RUNNING|PENDING) log "waiting for $key (job $JOB_ID, $JOB_STATE)"; return 1 ;;
    esac
    if [ "$(stalled_attempts "$attempts" 0)" -ge "$MAX_STALLED" ]; then
        gave_up "$key" "$key" "$attempts"
        return 1
    fi
    submit "$key" "$key" --job-name="$name" --output="$JOB_LOG_DIR/%x_%j.out" \
        --error="$JOB_LOG_DIR/%x_%j.out" "$@"
    return 1
}

ensure_env() {
    local ready=0
    [ -f "$ENV_DIR/$ENV_FLAG" ] && ready=1
    ensure_prereq "env::setup" "sx_setup_env" "$ready" "$STATE_DIR/env.attempts" \
        --partition="$ENV_PARTITION" --time=01:00:00 --mem=32G --gres="$GPU_GRES" \
        "$SELF" --setup-env
}

ensure_model() {
    local model="$1" ready=0
    [ -f "$MODEL_STORE/$model/.download_complete" ] && ready=1
    ensure_prereq "download::$model" "sx_dl_$(model_tag "$model")" "$ready" \
        "$STATE_DIR/download_$(model_tag "$model").attempts" \
        --partition="$DOWNLOAD_PARTITION" --time=04:00:00 --mem=32G --cpus-per-task=4 \
        "$SELF" --download "$model"
}

check_workspace() {
    command -v ws_list >/dev/null 2>&1 || return 0
    local info id days
    info="$(ws_list 2>/dev/null | awk -v d="$SELA_ROOT" '
        /^id:/ {id = $2}
        /workspace directory/ {hit = (index(d, $NF) == 1)}
        hit && /remaining time/ {print id, $4; exit}')"
    [ -n "$info" ] || return 0
    id="${info% *}"; days="${info#* }"
    [[ "$days" =~ ^[0-9]+$ ]] || return 0
    if [ "$days" -lt "$WS_WARN_DAYS" ]; then
        log "WARNING: the Lustre workspace '$id' holding $SELA_ROOT expires in $days days -- extend it: ws_extend $id 30"
    fi
}

# ---------------------------------------------------------------------------
# One pass over the whole selection matrix
# ---------------------------------------------------------------------------
run_pass() {
    ALL_DONE=0; PREREQ_WAITING=0
    log "===== Sela_beat pass start (user=$SLURM_USER dry_run=$DRY_RUN status_only=$STATUS_ONLY smoke=$SMOKE) ====="
    check_workspace
    snapshot_queue
    if [ "$QUEUE_OK" -eq 0 ] && [ "$STATUS_ONLY" -eq 0 ]; then
        log "WARNING: squeue unavailable this pass; skipping (will retry next cycle) to avoid duplicate submissions."
        return 0
    fi
    build_plan
    [ "$DRY_RUN" -eq 0 ] && [ "$STATUS_ONLY" -eq 0 ] && write_plan_file

    N_SUBMITTED=0; N_WOULD=0
    local n_total=0 n_finished=0 n_running=0 n_pending=0 n_waiting=0 n_gaveup=0 n_deferred=0
    local entry phase method tag knobs model dataset n seed run_dir n_shards isdef pyargs rest
    local i stem key name attempts progress label
    local -a pyarr
    declare -A done_run=() run_running=() run_pending=() run_stuck=() model_ok=()

    # 0) prerequisites: env, then every model of the plan
    local env_ok=0
    ensure_env && env_ok=1
    for entry in "${PLAN[@]}"; do
        IFS='|' read -r phase method tag knobs model rest <<< "$entry"
        [ -n "${model_ok[$model]:-}" ] && continue
        model_ok[$model]=0
        [ "$env_ok" -eq 1 ] && ensure_model "$model" && model_ok[$model]=1
        [ "${model_ok[$model]}" -eq 1 ] || PREREQ_WAITING=1
    done

    # 1) shards
    for entry in "${PLAN[@]}"; do
        IFS='|' read -r phase method tag knobs model dataset n seed run_dir n_shards isdef pyargs <<< "$entry"
        [ -n "${done_run[$run_dir]:-}" ] && continue          # same run in both phases
        done_run[$run_dir]=1
        run_running[$run_dir]=0; run_pending[$run_dir]=0; run_stuck[$run_dir]=0
        for (( i = 0; i < n_shards; i++ )); do
            n_total=$((n_total + 1))
            stem="$run_dir/shards/$(printf 'shard_%03dof%03d' "$i" "$n_shards")"
            key="shard::${run_dir#"$RUNS_DIR"/}::${i}of${n_shards}"
            label="${model##*/}/$dataset/n$n/$tag/seed$seed shard $((i + 1))/$n_shards"
            if [ -f "$stem.json" ]; then
                n_finished=$((n_finished + 1))
                if [ "$(journal_last_event "$key")" != FINISHED ] && [ -n "$(journal_last_line "$key")" ]; then
                    journal_append FINISHED "$key" "$(journal_last_jobid "$key")"
                    log "FINISHED  $label"
                fi
                continue
            fi
            if [ "${model_ok[$model]}" -ne 1 ]; then
                n_waiting=$((n_waiting + 1))
                continue
            fi
            name="sx_${method:0:6}_${dataset:0:4}_$(model_tag "$model")_$(printf '%s' "$key" | md5sum | cut -c1-6)"
            name="${name//+/-}"
            track_job "$key" "$name"
            case "$JOB_STATE" in
                RUNNING) n_running=$((n_running + 1)); run_running[$run_dir]=$(( run_running[$run_dir] + 1 )); continue ;;
                PENDING) n_pending=$((n_pending + 1)); run_pending[$run_dir]=$(( run_pending[$run_dir] + 1 )); continue ;;
            esac
            attempts="$stem.attempts"
            progress=0
            [ -f "$stem.partial.jsonl" ] && progress=$(wc -l < "$stem.partial.jsonl")
            if [ "$(stalled_attempts "$attempts" "$progress")" -ge "$MAX_STALLED" ]; then
                n_gaveup=$((n_gaveup + 1)); run_stuck[$run_dir]=$(( run_stuck[$run_dir] + 1 ))
                gave_up "$key" "$label" "$attempts"
                continue
            fi
            if [ "$N_SUBMITTED" -ge "$MAX_SUBMIT_PER_PASS" ]; then
                n_deferred=$((n_deferred + 1))
                continue
            fi
            read -r -a pyarr <<< "$pyargs"
            if submit "$key" "$label" --job-name="$name" --partition="$GPU_PARTITION" --time="$GPU_TIME" \
                    --mem="$GPU_MEM" --gres="$GPU_GRES" ${GPU_EXTRA_SBATCH[@]+"${GPU_EXTRA_SBATCH[@]}"} \
                    --output="$JOB_LOG_DIR/%x_%j.out" --error="$JOB_LOG_DIR/%x_%j.out" \
                    "$SELF" --worker "$attempts" -- "${pyarr[@]}" --shard_index "$i"; then
                run_pending[$run_dir]=$(( run_pending[$run_dir] + 1 ))
            fi
        done
    done
    log "summary: shards=$n_total finished=$n_finished running=$n_running pending=$n_pending submitted=$N_SUBMITTED would-submit=$N_WOULD waiting-for-env/model=$n_waiting gave-up=$n_gaveup deferred(cap)=$n_deferred"

    # 2) merge every run whose shards are all there
    local n_merged=0 n_runs=0 n_up_to_date=0 all
    declare -A merged_run=()
    for entry in "${PLAN[@]}"; do
        IFS='|' read -r phase method tag knobs model dataset n seed run_dir n_shards isdef pyargs <<< "$entry"
        [ -n "${merged_run[$run_dir]:-}" ] && continue
        merged_run[$run_dir]=1
        n_runs=$((n_runs + 1))
        all=1
        for (( i = 0; i < n_shards; i++ )); do
            [ -f "$run_dir/shards/$(printf 'shard_%03dof%03d' "$i" "$n_shards").json" ] || { all=0; break; }
        done
        [ "$all" -eq 1 ] || continue
        if [ -f "$run_dir/summary.json" ] && \
           [ -z "$(find "$run_dir/shards" -name 'shard_*.json' -newer "$run_dir/summary.json" 2>/dev/null | head -n 1)" ]; then
            n_up_to_date=$((n_up_to_date + 1))
            continue
        fi
        if [ "$DRY_RUN" -eq 1 ] || [ "$STATUS_ONLY" -eq 1 ]; then
            log "WOULD MERGE ${run_dir#"$RUNS_DIR"/}"
        elif "$SYS_PYTHON" -B "$REPO_ROOT/scripts/sela_tools.py" merge --run_dir "$run_dir" \
                --num_shards "$n_shards" >> "$BEAT_LOG" 2>&1; then
            n_merged=$((n_merged + 1)); n_up_to_date=$((n_up_to_date + 1))
            journal_append MERGED "run::${run_dir#"$RUNS_DIR"/}"
            log "MERGED    ${model##*/}/$dataset/n$n/$tag/seed$seed"
        else
            log "ERROR: merge failed for $run_dir (see $BEAT_LOG)"
        fi
    done
    [ "$n_merged" -gt 0 ] && log "merged $n_merged runs"
    ALL_DONE=0
    [ "$n_runs" -gt 0 ] && [ "$n_up_to_date" -eq "$n_runs" ] && ALL_DONE=1

    # 3) status table
    local status_tsv status_out err_log="$BEAT_LOG"
    [ "$DRY_RUN" -eq 1 ] && err_log=/dev/null
    status_tsv="$(mktemp)"
    declare -A listed=()
    for entry in "${PLAN[@]}"; do
        IFS='|' read -r phase method tag knobs model dataset n seed run_dir n_shards isdef pyargs <<< "$entry"
        [ -n "${listed[$run_dir]:-}" ] && continue
        listed[$run_dir]=1
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$phase" "$method" "$tag" "$model" \
            "$dataset" "$n" "$seed" "$run_dir" "$n_shards" "${run_running[$run_dir]:-0}" \
            "${run_pending[$run_dir]:-0}" "${run_stuck[$run_dir]:-0}" >> "$status_tsv"
    done
    status_out="$STATUS_FILE"
    [ "$DRY_RUN" -eq 1 ] && status_out="$(mktemp)"
    [ "$DRY_RUN" -eq 1 ] || mkdir -p "$TREE"
    if "$SYS_PYTHON" -B "$REPO_ROOT/scripts/sela_tools.py" status --plan "$status_tsv" \
            --out "$status_out" --interval "$(( CHECK_INTERVAL / 60 ))" 2>>"$err_log"; then
        if [ "$ONCE" -eq 1 ]; then cat "$status_out"; fi
        [ "$DRY_RUN" -eq 1 ] || log "status -> $status_out"
    fi
    rm -f "$status_tsv"
    [ "$DRY_RUN" -eq 1 ] && rm -f "$status_out"

    # 4) the report
    if [ "$AUTO_REPORT" -eq 1 ] && [ "$DRY_RUN" -eq 0 ] && [ "$STATUS_ONLY" -eq 0 ]; then
        SELA_ROOT="$SELA_ROOT" bash "$REPO_ROOT/reporting.sh" >> "$BEAT_LOG" 2>&1 \
            && log "report -> $REPO_ROOT/result/selar_report.md" \
            || log "WARNING: reporting.sh failed (see $LOG_DIR/reporting.log)"
    fi
    log "===== Sela_beat pass end ====="
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
validate_selection
canonicalize_knobs

if [ "$PLAN_ONLY" -ne 0 ]; then
    build_plan
    if [ "$PLAN_ONLY" -eq 1 ]; then print_plan; else plan_tsv; fi
    exit 0
fi

if [ "$DRY_RUN" -eq 0 ]; then
    mkdir -p "$TREE" "$LOG_DIR" "$JOB_LOG_DIR" "$STATE_DIR" "$MODEL_STORE" || {
        echo "ERROR: cannot create directories under $SELA_ROOT" >&2; exit 1; }
    [ -f "$JOURNAL" ] || printf '# Sela_beat journal -- <ts> | <EVENT> | <key> | job=<id>\n' > "$JOURNAL"
fi

if [ "$RETRY_FAILED" -eq 1 ] && [ "$DRY_RUN" -eq 0 ]; then
    n_reset=0
    while IFS= read -r f; do
        rm -f "$f"; n_reset=$((n_reset + 1))
    done < <(find "$RUNS_DIR" "$STATE_DIR" -name '*.attempts' 2>/dev/null)
    journal_append RESET "retry-failed"
    log "--retry-failed: reset $n_reset attempt counters"
fi

REINIT_AT=$(( TIME_LIMIT_SECONDS - REINIT_MARGIN ))   # ~23h50m into the job

# Flags the next instance inherits -- not --retry-failed, or every re-init
# would reset the stall counters and a crashing shard would never be given up.
FORWARD_ARGS=()
for a in ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}; do
    [ "$a" = "--retry-failed" ] || FORWARD_ARGS+=("$a")
done

# One line per (re-)initialization recording the config this instance runs with.
log "INIT Sela_beat job=${SLURM_JOB_ID:-standalone} root=$SELA_ROOT tree=$TREE phases=(${PHASES[*]}) methods=(${METHODS[*]}) combos=(${COMBOS[*]}) models=(${MODELS[*]}) datasets=(${DATASETS[*]}) seeds=(${SEEDS[*]}) n=$N_SAMPLES sweep=(${SWEEP_METHODS[*]}|${SWEEP_DATASETS[*]}|n=$SWEEP_N_SAMPLES|$SWEEP_MODE) preset=$SELAR_PRESET partition=$GPU_PARTITION interval=${CHECK_INTERVAL}s reinit_at=${REINIT_AT}s once=$ONCE dry_run=$DRY_RUN status_only=$STATUS_ONLY"

reinitialize() {
    # Hand off to a fresh instance before the wall clock ("initialize itself
    # again"). Under Slurm: submit a new Sela_beat job. Standalone: re-exec.
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        log "Elapsed ${SECONDS}s (~$(( SECONDS / 3600 ))h) -> submitting a fresh Sela_beat job."
        local newid
        newid="$(sbatch --parsable "$SELF" ${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"} 2>>"$BEAT_LOG")"
        log "Re-initialized as Slurm job ${newid:-<unknown>}. Exiting."
        exit 0
    else
        log "Elapsed ${SECONDS}s; re-initializing a fresh process (picks up edits)."
        exec "$SELF" ${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"}
    fi
}

# Under Slurm this job runs the copy of the script made at submission time.
# When Sela_beat.sh in the repo has been edited since (e.g. its SELECTION),
# hand over to a fresh job running the new version -- if that validates.
edited_and_valid() {
    [ -n "${SLURM_JOB_ID:-}" ] || return 1
    cmp -s "$0" "$SELF" && return 1
    if bash -n "$SELF" 2>>"$BEAT_LOG" && \
       SELA_ROOT="$SELA_ROOT" bash "$SELF" ${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"} --plan >/dev/null 2>>"$BEAT_LOG"; then
        log "Sela_beat.sh was edited -> handing over to a fresh job running the new version."
        return 0
    fi
    log "WARNING: Sela_beat.sh was edited but does not validate (see $BEAT_LOG) -- keeping the running version."
    return 1
}

# --status / --dry-run / --once: a single pass.
if [ "$ONCE" -eq 1 ]; then
    run_pass
    exit 0
fi

# Main loop: a pass every CHECK_INTERVAL, re-initializing near the 24h limit
# (or as soon as the script has been edited).
while true; do
    edited_and_valid && reinitialize
    run_pass
    if [ "$EXIT_WHEN_DONE" -eq 1 ] && [ "$ALL_DONE" -eq 1 ]; then
        log "every planned run is merged (EXIT_WHEN_DONE=1) -> exiting."
        exit 0
    fi

    remaining=$(( REINIT_AT - SECONDS ))
    if [ "$remaining" -le 0 ]; then
        reinitialize
    fi

    # Sleep one interval, but not past the re-init point.
    nap="$CHECK_INTERVAL"
    [ "$PREREQ_WAITING" -eq 1 ] && [ "$PREREQ_INTERVAL" -lt "$nap" ] && nap="$PREREQ_INTERVAL"
    [ "$remaining" -lt "$nap" ] && nap="$remaining"
    log "next check in ${nap}s (elapsed ${SECONDS}s, re-init at ${REINIT_AT}s)"
    sleep "$nap"
done

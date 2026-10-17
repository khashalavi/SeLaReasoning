# Environment setup

Nothing has to be installed by hand. On its first pass `Sela_beat.sh` submits
the environment build, then one download per model, and holds back all GPU
work until both are ready. This page explains what those jobs do and how to
redo them.

## The Python environment

| | |
|---|---|
| Location | `$SELA_ROOT/env` (Lustre) |
| Built by | `Sela_beat.sh --setup-env`, submitted as job `sx_setup_env` on `ENV_PARTITION` (`sgpu_devel`, 1 GPU, 32G, 1 h) |
| Base | module `Python/3.11.3-GCCcore-12.3.0` from the **AMD** EasyBuild tree |
| Contents | `torch==2.7.0` from the CUDA 12.8 index, then `requirements-sela.txt` |
| Check | imports torch, transformers, datasets, math_verify; a matmul on the GPU |
| Done marker | `$SELA_ROOT/env/.selar_env_complete` |
| Measured | 2 min 53 s on 2026-10-17; reported GPU: NVIDIA A100-SXM4-80GB |

`requirements-sela.txt` pins exactly what the CPU tests pass with:
`transformers==4.51.0`, `tokenizers==0.21.4`, `accelerate==1.12.0`,
`datasets==4.4.1`, `huggingface_hub==0.36.2`, `safetensors==0.8.0`,
`numpy==2.4.6`, `math-verify[antlr4_13_2]==0.9.0`,
`latex2sympy2_extended==1.11.0` and a few more. Transformers stays at 4.51.0
because the decoding loops rely on its `DynamicCache` API.

!!! warning "Why the build must run on a GPU node"
    All GPU nodes of marvin are AMD EPYC; the login node is Intel. Modules
    from the Intel EasyBuild tree crash with "Illegal instruction" on AMD. A
    venv records the interpreter it was created with. So the venv is created
    from the AMD tree's Python on a GPU node, and every job that uses it
    first:

    1. prepends `/opt/software/easybuild-AMD/modules/all` to `MODULEPATH`
       when `/proc/cpuinfo` says AMD,
    2. loads `Python/3.11.3-GCCcore-12.3.0`,
    3. exports `MKL_ENABLE_INSTRUCTIONS=AVX2 DNNL_MAX_CPU_ISA=AVX2`.

    This mirrors Disentangling-Reasoning's `new_env_a100`. The consequence:
    the venv **does not run on the login node**.

### Rebuilding

```bash
rm /lustre/scratch/data/s76kalav_hpc-memory/selareasoning/env/.selar_env_complete
```

The next pass submits `--setup-env` again. The job removes the incomplete
`env/` (only below `$SELA_ROOT`) and builds it from scratch. Change
`TORCH_VERSION`, `TORCH_INDEX_URL` or `PYTHON_MODULE` in `Sela_beat.sh`
first if needed.

## Model weights

| | |
|---|---|
| Location | `$SELA_ROOT/models/<org>/<name>/` in plain Hugging Face layout |
| Downloaded by | `Sela_beat.sh --download MODEL`, job `sx_dl_<model tag>` on `DOWNLOAD_PARTITION` (`sgpu_short`, CPU only, 32G, 4 CPUs, 4 h) |
| Files | everything except alternative formats (`*.h5`, `*.msgpack`, `*.onnx*`, `*.gguf`, `*.pth`, `original/`, `flax_model*`, `tf_model*`), and no `*.bin` when safetensors exist |
| Done marker | `.download_complete` inside the model folder |
| Measured | Qwen3-0.6B: 953 MB in 24 s. Qwen3-8B is about 16 GB |

A model is downloaded once and then loaded offline by every worker
(`HF_HUB_OFFLINE=1`). The Qwen3 and DeepSeek models are public; for gated
models, export `HF_TOKEN` in the shell that submits `Sela_beat.sh`.

To download again, delete the model folder (or only its `.download_complete`).

## Caches

Every job points all caches at Lustre, so nothing grows in the home directory:

| Variable | Location |
|---|---|
| `HF_HOME` | `$SELA_ROOT/hf_home` |
| `PIP_CACHE_DIR` | `$SELA_ROOT/cache/pip` |
| `PYTHONPYCACHEPREFIX` | `$SELA_ROOT/cache/pycache` (no `__pycache__` in the repository) |
| `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR` | `$SELA_ROOT/cache/triton`, `…/inductor` |
| `XDG_CACHE_HOME`, `CUDA_CACHE_PATH`, `MPLCONFIGDIR` | `$SELA_ROOT/cache/xdg`, `…/nv`, `…/mpl` |

The stdlib-only helpers (merge, status, report) run with the system
`/usr/bin/python3` (3.9) and `-B`, i.e. without writing bytecode.

## A CPU environment for the tests (optional)

The tests and the documentation build run on the login node. They need a
separate venv from the Intel tree's Python with the CPU build of torch:

```bash
source /etc/profile.d/modules.sh
module load Python/3.11.3-GCCcore-12.3.0
python3 -m venv /lustre/scratch/data/s76kalav_hpc-memory/selareasoning/cpu-env
source /lustre/scratch/data/s76kalav_hpc-memory/selareasoning/cpu-env/bin/activate
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-sela.txt pytest zensical
```

Then [run the tests](manual-runs.md#tests) or `zensical serve` for this site.

## The Lustre workspace

`/lustre/scratch/data/s76kalav_hpc-memory` is a workspace (id `memory`) with
an expiry date. On 2026-10-17 it had 47 days left (expires 2026-12-03) and
3 extensions. Every pass checks `ws_list` and logs a warning when fewer than
`WS_WARN_DAYS` (10) remain:

```bash
ws_list                 # remaining time
ws_extend memory 30     # extend by 30 days
```

An expired workspace takes the environment, the models **and all results**
with it.

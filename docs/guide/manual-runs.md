# Manual runs and tests

Everything the heart beat does can be done by hand: one shard, a merge, the
status table, the report. That is useful for debugging a single method or
trying a configuration outside the plan.

## One shard as a Slurm job

The heart beat submits shards as `Sela_beat.sh --worker`, which records the
attempt, activates the environment and runs `scripts/sela_run.py`. The same
works by hand:

```bash
cd ~/SeLaReasoning
S=/lustre/scratch/data/s76kalav_hpc-memory/selareasoning
RUN=$S/runs/manual/sem_try/seed41          # any directory; keep manual runs apart

sbatch --job-name=sx_manual --partition=sgpu_short --time=02:00:00 --mem=64G --gres=gpu:1 \
       --output=$S/logs/jobs/%x_%j.out \
       Sela_beat.sh --worker $RUN/shards/shard_000of001.attempts -- \
       --method sem --model_name Qwen/Qwen3-8B --model_path $S/models/Qwen/Qwen3-8B \
       --dataset_name aime_2025 --n_samples 30 --num_shards 1 --shard_index 0 \
       --run_dir $RUN --batch_size 15 --max_new_tokens 38912 --seed 41 \
       --topk 3 --tau 0.5 --sem_sim 0.85
```

Every flag of `sela_run.py` is listed in the [reference](../reference/sela-run.md).
Flags that are not given keep the Python defaults.

!!! tip "Manual runs and the report"
    The report reads `plan.tsv` for its tables. A manual run outside the plan
    appears only in the CSV `reports/all_runs.csv`. To get it into the tables,
    add its configuration to the SELECTION instead; the heart beat then finds
    the finished shards and does not run them again, as long as the run
    directory is the one the plan expects.

## Interactively on a GPU node

```bash
srun --partition=sgpu_devel --gres=gpu:1 --mem=64G --time=01:00:00 --pty bash
source /etc/profile.d/modules.sh
export MODULEPATH=/opt/software/easybuild-AMD/modules/all:$MODULEPATH
module load Python/3.11.3-GCCcore-12.3.0
source /lustre/scratch/data/s76kalav_hpc-memory/selareasoning/env/bin/activate
export MKL_ENABLE_INSTRUCTIONS=AVX2 DNNL_MAX_CPU_ISA=AVX2 HF_HUB_OFFLINE=1 \
       PYTHONPYCACHEPREFIX=/lustre/scratch/data/s76kalav_hpc-memory/selareasoning/cache/pycache
cd ~/SeLaReasoning
python scripts/sela_run.py --method dynk --model_name Qwen/Qwen3-0.6B \
    --model_path /lustre/scratch/data/s76kalav_hpc-memory/selareasoning/models/Qwen/Qwen3-0.6B \
    --dataset_name gsm8k --n_samples 4 --run_dir /lustre/scratch/data/s76kalav_hpc-memory/selareasoning/runs/manual/dynk_try \
    --batch_size 4 --max_new_tokens 512
```

## On the CPU

`--device cpu` runs the worker on the login node, given the CPU environment
from [setup](setup.md#a-cpu-environment-for-the-tests-optional). A real 8B model is far
too slow there. A tiny random model with the real tokenizer still exercises
the prompt, chat template, every method, grading, resuming and the output
format. Accuracy is 0, of course.

```python
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

src = "tok_src"
for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
          "generation_config.json", "config.json"):
    hf_hub_download("Qwen/Qwen3-0.6B", f, local_dir=src)
tok = AutoTokenizer.from_pretrained(src)
cfg = Qwen3Config(vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                  tie_word_embeddings=True, use_sliding_window=False, sliding_window=None,
                  eos_token_id=tok.eos_token_id)
Qwen3ForCausalLM(cfg).save_pretrained("qwen3-tiny")
tok.save_pretrained("qwen3-tiny")
```

```bash
python scripts/sela_run.py --method sem+dynk --model_name Qwen/Qwen3-0.6B --model_path qwen3-tiny \
    --dataset_name gsm8k --n_samples 4 --num_shards 2 --shard_index 0 \
    --batch_size 2 --max_new_tokens 16 --device cpu \
    --run_dir /lustre/scratch/data/s76kalav_hpc-memory/selareasoning/runs/manual/cpu_try
```

## Tests

```bash
python -m pytest -q -p no:cacheprovider tests/test_selarx.py     # 36 tests, ~11 s on the CPU
```

`tests/test_selarx.py` builds a tiny random Qwen3 (96-token vocabulary, 4
layers). Its output layer is scaled ×16, so about half of the steps are above
τ = 0.5 and both gate branches are exercised. Nothing is downloaded. The
tests check:

| Group | Tests |
|---|---|
| upstream equivalence | `cot` = `generate_cot`; `selar` + `batch_any` = `generate_selar` on a batch; per-row `selar` = `generate_selar` at batch size 1 (three rows) |
| reductions | layer α = 0 → CoT (lens and final gate); fhg θ = 10⁹ → CoT; fhg θ = −1 → SeLaR; ltr without horizon/hysteresis → SeLaR; dynk with k fixed → SeLaR; final gate fires like SeLaR |
| behaviour | layer steering changes the output (2 gates × 2 directions); ltr never exceeds M; hysteresis lengthens rollouts; the running threshold is recorded; semantic normalisation by log M |
| every method | the five ideas and four combinations with the lens gate, plus `layer`, `layer+ltr` and the five-way combination with the final gate: they run, the per-problem counters are consistent, the hooks are removed afterwards |
| building blocks | semantic entropy without merges, with merges, single transitive cluster; nucleus k; rejected method names and knob values |

## Merge, status and report by hand

```bash
S=/lustre/scratch/data/s76kalav_hpc-memory/selareasoning
python3 -B scripts/sela_tools.py merge --run_dir <run dir> --num_shards <n>
./Sela_beat.sh --status                      # status table from the journal and the queue
./reporting.sh                               # merges finished runs of the plan, writes the report
```

All three need only the system `python3`, so they also run on the login
node.

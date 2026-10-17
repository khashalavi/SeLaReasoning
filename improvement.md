# Research Blueprint: Next-Generation Training-Free Latent Reasoning (SeLaR Extensions)

This document provides a comprehensive, actionable execution plan for developing, implementing, and evaluating training-free extensions to **SeLaR (Selective Latent Reasoning)**.

Because these methods operate entirely during inference via PyTorch forward hooks and logit interventions, **no GPU backpropagation, model fine-tuning, or RL gradient updates are required**. All experiments can be executed on 1–2 consumer/workstation GPUs (e.g., RTX 4090 or A6000).

## 1. Experimental Rig & Implementation Foundation

### 1.1 Base Environment & Tooling

* **Framework:** PyTorch $\ge 2.2$, Hugging Face `transformers`, `accelerate`, `vllm` (for fast evaluation where applicable, though standard HF hooks are recommended for latent interventions).

* **Target Foundation Models:**

  * Primary: `Qwen/Qwen2.5-7B-Instruct` / `Qwen3-8B`

  * Secondary (Uncalibrated test): `DeepSeek-R1-Distill-Llama-8B`

* **Benchmark Suite:**

  * *Math & Symbolic:* GSM8K, MATH500, AIME 2024, AIME 2025

  * *Knowledge & STEM:* GPQA-Diamond

* **Evaluation Metrics:**

  1. **Accuracy (**$\%$**):** Final exact match / box-extracted answer accuracy.

  2. **Tokens Per Correct Answer (TPCA):**
     

     $$
     \text{TPCA} = \frac{\alpha \cdot T_c + (1 - \alpha) \cdot T_w}{\alpha}
     $$

     
     where $\alpha$ is accuracy, $T_c$ is average tokens on correct samples, and $T_w$ is average tokens on wrong samples.

  3. **Exploratory Step Activation Ratio (**$\%$**):** Percentage of total decoding steps where latent reasoning was triggered.

## 2. Idea 1: Layer-Wise Latent Steering (Hidden-State SeLaR)

### 2.1 Theoretical Motivation & Novelty

SeLaR constructs soft embeddings at the input embedding layer $E$. Modifying $E$ forces the perturbed vector through all shallow Transformer layers (0 to $L$), which can introduce semantic dispersion or dampening before reaching the middle/deep reasoning layers.

**Novelty:** Intercepting and applying selective contrastive steering **directly inside intermediate Transformer hidden layers** ($h^l$ at layer $l \in [L/3, 2L/3]$), enabling direct manipulation of the model's internal representation trajectory.

### 2.2 Mathematical Formulation

1. **Hidden State Extraction:** At step $t$, pass the sequence through the model up to layer $l^* = \lfloor \frac{L}{2} \rfloor$. Extract hidden state $h_t^{l^*} \in \mathbb{R}^{d}$.

2. **Logit Lens Projection:** Project $h_t^{l^*}$ through the unembedding head $W_U$ to estimate intermediate token distribution $p_t^{l^*}$:
   

   $$
   p_t^{l^*} = \text{softmax}(W_U \cdot \text{RMSNorm}(h_t^{l^*}))
   $$

3. **Layer Uncertainty Measurement:** Calculate truncated top-$k$ entropy $H_t^{l^*}$ over the top-$k$ tokens at layer $l^*$:
   

   $$
   \bar{H}_t^{l^*} = \frac{-\sum_{v \in \mathcal{V}_k} \hat{p}_t^{l^*}(v) \log \hat{p}_t^{l^*}(v)}{\log k}
   $$

4. **Layer-Steering Injection:** If $\bar{H}_t^{l^*} > \tau$, construct a continuous steering direction vector at layer $l^*$:
   

   $$
   \Delta_t^{l^*} = h_t^{l^*} - E(v_{t, \text{top1}}^{l^*})
   $$

   $$
   \tilde{h}_t^{l^*} = h_t^{l^*} + \bar{H}_t^{l^*} \cdot \frac{\Delta_t^{l^*}}{\Vert{}\Delta_t^{l^*}\Vert{}_2 + \epsilon} \cdot \Vert{}\Delta_t^{l^*}\Vert{}_2
   $$

5. Feed $\tilde{h}_t^{l^*}$ into layer $l^* + 1$ for the remaining forward pass.

### 2.3 Step-by-Step Implementation Guide

```
                      [ Input Sequence ]
                              │
                    Forward Layers 0 to l*
                              │
                   Extract Hidden State h_t^{l*}
                              │
                   Apply Logit Lens (W_U)
                              │
                   Compute Layer Entropy H_t^{l*}
                              │
                    Is H_t^{l*} > Threshold τ?
                     /                  \
                YES /                    \ NO
                   /                      \
   Apply Contrastive Steering       Pass Unmodified h_t^{l*}
   \tilde{h}_t^{l*} to Layer l*+1    to Layer l*+1
                   \                      /
                    \                    /
                  Forward Layers l*+1 to L
                              │
                  Sample Next Output Token

```

1. **PyTorch Hook Strategy:** Register a forward hook on `model.model.layers[l_star]`.

2. **Hook Execution:**

   * Intercept the layer output tensor `hidden_states`.

   * Check if decoding is in auto-regressive generation mode (sequence length = 1).

   * Apply logit lens projection using the model's `lm_head` and pre-unembedding layer norm.

   * Calculate top-$k$ normalized entropy.

   * If entropy exceeds $\tau$, modify the `hidden_states` in-place before returning to layer $l^* + 1$.

### 2.4 Hyperparameter Sweep & Ablation Matrix

| 

| **Variable** | **Search Range** | **Default** | 
| **Injection Layer (**$l^*$**)** | $[L/4, L/3, L/2, 2L/3]$ | $L/2$ (e.g., Layer 14 for 28-layer model) | 
| **Entropy Threshold (**$\tau$**)** | $[0.3, 0.4, 0.5, 0.6, 0.7]$ | $0.5$ | 
| **Steering Factor (**$\alpha$**)** | $[0.5, 1.0, 1.5, 2.0]$ | $1.0$ | 
| **Top-**$k$ **Candidates** | $[2, 3, 5]$ | $3$ | 

## 3. Idea 2: Semantic-Aware Adaptive Entropy Gating (Sem-SeLaR)

### 3.1 Theoretical Motivation & Novelty

Standard token-level entropy treats synonymous or formatting variations (e.g., `" 3"`, `"3"`, `"Three"`, `" 3."`) as distinct competing tokens, inflating calculated entropy. This causes false-positive activations on uncalibrated models (e.g., DeepSeek-R1-Distill-Llama-8B).

**Novelty:** Clustering top-$k$ candidate tokens by semantic equivalence before computing entropy, combined with a **dynamic running-window threshold** ($\tau_t$) that adapts to the model's baseline uncertainty over the reasoning chain.

### 3.2 Mathematical Formulation

#### Step A: Fast Semantic Clustering

1. Extract top-$k$ candidate tokens $\{v_1, v_2, \dots, v_k\}$ and their soft probabilities $\{p_1, p_2, \dots, p_k\}$.

2. Compute pairwise cosine similarity between candidate input embeddings: 

   $$
   S_{ij} = \frac{E(v_i) \cdot E(v_j)}{\Vert{}E(v_i)\Vert{}_2 \Vert{}E(v_j)\Vert{}_2}
   $$

3. Group tokens into semantic clusters $\mathcal{C}_m$ if $S_{ij} > \theta_{\text{sim}}$ (e.g., $\theta_{\text{sim}} = 0.82$).

4. Compute aggregated cluster probabilities: 

   $$
   P(\mathcal{C}_m) = \sum_{v_i \in \mathcal{C}_m} p_i
   $$

#### Step B: Semantic Entropy Calculation

$$
H_t^{\text{sem}} = -\sum_{m=1}^{M} P(\mathcal{C}_m) \log P(\mathcal{C}_m)
$$

 Normalize $H_t^{\text{sem}}$ by $\log M$, where $M$ is the number of active clusters ($M \le k$). If all top tokens collapse into 1 semantic cluster, $H_t^{\text{sem}} = 0$.

#### Step C: Dynamic Running-Window Thresholding

Instead of a static global threshold $\tau$, maintain a sliding history of the last $W$ steps ($W=15$): 

$$
\mu_t = \frac{1}{W} \sum_{i=1}^{W} H_{t-i}^{\text{sem}}, \quad \sigma_t = \sqrt{\frac{1}{W} \sum_{i=1}^{W} (H_{t-i}^{\text{sem}} - \mu_t)^2}
$$

$$
\tau_t = \mu_t + \gamma \cdot \sigma_t \quad (\text{where } \gamma \in [0.5, 1.5])
$$

Activate latent soft reasoning if $H_t^{\text{sem}} > \tau_t$.

### 3.3 Step-by-Step Implementation Guide

1. Precompute a normalized embedding lookup matrix $E_{\text{norm}} = \frac{E}{\Vert{}E\Vert{}_2}$ during initialization.

2. At decoding step $t$:

   * Extract logits, compute `topk(logits, k=5)`.

   * Retrieve $5 \times d$ embeddings from $E_{\text{norm}}$.

   * Compute $5 \times 5$ dot product matrix.

   * Apply Connected Components / Union-Find to group tokens exceeding threshold $\theta_{\text{sim}}$.

   * Sum probabilities per cluster and compute $H_t^{\text{sem}}$.

   * Update deque `history_entropy.append(H_t_sem)`.

   * Compare $H_t^{\text{sem}}$ with dynamic threshold $\tau_t$.

### 3.4 Hyperparameter Sweep & Ablation Matrix

| **Variable** | **Search Range** | **Default** | 
| **Similarity Threshold (**$\theta_{\text{sim}}$**)** | $[0.75, 0.80, 0.85, 0.90]$ | $0.82$ | 
| **Window Size (**$W$**)** | $[5, 10, 15, 20]$ | $15$ | 
| **Std Multiplier (**$\gamma$**)** | $[0.5, 1.0, 1.5]$ | $1.0$ | 

## 4. Idea 3: Multi-Step Latent Trajectory Rollouts (LTR-SeLaR)

### 4.1 Theoretical Motivation & Novelty

SeLaR executes a **1-step soft embedding intervention** before reverting to discrete token generation. However, complex mathematical and logic forks often require 2–4 consecutive reasoning steps to fully resolve ambiguity.

**Novelty:** Extending latent exploration across an adaptive multi-step horizon ($M \in [2, 4]$ steps) in continuous space whenever high entropy is detected, evaluating branch stability before committing back to discrete text.

### 4.2 Mathematical Formulation

1. **Trigger Condition:** At step $t$, if $H_t > \tau$, initiate a continuous rollout for $M$ steps ($m = 1, 2, \dots, M$).

2. **Soft Propagation Loop:**

   * For $m = 1 \dots M$: Construct regularized soft embedding $\bar{e}_{t+m-1}$: 

     $$
     \bar{e}_{t+m-1} = e_{t+m-1}^{\text{soft}} + \bar{H}_{t+m-1} \cdot \hat{\Delta}_{t+m-1} \cdot \Vert{}\Delta_{t+m-1}\Vert{}_2
     $$

      Feed $\bar{e}_{t+m-1}$ into the model to obtain logits for step $t+m$.

3. **Trajectory Collapse Criterion:**

   * Terminate continuous rollout early if $\bar{H}_{t+m} < \tau_{\text{exit}}$ (uncertainty resolved).

   * Otherwise, at step $t+M$, force a hard discrete token commit by taking $\arg\max(p_{t+M})$ or sampling from the final soft distribution.

### 4.3 Step-by-Step Implementation Guide

```
                 [ High Entropy Detected at Step t ]
                                 │
                   Initialize Rollout Counter m = 1
                                 │
                 ┌──────────────────────────────┐
                 │ Construct Soft Vector \bar{e}│
                 │ Forward Pass -> Get Logits   │
                 └──────────────┬───────────────┘
                                │
                   Compute New Entropy H_{t+m}
                                │
               Is H_{t+m} < \tau_exit OR m == M_max?
                /                                \
           YES /                                  \ NO
              /                                    \
   [ Collapse to Discrete Token ]             [ Increment m = m + 1 ]
   Resume Standard CoT Decoding               Continue Soft Forward Loop

```

1. Maintain a state flag `in_latent_rollout = False` and counter `rollout_step = 0`.

2. When $H_t > \tau$, set `in_latent_rollout = True`.

3. During `in_latent_rollout`:

   * Construct top-$k$ soft embedding from logits.

   * Apply contrastive regularization.

   * Pass custom embedding directly into `model(inputs_embeds=regularized_soft_embed)`.

   * Update KV-cache for the soft token step.

   * If step entropy drops below $\tau_{\text{exit}}$ or `rollout_step == M_max`, set `in_latent_rollout = False` and resume standard discrete token generation.

### 4.4 Hyperparameter Sweep & Ablation Matrix

| **Variable** | **Search Range** | **Default** | 
| **Max Rollout Horizon (**$M_{\text{max}}$**)** | $[2, 3, 4, 5]$ | $3$ | 
| **Exit Threshold (**$\tau_{\text{exit}}$**)** | $[0.15, 0.20, 0.25]$ | $0.20$ | 
| **Contrastive Decay Rate** | $[0.8, 0.9, 1.0]$ (Decay regularization over rollout) | $0.9$ | 

## 5. Idea 4: Functional Head Gating (Fact vs. Logic SeLaR)

### 5.1 Theoretical Motivation & Novelty

SeLaR improves multi-step math tasks (AIME) significantly, but shows flat or slightly negative gains on knowledge-intensive benchmarks like GPQA-Diamond. Continuous soft embeddings blur sharp factual lookup, while aiding procedural logic.

**Novelty:** Dynamically distinguishing between **Procedural/Logic Uncertainty** (where soft exploration helps) and **Factual/Recall Uncertainty** (where soft exploration hurts) using feed-forward network (FFN) vs. Attention activation norms.

### 5.2 Mathematical Formulation

1. **Activation Trait Extraction:** At step $t$, collect the intermediate activation norm ratio between Attention layers and Feed-Forward Networks (FFN) at the mid-to-late layers ($l \in [L/2, L]$): 

   $$
   R_{\text{act}} = \frac{\sum_{l=L/2}^{L} \Vert{}\mathbf{a}_{\text{attn}}^l\Vert{}_2}{\sum_{l=L/2}^{L} \Vert{}\mathbf{a}_{\text{FFN}}^l\Vert{}_2}
   $$

2. **Functional Gating:**

   * **High** $R_{\text{act}}$ **(Attention-Dominant):** Indicates active information routing and multi-step relational reasoning (Procedural Step).

   * **Low** $R_{\text{act}}$ **(FFN-Dominant):** Indicates parametric memory recall and factual extraction (Factual Step).

3. **Selective Gate:** 

   $$
   \text{Activate SeLaR} \iff (H_t > \tau) \quad \text{AND} \quad (R_{\text{act}} > \theta_{\text{reasoning}})
   $$

### 5.3 Step-by-Step Implementation Guide

1. Register forward hooks on `self_attn` output and `mlp` output for layers $L/2$ to $L$.

2. Compute L2 norm of `attn_output` and `mlp_output` per token step.

3. Compute step ratio $R_{\text{act}}$.

4. Evaluate combined trigger condition:

   * If both entropy is high AND $R_{\text{act}} > \theta_{\text{reasoning}}$, trigger contrastive soft embedding.

   * If entropy is high BUT $R_{\text{act}} \le \theta_{\text{reasoning}}$, fall back to standard discrete greedy/sampling decoding to preserve factual precision.

### 5.4 Hyperparameter Sweep & Ablation Matrix

| **Variable** | **Search Range** | **Default** | 
| **Reasoning Threshold (**$\theta_{\text{reasoning}}$**)** | $[0.8, 1.0, 1.2, 1.4]$ | $1.0$ | 
| **Monitored Layers** | $[L/2 \dots L, \text{All Layers}]$ | Layers $L/2$ to $L$ | 

## 6. Idea 5: Nucleus Dynamic-$K$ Soft Embedding (DynK-SeLaR)

### 6.1 Theoretical Motivation & Novelty

SeLaR uses a static top-$k$ parameter ($k=3$). However, candidate distributions vary across steps: some steps have 2 closely competing tokens (50%/50%), while others have 6 plausible options (15% each). A static $k=3$ either misses valid candidates or includes low-probability noise.

**Novelty:** Dynamically selecting $k_t$ per step using **Nucleus Cumulative Mass Thresholding** combined with top-$p$ probability bounds before constructing the regularized soft embedding.

### 6.2 Mathematical Formulation

1. Sort logits at step $t$ in descending order: $p_{(1)} \ge p_{(2)} \ge \dots \ge p_{(V)}$.

2. Compute cumulative probability mass: 

   $$
   F(m) = \sum_{j=1}^{m} p_{(j)}
   $$

3. Determine dynamic $k_t$: 

   $$
   k_t = \min \{ m \in \mathbb{N} \mid F(m) \ge p_{\text{mass}} \}
   $$

    Enforce bounds: $k_t \in [k_{\min}, k_{\max}]$ (e.g., $k_{\min}=2, k_{\max}=8$).

4. Renormalize probabilities over the dynamic set $\mathcal{V}_{k_t}$ and compute regularized soft embedding $\bar{e}_t$.

### 6.3 Step-by-Step Implementation Guide

1. At step $t$, compute `probs = softmax(logits, dim=-1)`.

2. Compute cumulative sum: `sorted_probs, sorted_indices = torch.sort(probs, descending=True)`.

3. Compute `cum_probs = torch.cumsum(sorted_probs, dim=-1)`.

4. Create mask: `mask = cum_probs - sorted_probs > p_mass`.

5. Set `k_t = clamp(first_true_index, min=k_min, max=k_max)`.

6. Construct soft embedding over top $k_t$ tokens.

### 6.4 Hyperparameter Sweep & Ablation Matrix

| **Variable** | **Search Range** | **Default** | 
| **Cumulative Mass (**$p_{\text{mass}}$**)** | $[0.80, 0.85, 0.90, 0.95]$ | $0.90$ | 
| **Min Candidates (**$k_{\min}$**)** | $[2]$ | $2$ | 
| **Max Candidates (**$k_{\max}$**)** | $[5, 8, 10]$ | $8$ | 

## 7. Master Comparison & Decision Matrix for Publication

Execute all 5 variants under the exact same benchmark suite and hardware environment. Populate the master evaluation table to select the winning configuration for paper submission:

| **Method Variant** | **GSM8K Acc (%)** | **MATH500 Acc (%)** | **AIME25 Acc (%)** | **GPQA Acc (%)** | **Avg Acc (%)** | **TPCA Improvement vs. CoT (%)** | **Activation Ratio (%)** | 
| **Baseline 1: Standard CoT** |  |  |  |  |  | 0.0% | 0.0% | 
| **Baseline 2: Vanilla SeLaR** |  |  |  |  |  |  | \~10.0% | 
| **Variant 1: Layer-Wise Steering (Hidden-State)** |  |  |  |  |  |  |  | 
| **Variant 2: Semantic Adaptive Gating (Sem-SeLaR)** |  |  |  |  |  |  |  | 
| **Variant 3: Multi-Step Rollouts (LTR-SeLaR)** |  |  |  |  |  |  |  | 
| **Variant 4: Functional Head Gating (Fact/Logic)** |  |  |  |  |  |  |  | 
| **Variant 5: Nucleus Dynamic-**$K$ **(DynK-SeLaR)** |  |  |  |  |  |  |  | 
| **Variant 6: Combined Top-2 (Best 1 + Best 2)** |  |  |  |  |  |  |  | 

## 8. Strategy for Scientific Paper Writing & Submission

Once the empirical results are populated:

1. **If Variant 1 (Layer-Wise Steering) wins:** Frame the paper as:

   > *"Steering LLM Thoughts from Within: Training-Free Intermediate Latent Reasoning in Transformer Hidden Spaces"*

   * **Core Pitch:** Input embedding modifications are lossy; hidden-state steering inside mid-transformer layers unlocks direct control over latent reasoning chains.

2. **If Variant 2 (Semantic Adaptive Gating) wins:** Frame the paper as:

   > *"Calibrated Latent Exploration: Semantic Entropy and Adaptive Thresholding for Robust Training-Free LLM Reasoning"*

   * **Core Pitch:** Standard entropy over-triggers on token surface-forms; semantic clustering stabilizes latent reasoning across both calibrated and uncalibrated LLM families.

3. **If Variant 3 (Multi-Step Rollouts) wins:** Frame the paper as:

   > *"Latent Trajectory Rollouts: Continuous Multi-Step Exploration for Complex Logical Deduction"*

   * **Core Pitch:** Single-step soft intervention is a local smoother; multi-step rollouts enable continuous lookahead search in continuous space without training.

4. **If Variant 4 (Functional Head Gating) wins:** Frame the paper as:

   > *"Knowing When to Speculate: Task-Adaptive Latent Reasoning via Attention-FFN Activation Routing"*

   * **Core Pitch:** Solves the core trade-off between procedural logic exploration and rigid factual recall.
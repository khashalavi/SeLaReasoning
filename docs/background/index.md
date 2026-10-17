# SeLaR in one page

*Fu & Luo, "SeLaR: Selective Latent Reasoning in Large Language Models",
ACL 2026.* The PDF is in `paper/2026.acl-long.320.pdf`.

## The problem

Chain-of-thought (CoT) decoding commits to one discrete token at every step
and discards the rest of the distribution. Training-free *latent reasoning*
methods such as Soft Thinking feed the model a probability-weighted mixture of
token embeddings instead. The paper names two weaknesses of doing that at
**every** step:

1. **Global activation** perturbs the many steps where the model is already
   sure, which destabilises the reasoning chain.
2. **Premature collapse**: soft embeddings drift towards the highest-probability
   token, so the "superposition" quickly degenerates into greedy decoding
   (Wu et al., 2025).

## The observation behind it

Over a CoT trace, the normalised top-k entropy has a **long tail**. Most steps
are near zero (*deterministic*), and a small tail is uncertain (*exploratory*).
For Qwen3-8B, at a threshold of 0.2:

| | GSM8K | MATH500 | GPQA | AIME24 | AIME25 |
|---|---:|---:|---:|---:|---:|
| deterministic steps | 80.91% | 81.74% | 73.60% | 80.72% | 82.66% |
| exploratory steps | 19.09% | 18.26% | 26.40% | 19.28% | 17.34% |

*(paper, Figure 1)*

## The method

### 1. Entropy over the top-k candidates

Let \(p_t\) be the next-token distribution after temperature and top-k / top-p
filtering, and \(\mathcal V_k\) its \(k\) most probable tokens. Renormalise over
them and take the entropy, normalised to \([0, 1]\):

$$
\hat p_t(v) = \frac{p_t(v)}{\sum_{u \in \mathcal V_k} p_t(u)}, \qquad
H_t = -\sum_{v \in \mathcal V_k} \hat p_t(v) \log \hat p_t(v), \qquad
\bar H_t = \operatorname{clamp}\!\left(\frac{H_t}{\log k},\, 0,\, 1\right)
$$

### 2. Entropy gate

$$
e_{t} =
\begin{cases}
E[x_t] & \text{if } \bar H_t < \tau \quad\text{(deterministic step: the sampled token)}\\[4pt]
\sum_{v \in \mathcal V_k} \hat p_t(v)\, E[v] & \text{if } \bar H_t \ge \tau \quad\text{(exploratory step: soft embedding)}
\end{cases}
$$

A token \(x_t\) is **always sampled and written to the output**, so the trace
stays readable. On exploratory steps, only the model's *next input* is soft.

### 3. Entropy-aware contrastive regularisation

The soft embedding is pushed away from the dominant token \(v^\ast\),
proportionally to the uncertainty:

$$
\Delta_t = e_t - E[v^\ast], \qquad
\hat\Delta_t = \frac{\Delta_t}{\lVert \Delta_t \rVert + \epsilon}, \qquad
\tilde e_t = e_t + \bar H_t \cdot \hat\Delta_t \cdot \lVert \Delta_t \rVert
$$

So \(\tilde e_t = e_t + \bar H_t\,(e_t - E[v^\ast])\): at \(\bar H_t \to 1\) the
mixture moves as far again *away* from the top-1 token as it already is.

!!! note "Not in the paper: the math-symbol guard"
    The official code forces a discrete input whenever the **sampled** token is
    a math symbol (digits, letters, operators, LaTeX commands; see
    `get_math_symbols_ids()` in `src/generation_utils.py`). SeLaR-X keeps this
    guard for every method (`math_guard=True`).

## Settings

| | |
|---|---|
| Models | Qwen3-1.7B, Qwen3-8B, Qwen3-32B; DeepSeek-R1-Distill-Llama-8B in the appendix |
| Decoding | temperature 0.6, top-p 0.95, top-k 20, min-p 0, thinking mode on |
| Budget | 32,768 new tokens (38,912 for AIME) |
| Hardware | 4× NVIDIA RTX PRO 6000, Hugging Face Transformers |
| SeLaR | k = 3 and a **per-benchmark** τ (Appendix B): GSM8K 0.6, MATH500 0.5, GPQA 0.7, AIME24 0.4, AIME25 0.5 |
| Prompt | `{question}\nPlease reason step by step, and make sure put your final answer within \boxed{}.` |

## Results

Accuracy (%), Table 1. Column order as in the paper:

| Qwen3-8B | GSM8K | MATH500 | GPQA | AIME24 | AIME25 | Avg |
|---|---:|---:|---:|---:|---:|---:|
| CoT (sampling) | 95.45 | **98.00** | 61.62 | 76.67 | 66.67 | 79.68 |
| CoT (greedy) | 95.22 | 96.20 | 55.05 | 70.00 | 63.33 | 75.96 |
| Soft Thinking | 94.92 | 95.80 | 57.58 | 70.00 | 66.67 | 76.99 |
| SwiR | 95.68 | 97.00 | **62.63** | 60.00 | 66.67 | 76.40 |
| **SeLaR** | **95.83** | 97.00 | 61.62 | **83.33** | **80.00** | **83.56** |

| Average over the five benchmarks | CoT (sampling) | SwiR | SeLaR |
|---|---:|---:|---:|
| Qwen3-1.7B | 60.96 | 57.94 | 61.71 |
| Qwen3-8B | 79.68 | 76.40 | 83.56 |
| Qwen3-32B | 82.38 | 84.30 | 84.83 |
| DeepSeek-R1-Distill-Llama-8B | 57.76 | 59.28 | 60.53 |

Most of the gain comes from AIME: on Qwen3-8B, +6.66 on AIME24 and +13.33 on
AIME25. Those are 2 and 4 problems out of 30.

### Ablations and sensitivity (Qwen3-8B)

| | GSM8K | MATH500 | GPQA | AIME24 | AIME25 | Avg |
|---|---:|---:|---:|---:|---:|---:|
| SeLaR (full, Table 1) | 95.83 | 97.00 | 61.62 | 83.33 | 80.00 | 83.56 |
| without selective activation | 95.14 | 95.80 | 57.58 | 76.67 | 66.67 | 78.37 |
| without contrastive regularisation | 94.92 | 96.20 | 57.58 | 70.00 | 60.00 | 75.74 |
| k = 3, τ = 0.5 everywhere (Table 3) | 95.60 | 97.00 | 55.05 | 76.67 | 80.00 | 80.86 |
| k = 5, τ = 0.5 everywhere (Table 3) | 95.30 | 96.40 | 61.62 | 76.67 | 53.33 | 76.66 |

Table 1 equals the per-benchmark best row of Table 3. With a single setting the
average gain over CoT shrinks from +3.88 to +1.18, as discussed in
[Fair comparisons](fair-comparison.md).

### How often the gate fires

| Exploratory steps | GSM8K | MATH500 | GPQA | AIME24 | AIME25 |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B (Fig. 5) | 10.4% | 9.6% | 6.2% | 13.8% | 10.2% |
| DeepSeek-R1-Distill-Llama-8B (Fig. 7) | 14.9% | 13.0% | 8.8% | 15.3% | 11.8% |

On average about one token in ten is exploratory. The less calibrated R1
distill fires more often and gains less.

### Efficiency: tokens per correct answer

The paper's headline cost metric, with accuracy \(\alpha\) and the average
token counts \(T_c\) / \(T_w\) of correct / wrong answers:

$$
\mathrm{TPCA} = \frac{\alpha\, T_c + (1-\alpha)\, T_w}{\alpha}
= \frac{\text{average tokens per problem}}{\text{accuracy}}
$$

SeLaR lowers TPCA against CoT on AIME24 by 19.2%, while SwiR raises it by
33.2%. On GPQA SeLaR's TPCA is 9.7% higher than CoT's.

### Why contrastive regularisation works (logit lens)

On 200 branching steps of AIME 2025, the authors compare the top-10
logit-lens tokens at every layer of a soft-input forward pass with those of
the top-1 and top-2 tokens fed alone. Without regularisation, the overlap with
top-1 rises from ~0.45 to ~0.73 in deep layers while the overlap with top-2
stays at ~0.40 (collapse). With regularisation, both stay comparable
(~0.55–0.60): two trajectories coexist.

## Limitations the paper names, and the ideas that answer them

| Limitation (paper §Limitations) | Answered by |
|---|---|
| Operates only in the token-embedding space; "future work should explore the hidden state space" | [Variant 1 · Layer steering](../methods/layer.md) |
| Less confident models trigger too many exploratory steps; "confidence-aware activation mechanisms or signals beyond entropy" | [Variant 2 · Semantic gating](../methods/sem.md), [Variant 4 · Functional gating](../methods/fhg.md) |
| One-step soft intervention (implicit in Algorithm 1) | [Variant 3 · Latent rollouts](../methods/ltr.md) |
| Fixed k = 3 for every step (Table 3: larger k dilutes the mixture) | [Variant 5 · Dynamic k](../methods/dynk.md) |
| Flat or negative gains on knowledge-heavy GPQA | [Variant 4 · Functional gating](../methods/fhg.md) |

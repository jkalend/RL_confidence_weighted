<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# 8. Confidence-Weighted Curriculum Learning for Synthetic RL

Target Internship: Test-Time Reinforcement Learning with Synthetic Data
Proposal: When a small (3B-8B) model generates artificial labels for entity classification, the quality of that synthetic supervision is highly variable. Instead of applying Policy Optimization (PO) to all synthetic data equally, this project would implement a Curriculum Learning approach. You would first extract the generative model's confidence scores (or self-consistency metrics) for each synthetic label. The RL loop would begin by adapting the model using only the highest-confidence synthetic data, gradually introducing lower-confidence examples as training progresses. The empirical assessment would measure whether this curriculum prevents the model from unlearning real-world patterns when adapting to synthetic noise.

make a thorough plan of implementation of this... no need to specify weeks

This is a comprehensive implementation plan for the **Confidence-Weighted Curriculum Learning for Synthetic RL** project. It is structured to be executed during a research internship, focusing on the technical integration of uncertainty quantification, curriculum design, and reinforcement learning.

### **Phase 1: Problem Definition \& Environment Setup**

**Objective:** Adapt a 3B-8B LLM to a new distribution (test-time) using its own high-confidence predictions, while mitigating the "collapsing" or "forgetting" effects of training on noisy synthetic data.

**1.1. Model \& Data Selection**

* **Model:** **Llama-3.1-8B-Instruct** or **Qwen-2.5-7B-Instruct**. These are state-of-the-art for this size and have strong instruction-following capabilities.
* **Task:** **Fine-grained Entity Classification** (e.g., Few-NERD or CrossNER). These datasets have hierarchical types, making "noise" (getting the sub-type wrong) common and measurable.
* **Data Split:**
    * *Source Domain (Pre-trained/Base capability):* General NER (e.g., CoNLL-03).
    * *Target Domain (Test-time adaptation):* Specialized domain (e.g., Biomedical or Legal NER).
    * *Calibration Set:* A small held-out set of labeled source data to calibrate confidence scores.

**1.2. Infrastructure Stack**

* **Training Framework:** `trl` (Transformer Reinforcement Learning) library by Hugging Face + `accelerate`.
* **Inference/Generation:** `vLLM` for high-throughput synthetic data generation.
* **Tracking:** `wandb` for monitoring KL divergence, reward trends, and curriculum stages.

***

### **Phase 2: Synthetic Data Pipeline \& Uncertainty Quantification**

**Goal:** Generate the "Target" dataset and annotate it with confidence scores to build the curriculum.

**2.1. Synthetic Label Generation**

* **Input:** Unlabeled sentences from the Target Domain.
* **Prompting:** Use Few-Shot Chain-of-Thought (CoT) prompting to encourage reasoning before labeling.
    * *Example:* "Extract entities... First, reason about the context... Then output JSON."
* **Generation:** For each input $x$, generate $K$ candidate outputs $\{y_1, ..., y_K\}$ (e.g., $K=8$) using a non-zero temperature ($T=0.7$).

**2.2. Confidence Metric Extraction (The "Weight")**
Implement two distinct metrics to compare:

1. **Sequence Probability (LogProb):**
    * Calculate the average log-probability of the generated label tokens.
    * *Pros:* Fast, built-in. *Cons:* Poorly calibrated for hallucinations.
2. **Semantic Entropy (Self-Consistency):**
    * Cluster the $K$ generations based on semantic equivalence (e.g., exact match of the extracted entity list).
    * The confidence score $C(x)$ is the proportion of generations belonging to the largest cluster (Majority Vote).
    * *Note:* This is the "gold standard" for reliability in reasoning tasks.[^1]

**2.3. Dataset Construction**
Create a master dataset `D_syn` where each entry is `(Input, Pseudo-Label, Confidence_Score)`.

* *Pseudo-Label* is the majority vote output.
* *Confidence_Score* is the consistency percentage (e.g., 0.8 if 8/10 matches).

***

### **Phase 3: Curriculum Design**

**Goal:** Design the scheduler that dictates *which* data the RL loop sees at step $t$.

**3.1. Pacing Functions**
Define a pacing function $g(t)$ that returns the percentage of data to use at training step $t$.

* **Step 1 (High Precision):** Start with top 10% confident data ($g(0) = 0.1$).
* **Step 2 (Expansion):** Linearly or logarithmically increase to include lower confidence data.
    * *Linear:* $g(t) = \min(1.0, 0.1 + \lambda \cdot t)$
    * *Root:* $g(t) = \min(1.0, \sqrt{t^2 + \text{start\_bias}})$
* **Filtering:** At step $t$, filter `D_syn` to keep samples where $Confidence\_Score \ge Percentile(1 - g(t))$.

**3.2. Dynamic vs. Static Curriculum**

* *Static (Easier/Baseline):* Pre-compute the bins. Epoch 1 uses Top-20%. Epoch 2 uses Top-40%.
* *Dynamic (Advanced):* Re-evaluate confidence during training. As the model adapts, its confidence on "hard" examples might change. (Start with Static for the MVP).

***

### **Phase 4: RL Implementation (Policy Optimization)**

**Goal:** Implement the RL loop that updates the policy $\pi_\theta$ using the curriculum-filtered data.

**4.1. Algorithm Selection**
Use **PPO (Proximal Policy Optimization)** or **GRPO (Group Relative Policy Optimization)**.

* *Why RL?* Unlike SFT, RL allows us to penalize the model less for "near misses" if we design the reward right, or strictly optimize for the *process* of following the synthetic constraints while maintaining KL constraints to the base model.
* *Correction for "Test-Time":* In this context, "RL" often implies treating the high-confidence pseudo-label as the ground truth reward signal.

**4.2. The Loop**

1. **Sample:** Draw a batch $B$ from the current curriculum slice of `D_syn`.
2. **Act:** Current policy $\pi_\theta$ generates responses for inputs in $B$.
3. **Reward:** Compute reward $R$.
    * *Sparse Reward:* +1 if generated response matches the `Pseudo-Label` exactly, 0 otherwise.
    * *Soft Reward:* Token-level overlap (F1 score) with the `Pseudo-Label`.
4. **Update:** PPO step to maximize expected reward.
    * **Crucial Constraint:** **KL Divergence Penalty**. Add a penalty $-\beta \log(\pi_\theta / \pi_{ref})$ where $\pi_{ref}$ is the original base model. This is the primary mechanism to prevent "unlearning".[^2]

**4.3. Anti-Forgetting Regularization (The "Unlearning" Check)**

* **Replay Buffer:** Mix in 5-10% real data (Source Domain) into every batch.
* **EWC (Elastic Weight Consolidation):** If Replay is not allowed (strict test-time setting), calculate Fisher Information on the base model and penalize changing important weights. (Start with KL penalty; it's standard in PPO).

***

### **Phase 5: Evaluation \& Metrics**

**Goal:** Prove that the curriculum approach is superior to naive training.

**5.1. Baselines**

1. **Zero-Shot:** Base model performance on Target Domain without adaptation.
2. **Naive SFT:** Fine-tune on *all* synthetic data (ignoring confidence).
3. **Naive RL:** PPO on *all* synthetic data (ignoring confidence).
4. **Threshold-Only:** Train only on Top-X% (no curriculum, just filtering).

**5.2. Metrics**

* **Adaptation Quality (Target Domain):** F1 Score on the *labeled* ground truth of the Target Domain (which the model never saw during training).
* **Forgetting (Source Domain):** F1 Score on the original Source Domain (e.g., CoNLL-03). A drop here indicates catastrophic forgetting.
* **Stability:** KL Divergence between $\pi_{final}$ and $\pi_{initial}$.

**5.3. Success Criteria**
The project is successful if the **Curriculum Method** achieves:

1. Higher Target F1 than Naive RL (indicating robustness to noise).
2. Higher Source F1 than Naive RL (indicating reduced forgetting).

### **Execution Timeline (Mental Check)**

* **Sprint 1:** Data generation (Llama-3 inference) + Confidence scoring code.
* **Sprint 2:** Implement PPO loop with `trl` + Static Curriculum (Epoch-based data loading).
* **Sprint 3:** Run baselines (Naive SFT, Naive RL) vs. Curriculum.
* **Sprint 4:** Analysis (Unlearning plots) + Report writing.


### **Methodological Nuances \& Pitfalls**

* **Reward Hacking:** The model might learn to output empty strings or short nonsense if the reward function isn't robust. Ensure the reward checks for valid JSON/Entity formatting.
* **Confidence Calibration:** 3B/8B models are often overconfident. Temperature scaling or simple calibration (Platt scaling) on the small calibration set might be necessary before sorting data for the curriculum.[^3]
* **Compute Limits:** Calculating "Self-Consistency" requires $K$ forward passes. If compute is tight, fall back to "LogProb of the top-1 generation" but normalize it by sequence length.
<span style="display:none">[^10][^11][^12][^13][^14][^15][^4][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://arxiv.org/abs/2510.01251

[^2]: https://arxiv.org/html/2501.13669v2

[^3]: https://arxiv.org/abs/2404.15993

[^4]: https://arxiv.org/abs/2509.03581

[^5]: https://openreview.net/forum?id=VuVhgEiu20

[^6]: https://www.databricks.com/blog/tao-using-test-time-compute-train-efficient-llms-without-labeled-data

[^7]: https://cs224r.stanford.edu/projects/pdfs/CS224R_Final_Report___Faierman__Jia__Hsu.pdf

[^8]: https://intoai.pub/p/test-time-reinforcement-learning

[^9]: https://arxiv.org/html/2410.13674v3

[^10]: https://aclanthology.org/2024.acl-long.77.pdf

[^11]: https://syncedreview.com/2024/07/01/achieving-8x-performance-gains-with-reinforcement-learning-on-synthetic-data-in-large-language-models/

[^12]: https://arxiv.org/abs/2504.09710

[^13]: https://aclanthology.org/anthology-files/pdf/emnlp/2025.emnlp-main.544.pdf

[^14]: https://cs224r.stanford.edu/projects/pdfs/CS224R_Final_Project__1_1.pdf

[^15]: https://arxiv.org/html/2509.01213v1


# Project Summary: Confidence-Weighted Curriculum Learning for Synthetic RL

## Overview

This project implements a **test-time reinforcement learning** pipeline for adapting a 3B–8B language model to a new domain (fine-grained entity classification) using its own synthetic labels. The key innovation is a **confidence-weighted curriculum**: the RL loop starts with only the highest-confidence pseudo-labels and gradually introduces lower-confidence examples, aiming to reduce catastrophic forgetting and improve robustness to noisy synthetic supervision.

## Why These Implementation Steps Were Taken

### 1. Unsloth + QLoRA + GRPO

- **Unsloth**: Chosen for 2–4x faster training and ~90% lower memory use than standard TRL + Flash Attention. Essential for fitting an 8B model on a single RTX 4090 (24GB VRAM).
- **QLoRA (4-bit)**: Enables training 7B–8B models on consumer GPUs by quantizing the base model to 4-bit while training only low-rank LoRA adapters.
- **GRPO (Group Relative Policy Optimization)**: Preferred over PPO because it does not require a separate critic or reference model in memory. GRPO computes advantages within each group of generations, reducing VRAM and simplifying the setup.

### 2. Self-Consistency for Confidence

- **Semantic entropy (self-consistency)** was implemented as the main confidence metric: for each input, we generate K outputs and take the proportion of generations in the largest cluster as the confidence score.
- This is more reliable than sequence log-probability for reasoning-style tasks, as log-probs are often poorly calibrated and can be high for hallucinations.

### 3. Static Curriculum

- A **static curriculum** was used: pre-computed bins (e.g., top 10%, 20%, 40%, …) per epoch. The model trains on expanding subsets of data ordered by confidence.
- Dynamic curriculum (re-evaluating confidence during training) was deferred as a future improvement.

### 4. Soft Reward + Format Penalty

- **Soft reward**: Token-level F1 overlap between the model output and the pseudo-label, to allow partial credit for near-correct answers.
- **Format reward**: Bonus for valid JSON entity output to discourage reward hacking (e.g., empty or malformed outputs).

### 5. Source vs. Target Evaluation

- **Target F1**: Adaptation quality on the target domain (e.g., Few-NERD).
- **Source F1**: Forgetting on the source domain (e.g., CoNLL-03). A drop here indicates catastrophic forgetting.

## What Could Be Improved

1. **Dynamic curriculum**: Recompute confidence as the model adapts; “hard” examples may become easier over time.
2. **Replay buffer**: Mix 5–10% real source-domain data into each batch to further reduce forgetting.
3. **Calibration**: Apply temperature scaling or Platt scaling on a small calibration set to improve confidence calibration.
4. **vLLM for generation**: Use vLLM for synthetic label generation to speed up the data pipeline.
5. **Larger K**: Increase the number of generations per input (e.g., K=16) for more stable self-consistency estimates.
6. **Baselines**: Add explicit runs for Zero-Shot, Naive SFT, Naive RL, and Threshold-Only for quantitative comparison.

## What Could Be Changed

1. **Confidence metric**: Compare or combine self-consistency with log-probability or other uncertainty metrics.
2. **Pacing function**: Experiment with root, logarithmic, or step-based schedules instead of linear.
3. **Reward design**: Try sparse (exact match) vs. soft (F1) rewards, or add length/format penalties.
4. **Model choice** (generation only; training uses smaller models). Unsloth preferred for quality:
   - `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen3-8B-Instruct`, `Qwen/Qwen3.5-27B`, `meta-llama/Llama-3.1-8B-Instruct`: Via Unsloth (default).
   - `zai-org/GLM-4.7-Flash` (30B), `openai/gpt-oss-20b`: Use Unsloth for best results; Ollama optional via `--backend ollama`.
5. **Dataset**: Switch from Few-NERD to CrossNER or another fine-grained NER benchmark.

## What the Results Mean

- **Higher Target F1 than Naive RL**: The curriculum helps the model adapt to the target domain more effectively, likely by reducing exposure to low-quality pseudo-labels early in training.
- **Higher Source F1 than Naive RL**: The curriculum reduces catastrophic forgetting, as the model sees more reliable supervision first and is less destabilized by noisy data.
- **KL divergence**: Monitoring KL to the reference model indicates how far the policy has drifted; very high KL may signal overfitting or collapse.
- **Curriculum stage plots**: Tracking reward and F1 across curriculum stages shows whether the expansion from high- to low-confidence data is beneficial or harmful.

## File Structure

| Path | Purpose |
|------|---------|
| `src/config.py` | Configuration dataclasses |
| `src/data/generate_synthetic.py` | Synthetic label generation with self-consistency, incremental save, resume, multi-model |
| `src/data/confidence.py` | Confidence metrics (self-consistency, logprob) |
| `src/curriculum/scheduler.py` | Pacing function and curriculum filtering |
| `src/reward.py` | GRPO reward functions (format + overlap) |
| `src/train.py` | GRPO training with curriculum stages |
| `src/evaluate.py` | Source/Target F1 evaluation |
| `scripts/run_*.py` | Entry points for generate, train, evaluate |
| `scripts/download_models.py` | Pre-download models to HuggingFace cache |
| `run_pipeline.py` | End-to-end pipeline |

## Running the Project

```powershell
# 1. Install (PyTorch cu130 + requirements)
.\scripts\install.ps1

# 1b. Optional: pre-download models (no GPU)
python scripts/download_models.py
# Or: python scripts/download_models.py Qwen/Qwen3.5-27B zai-org/GLM-4.7-Flash
# Or: python scripts/download_models.py --all

# 2. Generate synthetic data (or use mock for testing)
python scripts/run_generate.py --max-samples 500
# Larger models (Unsloth): --model Qwen/Qwen3.5-27B or --model zai-org/GLM-4.7-Flash
# python scripts/create_mock_synthetic.py  # no GPU needed

# 3. Train
python scripts/run_train.py --epochs 3

# 4. Evaluate
python scripts/run_evaluate.py
```

## Synthetic Generation: Crash Resilience and Multi-Model

The generation script saves after each sample, so a crash loses at most one sample. Use `--resume` to continue from existing output.

Resume skips `(input, model)` pairs already in the file, enabling two workflows:

- **Same inputs, different models**: Run Qwen3 for 300 samples, then Llama with `--resume` for the same 300 → 600 records (300 inputs × 2 models).
- **Different inputs per model**: Run Qwen3 for 300, then Llama with `--resume` and `--max-samples 600` → 600 records (first 300 from Qwen, next 300 from Llama).

Each record includes a `model` field so you can filter or compare by model downstream.

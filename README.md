# Confidence-Weighted Curriculum Learning for Synthetic RL

Test-time RL adaptation using synthetic entity labels with a confidence-based curriculum. Uses GRPO + Unsloth + QLoRA for memory-efficient training on a single RTX 4090 (24GB VRAM).

## Setup

```powershell
# Create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA 13.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Install dependencies
pip install -r requirements.txt
```

Or run `.\scripts\install.ps1` from project root.

## Usage

### 1. Generate synthetic labels

```powershell
python scripts/run_generate.py --model Qwen/Qwen2.5-7B-Instruct --max-samples 500
```

**Crash resilience**: Output is saved after each sample. If the run crashes, use `--resume` to continue:

```powershell
python scripts/run_generate.py --max-samples 500 --resume
```

**Multi-model in one file**: Resume skips `(input, model)` pairs already present, so you can mix models:

```powershell
# 300 from Qwen3
python scripts/run_generate.py --model Qwen/Qwen3-8B-Instruct --max-samples 300

# 300 more from Llama on the same inputs (600 total)
python scripts/run_generate.py --model meta-llama/Llama-3.1-8B-Instruct --max-samples 300 --resume

# Or different inputs: 300 Qwen + 300 Llama on next 300
python scripts/run_generate.py --model meta-llama/Llama-3.1-8B-Instruct --max-samples 600 --resume
```

Each record includes a `model` field for filtering or comparison.

### 2. Train with curriculum GRPO

```powershell
python scripts/run_train.py --synthetic data/synthetic_target.json --epochs 3
```

### 3. Evaluate

```powershell
python scripts/run_evaluate.py --checkpoint output/checkpoints/final
```

### Full pipeline

```powershell
python run_pipeline.py --max-samples 100 --epochs 2
```

## Project structure

```
src/
  config.py       # Configuration
  data/           # Data loading, generation, confidence metrics
  curriculum/      # Curriculum scheduler
  reward.py       # GRPO reward functions
  train.py        # GRPO training loop
  evaluate.py     # Source/Target F1 evaluation
scripts/
  install.ps1
  run_generate.py
  run_train.py
  run_evaluate.py
```

## Baselines

To compare curriculum vs naive training, run with `--no-curriculum` (future) or use different configs.

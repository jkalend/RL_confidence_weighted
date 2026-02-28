"""Configuration for Confidence-Weighted Curriculum RL."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"


@dataclass
class DataConfig:
    """Data and dataset configuration."""

    # Source domain (general NER, e.g. CoNLL-03)
    source_dataset: str = "conll2003"
    source_split: str = "test"

    # Target domain (specialized NER, e.g. Few-NERD, CrossNER)
    target_dataset: str = "DFKI-SLT/few-nerd"
    target_split: str = "train"  # train, validation, or test (not "supervised" - that's a config)
    target_subset: str | None = "inter"  # inter, intra, or None for full
    source_subset: str | None = None

    # Synthetic generation
    num_generations_per_input: int = 8
    generation_temperature: float = 0.7
    max_new_tokens: int = 256 * 8

    # Ollama-only: context length (num_ctx). None = use Ollama app default.
    ollama_num_ctx: int | None = None

    # Confidence metric: "self_consistency" or "logprob"
    confidence_metric: Literal["self_consistency", "logprob"] = "self_consistency"


@dataclass
class CurriculumConfig:
    """Curriculum scheduling configuration."""

    # Static curriculum: percentiles per epoch
    # Epoch 0: top 10%, Epoch 1: top 20%, ...
    start_percentile: float = 0.1  # g(0) = 0.1
    end_percentile: float = 1.0
    schedule: Literal["linear", "root", "step"] = "linear"

    # For linear: g(t) = min(1, start + lambda * t)
    lambda_factor: float = 0.1

    # For step: pre-compute bins [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    step_bins: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.4, 0.6, 0.8, 1.0])


@dataclass
class ModelConfig:
    """Model configuration."""

    # Qwen2.5-7B or Qwen2.5-4B for memory efficiency
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    use_4bit: bool = True
    load_in_4bit: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


@dataclass
class GRPOConfig:
    """GRPO training configuration."""

    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6
    max_length: int = 512

    # GRPO-specific
    num_generations: int = 4  # group size for GRPO
    beta: float = 0.01  # KL penalty coefficient

    # Replay buffer: mix in source domain to prevent forgetting
    replay_ratio: float = 0.1  # 10% real data per batch

    # Wandb
    report_to: str = "wandb"
    run_name: str | None = None
    bf16: bool | None = None


@dataclass
class Config:
    """Master configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)

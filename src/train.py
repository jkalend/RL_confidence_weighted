"""GRPO training with curriculum learning."""

import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from src.config import Config, CHECKPOINT_DIR, DATA_DIR
from src.data.prompts import build_entity_extraction_prompt
from src.curriculum import get_curriculum_dataset
from src.reward import entity_reward_soft


def _synthetic_to_hf_dataset(D_syn: list[dict], epoch: int, total_epochs: int, config: Config) -> Dataset:
    """Convert curriculum-filtered D_syn to HuggingFace Dataset with prompt column."""
    filtered = get_curriculum_dataset(D_syn, config.curriculum, epoch, total_epochs)

    prompts = []
    pseudo_labels = []
    for s in filtered:
        prompt = build_entity_extraction_prompt(s["input"])
        prompts.append(prompt)
        pseudo_labels.append(s["pseudo_label"])

    return Dataset.from_dict({"prompt": prompts, "pseudo_label": pseudo_labels})


def load_synthetic_for_training(path: Path | str) -> list[dict[str, Any]]:
    """Load synthetic dataset from JSON."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _is_bf16_supported():
    """Check if hardware supports BF16."""
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False


def run_grpo_training(
    config: Config | None = None,
    synthetic_path: Path | str | None = None,
    use_unsloth: bool = True,
) -> None:
    """
    Run GRPO training with curriculum.
    Uses Unsloth for 4-bit model + TRL GRPOTrainer.
    """
    config = config or Config()
    synthetic_path = Path(synthetic_path) if synthetic_path else DATA_DIR / "synthetic_target.json"

    D_syn = load_synthetic_for_training(synthetic_path)
    total_epochs = config.grpo.num_train_epochs

    # Load model
    if use_unsloth:
        model, tokenizer = _load_unsloth_model(config)
    else:
        model, tokenizer = _load_transformers_model(config)

    from trl import GRPOConfig, GRPOTrainer

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Curriculum: train one epoch per stage, each with expanding data slice
    trainer = None
    for epoch in range(total_epochs):
        train_ds = _synthetic_to_hf_dataset(D_syn, epoch, total_epochs, config)
        if len(train_ds) == 0:
            continue

        training_args = GRPOConfig(
            output_dir=str(CHECKPOINT_DIR / f"stage_{epoch}"),
            num_train_epochs=1,
            per_device_train_batch_size=config.grpo.per_device_train_batch_size,
            gradient_accumulation_steps=config.grpo.gradient_accumulation_steps,
            learning_rate=config.grpo.learning_rate,
            max_length=config.grpo.max_length,
            beta=config.grpo.beta,
            num_generations=config.grpo.num_generations,
            report_to=config.grpo.report_to,
            run_name=f"{config.grpo.run_name or 'curriculum-grpo'}-stage{epoch}",
            bf16=config.grpo.bf16 if config.grpo.bf16 is not None else _is_bf16_supported(),
            gradient_checkpointing=True,
        )

        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            reward_funcs=entity_reward_soft,
            train_dataset=train_ds,
        )

        trainer.train()
        trainer.save_model(str(CHECKPOINT_DIR / f"stage_{epoch}"))
        # Model stays in memory for next curriculum stage

    if trainer:
        trainer.save_model(str(CHECKPOINT_DIR / "final"))
    else:
        # Create placeholder checkpoint if no training occurred
        final_dir = CHECKPOINT_DIR / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        with open(final_dir / "README.md", "w") as f:
            f.write("# Placeholder Checkpoint\nNo training steps were performed.")
        print(f"Warning: No training occurred. Created placeholder at {final_dir}")


def _load_unsloth_model(config: Config):
    """Load 4-bit model with Unsloth."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.model_name,
        max_seq_length=config.grpo.max_length,
        load_in_4bit=config.model.use_4bit,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        target_modules=config.model.lora_target_modules,
    )
    return model, tokenizer


def _load_transformers_model(config: Config):
    """Fallback: load with transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        load_in_4bit=config.model.use_4bit,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer

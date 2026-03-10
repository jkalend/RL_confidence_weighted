"""GRPO training with curriculum learning."""

import json
import sys
from types import MethodType
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset

from src.config import Config, CHECKPOINT_DIR, DATA_DIR


def _patch_grpo_logits_vs_hidden_states(trainer):
    """Fix Unsloth GRPO bug: chunked_hidden_states_selective_log_softmax receives logits
    but expects hidden_states, causing matmul shape error (79x248320 @ 2560x248320).
    Wrap _get_per_token_logps_and_entropies to fix the logits path."""
    original = trainer._get_per_token_logps_and_entropies

    def patched(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None,
                compute_entropy=False, compute_efficient=False, *args, **kwargs):
        return original(model, input_ids, attention_mask, logits_to_keep, batch_size,
                        compute_entropy, compute_efficient, *args, **kwargs)

    # Patch the function used inside _get_per_token_logps_and_entropies
    # by finding and patching chunked_hidden_states_selective_log_softmax in the method's module
    cls = type(trainer)
    mod = sys.modules.get(cls.__module__)
    if mod is None or not hasattr(mod, "chunked_hidden_states_selective_log_softmax"):
        return
    _orig = mod.chunked_hidden_states_selective_log_softmax

    def _patched(hidden_states, lm_head, index, chunks=4, logit_scale_multiply=0.0,
                 logit_scale_divide=0.0, logit_softcapping=0.0, temperature=1.0):
        vocab_size = lm_head.shape[0]
        if hidden_states.shape[-1] == vocab_size:
            _logits = hidden_states.to(torch.float32)
            if logit_scale_multiply != 0.0:
                _logits = _logits * logit_scale_multiply
            if logit_scale_divide != 0.0:
                _logits = _logits / logit_scale_divide
            if logit_softcapping != 0.0:
                _logits = _logits * torch.tanh(_logits / logit_softcapping)
            if temperature != 1.0:
                _logits = _logits / temperature
            return mod.chunked_selective_log_softmax(_logits, index)
        return _orig(hidden_states, lm_head, index, chunks,
                     logit_scale_multiply, logit_scale_divide,
                     logit_softcapping, temperature)

    mod.chunked_hidden_states_selective_log_softmax = _patched


def _patch_grpo_logprob_length_alignment(trainer):
    """Align cached per-token logprobs to completion_mask length.

    Unsloth GRPO can provide old/ref/sampling logprobs with prompt+completion length
    while completion_mask stays completion-only, causing shape mismatch in compute_loss.
    """
    original = trainer.compute_loss

    def patched(self, model, inputs, *args, **kwargs):
        completion_mask = inputs.get("completion_mask")
        target_len = None
        if isinstance(completion_mask, torch.Tensor) and completion_mask.dim() >= 2:
            target_len = completion_mask.shape[1]

        if target_len is not None and target_len > 0:
            for key in ("old_per_token_logps", "ref_per_token_logps", "sampling_per_token_logps"):
                value = inputs.get(key)
                if not isinstance(value, torch.Tensor) or value.dim() < 2:
                    continue
                current_len = value.shape[1]
                if current_len == target_len:
                    continue
                if current_len > target_len:
                    inputs[key] = value[:, -target_len:]
                else:
                    pad = torch.zeros(
                        value.shape[0],
                        target_len - current_len,
                        device=value.device,
                        dtype=value.dtype,
                    )
                    inputs[key] = torch.cat([pad, value], dim=1)

        return original(model, inputs, *args, **kwargs)

    trainer.compute_loss = MethodType(patched, trainer)
from src.data.prompts import build_entity_extraction_prompt
from src.curriculum import get_curriculum_dataset
from src.reward import entity_reward_soft


def _disable_gradient_checkpointing_everywhere(model) -> None:
    """Best-effort disable across wrapped PEFT/Unsloth/Transformers model layers."""
    seen = set()
    stack = [model]
    while stack:
        current = stack.pop()
        if current is None:
            continue
        ident = id(current)
        if ident in seen:
            continue
        seen.add(ident)

        if hasattr(current, "gradient_checkpointing_disable"):
            try:
                current.gradient_checkpointing_disable()
            except Exception:
                pass
        if hasattr(current, "gradient_checkpointing"):
            try:
                current.gradient_checkpointing = False
            except Exception:
                pass
        if hasattr(current, "config") and hasattr(current.config, "gradient_checkpointing"):
            try:
                current.config.gradient_checkpointing = False
            except Exception:
                pass

        for attr in ("model", "base_model", "language_model"):
            if hasattr(current, attr):
                try:
                    stack.append(getattr(current, attr))
                except Exception:
                    pass


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
            max_prompt_length=config.grpo.max_prompt_length,
            max_completion_length=config.grpo.max_completion_length,
            generation_kwargs={"max_new_tokens": config.grpo.max_completion_length},
            beta=config.grpo.beta,
            num_generations=config.grpo.num_generations,
            report_to=config.grpo.report_to,
            run_name=f"{config.grpo.run_name or 'curriculum-grpo'}-stage{epoch}",
            bf16=config.grpo.bf16 if config.grpo.bf16 is not None else _is_bf16_supported(),
            gradient_checkpointing=False,  # True causes apply_rotary_pos_emb shape mismatch with Qwen3.5
        )
        # Unsloth wrapper reads this field dynamically via getattr(..., True).
        setattr(training_args, "gradient_checkpointing", False)
        setattr(training_args, "gradient_checkpointing_kwargs", None)

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            reward_funcs=entity_reward_soft,
            train_dataset=train_ds,
        )

        if use_unsloth:
            _patch_grpo_logits_vs_hidden_states(trainer)
            _patch_grpo_logprob_length_alignment(trainer)
            if hasattr(trainer, "args"):
                setattr(trainer.args, "gradient_checkpointing", False)
                setattr(trainer.args, "gradient_checkpointing_kwargs", None)
            # Disable gradient checkpointing to avoid apply_rotary_pos_emb shape mismatch with Qwen3.5
            if hasattr(trainer.model, "for_training"):
                trainer.model.for_training(use_gradient_checkpointing=False)
            _disable_gradient_checkpointing_everywhere(trainer.model)

        trainer.train()
        trainer.save_model(str(CHECKPOINT_DIR / f"stage_{epoch}"))
        # Model stays in memory for next curriculum stage

    if trainer:
        trainer.save_model(str(CHECKPOINT_DIR / "final"))
    else:
        # Create placeholder checkpoint if no training occurred
        final_dir = CHECKPOINT_DIR / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        # Save minimal model and tokenizer files so run_evaluation can load it
        if use_unsloth:
            model, tokenizer = _load_unsloth_model(config)
        else:
            model, tokenizer = _load_transformers_model(config)
        
        model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        
        with open(final_dir / "README.md", "w") as f:
            f.write("# Placeholder Checkpoint\nNo training steps were performed.")
        print(f"Warning: No training occurred. Created placeholder at {final_dir}")


def _load_unsloth_model(config: Config):
    """Load 4-bit model with Unsloth."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.model_name,
        max_seq_length=config.grpo.max_prompt_length + config.grpo.max_completion_length,
        load_in_4bit=config.model.use_4bit,
        dtype="bfloat16" if _is_bf16_supported() else None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        target_modules=config.model.lora_target_modules,
        # Qwen3.5 + Unsloth GRPO can hit rotary shape mismatches when checkpointing is on.
        # Disable at model patch time to avoid torch.utils.checkpoint path entirely.
        use_gradient_checkpointing=False,
    )
    _patch_generation_config(model, tokenizer, config)
    return model, tokenizer


def _load_transformers_model(config: Config):
    """Fallback: load with transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name, trust_remote_code=True)

    bnb_config = None
    if config.model.use_4bit:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    _patch_generation_config(model, tokenizer, config)
    return model, tokenizer


def _patch_generation_config(model, tokenizer, config: Config) -> None:
    """Replace the model's generation_config with a clean one.

    Models like Qwen3 ship with max_length=40960, temperature=0.6, top_p=0.95
    in generation_config.json. Replacing the object entirely avoids the
    transformers warning about modified defaults and ensures our max_new_tokens
    is the sole length constraint regardless of trainer kwargs.
    """
    from transformers import GenerationConfig

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model.generation_config = GenerationConfig(
        max_new_tokens=config.grpo.max_completion_length,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
    )

"""Generate synthetic labels with confidence scores."""

import json
import os
import torch
from pathlib import Path
from typing import Any, Literal

from tqdm import tqdm

from src.config import DATA_DIR, DataConfig
from src.data.loaders import load_ner_dataset, entities_to_json
from src.data.confidence import compute_self_consistency
from src.data.prompts import build_entity_extraction_prompt

Backend = Literal["unsloth", "ollama", "transformers"]


def generate_synthetic_dataset(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    config: DataConfig | None = None,
    output_path: Path | str | None = None,
    max_samples: int | None = 100,
    backend: Backend = "ollama",
    resume: bool = False,
) -> list[dict[str, Any]]:
    """
    Generate synthetic entity labels for target domain using Few-Shot CoT.
    Returns D_syn: list of {input, model, pseudo_label, confidence_score, generations}.
    Saves after each sample for crash resilience. Use resume=True to continue from existing output.
    Multi-model: resume skips (input, model) pairs, so you can mix models in one file
    (same inputs from different models, or different inputs per model).
    """
    config = config or DataConfig()
    output_path = Path(output_path) if output_path else DATA_DIR / "synthetic_target.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep user-facing model name for storage (before alias resolution)
    model_name_for_record = model_name
    # Resolve model aliases for loading (e.g. Qwen3-8B-Instruct -> Qwen3-8B)
    model_name = _MODEL_ALIASES.get(model_name, model_name)

    # Load target domain (unlabeled sentences - we use train split and ignore labels for generation)
    samples = load_ner_dataset(config, domain="target")
    if max_samples:
        samples = samples[:max_samples]

    # Resume: skip (input, model) pairs already in file. Supports multi-model: same inputs
    # from different models, or different inputs per model.
    D_syn = []
    processed_keys: set[tuple[str, str]] = set()
    if resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            D_syn = json.load(f)
        processed_keys = {(d["input"], d.get("model", "unknown")) for d in D_syn}

    K = config.num_generations_per_input
    temperature = config.generation_temperature
    max_new_tokens = config.max_new_tokens

    # Load model or resolve Ollama model name
    model, tokenizer = None, None
    ollama_model: str | None = None
    if backend == "ollama":
        ollama_model = _OLLAMA_MODEL_MAP.get(model_name, model_name)
    elif backend == "unsloth":
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = _load_unsloth_model(model_name, config)
        except (ImportError, RuntimeError):
            model, tokenizer = _load_transformers_model(model_name)
    else:
        model, tokenizer = _load_transformers_model(model_name)

    if model is not None:
        model.eval()

    def _save():
        temp_path = output_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(D_syn, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, output_path)

    for sample in tqdm(samples, desc="Generating synthetic labels"):
        sentence = sample["sentence"]
        if (sentence, model_name_for_record) in processed_keys:
            continue

        prompt = build_entity_extraction_prompt(sentence)

        # Generate K candidates
        if backend == "ollama":
            generations = _generate_k_outputs_ollama(
                ollama_model, prompt, K, temperature, max_new_tokens, num_ctx=config.ollama_num_ctx
            )
        else:
            generations = _generate_k_outputs(
                model, tokenizer, prompt, K, temperature, max_new_tokens
            )

        # Compute confidence (self-consistency)
        pseudo_label, confidence = compute_self_consistency(generations)

        # If we have ground truth, we could compare - but for test-time we don't
        D_syn.append({
            "input": sentence,
            "model": model_name_for_record,
            "pseudo_label": pseudo_label,
            "confidence_score": confidence,
            "generations": generations,
            "tokens": sample.get("tokens", []),
        })
        processed_keys.add((sentence, model_name_for_record))

        # Save after each sample for crash resilience
        _save()

    return D_syn


# HuggingFace model aliases (for loading). Prefer Unsloth for all models including larger ones.
# gpt-oss:20b is Ollama format; map to HF repo for unsloth/transformers backends.
_MODEL_ALIASES = {
    "Qwen/Qwen3-8B-Instruct": "Qwen/Qwen3-8B",
    "Qwen/Qwen3.5-27B": "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-35B": "Qwen/Qwen3.5-35B",
    "zai-org/GLM-4.7-Flash": "zai-org/GLM-4.7-Flash",
    "openai/gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss:20b": "openai/gpt-oss-20b",
}

# Map HuggingFace model names to Ollama model names (for --backend ollama). Unsloth preferred for quality.
_OLLAMA_MODEL_MAP = {
    "Qwen/Qwen3.5-27B": "qwen3.5:27b",
    "Qwen/Qwen3.5-35B": "qwen3.5:35b",
    "zai-org/GLM-4.7-Flash": "glm-4.7-flash",
    "GLM-4.7-Flash": "glm-4.7-flash",
    "glm-4.7-flash": "glm-4.7-flash",
    "openai/gpt-oss-20b": "gpt-oss:20b",
    "gpt-oss-20b": "gpt-oss:20b",
    "gpt-oss:20b": "gpt-oss:20b",
}


def _load_unsloth_model(model_name: str, config: DataConfig):
    """Load model with Unsloth for fast inference."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=getattr(config, "max_seq_length", 512),
        load_in_4bit=True,
        dtype=None,  # auto
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def _load_transformers_model(model_name: str):
    """Fallback: load with standard transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def _generate_k_outputs(model, tokenizer, prompt: str, K: int, temperature: float, max_new_tokens: int) -> list[str]:
    """Generate K outputs with temperature sampling."""
    from transformers import GenerationConfig

    # Chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = []
    for _ in range(K):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        outputs.append(decoded.strip())

    return outputs


def _generate_k_outputs_ollama(
    model_name: str,
    prompt: str,
    K: int,
    temperature: float,
    max_new_tokens: int,
    num_ctx: int | None = None,
) -> list[str]:
    """Generate K outputs via Ollama API. Requires Ollama running locally with model pulled."""
    try:
        from ollama import Client
    except ImportError:
        raise ImportError(
            "Ollama backend requires: pip install ollama. "
            "Also ensure Ollama is running and the model is pulled (e.g. ollama pull glm-4.7-flash)."
        ) from None

    options: dict = {"temperature": temperature, "num_predict": max_new_tokens}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx

    client = Client()
    messages = [{"role": "user", "content": prompt}]
    outputs = []
    for _ in range(K):
        response = client.chat(
            model=model_name,
            messages=messages,
            stream=False,
            options=options,
        )
        content = response.get("message", {}).get("content", "")
        outputs.append(content.strip())
    return outputs

"""Evaluation: Target F1 (adaptation) and Source F1 (forgetting)."""

import json
from pathlib import Path
from typing import Any

from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

from src.config import Config, CHECKPOINT_DIR, DATA_DIR
from src.data.loaders import load_conll2003, load_few_nerd, _extract_entities_from_bio
from src.data.prompts import build_entity_extraction_prompt


def _predict_entities(model, tokenizer, sentence: str, max_new_tokens: int = 256) -> list[dict]:
    """Run model inference and parse entity output."""
    prompt = build_entity_extraction_prompt(sentence)

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with __import__("torch").no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Parse JSON from output
    start = decoded.find("[")
    end = decoded.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            parsed = json.loads(decoded[start:end])
            if isinstance(parsed, list):
                return [
                    {"text": str(e.get("text", "")), "type": str(e.get("type", "O"))}
                    for e in parsed
                    if isinstance(e, dict)
                ]
        except json.JSONDecodeError:
            pass
    return []


def _entities_to_bio_tags(tokens: list[str], entities: list[dict]) -> list[str]:
    """Convert entity spans to BIO tag sequence. Uses simple word alignment."""
    tags = ["O"] * len(tokens)
    for ent in entities:
        text = ent.get("text", "")
        etype = ent.get("type", "O")
        span_tokens = text.split()
        if not span_tokens:
            continue
        # Find span in token sequence
        for i in range(len(tokens) - len(span_tokens) + 1):
            if tokens[i:i + len(span_tokens)] == span_tokens:
                for j in range(len(span_tokens)):
                    tags[i + j] = f"B-{etype}" if j == 0 else f"I-{etype}"
                break
    return tags


def _tags_from_sample(sample: dict) -> list[str]:
    """Get BIO tags from sample using label_names."""
    label_names = sample.get("label_names", ["O"])
    tags = sample.get("ner_tags", [])
    if not tags:
        return ["O"] * len(sample["tokens"])
    return [label_names[t] if t < len(label_names) else "O" for t in tags]


def evaluate_model(
    model,
    tokenizer,
    source_samples: list[dict] | None = None,
    target_samples: list[dict] | None = None,
    max_eval: int = 200,
) -> dict[str, Any]:
    """
    Evaluate on source (forgetting) and target (adaptation) domains.
    Returns dict with source_f1, target_f1, and per-domain reports.
    """
    results = {}

    if source_samples:
        source_samples = source_samples[:max_eval]
        preds = []
        for s in source_samples:
            ents = _predict_entities(model, tokenizer, s["sentence"])
            preds.append(ents)

        true_tags = []
        pred_tags = []
        for s, p in zip(source_samples, preds, strict=True):
            true_tags.append(_tags_from_sample(s))
            pred_tags.append(_entities_to_bio_tags(s["tokens"], p))

        results["source_f1"] = f1_score(true_tags, pred_tags)
        results["source_precision"] = precision_score(true_tags, pred_tags)
        results["source_recall"] = recall_score(true_tags, pred_tags)
        results["source_report"] = classification_report(true_tags, pred_tags)

    if target_samples:
        target_samples = target_samples[:max_eval]
        preds = []
        for s in target_samples:
            ents = _predict_entities(model, tokenizer, s["sentence"])
            preds.append(ents)

        true_tags = []
        pred_tags = []
        for s, p in zip(target_samples, preds, strict=True):
            true_tags.append(_tags_from_sample(s))
            pred_tags.append(_entities_to_bio_tags(s["tokens"], p))

        results["target_f1"] = f1_score(true_tags, pred_tags)
        results["target_precision"] = precision_score(true_tags, pred_tags)
        results["target_recall"] = recall_score(true_tags, pred_tags)
        results["target_report"] = classification_report(true_tags, pred_tags)

    return results


def run_evaluation(
    checkpoint_path: Path | str | None = None,
    config: Config | None = None,
    max_eval: int = 200,
) -> dict[str, Any]:
    """Load model from checkpoint and run full evaluation."""
    config = config or Config()
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else CHECKPOINT_DIR / "final"

    # Load model
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(checkpoint_path),
            max_seq_length=config.grpo.max_length if config else 512,
            load_in_4bit=config.model.use_4bit if config else True,
        )
        FastLanguageModel.for_inference(model)
    except (ImportError, ModuleNotFoundError, OSError, ValueError):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_path),
            load_in_4bit=config.model.use_4bit if config else True,
            device_map="auto",
            trust_remote_code=True,
        )

    source_samples = load_conll2003(config.data.source_split if config else "test")
    target_samples = load_few_nerd(
        split="test",  # Use held-out test split for evaluation
        subset=config.data.target_subset if config else "inter",
        max_samples=max_eval
    )

    return evaluate_model(model, tokenizer, source_samples, target_samples, max_eval)

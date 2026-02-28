"""Data loaders for NER datasets."""

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from src.config import DATA_DIR, DataConfig


def _tokens_to_sentence(tokens: list[str]) -> str:
    """Join tokens into a sentence, handling subword spacing."""
    if not tokens:
        return ""
    # Simple join - datasets often have proper tokenization
    return " ".join(tokens)


def _extract_entities_from_bio(tokens: list[str], tags: list[str | int], label_names: list[str]) -> list[dict]:
    """Convert BIO/BIOES tags to entity spans."""
    entities = []
    current_entity: dict | None = None
    current_tokens: list[str] = []

    for i, (token, tag_idx) in enumerate(zip(tokens, tags)):
        if isinstance(tag_idx, str):
            label = tag_idx
        else:
            label = label_names[tag_idx] if tag_idx < len(label_names) else "O"

        label_type = label[2:] if len(label) > 2 else ""
        if (label.startswith("B-") or label.startswith("I-")) and (
            current_entity is None or label_type != current_entity["type"]
        ):
            if current_entity:
                current_entity["text"] = " ".join(current_tokens)
                entities.append(current_entity)
            current_entity = {"type": label_type, "start": i, "end": i + 1}
            current_tokens = [token]
        elif label.startswith("I-") and current_entity:
            current_tokens.append(token)
            current_entity["end"] = i + 1
        else:
            if current_entity:
                current_entity["text"] = " ".join(current_tokens)
                entities.append(current_entity)
                current_entity = None
                current_tokens = []

    if current_entity:
        current_entity["text"] = " ".join(current_tokens)
        entities.append(current_entity)

    return entities


def load_conll2003(split: str = "test") -> list[dict[str, Any]]:
    """Load CoNLL-2003 dataset (source domain)."""
    ds = load_dataset("conll2003", split=split)
    label_names = ds.features["ner_tags"].feature.names

    samples = []
    for ex in ds:
        tokens = ex["tokens"]
        tags = ex["ner_tags"]
        sentence = _tokens_to_sentence(tokens)
        entities = _extract_entities_from_bio(tokens, tags, label_names)
        samples.append({
            "tokens": tokens,
            "sentence": sentence,
            "entities": entities,
            "ner_tags": tags,
            "label_names": label_names,
        })
    return samples


def load_few_nerd(split: str = "train", subset: str | None = "inter", max_samples: int | None = None) -> list[dict[str, Any]]:
    """Load Few-NERD dataset (target domain). Config: inter, intra, or supervised."""
    config_name = subset if subset in ("inter", "intra", "supervised") else "inter"
    ds = load_dataset("DFKI-SLT/few-nerd", config_name, split=split)

    # Few-NERD uses fine_ner_tags
    if "fine_ner_tags" in ds.features:
        label_feat = ds.features["fine_ner_tags"]
    else:
        label_feat = ds.features["ner_tags"]
    label_names = label_feat.feature.names if hasattr(label_feat.feature, "names") else [f"tag_{i}" for i in range(67)]

    samples = []
    for i, ex in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        tokens = ex["tokens"]
        tags = ex.get("fine_ner_tags", ex.get("ner_tags", []))
        sentence = _tokens_to_sentence(tokens)
        entities = _extract_entities_from_bio(tokens, tags, label_names)
        samples.append({
            "tokens": tokens,
            "sentence": sentence,
            "entities": entities,
            "ner_tags": tags,
            "label_names": label_names,
        })
    return samples


def load_ner_dataset(config: DataConfig, domain: str = "target") -> list[dict[str, Any]]:
    """Load NER dataset based on config."""
    dataset_name = config.source_dataset if domain == "source" else config.target_dataset
    split = config.source_split if domain == "source" else config.target_split
    subset = config.source_subset if domain == "source" else config.target_subset

    if dataset_name == "conll2003":
        return load_conll2003(split=split)
    elif dataset_name == "few-nerd" or dataset_name.endswith("/few-nerd"):
        return load_few_nerd(split=split, subset=subset)
    else:
        # Fallback to defaults or raise error
        if domain == "source":
            return load_conll2003(split)
        else:
            return load_few_nerd(split=split, subset=subset)


def entities_to_json(entities: list[dict]) -> str:
    """Convert entity list to canonical JSON string for comparison."""
    simplified = [{"text": e["text"], "type": e["type"]} for e in entities]
    return json.dumps(simplified, sort_keys=True)


def load_synthetic_dataset(path: Path | str) -> list[dict[str, Any]]:
    """Load pre-generated synthetic dataset with confidence scores."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Synthetic dataset not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return data

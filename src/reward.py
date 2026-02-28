"""Reward functions for GRPO entity extraction."""

import json
import re
from typing import Any


def _extract_json_from_completion(completion: str) -> str | None:
    """Extract JSON array from model output."""
    if isinstance(completion, list):
        # Conversational format: [{"role": "assistant", "content": "..."}]
        content = completion[0].get("content", "") if completion else ""
    else:
        content = str(completion)

    start = content.find("[")
    end = content.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            parsed = json.loads(content[start:end])
            if isinstance(parsed, list):
                simplified = [
                    {"text": str(e.get("text", e.get("entity", ""))), "type": str(e.get("type", e.get("label", "O")))}
                    for e in parsed
                    if isinstance(e, dict)
                ]
                return json.dumps(simplified, sort_keys=True)
        except json.JSONDecodeError:
            pass
    return None


def _format_reward(completion: Any) -> float:
    """+1 if valid JSON entity format, 0 otherwise."""
    extracted = _extract_json_from_completion(completion)
    return 1.0 if extracted is not None else 0.0


def _entity_overlap_f1(pred_json: str | None, ref_json: str) -> float:
    """Compute token-level F1 overlap between predicted and reference entity sets."""
    if pred_json is None:
        return 0.0

    def parse_to_set(s: str) -> set[tuple[str, str]]:
        try:
            items = json.loads(s)
            return {(str(e.get("text", "")), str(e.get("type", "O"))) for e in items if isinstance(e, dict)}
        except (json.JSONDecodeError, TypeError):
            return set()

    pred_set = parse_to_set(pred_json)
    ref_set = parse_to_set(ref_json)

    if not ref_set:
        return 1.0 if not pred_set else 0.0
    if not pred_set:
        return 0.0

    tp = len(pred_set & ref_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(ref_set) if ref_set else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _sparse_reward(pred_json: str | None, ref_json: str) -> float:
    """+1 if exact match, 0 otherwise."""
    if pred_json is None:
        return 0.0
    return 1.0 if pred_json == ref_json else 0.0


def make_entity_reward_func(
    reward_type: str = "soft",
    format_weight: float = 0.2,
    accuracy_weight: float = 0.8,
) -> callable:
    """
    Create reward function for entity extraction.
    reward_type: "soft" (F1 overlap) or "sparse" (exact match)
    Dataset must have 'pseudo_label' column.
    """

    def reward_func(completions, pseudo_label=None, **kwargs):
        if pseudo_label is None:
            pseudo_label = [""] * len(completions)
        if not isinstance(pseudo_label, list):
            pseudo_label = [pseudo_label] * len(completions)

        rewards = []
        for completion, ref in zip(completions, pseudo_label):
            content = completion[0]["content"] if isinstance(completion, list) else completion
            pred_json = _extract_json_from_completion(content)

            fmt_r = _format_reward(content)
            if reward_type == "sparse":
                acc_r = _sparse_reward(pred_json, ref)
            else:
                acc_r = _entity_overlap_f1(pred_json, ref)

            total = format_weight * fmt_r + accuracy_weight * acc_r
            rewards.append(float(total))

        return rewards

    return reward_func


def entity_reward_soft(completions, pseudo_label=None, **kwargs):
    """Soft reward: format + F1 overlap. Use with dataset column 'pseudo_label'."""
    func = make_entity_reward_func(reward_type="soft", format_weight=0.2, accuracy_weight=0.8)
    return func(completions=completions, pseudo_label=pseudo_label, **kwargs)


def entity_reward_sparse(completions, pseudo_label=None, **kwargs):
    """Sparse reward: format + exact match. Use with dataset column 'pseudo_label'."""
    func = make_entity_reward_func(reward_type="sparse", format_weight=0.2, accuracy_weight=0.8)
    return func(completions=completions, pseudo_label=pseudo_label, **kwargs)

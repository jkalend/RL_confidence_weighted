"""Confidence metrics for synthetic labels."""

import json
import math
from collections import Counter
from typing import Any


def _normalize_entity_output(raw: str) -> str | None:
    """Extract and normalize entity JSON from model output."""
    raw = raw.strip()
    # Try to find JSON in output
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            parsed = json.loads(raw[start:end])
            if isinstance(parsed, list):
                simplified = []
                for e in parsed:
                    if isinstance(e, dict):
                        text_value = str(e.get("text", e.get("entity", "")))
                        type_value = str(e.get("type", e.get("label", "O")))
                        simplified.append({"text": text_value, "type": type_value})
                return json.dumps(simplified, sort_keys=True)
        except json.JSONDecodeError:
            pass
    return None


def _semantic_equivalent(a: str | None, b: str | None) -> bool:
    """Check if two entity outputs are semantically equivalent."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a == b


def compute_self_consistency(generations: list[str]) -> tuple[str, float]:
    """
    Compute self-consistency (semantic entropy) confidence.
    Returns (majority_vote_output, confidence_score).
    Confidence = proportion of generations in largest cluster.
    """
    normalized = [_normalize_entity_output(g) for g in generations]
    valid = [n for n in normalized if n is not None]

    if not valid:
        # No valid JSON - use raw majority or first
        if not generations:
            return "", 0.0
        counter = Counter(generations)
        majority = counter.most_common(1)[0][0]
        return majority, 0.0

    counter = Counter(valid)
    majority_output = counter.most_common(1)[0][0]
    count = counter[majority_output]
    confidence = count / len(generations)

    return majority_output, confidence


def compute_logprob_confidence(logprobs_list: list[list[float]]) -> float:
    """
    Compute average log-probability confidence.
    logprobs_list: list of token logprobs per generation.
    Returns mean of mean logprobs per sequence (higher = more confident).
    """
    if not logprobs_list:
        return 0.0

    mean_logprobs = []
    for logprobs in logprobs_list:
        if logprobs:
            mean_logprobs.append(sum(logprobs) / len(logprobs))
        else:
            mean_logprobs.append(float("-inf"))

    # Normalize to [0, 1] using softmax-like or min-max
    max_lp = max(mean_logprobs)
    min_lp = min(mean_logprobs)

    if math.isinf(max_lp) and max_lp < 0:
        return 0.0

    if max_lp == min_lp:
        return 1.0
    # Use spread-based confidence: tighter spreads produce values near 1
    spread = max_lp - min_lp
    return math.exp(-spread)

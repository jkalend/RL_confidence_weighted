"""Curriculum scheduler for confidence-weighted data filtering."""

import math
from typing import Any

from torch.utils.data import Dataset

from src.config import CurriculumConfig


class CurriculumScheduler:
    """
    Pacing function g(t) that returns the fraction of data to use at step t.
    Filtering: keep samples where confidence_score >= Percentile(1 - g(t)).
    """

    def __init__(self, config: CurriculumConfig, total_steps: int):
        self.config = config
        self.total_steps = total_steps

    def g(self, step: int) -> float:
        """Return fraction of data to include at this step (0 to 1)."""
        t = step / max(1, self.total_steps - 1)  # normalize to [0, 1]

        if self.config.schedule == "linear":
            # g(t) = min(1, start + lambda * t)
            return min(1.0, self.config.start_percentile + self.config.lambda_factor * t)
        elif self.config.schedule == "root":
            # g(t) = min(1, sqrt(t^2 + start_bias))
            return min(1.0, math.sqrt(t * t + self.config.start_percentile))
        elif self.config.schedule == "step":
            # Pre-computed bins
            idx = min(int(t * len(self.config.step_bins)), len(self.config.step_bins) - 1)
            return self.config.step_bins[idx]
        else:
            return min(1.0, self.config.start_percentile + self.config.lambda_factor * t)

    def get_percentile_threshold(self, step: int, sorted_confidences: list[float]) -> float:
        """
        At step t, we include top g(t) fraction of data.
        So we need confidence >= percentile at (1 - g(t)).
        E.g. g(t)=0.2 means we use top 20% -> threshold = 80th percentile of confidence.
        """
        g_t = self.g(step)
        # We want top g(t) fraction -> threshold is the (1-g(t)) percentile
        idx = int((1 - g_t) * len(sorted_confidences))
        idx = max(0, min(idx, len(sorted_confidences) - 1))
        return sorted_confidences[idx]


def get_curriculum_dataset(
    D_syn: list[dict[str, Any]],
    config: CurriculumConfig,
    epoch: int,
    total_epochs: int,
) -> list[dict[str, Any]]:
    """
    Filter D_syn to the curriculum slice for the given epoch.
    Static curriculum: epoch 0 = top 10%, epoch 1 = top 20%, ...
    """
    total_steps = total_epochs
    scheduler = CurriculumScheduler(config, total_steps)
    g_t = scheduler.g(epoch)

    # Sort by confidence descending (high confidence first)
    sorted_by_conf = sorted(D_syn, key=lambda x: x["confidence_score"], reverse=True)
    n_keep = max(1, int(g_t * len(sorted_by_conf)))
    return sorted_by_conf[:n_keep]


class CurriculumDataset(Dataset):
    """PyTorch Dataset that yields curriculum-filtered samples for a given epoch."""

    def __init__(
        self,
        D_syn: list[dict[str, Any]],
        config: CurriculumConfig,
        epoch: int,
        total_epochs: int,
    ):
        self.samples = get_curriculum_dataset(D_syn, config, epoch, total_epochs)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]

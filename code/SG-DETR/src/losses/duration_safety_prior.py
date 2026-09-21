"""Calibration loss for the decoupled sample-duration safety prior."""

from typing import Any, Dict, List

import torch
import torch.nn.functional as func

from src.losses.losses import SetCriterion


class DurationSafetyPriorOnlySetCriterion(SetCriterion):
    """Train only the absolute-duration classifier used by safety routing."""

    def forward(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        meta: List[Dict[str, Any]],
        matching: Dict[str, Any],
    ) -> Dict[str, Any]:
        del targets, matching
        if "duration_safety_prior_logits" not in outputs:
            return {}
        logits = outputs["duration_safety_prior_logits"]
        low, high = self.query_prior_absolute_thresholds
        labels = []
        for sample_meta in meta:
            durations = [end - start for start, end in sample_meta["relevant_windows"]]
            mean_duration = sum(durations) / len(durations)
            labels.append(0 if mean_duration < low else 1 if mean_duration < high else 2)
        target = torch.tensor(labels, device=logits.device)
        counts = torch.bincount(target, minlength=3).clamp_min(1)
        weights = target.numel() / (3.0 * counts.to(logits.dtype))
        return {"loss_duration_safety_prior": func.cross_entropy(logits, target, weight=weights)}

"""AP-aligned expert calibration under soft duration responsibilities."""

from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as func

from src.losses.iou_oracle_routing import _smooth_ap, max_candidate_iou
from src.losses.losses import SetCriterion
from src.losses.soft_duration_experts import soft_duration_targets


class SoftDurationAPExpertsOnlySetCriterion(SetCriterion):
    """Optimize each expert's candidate ranking for its soft duration region."""

    def __init__(
        self,
        *args: Any,
        duration_centers_seconds: Sequence[float] = (5.0, 17.320508, 60.0),
        duration_target_beta: float = 2.0,
        duration_ap_rank_temperature: float = 0.05,
        duration_ap_thresholds: Sequence[float] = (0.5, 0.7, 0.9),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.duration_centers_seconds = tuple(duration_centers_seconds)
        self.duration_target_beta = duration_target_beta
        self.duration_ap_rank_temperature = duration_ap_rank_temperature
        self.duration_ap_thresholds = tuple(duration_ap_thresholds)

    def forward(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        meta: List[Dict[str, Any]],
        matching: Dict[str, Any],
    ) -> Dict[str, Any]:
        del matching
        required = {
            "expert_quality_residuals",
            "base_class_scores",
            "base_quality_scores",
            "pred_spans",
        }
        if not required.issubset(outputs):
            return {}

        residuals = outputs["expert_quality_residuals"]
        responsibilities = soft_duration_targets(
            meta,
            residuals.device,
            self.duration_centers_seconds,
            self.duration_target_beta,
        ).detach()
        expert_scores = 0.5 * (
            func.logsigmoid(outputs["base_class_scores"].detach() + residuals)
            + func.logsigmoid(outputs["base_quality_scores"].detach() + residuals)
        )

        numerator = residuals.new_zeros(())
        denominator = residuals.new_zeros(())
        for sample_idx, target in enumerate(targets["span_labels"]):
            candidate_iou = max_candidate_iou(
                outputs["pred_spans"][sample_idx].detach(),
                target["spans"],
            )
            for expert_idx in range(expert_scores.shape[-1]):
                threshold_losses = []
                scores = expert_scores[sample_idx, :, expert_idx]
                for threshold in self.duration_ap_thresholds:
                    positive = candidate_iou >= threshold
                    negative = candidate_iou < threshold
                    if positive.any() and negative.any():
                        threshold_losses.append(
                            1.0 - _smooth_ap(
                                scores[positive],
                                scores[negative],
                                self.duration_ap_rank_temperature,
                            ),
                        )
                if threshold_losses:
                    weight = responsibilities[sample_idx, expert_idx]
                    numerator = numerator + weight * torch.stack(threshold_losses).mean()
                    denominator = denominator + weight

        return {"loss_soft_duration_ap": numerator / denominator.clamp_min(1.0)}

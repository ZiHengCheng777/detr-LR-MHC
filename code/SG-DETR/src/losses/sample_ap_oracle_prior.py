"""Sample-level expert supervision from all-query counterfactual Smooth-AP."""

from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as func
from torch import Tensor

from src.losses.iou_oracle_routing import _smooth_ap, max_candidate_iou
from src.losses.losses import SetCriterion


def sample_expert_ap_targets(
    outputs: Dict[str, Any],
    targets: Dict[str, Any],
    ap_temperature: float,
    rank_temperature: float,
    thresholds: Sequence[float] = (0.5, 0.7),
) -> Tensor:
    """Build a soft best-expert label from each expert's candidate ranking."""
    residuals = outputs["expert_quality_residuals"].detach()
    expert_scores = 0.5 * (
        func.logsigmoid(outputs["base_class_scores"].detach() + residuals)
        + func.logsigmoid(outputs["base_quality_scores"].detach() + residuals)
    )
    sample_targets = []
    for sample_idx, target in enumerate(targets["span_labels"]):
        candidate_iou = max_candidate_iou(
            outputs["pred_spans"][sample_idx],
            target["spans"],
        )
        expert_utilities = []
        for expert_idx in range(expert_scores.shape[-1]):
            threshold_aps = []
            scores = expert_scores[sample_idx, :, expert_idx]
            for threshold in thresholds:
                positive = candidate_iou >= threshold
                negative = candidate_iou < threshold
                if positive.any() and negative.any():
                    threshold_aps.append(
                        _smooth_ap(scores[positive], scores[negative], rank_temperature),
                    )
            expert_utilities.append(
                torch.stack(threshold_aps).mean()
                if threshold_aps
                else scores.new_zeros(())
            )
        utilities = torch.stack(expert_utilities)
        sample_targets.append(torch.softmax(utilities / ap_temperature, dim=-1))
    return torch.stack(sample_targets).detach()


class SampleAPOraclePriorOnlySetCriterion(SetCriterion):
    """Train the sample prior to predict the best counterfactual expert."""

    def __init__(
        self,
        *args: Any,
        sample_ap_oracle_temperature: float = 0.05,
        sample_ap_rank_temperature: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sample_ap_oracle_temperature = sample_ap_oracle_temperature
        self.sample_ap_rank_temperature = sample_ap_rank_temperature

    def forward(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        meta: List[Dict[str, Any]],
        matching: Dict[str, Any],
    ) -> Dict[str, Any]:
        del meta, matching
        required = {
            "duration_safety_prior_logits",
            "expert_quality_residuals",
            "base_class_scores",
            "base_quality_scores",
            "pred_spans",
        }
        if not required.issubset(outputs):
            return {}
        target_probs = sample_expert_ap_targets(
            outputs,
            targets,
            self.sample_ap_oracle_temperature,
            self.sample_ap_rank_temperature,
        )
        logits = outputs["duration_safety_prior_logits"]
        loss = -(target_probs * func.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        return {"loss_sample_ap_oracle_prior": loss}

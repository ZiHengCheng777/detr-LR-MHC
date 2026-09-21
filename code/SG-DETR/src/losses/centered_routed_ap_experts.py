"""Direct AP-gain training for centered soft-routed experts."""

from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as func

from src.losses.balanced_counterfactual_rank import duration_group
from src.losses.iou_oracle_routing import _smooth_ap, max_candidate_iou
from src.losses.losses import SetCriterion


class CenteredRoutedAPExpertsOnlySetCriterion(SetCriterion):
    """Optimize routed AP against the fixed uniform counterfactual."""

    def __init__(
        self,
        *args: Any,
        routed_ap_rank_temperature: float = 0.05,
        routed_ap_safety_temperature: float = 0.02,
        routed_ap_margin: float = 0.002,
        routed_ap_thresholds: Sequence[float] = (0.5, 0.7, 0.9),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.routed_ap_rank_temperature = routed_ap_rank_temperature
        self.routed_ap_safety_temperature = routed_ap_safety_temperature
        self.routed_ap_margin = routed_ap_margin
        self.routed_ap_thresholds = tuple(routed_ap_thresholds)

    def forward(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        meta: List[Dict[str, Any]],
        matching: Dict[str, Any],
    ) -> Dict[str, Any]:
        del matching
        required = {
            "duration_safety_prior_logits",
            "expert_quality_residuals",
            "base_class_scores",
            "base_quality_scores",
            "pred_spans",
        }
        if not required.issubset(outputs):
            return {}

        residuals = outputs["expert_quality_residuals"]
        centered = residuals - residuals.mean(dim=-1, keepdim=True)
        route = torch.softmax(outputs["duration_safety_prior_logits"].detach(), dim=-1)
        routed_residual = (route.unsqueeze(1) * centered).sum(dim=-1)
        base_class = outputs["base_class_scores"].squeeze(-1).detach()
        base_quality = outputs["base_quality_scores"].squeeze(-1).detach()
        routed_score = 0.5 * (
            func.logsigmoid(base_class + routed_residual)
            + func.logsigmoid(base_quality + routed_residual)
        )
        uniform_score = 0.5 * (
            func.logsigmoid(base_class) + func.logsigmoid(base_quality)
        )

        grouped: List[List[torch.Tensor]] = [[], [], []]
        for sample_idx, target in enumerate(targets["span_labels"]):
            candidate_iou = max_candidate_iou(
                outputs["pred_spans"][sample_idx].detach(),
                target["spans"],
            )
            threshold_losses = []
            for threshold in self.routed_ap_thresholds:
                positive = candidate_iou >= threshold
                negative = candidate_iou < threshold
                if not positive.any() or not negative.any():
                    continue
                routed_ap = _smooth_ap(
                    routed_score[sample_idx, positive],
                    routed_score[sample_idx, negative],
                    self.routed_ap_rank_temperature,
                )
                uniform_ap = _smooth_ap(
                    uniform_score[sample_idx, positive],
                    uniform_score[sample_idx, negative],
                    self.routed_ap_rank_temperature,
                ).detach()
                deficit = uniform_ap - routed_ap + self.routed_ap_margin
                safety = self.routed_ap_safety_temperature * func.softplus(
                    deficit / self.routed_ap_safety_temperature,
                )
                threshold_losses.append((1.0 - routed_ap) + safety)
            if threshold_losses:
                grouped[duration_group(meta[sample_idx])].append(
                    torch.stack(threshold_losses).mean(),
                )

        group_means = [torch.stack(values).mean() for values in grouped if values]
        loss = torch.stack(group_means).mean() if group_means else residuals.sum() * 0
        return {"loss_centered_routed_ap": loss}

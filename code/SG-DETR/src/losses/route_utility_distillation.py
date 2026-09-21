"""Distill per-query expert utility into the adaptive soft router."""

from typing import Any, Dict, List

import torch
import torch.nn.functional as func
from torch import Tensor

from src.losses.losses import SetCriterion


def route_utility_distillation_loss(
    outputs: Dict[str, Any],
    indices: List[Any],
    temperature: float = 0.02,
    hard_negatives: int = 16,
    minimum_utility_range: float = 1e-4,
    negative_weight: float = 1.0,
) -> Tensor:
    """Teach the router which frozen expert improves each query's ranking.

    Expert utility is measured in the same log-score space used by evaluation.
    A positive query benefits from a higher expert score, while a hard negative
    benefits from a lower one.  The targets are detached, so only the router is
    optimized when this criterion is used for second-stage calibration.
    """
    route_logits: Tensor = outputs["route_logits"]
    expert_residuals: Tensor = outputs["expert_quality_residuals"].detach()
    base_class: Tensor = outputs["base_class_scores"].squeeze(-1).detach()
    base_quality: Tensor = outputs["base_quality_scores"].squeeze(-1).detach()

    expert_class = base_class.unsqueeze(-1) + expert_residuals
    expert_quality = base_quality.unsqueeze(-1) + expert_residuals
    expert_scores = 0.5 * (
        func.logsigmoid(expert_class) + func.logsigmoid(expert_quality)
    )
    uniform_residual = expert_residuals.mean(dim=-1)
    uniform_scores = 0.5 * (
        func.logsigmoid(base_class + uniform_residual)
        + func.logsigmoid(base_quality + uniform_residual)
    )
    expert_gain = expert_scores - uniform_scores.unsqueeze(-1)

    losses = []
    weights = []
    for sample_idx, (positive_idx, _) in enumerate(indices):
        positive_idx = positive_idx.to(route_logits.device)
        if positive_idx.numel() == 0:
            continue

        negative_mask = torch.ones(
            route_logits.shape[1],
            dtype=torch.bool,
            device=route_logits.device,
        )
        negative_mask[positive_idx] = False
        negative_idx = torch.nonzero(negative_mask, as_tuple=False).squeeze(-1)
        if negative_idx.numel() > 0:
            count = min(hard_negatives, negative_idx.numel())
            hard_order = torch.topk(uniform_scores[sample_idx, negative_idx], count).indices
            negative_idx = negative_idx[hard_order]

        selected_idx = torch.cat((positive_idx, negative_idx))
        signs = torch.cat(
            (
                torch.ones_like(positive_idx, dtype=route_logits.dtype),
                -torch.ones_like(negative_idx, dtype=route_logits.dtype),
            ),
        )
        utility = signs.unsqueeze(-1) * expert_gain[sample_idx, selected_idx]
        utility_range = utility.amax(dim=-1) - utility.amin(dim=-1)
        informative = utility_range >= minimum_utility_range
        if not informative.any():
            continue

        target = func.softmax(utility[informative] / temperature, dim=-1)
        log_probability = func.log_softmax(route_logits[sample_idx, selected_idx][informative], dim=-1)
        per_query = -(target * log_probability).sum(dim=-1)
        query_weights = utility_range[informative] / utility_range[informative].mean().clamp_min(1e-8)
        selected_signs = signs[informative]
        query_weights = query_weights * torch.where(
            selected_signs > 0,
            torch.ones_like(selected_signs),
            torch.full_like(selected_signs, negative_weight),
        )
        losses.append(per_query)
        weights.append(query_weights)

    if not losses:
        return route_logits.sum() * 0
    all_losses = torch.cat(losses)
    all_weights = torch.cat(weights)
    return (all_losses * all_weights).sum() / all_weights.sum().clamp_min(1e-8)


class RouteUtilityDistillationSetCriterion(SetCriterion):
    """Calibration criterion that optimizes only query-level router utility."""

    def __init__(
        self,
        *args: Any,
        route_utility_temperature: float = 0.02,
        route_utility_hard_negatives: int = 16,
        route_utility_minimum_range: float = 1e-4,
        route_utility_negative_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.route_utility_temperature = route_utility_temperature
        self.route_utility_hard_negatives = route_utility_hard_negatives
        self.route_utility_minimum_range = route_utility_minimum_range
        self.route_utility_negative_weight = route_utility_negative_weight

    def forward(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        meta: List[Dict[str, Any]],
        matching: Dict[str, Any],
    ) -> Dict[str, Any]:
        del targets, meta
        losses = {}
        required = {
            "route_logits",
            "expert_quality_residuals",
            "base_class_scores",
            "base_quality_scores",
        }
        if required.issubset(outputs):
            losses["loss_route_utility"] = route_utility_distillation_loss(
                outputs,
                matching["positive"]["indices"],
                temperature=self.route_utility_temperature,
                hard_negatives=self.route_utility_hard_negatives,
                minimum_utility_range=self.route_utility_minimum_range,
                negative_weight=self.route_utility_negative_weight,
            )
        return losses

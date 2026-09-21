"""Duration-balanced Smooth-AP improvement over uniform routing."""

from typing import Any, Dict, List

import torch
import torch.nn.functional as func
from torch import Tensor

from src.losses.balanced_counterfactual_rank import duration_group
from src.losses.losses import SetCriterion


def smooth_average_precision(
    positive_scores: Tensor,
    negative_scores: Tensor,
    temperature: float,
) -> Tensor:
    """Approximate AP from differentiable positive-vs-negative ranks."""
    outranking = torch.sigmoid(
        (negative_scores.unsqueeze(0) - positive_scores.unsqueeze(1)) / temperature,
    ).sum(dim=1)
    return (1.0 / (1.0 + outranking)).mean()


def balanced_counterfactual_ap_loss(
    outputs: Dict[str, Any],
    indices: List[Any],
    meta: List[Dict[str, Any]],
    rank_temperature: float = 0.02,
    improvement_margin: float = 0.002,
    hinge_temperature: float = 0.01,
    hard_negatives: int = 32,
) -> Tensor:
    """Require routed Smooth-AP to exceed the uniform counterfactual per group."""
    actual = 0.5 * (
        func.logsigmoid(outputs["pred_logits"].squeeze(-1))
        + func.logsigmoid(outputs["pred_quality_scores"].squeeze(-1))
    )
    uniform = 0.5 * (
        func.logsigmoid(outputs["pred_uniform_logits"].squeeze(-1))
        + func.logsigmoid(outputs["pred_uniform_quality_scores"].squeeze(-1))
    ).detach()
    grouped: List[List[Tensor]] = [[], [], []]
    for sample_idx, (positive_idx, _) in enumerate(indices):
        positive_idx = positive_idx.to(actual.device)
        if positive_idx.numel() == 0:
            continue
        negative_mask = torch.ones(actual.shape[1], dtype=torch.bool, device=actual.device)
        negative_mask[positive_idx] = False
        negative_idx = torch.nonzero(negative_mask, as_tuple=False).squeeze(-1)
        if negative_idx.numel() == 0:
            continue
        count = min(hard_negatives, negative_idx.numel())
        hard_order = torch.topk(uniform[sample_idx, negative_idx], count).indices
        hard_idx = negative_idx[hard_order]
        actual_ap = smooth_average_precision(
            actual[sample_idx, positive_idx],
            actual[sample_idx, hard_idx],
            rank_temperature,
        )
        uniform_ap = smooth_average_precision(
            uniform[sample_idx, positive_idx],
            uniform[sample_idx, hard_idx],
            rank_temperature,
        )
        deficit = uniform_ap - actual_ap + improvement_margin
        grouped[duration_group(meta[sample_idx])].append(
            hinge_temperature * func.softplus(deficit / hinge_temperature),
        )
    means = [torch.stack(values).mean() for values in grouped if values]
    if not means:
        return actual.sum() * 0
    return torch.stack(means).mean()


class BalancedCounterfactualAPOnlySetCriterion(SetCriterion):
    """Router-only criterion using duration-balanced Smooth-AP."""

    def __init__(
        self,
        *args: Any,
        counterfactual_ap_rank_temperature: float = 0.02,
        counterfactual_ap_margin: float = 0.002,
        counterfactual_ap_hinge_temperature: float = 0.01,
        counterfactual_ap_hard_negatives: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.counterfactual_ap_rank_temperature = counterfactual_ap_rank_temperature
        self.counterfactual_ap_margin = counterfactual_ap_margin
        self.counterfactual_ap_hinge_temperature = counterfactual_ap_hinge_temperature
        self.counterfactual_ap_hard_negatives = counterfactual_ap_hard_negatives

    def forward(self, outputs, targets, meta, matching):
        del targets
        required = {
            "pred_logits",
            "pred_quality_scores",
            "pred_uniform_logits",
            "pred_uniform_quality_scores",
        }
        if not required.issubset(outputs):
            return {}
        return {
            "loss_balanced_counterfactual_ap": balanced_counterfactual_ap_loss(
                outputs,
                matching["positive"]["indices"],
                meta,
                rank_temperature=self.counterfactual_ap_rank_temperature,
                improvement_margin=self.counterfactual_ap_margin,
                hinge_temperature=self.counterfactual_ap_hinge_temperature,
                hard_negatives=self.counterfactual_ap_hard_negatives,
            ),
        }

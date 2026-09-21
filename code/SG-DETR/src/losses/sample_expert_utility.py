"""Distill sample-level expert ranking utility into the soft route prior."""

from typing import Any, Dict, List

import torch
import torch.nn.functional as func
from torch import Tensor

from src.losses.losses import SetCriterion


def sample_expert_utility_loss(
    outputs: Dict[str, Any],
    indices: List[Any],
    temperature: float = 0.02,
    hard_negatives: int = 16,
) -> Tensor:
    """Supervise each sample prior with detached per-expert ranking gaps."""
    prior_logits: Tensor = outputs["query_length_prior_logits"]
    residuals: Tensor = outputs["expert_quality_residuals"].detach()
    base_class = outputs["base_class_scores"].squeeze(-1).detach()
    base_quality = outputs["base_quality_scores"].squeeze(-1).detach()

    expert_score = 0.5 * (
        func.logsigmoid(base_class.unsqueeze(-1) + residuals)
        + func.logsigmoid(base_quality.unsqueeze(-1) + residuals)
    )
    uniform_residual = residuals.mean(dim=-1)
    uniform_score = 0.5 * (
        func.logsigmoid(base_class + uniform_residual)
        + func.logsigmoid(base_quality + uniform_residual)
    )

    losses = []
    for sample_idx, (positive_idx, _) in enumerate(indices):
        positive_idx = positive_idx.to(prior_logits.device)
        if positive_idx.numel() == 0:
            continue
        negative_mask = torch.ones(
            uniform_score.shape[1],
            dtype=torch.bool,
            device=prior_logits.device,
        )
        negative_mask[positive_idx] = False
        negative_idx = torch.nonzero(negative_mask, as_tuple=False).squeeze(-1)
        if negative_idx.numel() == 0:
            continue
        count = min(hard_negatives, negative_idx.numel())
        hard_order = torch.topk(uniform_score[sample_idx, negative_idx], count).indices
        hard_idx = negative_idx[hard_order]

        utility = (
            expert_score[sample_idx, positive_idx].mean(dim=0)
            - expert_score[sample_idx, hard_idx].mean(dim=0)
        )
        target = func.softmax(utility / temperature, dim=-1).detach()
        losses.append(
            -(target * func.log_softmax(prior_logits[sample_idx], dim=-1)).sum(),
        )

    if not losses:
        return prior_logits.sum() * 0
    return torch.stack(losses).mean()


class SampleExpertUtilityOnlySetCriterion(SetCriterion):
    """Calibration criterion that trains only the sample expert selector."""

    def __init__(
        self,
        *args: Any,
        sample_expert_utility_temperature: float = 0.02,
        sample_expert_utility_hard_negatives: int = 16,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sample_expert_utility_temperature = sample_expert_utility_temperature
        self.sample_expert_utility_hard_negatives = sample_expert_utility_hard_negatives

    def forward(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        meta: List[Dict[str, Any]],
        matching: Dict[str, Any],
    ) -> Dict[str, Any]:
        del targets, meta
        required = {
            "query_length_prior_logits",
            "expert_quality_residuals",
            "base_class_scores",
            "base_quality_scores",
        }
        if not required.issubset(outputs):
            return {}
        return {
            "loss_sample_expert_utility": sample_expert_utility_loss(
                outputs,
                matching["positive"]["indices"],
                temperature=self.sample_expert_utility_temperature,
                hard_negatives=self.sample_expert_utility_hard_negatives,
            ),
        }

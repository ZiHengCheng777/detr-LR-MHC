"""Duration-balanced routed-vs-uniform counterfactual ranking objective."""

from typing import Any, Dict, List

import torch
import torch.nn.functional as func
from torch import Tensor

from src.losses.losses import SetCriterion


def duration_group(sample_meta: Dict[str, Any]) -> int:
    durations = [end - start for start, end in sample_meta["relevant_windows"]]
    mean_duration = sum(durations) / len(durations)
    return 0 if mean_duration < 10 else 1 if mean_duration < 30 else 2


def balanced_counterfactual_rank_loss(
    outputs: Dict[str, Any],
    indices: List[Any],
    meta: List[Dict[str, Any]],
    margin: float = 0.002,
    temperature: float = 0.01,
    ranking_margin: float = 0.2,
    hard_negatives: int = 16,
) -> Tensor:
    """Require routed pair ranking to beat its uniform counterfactual per group."""
    actual = 0.5 * (
        func.logsigmoid(outputs["pred_logits"].squeeze(-1))
        + func.logsigmoid(outputs["pred_quality_scores"].squeeze(-1))
    )
    uniform = 0.5 * (
        func.logsigmoid(outputs["pred_uniform_logits"].squeeze(-1))
        + func.logsigmoid(outputs["pred_uniform_quality_scores"].squeeze(-1))
    ).detach()

    grouped_losses: List[List[Tensor]] = [[], [], []]
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

        actual_pair = func.softplus(
            ranking_margin
            - actual[sample_idx, positive_idx].unsqueeze(1)
            + actual[sample_idx, hard_idx].unsqueeze(0),
        ).mean()
        uniform_pair = func.softplus(
            ranking_margin
            - uniform[sample_idx, positive_idx].unsqueeze(1)
            + uniform[sample_idx, hard_idx].unsqueeze(0),
        ).mean()
        excess = actual_pair - uniform_pair + margin
        grouped_losses[duration_group(meta[sample_idx])].append(
            temperature * func.softplus(excess / temperature),
        )

    group_means = [torch.stack(values).mean() for values in grouped_losses if values]
    if not group_means:
        return actual.sum() * 0
    return torch.stack(group_means).mean()


class BalancedCounterfactualRankOnlySetCriterion(SetCriterion):
    """Router-only criterion using group-balanced counterfactual ranking."""

    def __init__(
        self,
        *args: Any,
        counterfactual_margin: float = 0.002,
        counterfactual_temperature: float = 0.01,
        counterfactual_hard_negatives: int = 16,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.counterfactual_margin = counterfactual_margin
        self.counterfactual_temperature = counterfactual_temperature
        self.counterfactual_hard_negatives = counterfactual_hard_negatives

    def forward(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        meta: List[Dict[str, Any]],
        matching: Dict[str, Any],
    ) -> Dict[str, Any]:
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
            "loss_balanced_counterfactual_rank": balanced_counterfactual_rank_loss(
                outputs,
                matching["positive"]["indices"],
                meta,
                margin=self.counterfactual_margin,
                temperature=self.counterfactual_temperature,
                ranking_margin=self.expert_ranking_margin,
                hard_negatives=self.counterfactual_hard_negatives,
            ),
        }

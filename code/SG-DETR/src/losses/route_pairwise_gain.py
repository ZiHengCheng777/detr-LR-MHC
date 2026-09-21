"""Pairwise objective that trains routed corrections against uniform ranking."""

from typing import Any, Dict, List

import torch
import torch.nn.functional as func
from torch import Tensor

from src.losses.losses import SetCriterion, create_targets_k_repeats
from src.losses.route_utility_distillation import route_utility_distillation_loss


def route_pairwise_gain_loss(
    outputs: Dict[str, Any],
    targets: Dict[str, Any],
    indices: List[Any],
    margin: float = 0.02,
    temperature: float = 0.02,
    hard_negatives: int = 8,
    width_power: float = 0.5,
) -> Tensor:
    """Make routed corrections improve positive-vs-hard-negative score gaps.

    The correction is measured relative to the uniform-expert counterfactual,
    so this objective cannot be satisfied by shifting all expert outputs equally.
    """
    if "route_correction_scores" in outputs:
        corrections: Tensor = outputs["route_correction_scores"].squeeze(-1)
    else:
        corrections = (
            outputs["route_gate_scores"] * outputs["route_delta_scores"]
        ).squeeze(-1)
    uniform_scores = 0.5 * (
        func.logsigmoid(outputs["pred_uniform_logits"].squeeze(-1))
        + func.logsigmoid(outputs["pred_uniform_quality_scores"].squeeze(-1))
    ).detach()
    losses = []
    weights = []
    for sample_idx, (positive_idx, target_idx) in enumerate(indices):
        if positive_idx.numel() == 0:
            continue
        positive_idx = positive_idx.to(corrections.device)
        target_idx = target_idx.to(corrections.device)
        negative_mask = torch.ones(
            corrections.shape[1],
            dtype=torch.bool,
            device=corrections.device,
        )
        negative_mask[positive_idx] = False
        negative_idx = torch.nonzero(negative_mask, as_tuple=False).squeeze(-1)
        if negative_idx.numel() == 0:
            continue
        hard_count = min(hard_negatives, negative_idx.numel())
        hard_order = torch.topk(
            uniform_scores[sample_idx, negative_idx],
            hard_count,
        ).indices
        hard_idx = negative_idx[hard_order]
        correction_gap = (
            corrections[sample_idx, positive_idx].unsqueeze(1)
            - corrections[sample_idx, hard_idx].unsqueeze(0)
        )
        pair_loss = temperature * func.softplus(
            (margin - correction_gap) / temperature,
        )
        losses.append(pair_loss.mean(dim=1))
        target_widths = targets["span_labels"][sample_idx]["spans"][target_idx, 1]
        weights.append(target_widths.to(corrections.device).clamp_min(0.02).pow(-width_power))

    if not losses:
        return corrections.sum() * 0
    loss = torch.cat(losses)
    weight = torch.cat(weights)
    weight = weight / weight.mean().clamp_min(torch.finfo(weight.dtype).eps)
    return (loss * weight).mean()


def route_gate_relevance_loss(
    outputs: Dict[str, Any],
    indices: List[Any],
    hard_negatives: int = 8,
) -> Tensor:
    """Separate matched queries from the strongest uniform hard negatives."""
    gates: Tensor = outputs["route_gate_scores"].squeeze(-1).clamp(1e-6, 1 - 1e-6)
    uniform_scores = 0.5 * (
        func.logsigmoid(outputs["pred_uniform_logits"].squeeze(-1))
        + func.logsigmoid(outputs["pred_uniform_quality_scores"].squeeze(-1))
    ).detach()
    positive_losses = []
    negative_losses = []
    for sample_idx, (positive_idx, _) in enumerate(indices):
        if positive_idx.numel() == 0:
            continue
        positive_idx = positive_idx.to(gates.device)
        negative_mask = torch.ones(
            gates.shape[1],
            dtype=torch.bool,
            device=gates.device,
        )
        negative_mask[positive_idx] = False
        negative_idx = torch.nonzero(negative_mask, as_tuple=False).squeeze(-1)
        if negative_idx.numel() == 0:
            continue
        hard_count = min(hard_negatives, negative_idx.numel())
        hard_order = torch.topk(
            uniform_scores[sample_idx, negative_idx],
            hard_count,
        ).indices
        hard_idx = negative_idx[hard_order]
        positive_losses.append(-gates[sample_idx, positive_idx].log().mean())
        negative_losses.append(-(1 - gates[sample_idx, hard_idx]).log().mean())

    if not positive_losses:
        return gates.sum() * 0
    return 0.5 * (
        torch.stack(positive_losses).mean()
        + torch.stack(negative_losses).mean()
    )


class PairwiseGainSetCriterion(SetCriterion):
    """Set criterion with a routing-only pairwise ranking objective."""

    def __init__(
        self,
        *args: Any,
        route_pairwise_margin: float = 0.02,
        route_pairwise_temperature: float = 0.02,
        route_pairwise_hard_negatives: int = 8,
        route_pairwise_width_power: float = 0.5,
        route_utility_temperature: float = 0.02,
        route_utility_minimum_range: float = 1e-4,
        route_utility_negative_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.route_pairwise_margin = route_pairwise_margin
        self.route_pairwise_temperature = route_pairwise_temperature
        self.route_pairwise_hard_negatives = route_pairwise_hard_negatives
        self.route_pairwise_width_power = route_pairwise_width_power
        self.route_utility_temperature = route_utility_temperature
        self.route_utility_minimum_range = route_utility_minimum_range
        self.route_utility_negative_weight = route_utility_negative_weight

    def forward(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        meta: List[Dict[str, Any]],
        matching: Dict[str, Any],
    ) -> Dict[str, Any]:
        losses = super().forward(outputs, targets, meta, matching)
        required = {
            "route_gate_scores",
            "route_delta_scores",
            "pred_uniform_logits",
            "pred_uniform_quality_scores",
        }
        missing = required.difference(outputs)
        if missing and self.weight_dict.get("loss_route_pairwise_gain", 0) > 0:
            raise RuntimeError(
                "counterfactual routing loss requires model outputs: "
                + ", ".join(sorted(missing)),
            )
        if missing:
            return losses
        retrieval_targets = (
            targets
            if self.one2one
            else create_targets_k_repeats(targets, self.target_repeat)
        )
        losses["loss_route_pairwise_gain"] = route_pairwise_gain_loss(
            outputs,
            retrieval_targets,
            matching["positive"]["indices"],
            margin=self.route_pairwise_margin,
            temperature=self.route_pairwise_temperature,
            hard_negatives=self.route_pairwise_hard_negatives,
            width_power=self.route_pairwise_width_power,
        )
        utility_required = {
            "route_logits",
            "expert_quality_residuals",
            "base_class_scores",
            "base_quality_scores",
        }
        utility_missing = utility_required.difference(outputs)
        if utility_missing and self.weight_dict.get("loss_route_utility", 0) > 0:
            raise RuntimeError(
                "route utility loss requires model outputs: "
                + ", ".join(sorted(utility_missing)),
            )
        if not utility_missing:
            losses["loss_route_utility"] = route_utility_distillation_loss(
                outputs,
                matching["positive"]["indices"],
                temperature=self.route_utility_temperature,
                hard_negatives=self.route_pairwise_hard_negatives,
                minimum_utility_range=self.route_utility_minimum_range,
                negative_weight=self.route_utility_negative_weight,
            )
        losses["loss_route_gate_relevance"] = route_gate_relevance_loss(
            outputs,
            matching["positive"]["indices"],
            hard_negatives=self.route_pairwise_hard_negatives,
        )
        return losses

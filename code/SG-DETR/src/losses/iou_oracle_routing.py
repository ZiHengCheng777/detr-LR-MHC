"""Query-level routing supervision derived from all predicted candidate spans."""

from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as func
from torch import Tensor

from src.losses.balanced_counterfactual_rank import duration_group
from src.losses.losses import SetCriterion
from src.utils.span_utils import span_cxw_to_xx, temporal_iou


def max_candidate_iou(pred_spans: Tensor, gt_spans: Tensor) -> Tensor:
    """Return each candidate's maximum IoU with any ground-truth span."""
    if gt_spans.numel() == 0:
        return pred_spans.new_zeros(pred_spans.shape[0])
    ious, _ = temporal_iou(
        span_cxw_to_xx(pred_spans.detach()),
        span_cxw_to_xx(gt_spans.to(pred_spans.device)),
    )
    return ious.amax(dim=1).detach()


def _smooth_ap(positive_scores: Tensor, negative_scores: Tensor, temperature: float) -> Tensor:
    outranking = torch.sigmoid(
        (negative_scores.unsqueeze(0) - positive_scores.unsqueeze(1)) / temperature,
    ).sum(dim=1)
    return (1.0 / (1.0 + outranking)).mean()


def _expert_oracle_distribution(
    outputs: Dict[str, Any],
    sample_idx: int,
    candidate_iou: Tensor,
    temperature: float,
) -> Tensor:
    residuals = outputs["expert_quality_residuals"][sample_idx]
    expert_quality = outputs["base_quality_scores"][sample_idx] + residuals
    target = candidate_iou.unsqueeze(-1).expand_as(expert_quality)
    quality_cost = func.binary_cross_entropy_with_logits(
        expert_quality.detach(),
        target,
        reduction="none",
    )

    # The deployed residual also changes classification. Reward experts that
    # separate high-IoU candidates from background instead of fitting IoU alone.
    expert_class = outputs["base_class_scores"][sample_idx] + residuals
    relevance = (candidate_iou >= 0.5).to(expert_class.dtype).unsqueeze(-1).expand_as(expert_class)
    class_cost = func.binary_cross_entropy_with_logits(
        expert_class.detach(),
        relevance,
        reduction="none",
    )
    return torch.softmax(-(quality_cost + class_cost) / temperature, dim=-1)


def _gate_oracle_loss(
    outputs: Dict[str, Any],
    sample_idx: int,
    candidate_iou: Tensor,
    temperature: float,
    target_mode: str,
    utility_margin: float,
) -> Tensor:
    """Supervise whether applying the complete routed correction beats uniform."""
    uniform_class = outputs["pred_uniform_logits"][sample_idx].detach()
    uniform_quality = outputs["pred_uniform_quality_scores"][sample_idx].detach()
    full_class = uniform_class + outputs["route_delta_scores"][sample_idx].detach()
    full_quality = uniform_quality + outputs["route_delta_scores"][sample_idx].detach()
    relevance = (candidate_iou >= 0.5).to(uniform_class.dtype).unsqueeze(-1)
    quality_target = candidate_iou.unsqueeze(-1)

    if target_mode == "cost":
        uniform_cost = func.binary_cross_entropy_with_logits(
            uniform_class,
            relevance,
            reduction="none",
        ) + func.binary_cross_entropy_with_logits(
            uniform_quality,
            quality_target,
            reduction="none",
        )
        routed_cost = func.binary_cross_entropy_with_logits(
            full_class,
            relevance,
            reduction="none",
        ) + func.binary_cross_entropy_with_logits(
            full_quality,
            quality_target,
            reduction="none",
        )
        gate_target = torch.sigmoid((uniform_cost - routed_cost) / temperature)
    elif target_mode == "rank_utility":
        uniform_candidate_rank = 0.5 * (
            func.logsigmoid(uniform_class) + func.logsigmoid(uniform_quality)
        )
        routed_candidate_rank = 0.5 * (
            func.logsigmoid(full_class) + func.logsigmoid(full_quality)
        )
        direction = relevance.mul(2).sub(1)
        signed_utility = direction * (routed_candidate_rank - uniform_candidate_rank)
        gate_target = torch.sigmoid(
            (signed_utility - utility_margin) / temperature,
        )
    else:
        raise ValueError(f"unknown IoU gate target mode: {target_mode}")
    gate_prediction = outputs["route_gate_scores"][sample_idx].clamp(1e-5, 1 - 1e-5)
    gate_loss = func.binary_cross_entropy(gate_prediction, gate_target, reduction="none").squeeze(-1)

    # High-IoU and high-ranked candidates dominate moment AP, but retain a
    # background term so the gate learns when not to apply a correction.
    uniform_rank = 0.5 * (
        func.logsigmoid(uniform_class.squeeze(-1))
        + func.logsigmoid(uniform_quality.squeeze(-1))
    )
    rank_weight = torch.softmax(uniform_rank / 0.1, dim=0) * uniform_rank.numel()
    weights = (0.2 + 0.8 * candidate_iou) * (0.5 + 0.5 * rank_weight)
    return (gate_loss * weights).sum() / weights.sum().clamp_min(1e-6)


def iou_oracle_routing_loss(
    outputs: Dict[str, Any],
    targets: Dict[str, Any],
    meta: List[Dict[str, Any]],
    oracle_temperature: float = 0.05,
    positive_iou: float = 0.3,
    uniform_margin: float = 0.002,
    safety_temperature: float = 0.01,
    ranking_temperature: float = 0.02,
    hard_negatives: int = 32,
    gate_oracle_temperature: float = 0.05,
    gate_oracle_weight: float = 1.0,
    gate_target_mode: str = "cost",
    gate_utility_margin: float = 0.0,
    direct_rank_weight: float = 0.0,
    direct_rank_margin: float = 0.0,
    oracle_weight: float = 1.0,
    safety_weight: float = 1.0,
    thresholds: Sequence[float] = (0.5, 0.7),
) -> Tensor:
    """Fit soft expert-oracle labels and prevent worse ranking than uniform."""
    route_log_probs = func.log_softmax(outputs["route_logits"], dim=-1)
    actual = 0.5 * (
        func.logsigmoid(outputs["pred_logits"].squeeze(-1))
        + func.logsigmoid(outputs["pred_quality_scores"].squeeze(-1))
    )
    uniform = 0.5 * (
        func.logsigmoid(outputs["pred_uniform_logits"].squeeze(-1))
        + func.logsigmoid(outputs["pred_uniform_quality_scores"].squeeze(-1))
    ).detach()

    grouped: List[List[Tensor]] = [[], [], []]
    for sample_idx, target in enumerate(targets["span_labels"]):
        candidate_iou = max_candidate_iou(
            outputs["pred_spans"][sample_idx],
            target["spans"],
        )
        oracle = _expert_oracle_distribution(
            outputs,
            sample_idx,
            candidate_iou,
            oracle_temperature,
        )
        weights = 0.25 + 0.75 * candidate_iou
        useful = candidate_iou >= positive_iou
        if useful.any():
            oracle_loss = -(
                oracle[useful] * route_log_probs[sample_idx, useful]
            ).sum(dim=-1)
            oracle_loss = (oracle_loss * weights[useful]).sum() / weights[useful].sum()
        else:
            oracle_loss = route_log_probs[sample_idx].sum() * 0

        safety_terms: List[Tensor] = []
        direct_rank_terms: List[Tensor] = []
        for threshold in thresholds:
            positive = candidate_iou >= threshold
            negative = candidate_iou < threshold
            if not positive.any() or not negative.any():
                continue
            negative_idx = torch.nonzero(negative, as_tuple=False).squeeze(-1)
            count = min(hard_negatives, negative_idx.numel())
            hard_order = torch.topk(uniform[sample_idx, negative_idx], count).indices
            hard_idx = negative_idx[hard_order]
            actual_ap = _smooth_ap(
                actual[sample_idx, positive],
                actual[sample_idx, hard_idx],
                ranking_temperature,
            )
            uniform_ap = _smooth_ap(
                uniform[sample_idx, positive],
                uniform[sample_idx, hard_idx],
                ranking_temperature,
            )
            deficit = uniform_ap - actual_ap + uniform_margin
            safety_terms.append(safety_temperature * func.softplus(deficit / safety_temperature))
            positive_scores = actual[sample_idx, positive]
            negative_scores = actual[sample_idx, hard_idx]
            pairwise_deficit = (
                negative_scores.unsqueeze(0)
                - positive_scores.unsqueeze(1)
                + direct_rank_margin
            )
            direct_rank_terms.append(
                ranking_temperature
                * func.softplus(pairwise_deficit / ranking_temperature).mean(),
            )
        safety_loss = torch.stack(safety_terms).mean() if safety_terms else oracle_loss * 0
        direct_rank_loss = (
            torch.stack(direct_rank_terms).mean()
            if direct_rank_terms
            else oracle_loss * 0
        )
        gate_loss = oracle_loss * 0
        if gate_oracle_weight > 0 and {"route_gate_scores", "route_delta_scores"}.issubset(outputs):
            gate_loss = _gate_oracle_loss(
                outputs,
                sample_idx,
                candidate_iou,
                gate_oracle_temperature,
                gate_target_mode,
                gate_utility_margin,
            )
        grouped[duration_group(meta[sample_idx])].append(
            oracle_weight * oracle_loss
            + safety_weight * safety_loss
            + gate_oracle_weight * gate_loss
            + direct_rank_weight * direct_rank_loss,
        )

    group_means = [torch.stack(values).mean() for values in grouped if values]
    if not group_means:
        return route_log_probs.sum() * 0
    return torch.stack(group_means).mean()


class IoUOracleRoutingOnlySetCriterion(SetCriterion):
    """Router-only criterion using all-query IoU-derived expert labels."""

    def __init__(
        self,
        *args: Any,
        iou_oracle_temperature: float = 0.05,
        iou_oracle_positive_iou: float = 0.3,
        iou_oracle_uniform_margin: float = 0.002,
        iou_oracle_safety_temperature: float = 0.01,
        iou_ranking_temperature: float = 0.02,
        iou_hard_negatives: int = 32,
        iou_gate_oracle_temperature: float = 0.05,
        iou_gate_oracle_weight: float = 1.0,
        iou_gate_target_mode: str = "cost",
        iou_gate_utility_margin: float = 0.0,
        iou_direct_rank_weight: float = 0.0,
        iou_direct_rank_margin: float = 0.0,
        iou_oracle_weight: float = 1.0,
        iou_safety_weight: float = 1.0,
        iou_thresholds: Sequence[float] = (0.5, 0.7),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.iou_oracle_temperature = iou_oracle_temperature
        self.iou_oracle_positive_iou = iou_oracle_positive_iou
        self.iou_oracle_uniform_margin = iou_oracle_uniform_margin
        self.iou_oracle_safety_temperature = iou_oracle_safety_temperature
        self.iou_ranking_temperature = iou_ranking_temperature
        self.iou_hard_negatives = iou_hard_negatives
        self.iou_gate_oracle_temperature = iou_gate_oracle_temperature
        self.iou_gate_oracle_weight = iou_gate_oracle_weight
        self.iou_gate_target_mode = iou_gate_target_mode
        self.iou_gate_utility_margin = iou_gate_utility_margin
        self.iou_direct_rank_weight = iou_direct_rank_weight
        self.iou_direct_rank_margin = iou_direct_rank_margin
        self.iou_oracle_weight = iou_oracle_weight
        self.iou_safety_weight = iou_safety_weight
        self.iou_thresholds = tuple(iou_thresholds)

    def forward(self, outputs, targets, meta, matching):
        del matching
        required = {
            "pred_logits",
            "pred_quality_scores",
            "pred_uniform_logits",
            "pred_uniform_quality_scores",
            "pred_spans",
            "route_logits",
            "expert_quality_residuals",
            "base_quality_scores",
            "base_class_scores",
        }
        if not required.issubset(outputs):
            return {}
        return {
            "loss_iou_oracle_routing": iou_oracle_routing_loss(
                outputs,
                targets,
                meta,
                oracle_temperature=self.iou_oracle_temperature,
                positive_iou=self.iou_oracle_positive_iou,
                uniform_margin=self.iou_oracle_uniform_margin,
                safety_temperature=self.iou_oracle_safety_temperature,
                ranking_temperature=self.iou_ranking_temperature,
                hard_negatives=self.iou_hard_negatives,
                gate_oracle_temperature=self.iou_gate_oracle_temperature,
                gate_oracle_weight=self.iou_gate_oracle_weight,
                gate_target_mode=self.iou_gate_target_mode,
                gate_utility_margin=self.iou_gate_utility_margin,
                direct_rank_weight=self.iou_direct_rank_weight,
                direct_rank_margin=self.iou_direct_rank_margin,
                oracle_weight=self.iou_oracle_weight,
                safety_weight=self.iou_safety_weight,
                thresholds=self.iou_thresholds,
            ),
        }

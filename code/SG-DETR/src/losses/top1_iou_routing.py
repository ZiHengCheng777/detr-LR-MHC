"""Listwise top-1 supervision for routed moment scores."""

from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as func
from torch import Tensor

from src.losses.balanced_counterfactual_rank import duration_group
from src.losses.iou_oracle_routing import max_candidate_iou
from src.losses.losses import SetCriterion, create_targets_k_repeats


def top1_iou_routing_loss(
    outputs: Dict[str, Any],
    targets: Dict[str, Any],
    meta: List[Dict[str, Any]],
    score_temperature: float = 0.1,
    safety_margin: float = 0.0,
    safety_temperature: float = 0.02,
    direct_weight: float = 1.0,
) -> Tensor:
    """Rank the candidate with maximum GT IoU above all other queries."""
    routed_score = 0.5 * (
        func.logsigmoid(outputs["pred_logits"].squeeze(-1))
        + func.logsigmoid(outputs["pred_quality_scores"].squeeze(-1))
    )
    uniform_quality = outputs.get(
        "pred_uniform_quality_scores",
        outputs["pred_quality_scores"],
    )
    uniform_score = 0.5 * (
        func.logsigmoid(outputs["pred_uniform_logits"].squeeze(-1))
        + func.logsigmoid(uniform_quality.squeeze(-1))
    ).detach()

    grouped: List[List[Tensor]] = [[], [], []]
    for sample_idx, target in enumerate(targets["span_labels"]):
        candidate_iou = max_candidate_iou(
            outputs["pred_spans"][sample_idx],
            target["spans"],
        )
        if candidate_iou.numel() == 0:
            continue
        best_idx = candidate_iou.argmax().view(1)
        routed_ce = func.cross_entropy(
            routed_score[sample_idx].unsqueeze(0) / score_temperature,
            best_idx,
        )
        uniform_ce = func.cross_entropy(
            uniform_score[sample_idx].unsqueeze(0) / score_temperature,
            best_idx,
        )
        deficit = routed_ce - uniform_ce + safety_margin
        safety = safety_temperature * func.softplus(deficit / safety_temperature)
        grouped[duration_group(meta[sample_idx])].append(
            direct_weight * routed_ce + safety,
        )

    group_means = [torch.stack(values).mean() for values in grouped if values]
    if not group_means:
        return routed_score.sum() * 0
    return torch.stack(group_means).mean()


class Top1IoURoutingOnlySetCriterion(SetCriterion):
    """Train LR-MHC modules against the final top-1 retrieval objective."""

    def __init__(
        self,
        *args: Any,
        top1_score_temperature: float = 0.1,
        top1_safety_margin: float = 0.0,
        top1_safety_temperature: float = 0.02,
        top1_direct_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.top1_score_temperature = top1_score_temperature
        self.top1_safety_margin = top1_safety_margin
        self.top1_safety_temperature = top1_safety_temperature
        self.top1_direct_weight = top1_direct_weight

    def forward(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        meta: List[Dict[str, Any]],
        matching: Dict[str, Any],
    ) -> Dict[str, Tensor]:
        del matching
        required = {
            "pred_logits",
            "pred_quality_scores",
            "pred_uniform_logits",
            "pred_spans",
        }
        if not required.issubset(outputs):
            return {}
        return {
            "loss_top1_iou_routing": top1_iou_routing_loss(
                outputs,
                targets,
                meta,
                score_temperature=self.top1_score_temperature,
                safety_margin=self.top1_safety_margin,
                safety_temperature=self.top1_safety_temperature,
                direct_weight=self.top1_direct_weight,
            ),
        }


def routed_ap_loss(
    outputs: Dict[str, Any],
    targets: Dict[str, Any],
    meta: List[Dict[str, Any]],
    rank_temperature: float = 0.05,
    iou_thresholds: Sequence[float] = (
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
    ),
    safety_margin: float = 0.0,
    safety_temperature: float = 0.02,
    direct_weight: float = 1.0,
) -> Tensor:
    """Optimize final routed AP while balancing the three duration groups."""
    routed_score = 0.5 * (
        func.logsigmoid(outputs["pred_logits"].squeeze(-1))
        + func.logsigmoid(outputs["pred_quality_scores"].squeeze(-1))
    )
    uniform_quality = outputs.get(
        "pred_uniform_quality_scores",
        outputs["pred_quality_scores"],
    )
    uniform_score = 0.5 * (
        func.logsigmoid(outputs["pred_uniform_logits"].squeeze(-1))
        + func.logsigmoid(uniform_quality.squeeze(-1))
    ).detach()

    candidate_ious = []
    for sample_idx, target in enumerate(targets["span_labels"]):
        candidate_ious.append(
            max_candidate_iou(
                outputs["pred_spans"][sample_idx].detach(),
                target["spans"],
            ),
        )

    candidate_iou = torch.stack(candidate_ious)
    thresholds = candidate_iou.new_tensor(iou_thresholds)[None, :, None]
    positive = candidate_iou[:, None, :] >= thresholds
    negative = ~positive
    pair_mask = positive[:, :, :, None] & negative[:, :, None, :]
    valid = pair_mask.any(dim=(-1, -2))

    def pairwise_loss(scores: Tensor) -> Tensor:
        score_difference = (
            scores[:, None, :, None] - scores[:, None, None, :]
        ) / rank_temperature
        losses = func.softplus(-score_difference) * pair_mask
        return losses.sum(dim=(-1, -2)) / pair_mask.sum(
            dim=(-1, -2),
        ).clamp_min(1)

    routed_values = pairwise_loss(routed_score)
    uniform_values = pairwise_loss(uniform_score)
    deficit = routed_values - uniform_values + safety_margin
    values = direct_weight * routed_values + safety_temperature * func.softplus(
        deficit / safety_temperature,
    )
    values = (values * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)

    grouped: List[List[Tensor]] = [[], [], []]
    for sample_idx, sample_value in enumerate(values):
        if valid[sample_idx].any():
            grouped[duration_group(meta[sample_idx])].append(sample_value)
    group_means = [torch.stack(group).mean() for group in grouped if group]
    if not group_means:
        return routed_score.sum() * 0
    return torch.stack(group_means).mean()


class APTop1RoutingOnlySetCriterion(Top1IoURoutingOnlySetCriterion):
    """Jointly optimize duration-balanced AP and top-1 query selection."""

    def __init__(
        self,
        *args: Any,
        routed_ap_rank_temperature: float = 0.05,
        routed_ap_iou_thresholds: Sequence[float] = (
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
        ),
        routed_ap_safety_margin: float = 0.0,
        routed_ap_safety_temperature: float = 0.02,
        routed_ap_direct_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.routed_ap_rank_temperature = routed_ap_rank_temperature
        self.routed_ap_iou_thresholds = tuple(routed_ap_iou_thresholds)
        self.routed_ap_safety_margin = routed_ap_safety_margin
        self.routed_ap_safety_temperature = routed_ap_safety_temperature
        self.routed_ap_direct_weight = routed_ap_direct_weight

    def forward(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        meta: List[Dict[str, Any]],
        matching: Dict[str, Any],
    ) -> Dict[str, Tensor]:
        losses = super().forward(outputs, targets, meta, matching)
        required = {
            "pred_logits",
            "pred_quality_scores",
            "pred_uniform_logits",
            "pred_spans",
        }
        if required.issubset(outputs):
            losses["loss_routed_ap"] = routed_ap_loss(
                outputs,
                targets,
                meta,
                rank_temperature=self.routed_ap_rank_temperature,
                iou_thresholds=self.routed_ap_iou_thresholds,
                safety_margin=self.routed_ap_safety_margin,
                safety_temperature=self.routed_ap_safety_temperature,
                direct_weight=self.routed_ap_direct_weight,
            )
        return losses


class UnifiedLRMHCSetCriterion(APTop1RoutingOnlySetCriterion):
    """Train only the routed ranking, expert, and optional soft-alignment terms."""

    def forward(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        meta: List[Dict[str, Any]],
        matching: Dict[str, Any],
    ) -> Dict[str, Tensor]:
        losses = super().forward(outputs, targets, meta, matching)
        outputs_without_aux = {
            key: value for key, value in outputs.items() if key != "aux_outputs"
        }
        retrieval_targets = (
            targets
            if self.one2one
            else create_targets_k_repeats(targets, self.target_repeat)
        )
        indices = matching["positive"]["indices"]
        losses.update(
            self._route_alignment_loss(
                outputs_without_aux,
                retrieval_targets,
                indices,
            ),
        )
        losses.update(
            self._expert_quality_loss(
                outputs_without_aux,
                retrieval_targets,
                indices,
            ),
        )
        losses.update(
            self._expert_specialization_loss(
                outputs_without_aux,
                retrieval_targets,
                indices,
            ),
        )
        losses.update(
            self._route_safety_loss(
                outputs_without_aux,
                retrieval_targets,
                indices,
            ),
        )
        return losses


class UnifiedTop1LRMHCSetCriterion(Top1IoURoutingOnlySetCriterion):
    """Train routed top-1 ranking with an optional soft width-alignment term."""

    def forward(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        meta: List[Dict[str, Any]],
        matching: Dict[str, Any],
    ) -> Dict[str, Tensor]:
        losses = super().forward(outputs, targets, meta, matching)
        outputs_without_aux = {
            key: value for key, value in outputs.items() if key != "aux_outputs"
        }
        retrieval_targets = (
            targets
            if self.one2one
            else create_targets_k_repeats(targets, self.target_repeat)
        )
        losses.update(
            self._route_alignment_loss(
                outputs_without_aux,
                retrieval_targets,
                matching["positive"]["indices"],
            ),
        )
        return losses


class FixedRoutingSetCriterion(SetCriterion):
    """Run the standard DETR losses while accepting unified-run ranking options."""

    def __init__(
        self,
        *args: Any,
        top1_score_temperature: float = 0.1,
        top1_safety_margin: float = 0.0,
        top1_safety_temperature: float = 0.02,
        top1_direct_weight: float = 1.0,
        routed_ap_rank_temperature: float = 0.05,
        routed_ap_safety_margin: float = 0.0,
        routed_ap_safety_temperature: float = 0.02,
        routed_ap_direct_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        del (
            top1_score_temperature,
            top1_safety_margin,
            top1_safety_temperature,
            top1_direct_weight,
            routed_ap_rank_temperature,
            routed_ap_safety_margin,
            routed_ap_safety_temperature,
            routed_ap_direct_weight,
        )
        super().__init__(*args, **kwargs)

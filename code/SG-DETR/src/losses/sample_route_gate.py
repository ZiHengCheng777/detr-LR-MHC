"""Sample-level supervision for selectively enabling the soft router."""

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as func
from torch import Tensor

from src.losses.balanced_counterfactual_rank import duration_group
from src.losses.iou_oracle_routing import max_candidate_iou
from src.losses.losses import SetCriterion


def _smooth_ap(
    positive_scores: Tensor,
    negative_scores: Tensor,
    temperature: float,
) -> Tensor:
    outranking = torch.sigmoid(
        (negative_scores.unsqueeze(0) - positive_scores.unsqueeze(1)) / temperature,
    ).sum(dim=1)
    return (1.0 / (1.0 + outranking)).mean()


def sample_route_gate_loss(
    outputs: Dict[str, Any],
    indices: List[Any],
    temperature: float = 0.002,
    hard_negatives: int = 16,
    balance_targets: bool = False,
    meta: Optional[List[Dict[str, Any]]] = None,
    balance_duration_groups: bool = False,
    retrieval_targets: Optional[Dict[str, Any]] = None,
    target_mode: str = "matched_gap",
    ap_rank_temperature: float = 0.02,
    ap_iou_thresholds: Sequence[float] = (0.5, 0.7),
    return_diagnostics: bool = False,
) -> Any:
    """Predict whether the ungated routed correction improves sample ranking."""
    gate_logits: Tensor = outputs["sample_route_gate_logits"]
    correction = (
        outputs["base_route_gate_scores"] * outputs["route_delta_scores"]
    ).squeeze(-1).detach()
    uniform_class = outputs["pred_uniform_logits"].squeeze(-1).detach()
    uniform_quality = outputs["pred_uniform_quality_scores"].squeeze(-1).detach()
    uniform_score = 0.5 * (
        func.logsigmoid(uniform_class) + func.logsigmoid(uniform_quality)
    )
    candidate_score = 0.5 * (
        func.logsigmoid(uniform_class + correction)
        + func.logsigmoid(uniform_quality + correction)
    )

    gate_targets = []
    selected_logits = []
    duration_groups = []
    for sample_idx, (positive_idx, _) in enumerate(indices):
        positive_idx = positive_idx.to(gate_logits.device)
        if positive_idx.numel() == 0:
            continue
        negative_mask = torch.ones(
            uniform_score.shape[1],
            dtype=torch.bool,
            device=gate_logits.device,
        )
        negative_mask[positive_idx] = False
        negative_idx = torch.nonzero(negative_mask, as_tuple=False).squeeze(-1)
        if negative_idx.numel() == 0:
            continue
        count = min(hard_negatives, negative_idx.numel())
        hard_order = torch.topk(uniform_score[sample_idx, negative_idx], count).indices
        hard_idx = negative_idx[hard_order]

        if target_mode == "matched_gap":
            uniform_gap = (
                uniform_score[sample_idx, positive_idx].mean()
                - uniform_score[sample_idx, hard_idx].mean()
            )
            candidate_gap = (
                candidate_score[sample_idx, positive_idx].mean()
                - candidate_score[sample_idx, hard_idx].mean()
            )
            improvement = candidate_gap - uniform_gap
        elif target_mode == "iou_ap":
            if retrieval_targets is None:
                raise ValueError("iou_ap sample gate targets require retrieval targets")
            candidate_iou = max_candidate_iou(
                outputs["pred_spans"][sample_idx],
                retrieval_targets["span_labels"][sample_idx]["spans"],
            )
            improvements = []
            for iou_threshold in ap_iou_thresholds:
                positive = candidate_iou >= iou_threshold
                negative = ~positive
                if not positive.any() or not negative.any():
                    continue
                uniform_ap = _smooth_ap(
                    uniform_score[sample_idx, positive],
                    uniform_score[sample_idx, negative],
                    ap_rank_temperature,
                )
                candidate_ap = _smooth_ap(
                    candidate_score[sample_idx, positive],
                    candidate_score[sample_idx, negative],
                    ap_rank_temperature,
                )
                improvements.append(candidate_ap - uniform_ap)
            if not improvements:
                continue
            improvement = torch.stack(improvements).mean()
        else:
            raise ValueError(f"unknown sample route gate target mode: {target_mode}")
        gate_targets.append(torch.sigmoid(improvement / temperature))
        selected_logits.append(gate_logits[sample_idx])
        duration_groups.append(duration_group(meta[sample_idx]) if meta is not None else 0)

    if not gate_targets:
        zero = gate_logits.sum() * 0
        diagnostics = (zero,) * 7
        return diagnostics if return_diagnostics else zero
    selected_logits_tensor = torch.stack(selected_logits)
    target_tensor = torch.stack(gate_targets).detach()
    per_sample_loss = func.binary_cross_entropy_with_logits(
        selected_logits_tensor,
        target_tensor,
        reduction="none",
    )
    if balance_duration_groups:
        group_losses = []
        group_ids = torch.as_tensor(duration_groups, device=target_tensor.device)
        for group_idx in range(3):
            mask = group_ids == group_idx
            if not mask.any():
                continue
            group_targets = target_tensor[mask]
            group_raw_loss = per_sample_loss[mask]
            if balance_targets:
                positive_mass = group_targets.mean().clamp(1e-3, 1 - 1e-3)
                weights = (
                    group_targets * (1 - positive_mass)
                    + (1 - group_targets) * positive_mass
                )
                group_losses.append(
                    (group_raw_loss * weights).sum() / weights.sum().clamp_min(1e-6),
                )
            else:
                group_losses.append(group_raw_loss.mean())
        loss = torch.stack(group_losses).mean()
    elif balance_targets:
        positive_mass = target_tensor.mean().clamp(1e-3, 1 - 1e-3)
        weights = (
            target_tensor * (1 - positive_mass)
            + (1 - target_tensor) * positive_mass
        )
        loss = (per_sample_loss * weights).sum() / weights.sum().clamp_min(1e-6)
    else:
        loss = per_sample_loss.mean()
    if not return_diagnostics:
        return loss
    group_ids = torch.as_tensor(duration_groups, device=target_tensor.device)
    group_positive_rates = []
    for group_idx in range(3):
        group_targets = target_tensor[group_ids == group_idx]
        group_positive_rates.append(
            (group_targets >= 0.5).float().mean()
            if group_targets.numel()
            else target_tensor.new_zeros(()),
        )
    return (
        loss,
        target_tensor.mean(),
        (target_tensor >= 0.5).float().mean(),
        torch.sigmoid(selected_logits_tensor).mean(),
        *group_positive_rates,
    )


class SampleRouteGateOnlySetCriterion(SetCriterion):
    """Calibration criterion that trains only the sample-level route gate."""

    def __init__(
        self,
        *args: Any,
        sample_route_gate_temperature: float = 0.002,
        sample_route_gate_hard_negatives: int = 16,
        sample_route_gate_balance_targets: bool = False,
        sample_route_gate_balance_duration_groups: bool = False,
        sample_route_gate_target_mode: str = "matched_gap",
        sample_route_gate_ap_rank_temperature: float = 0.02,
        sample_route_gate_ap_iou_thresholds: Sequence[float] = (0.5, 0.7),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sample_route_gate_temperature = sample_route_gate_temperature
        self.sample_route_gate_hard_negatives = sample_route_gate_hard_negatives
        self.sample_route_gate_balance_targets = sample_route_gate_balance_targets
        self.sample_route_gate_balance_duration_groups = (
            sample_route_gate_balance_duration_groups
        )
        self.sample_route_gate_target_mode = sample_route_gate_target_mode
        self.sample_route_gate_ap_rank_temperature = sample_route_gate_ap_rank_temperature
        self.sample_route_gate_ap_iou_thresholds = tuple(
            sample_route_gate_ap_iou_thresholds,
        )

    def forward(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        meta: List[Dict[str, Any]],
        matching: Dict[str, Any],
    ) -> Dict[str, Any]:
        required = {
            "sample_route_gate_logits",
            "base_route_gate_scores",
            "route_delta_scores",
            "pred_uniform_logits",
            "pred_uniform_quality_scores",
        }
        if not required.issubset(outputs):
            return {}
        (
            loss,
            target_mean,
            positive_rate,
            prediction_mean,
            short_positive_rate,
            middle_positive_rate,
            long_positive_rate,
        ) = sample_route_gate_loss(
            outputs,
            matching["positive"]["indices"],
            temperature=self.sample_route_gate_temperature,
            hard_negatives=self.sample_route_gate_hard_negatives,
            balance_targets=self.sample_route_gate_balance_targets,
            meta=meta,
            balance_duration_groups=self.sample_route_gate_balance_duration_groups,
            retrieval_targets=targets,
            target_mode=self.sample_route_gate_target_mode,
            ap_rank_temperature=self.sample_route_gate_ap_rank_temperature,
            ap_iou_thresholds=self.sample_route_gate_ap_iou_thresholds,
            return_diagnostics=True,
        )
        return {
            "loss_sample_route_gate": loss,
            "sample_route_target_mean": target_mean,
            "sample_route_target_positive_rate": positive_rate,
            "sample_route_prediction_mean": prediction_mean,
            "sample_route_short_positive_rate": short_positive_rate,
            "sample_route_middle_positive_rate": middle_positive_rate,
            "sample_route_long_positive_rate": long_positive_rate,
        }

"""Soft absolute-duration supervision for LR-MHC experts and router."""

import math
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as func
from torch import Tensor

from src.losses.losses import SetCriterion


def soft_duration_targets(
    meta: List[Dict[str, Any]],
    device: torch.device,
    centers_seconds: Sequence[float],
    target_beta: float,
) -> Tensor:
    """Map GT durations to overlapping expert responsibilities in log time."""
    durations = []
    for sample_meta in meta:
        lengths = [float(end - start) for start, end in sample_meta["relevant_windows"]]
        durations.append(sum(lengths) / len(lengths))
    log_duration = torch.tensor(durations, device=device).clamp_min(1e-3).log().unsqueeze(-1)
    centers = torch.tensor(centers_seconds, device=device).clamp_min(1e-3).log()
    return torch.softmax(-target_beta * (log_duration - centers).square(), dim=-1)


class SoftDurationExpertsOnlySetCriterion(SetCriterion):
    """Train duration experts and a sample router while freezing the base DETR."""

    def __init__(
        self,
        *args: Any,
        duration_centers_seconds: Sequence[float] = (5.0, math.sqrt(300.0), 60.0),
        duration_target_beta: float = 2.0,
        duration_rank_margin: float = 0.1,
        duration_hard_negatives: int = 16,
        duration_balance_prior: bool = False,
        duration_balance_advantage: bool = False,
        duration_route_advantage_margin: float = 0.02,
        duration_route_advantage_temperature: float = 0.05,
        duration_advantage_trainable_experts: Sequence[int] = (0, 1, 2),
        duration_advantage_contrast_expert: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if len(duration_centers_seconds) != 3 or any(value <= 0 for value in duration_centers_seconds):
            raise ValueError("duration_centers_seconds must contain three positive values")
        self.duration_centers_seconds = tuple(duration_centers_seconds)
        self.duration_target_beta = duration_target_beta
        self.duration_rank_margin = duration_rank_margin
        self.duration_hard_negatives = duration_hard_negatives
        self.duration_balance_prior = duration_balance_prior
        self.duration_balance_advantage = duration_balance_advantage
        if duration_route_advantage_temperature <= 0:
            raise ValueError("duration_route_advantage_temperature must be positive")
        self.duration_route_advantage_margin = duration_route_advantage_margin
        self.duration_route_advantage_temperature = duration_route_advantage_temperature
        trainable_experts = tuple(int(index) for index in duration_advantage_trainable_experts)
        if not trainable_experts or any(index not in (0, 1, 2) for index in trainable_experts):
            raise ValueError("duration_advantage_trainable_experts must select experts 0, 1, or 2")
        self.duration_advantage_trainable_experts = trainable_experts
        if duration_advantage_contrast_expert is not None and duration_advantage_contrast_expert not in (0, 1, 2):
            raise ValueError("duration_advantage_contrast_expert must be 0, 1, or 2")
        self.duration_advantage_contrast_expert = duration_advantage_contrast_expert

    def forward(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        meta: List[Dict[str, Any]],
        matching: Dict[str, Any],
    ) -> Dict[str, Any]:
        del targets
        required = {
            "duration_safety_prior_logits",
            "expert_quality_residuals",
            "base_class_scores",
            "base_quality_scores",
        }
        if not required.issubset(outputs):
            return {}

        residuals = outputs["expert_quality_residuals"]
        target_probs = soft_duration_targets(
            meta,
            residuals.device,
            self.duration_centers_seconds,
            self.duration_target_beta,
        ).detach()
        class_weights = None
        if self.duration_balance_prior or self.duration_balance_advantage:
            target_mass = target_probs.sum(dim=0).clamp_min(1e-3)
            class_weights = target_probs.shape[0] / (target_probs.shape[1] * target_mass)
        prior_terms = target_probs * func.log_softmax(
            outputs["duration_safety_prior_logits"], dim=-1,
        )
        if self.duration_balance_prior:
            assert class_weights is not None
            prior_terms = prior_terms * class_weights
        prior_loss = -prior_terms.sum(dim=-1).mean()

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

        rank_numerator = residuals.new_zeros(())
        rank_denominator = residuals.new_zeros(())
        advantage_numerator = residuals.new_zeros(())
        advantage_denominator = residuals.new_zeros(())
        specialization_losses = []
        for sample_idx, (positive_idx, _) in enumerate(matching["positive"]["indices"]):
            positive_idx = positive_idx.to(residuals.device)
            if positive_idx.numel() == 0:
                continue
            negative_mask = torch.ones(
                residuals.shape[1], dtype=torch.bool, device=residuals.device,
            )
            negative_mask[positive_idx] = False
            negative_idx = torch.nonzero(negative_mask, as_tuple=False).squeeze(-1)
            if negative_idx.numel() == 0:
                continue
            count = min(self.duration_hard_negatives, negative_idx.numel())
            hard_order = torch.topk(uniform_score[sample_idx, negative_idx].detach(), count).indices
            hard_idx = negative_idx[hard_order]
            pair_loss = func.softplus(
                self.duration_rank_margin
                - expert_score[sample_idx, positive_idx].unsqueeze(1)
                + expert_score[sample_idx, hard_idx].unsqueeze(0),
            ).mean(dim=(0, 1))
            rank_numerator = rank_numerator + (pair_loss * target_probs[sample_idx]).sum()
            rank_denominator = rank_denominator + target_probs[sample_idx].sum()

            if self.duration_advantage_contrast_expert is None:
                gradient_mask = residuals.new_zeros(3)
                gradient_mask[list(self.duration_advantage_trainable_experts)] = 1
                advantage_residuals = residuals[sample_idx].detach() + gradient_mask * (
                    residuals[sample_idx] - residuals[sample_idx].detach()
                )
            else:
                contrast = residuals.new_full((3,), -0.5)
                contrast[self.duration_advantage_contrast_expert] = 1.0
                contrast_score = (residuals[sample_idx] * contrast).sum(dim=-1, keepdim=True)
                contrast_score = contrast_score / contrast.square().sum()
                advantage_residuals = residuals[sample_idx].detach() + contrast * (
                    contrast_score - contrast_score.detach()
                )
            centered_residuals = advantage_residuals - advantage_residuals.mean(
                dim=-1,
                keepdim=True,
            )
            routed_delta = (centered_residuals * target_probs[sample_idx]).sum(dim=-1)
            routed_pair_advantage = (
                routed_delta[positive_idx].unsqueeze(1)
                - routed_delta[hard_idx].unsqueeze(0)
            )
            temperature = self.duration_route_advantage_temperature
            advantage_loss = temperature * func.softplus(
                (self.duration_route_advantage_margin - routed_pair_advantage)
                / temperature,
            ).mean()
            advantage_weight = residuals.new_ones(())
            if self.duration_balance_advantage:
                assert class_weights is not None
                advantage_weight = (target_probs[sample_idx] * class_weights).sum()
            advantage_numerator = advantage_numerator + advantage_weight * advantage_loss
            advantage_denominator = advantage_denominator + advantage_weight

            positive_residuals = residuals[sample_idx, positive_idx]
            specialization_losses.append(
                -(
                    target_probs[sample_idx]
                    * func.log_softmax(
                        positive_residuals / self.expert_specialization_temperature,
                        dim=-1,
                    )
                ).sum(dim=-1).mean(),
            )

        zero = residuals.sum() * 0
        rank_loss = rank_numerator / rank_denominator.clamp_min(1.0)
        specialization_loss = (
            torch.stack(specialization_losses).mean() if specialization_losses else zero
        )
        advantage_loss = advantage_numerator / advantage_denominator.clamp_min(1.0)
        return {
            "loss_soft_duration_prior": prior_loss,
            "loss_soft_duration_rank": rank_loss,
            "loss_soft_duration_route_advantage": advantage_loss,
            "loss_soft_duration_specialization": specialization_loss,
        }

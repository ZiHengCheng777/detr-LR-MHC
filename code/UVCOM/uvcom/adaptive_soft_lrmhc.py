"""Adaptive soft length routing for DETR query classification."""

import math

import torch
from torch import nn
import torch.nn.functional as F


def balanced_expert_rank_loss(
    base_margin,
    expert_residuals,
    targets,
    indices,
    centers=(-3.1781, -2.1253, -1.3863),
    target_beta=2.0,
    margin=0.2,
    hard_negatives=8,
):
    """Train each duration expert to rank matched queries above hard negatives."""
    base_margin = base_margin.detach()
    expert_scores = base_margin.unsqueeze(-1) + expert_residuals
    numerators = expert_residuals.new_zeros(expert_residuals.shape[-1])
    denominators = expert_residuals.new_zeros(expert_residuals.shape[-1])

    for sample_idx, (src_idx, target_idx) in enumerate(indices):
        src_idx = src_idx.to(expert_residuals.device)
        target_idx = target_idx.to(expert_residuals.device)
        if src_idx.numel() == 0:
            continue
        negative_mask = torch.ones(
            expert_residuals.shape[1],
            dtype=torch.bool,
            device=expert_residuals.device,
        )
        negative_mask[src_idx] = False
        negative_idx = torch.nonzero(negative_mask, as_tuple=False).squeeze(-1)
        if negative_idx.numel() == 0:
            continue

        hard_count = min(hard_negatives, negative_idx.numel())
        hard_order = torch.topk(
            base_margin[sample_idx, negative_idx], hard_count
        ).indices
        hard_idx = negative_idx[hard_order]
        pair_loss = F.softplus(
            margin
            - expert_scores[sample_idx, src_idx].unsqueeze(1)
            + expert_scores[sample_idx, hard_idx].unsqueeze(0)
        ).mean(dim=1)

        target_widths = targets["span_labels"][sample_idx]["spans"][
            target_idx, 1
        ].clamp(1e-4, 1.0 - 1e-4)
        target_centers = expert_residuals.new_tensor(centers)
        responsibilities = (
            -target_beta
            * (torch.logit(target_widths).unsqueeze(-1) - target_centers).square()
        ).softmax(dim=-1).detach()
        numerators += (pair_loss * responsibilities).sum(dim=0)
        denominators += responsibilities.sum(dim=0)

    return (numerators / denominators.clamp_min(1.0)).mean()


def routed_residual_rank_loss(
    base_margin,
    applied_residual,
    indices,
    margin=0.02,
    temperature=0.05,
    hard_negatives=8,
):
    """Rank matched queries above hard negatives using deployed scores."""
    base_margin = base_margin.detach()
    applied_residual = applied_residual.squeeze(-1)
    corrected_margin = base_margin + applied_residual
    sample_losses = []

    for sample_idx, (src_idx, _) in enumerate(indices):
        src_idx = src_idx.to(applied_residual.device)
        if src_idx.numel() == 0:
            continue
        negative_mask = torch.ones(
            applied_residual.shape[1],
            dtype=torch.bool,
            device=applied_residual.device,
        )
        negative_mask[src_idx] = False
        negative_idx = torch.nonzero(negative_mask, as_tuple=False).squeeze(-1)
        if negative_idx.numel() == 0:
            continue

        hard_count = min(hard_negatives, negative_idx.numel())
        hard_order = torch.topk(
            base_margin[sample_idx, negative_idx], hard_count
        ).indices
        hard_idx = negative_idx[hard_order]
        corrected_advantage = (
            corrected_margin[sample_idx, src_idx].unsqueeze(1)
            - corrected_margin[sample_idx, hard_idx].unsqueeze(0)
        )
        sample_losses.append(
            temperature * F.softplus(
                (margin - corrected_advantage) / temperature
            ).mean()
        )

    if not sample_losses:
        return applied_residual.sum() * 0.0
    return torch.stack(sample_losses).mean()


def query_gate_oracle_loss(
    base_margin,
    routed_residual,
    query_gate,
    indices,
    hard_negatives=8,
    minimum_impact=1e-4,
):
    """Open a query gate only when its detached correction has a useful sign."""
    base_margin = base_margin.detach()
    residual = routed_residual.detach().squeeze(-1)
    gates = query_gate.squeeze(-1).clamp(1e-6, 1.0 - 1e-6)
    group_losses = []
    for sample_idx, (positive_idx, _) in enumerate(indices):
        positive_idx = positive_idx.to(gates.device)
        if positive_idx.numel() == 0:
            continue
        negative_mask = torch.ones(gates.shape[1], dtype=torch.bool, device=gates.device)
        negative_mask[positive_idx] = False
        negative_idx = torch.nonzero(negative_mask, as_tuple=False).squeeze(-1)
        if negative_idx.numel() == 0:
            continue
        hard_count = min(hard_negatives, negative_idx.numel())
        hard_idx = negative_idx[torch.topk(
            base_margin[sample_idx, negative_idx], hard_count
        ).indices]
        sample_losses = []
        for query_idx, beneficial_sign in ((positive_idx, 1.0), (hard_idx, -1.0)):
            impact = residual[sample_idx, query_idx]
            informative = impact.abs() >= minimum_impact
            if not informative.any():
                continue
            impact = impact[informative]
            target = (beneficial_sign * impact > 0).to(gates.dtype)
            loss = F.binary_cross_entropy(
                gates[sample_idx, query_idx][informative], target, reduction="none"
            )
            weight = impact.abs()
            weight = weight / weight.mean().clamp_min(torch.finfo(weight.dtype).eps)
            sample_losses.append((loss * weight.clamp(max=4.0)).mean())
        if sample_losses:
            group_losses.append(torch.stack(sample_losses).mean())
    if not group_losses:
        return gates.sum() * 0.0
    return torch.stack(group_losses).mean()


def quality_aware_routed_rank_loss(
    base_margin,
    applied_residual,
    pred_spans,
    targets,
    minimum_quality_gap=0.10,
    score_margin=0.05,
    temperature=0.10,
    balance_durations=False,
    video_durations=None,
    duration_centers=(-3.1781, -2.1253, -1.3863),
    duration_beta=2.0,
    short_boundary=10.0,
    long_boundary=30.0,
    short_temperature=2.0,
    long_temperature=4.0,
    cross_target_positive_threshold=0.10,
    return_group_losses=False,
):
    """Rank routed query scores by their detached localization IoU quality."""
    base_margin = base_margin.detach()
    corrected_margin = base_margin + applied_residual.squeeze(-1)
    pred_spans = pred_spans.detach()
    sample_losses = []
    duration_responsibilities = []
    group_loss_sums = applied_residual.new_zeros(3)
    group_loss_weights = applied_residual.new_zeros(3)

    for sample_idx, target in enumerate(targets["span_labels"]):
        target_spans = target["spans"].to(pred_spans)
        if target_spans.numel() == 0:
            continue
        pred_xx = torch.stack(
            (
                pred_spans[sample_idx, :, 0] - pred_spans[sample_idx, :, 1] / 2,
                pred_spans[sample_idx, :, 0] + pred_spans[sample_idx, :, 1] / 2,
            ),
            dim=-1,
        )
        target_xx = torch.stack(
            (
                target_spans[:, 0] - target_spans[:, 1] / 2,
                target_spans[:, 0] + target_spans[:, 1] / 2,
            ),
            dim=-1,
        )
        left = torch.maximum(pred_xx[:, None, 0], target_xx[None, :, 0])
        right = torch.minimum(pred_xx[:, None, 1], target_xx[None, :, 1])
        intersection = (right - left).clamp_min(0.0)
        union = (
            (pred_xx[:, 1] - pred_xx[:, 0]).clamp_min(0.0)[:, None]
            + (target_xx[:, 1] - target_xx[:, 0]).clamp_min(0.0)[None, :]
            - intersection
        )
        target_quality = intersection / union.clamp_min(1e-6)

        if balance_durations and video_durations is not None:
            duration_seconds = (
                target_spans[:, 1] * video_durations[sample_idx].detach()
            )
            short_weight = torch.sigmoid(
                (short_boundary - duration_seconds) / short_temperature
            )
            long_weight = torch.sigmoid(
                (duration_seconds - long_boundary) / long_temperature
            )
            middle_weight = (1.0 - short_weight) * (1.0 - long_weight)
            target_responsibilities = torch.stack(
                (short_weight, middle_weight, long_weight), dim=-1
            )
            target_responsibilities = target_responsibilities / (
                target_responsibilities.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            )

            # A query matching another GT window is not a valid negative. This is
            # important for QVHighlights samples containing moments of mixed lengths.
            all_target_quality = target_quality.amax(dim=1)
            valid_negative = all_target_quality <= cross_target_positive_threshold
            scores = corrected_margin[sample_idx]
            score_advantage = scores[:, None] - scores[None, :]
            pair_loss = temperature * F.softplus(
                (score_margin - score_advantage) / temperature
            )

            for group_idx in range(3):
                group_quality = (
                    target_quality
                    * target_responsibilities[:, group_idx].unsqueeze(0)
                ).amax(dim=1)
                quality_gap = group_quality[:, None] - group_quality[None, :]
                valid_pairs = (
                    (quality_gap >= minimum_quality_gap)
                    & valid_negative.unsqueeze(0)
                )
                if not valid_pairs.any():
                    continue
                pair_weight = quality_gap.clamp_min(0.0)
                group_sample_loss = (
                    pair_loss[valid_pairs] * pair_weight[valid_pairs]
                ).sum() / pair_weight[valid_pairs].sum().clamp_min(1e-6)
                # Smooth union probability mirrors eval membership: a sample
                # contributes when any of its GT windows belongs to this group.
                group_presence = 1.0 - torch.prod(
                    1.0 - target_responsibilities[:, group_idx]
                )
                group_loss_sums[group_idx] += group_sample_loss * group_presence
                group_loss_weights[group_idx] += group_presence
            continue

        quality = target_quality.amax(dim=1)
        quality_gap = quality[:, None] - quality[None, :]
        valid_pairs = quality_gap >= minimum_quality_gap
        if not valid_pairs.any():
            continue

        scores = corrected_margin[sample_idx]
        score_advantage = scores[:, None] - scores[None, :]
        pair_loss = temperature * F.softplus(
            (score_margin - score_advantage) / temperature
        )
        pair_weight = quality_gap.clamp_min(0.0)
        sample_losses.append(
            (pair_loss[valid_pairs] * pair_weight[valid_pairs]).sum()
            / pair_weight[valid_pairs].sum().clamp_min(1e-6)
        )
        target_width = target_spans[:, 1].mean().clamp(1e-4, 1.0 - 1e-4)
        if video_durations is None:
            centers = pred_spans.new_tensor(duration_centers)
            responsibility = (
                -duration_beta * (torch.logit(target_width) - centers).square()
            ).softmax(0)
        else:
            duration_seconds = target_width * video_durations[sample_idx].detach()
            short_weight = torch.sigmoid(
                (short_boundary - duration_seconds) / short_temperature
            )
            long_weight = torch.sigmoid(
                (duration_seconds - long_boundary) / long_temperature
            )
            middle_weight = (1.0 - short_weight) * (1.0 - long_weight)
            responsibility = torch.stack(
                (short_weight, middle_weight, long_weight)
            )
            responsibility = responsibility / responsibility.sum().clamp_min(1e-6)
        duration_responsibilities.append(responsibility)

    if not sample_losses:
        active_groups = group_loss_weights > 0
        if active_groups.any():
            group_losses = group_loss_sums / group_loss_weights.clamp_min(1e-6)
            balanced_loss = group_losses[active_groups].mean()
            if return_group_losses:
                return balanced_loss, group_losses
            return balanced_loss
        zero = applied_residual.sum() * 0.0
        if return_group_losses:
            return zero, zero.expand(3)
        return zero
    sample_losses = torch.stack(sample_losses)
    if not balance_durations:
        return sample_losses.mean()
    responsibilities = torch.stack(duration_responsibilities)
    group_losses = (
        (sample_losses.unsqueeze(-1) * responsibilities).sum(dim=0)
        / responsibilities.sum(dim=0).clamp_min(1e-6)
    )
    return group_losses.mean()



class AdaptiveSoftLRMHC(nn.Module):
    """Mix three residual experts using detached query and width features."""

    def __init__(
        self,
        hidden_dim,
        num_classes=2,
        foreground_index=1,
        router_hidden_dim=64,
        route_beta=1.0,
        prototype_init=(-3.1781, -2.1253, -1.3863),
        min_gap=0.05,
        gate_init=0.5,
        uniform_routing=False,
        hard_routing=False,
        learn_prototypes=True,
        use_gate=True,
        detach_width=True,
        use_query_gate=False,
        query_gate_hidden_dim=32,
        query_gate_init=0.2,
        use_route_confidence_gate=False,
        output_scale=1.0,
        eval_output_scale=None,
        use_span_refine=False,
        span_output_scale=0.1,
        eval_span_output_scale=None,
        span_expert_scale_init=(1.0, 1.0, 1.0),
    ):
        super().__init__()
        if len(prototype_init) != 3:
            raise ValueError("prototype_init must contain three values")
        centers = torch.as_tensor(prototype_init, dtype=torch.float32)
        if not torch.all(centers[1:] > centers[:-1]):
            raise ValueError("prototype_init must be strictly increasing")
        gaps = centers[1:] - centers[:-1] - min_gap
        if not torch.all(gaps > 0):
            raise ValueError("prototype gaps must be larger than min_gap")
        if not 0.0 < gate_init < 1.0:
            raise ValueError("gate_init must be in (0, 1)")
        if use_query_gate and not 0.0 < query_gate_init < 1.0:
            raise ValueError("query_gate_init must be in (0, 1)")
        if output_scale < 0.0:
            raise ValueError("output_scale must be non-negative")
        if eval_output_scale is not None and eval_output_scale < 0.0:
            raise ValueError("eval_output_scale must be non-negative")
        if span_output_scale < 0.0:
            raise ValueError("span_output_scale must be non-negative")
        if eval_span_output_scale is not None and eval_span_output_scale < 0.0:
            raise ValueError("eval_span_output_scale must be non-negative")
        span_scale_init = torch.as_tensor(
            span_expert_scale_init, dtype=torch.float32
        )
        if span_scale_init.numel() != 3 or not torch.all(span_scale_init > 0):
            raise ValueError("span_expert_scale_init must contain three positive values")
        if uniform_routing and hard_routing:
            raise ValueError("uniform_routing and hard_routing are mutually exclusive")

        self.num_classes = num_classes
        self.foreground_index = foreground_index
        self.route_beta = float(route_beta)
        self.min_gap = float(min_gap)
        self.uniform_routing = uniform_routing
        self.hard_routing = hard_routing
        self.use_gate = use_gate
        self.detach_width = detach_width
        self.use_query_gate = use_query_gate
        self.use_route_confidence_gate = use_route_confidence_gate
        self.output_scale = float(output_scale)
        self.eval_output_scale = (
            None if eval_output_scale is None else float(eval_output_scale)
        )
        self.use_span_refine = use_span_refine
        self.span_output_scale = float(span_output_scale)
        self.eval_span_output_scale = (
            None
            if eval_span_output_scale is None
            else float(eval_span_output_scale)
        )

        self.center_start = nn.Parameter(centers[:1].clone())
        self.gap_raw = nn.Parameter(torch.log(torch.expm1(gaps)))
        self.center_start.requires_grad_(learn_prototypes)
        self.gap_raw.requires_grad_(learn_prototypes)

        self.experts = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(3)])
        for expert in self.experts:
            nn.init.normal_(expert.weight, std=1e-4)
            nn.init.zeros_(expert.bias)

        if use_span_refine:
            self.span_experts = nn.ModuleList(
                [nn.Linear(hidden_dim, 2) for _ in range(3)]
            )
            for expert in self.span_experts:
                nn.init.normal_(expert.weight, std=1e-4)
                nn.init.zeros_(expert.bias)
            self.span_expert_scale_raw = nn.Parameter(
                torch.log(torch.expm1(span_scale_init))
            )

        self.router = nn.Sequential(
            nn.Linear(hidden_dim + 1, router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, 3),
        )
        nn.init.xavier_uniform_(self.router[0].weight)
        nn.init.zeros_(self.router[0].bias)
        nn.init.zeros_(self.router[2].weight)
        nn.init.zeros_(self.router[2].bias)

        gate_logit = math.log(gate_init / (1.0 - gate_init))
        self.gate_logits = nn.Parameter(torch.full((3,), gate_logit))
        if use_query_gate:
            self.query_gate = nn.Sequential(
                nn.Linear(hidden_dim + 7, query_gate_hidden_dim),
                nn.GELU(),
                nn.Linear(query_gate_hidden_dim, 1),
            )
            nn.init.xavier_uniform_(self.query_gate[0].weight)
            nn.init.zeros_(self.query_gate[0].bias)
            nn.init.zeros_(self.query_gate[2].weight)
            query_gate_logit = math.log(query_gate_init / (1.0 - query_gate_init))
            nn.init.constant_(self.query_gate[2].bias, query_gate_logit)

    def prototypes(self):
        gaps = F.softplus(self.gap_raw) + self.min_gap
        return torch.cat((self.center_start, self.center_start + gaps.cumsum(0)))

    def forward(self, query_features, base_logits, predicted_width, base_spans=None):
        if query_features.shape[:-1] != base_logits.shape[:-1]:
            raise ValueError("query_features and base_logits must share leading dimensions")
        width = predicted_width
        if self.detach_width:
            width = width.detach()
        width = width.clamp(1e-4, 1.0 - 1e-4)
        width_logit = torch.logit(width).unsqueeze(-1)
        width_logit = width_logit.expand(*query_features.shape[:-1], 1)

        prototypes = self.prototypes().to(width_logit)
        prior_logits = -self.route_beta * (width_logit - prototypes).square()
        router_input = torch.cat((query_features.detach(), width_logit), dim=-1)
        route_logits = prior_logits + self.router(router_input)
        if self.uniform_routing:
            route_weights = torch.full_like(route_logits, 1.0 / 3.0)
        elif self.hard_routing:
            route_weights = F.one_hot(
                prior_logits.argmax(dim=-1), num_classes=3
            ).to(route_logits)
        else:
            route_weights = route_logits.softmax(dim=-1)

        expert_residuals = torch.cat(
            [expert(query_features) for expert in self.experts], dim=-1
        )
        routed_residual = (route_weights * expert_residuals).sum(dim=-1, keepdim=True)
        query_gate = torch.ones_like(routed_residual)
        if self.use_query_gate and not self.uniform_routing and self.use_gate:
            query_gate_input = torch.cat(
                (
                    query_features.detach(), width_logit, route_weights.detach(),
                    routed_residual.detach(), base_logits.detach(),
                ),
                dim=-1,
            )
            query_gate = self.query_gate(query_gate_input).sigmoid()
            route_gate = query_gate
        elif self.uniform_routing or not self.use_gate:
            route_gate = torch.ones_like(routed_residual)
        else:
            route_gate = (
                route_weights * self.gate_logits.sigmoid()
            ).sum(dim=-1, keepdim=True)
        if self.use_route_confidence_gate and not self.uniform_routing:
            entropy = -(route_weights * route_weights.clamp_min(1e-8).log()).sum(
                dim=-1, keepdim=True
            )
            confidence = (1.0 - entropy / math.log(3.0)).clamp(0.0, 1.0)
            route_gate = (2.0 * route_gate * confidence.detach().sqrt()).clamp(max=1.0)
        output_scale = (
            self.eval_output_scale
            if not self.training and self.eval_output_scale is not None
            else self.output_scale
        )
        residual = output_scale * route_gate * routed_residual

        corrected_logits = base_logits.clone()
        corrected_logits[..., self.foreground_index:self.foreground_index + 1] += residual
        corrected_spans = base_spans
        expert_span_deltas = None
        routed_span_delta = None
        applied_span_delta = None
        if self.use_span_refine:
            if base_spans is None:
                raise ValueError("base_spans are required when span refinement is enabled")
            expert_span_deltas = torch.stack(
                [expert(query_features) for expert in self.span_experts], dim=-2
            )
            span_expert_scales = F.softplus(self.span_expert_scale_raw).to(
                expert_span_deltas
            )
            routed_span_delta = (
                route_weights.unsqueeze(-1)
                * span_expert_scales.view(*([1] * (route_weights.ndim - 1)), 3, 1)
                * expert_span_deltas
            ).sum(dim=-2)
            span_output_scale = (
                self.eval_span_output_scale
                if not self.training and self.eval_span_output_scale is not None
                else self.span_output_scale
            )
            applied_span_delta = span_output_scale * route_gate * routed_span_delta
            base_span_logits = torch.logit(base_spans.clamp(1e-4, 1.0 - 1e-4))
            corrected_spans = (base_span_logits + applied_span_delta).sigmoid()
        diagnostics = {
            "base_logits": base_logits,
            "route_logits": route_logits,
            "route_weights": route_weights,
            "route_prototypes": prototypes.view(
                *([1] * (route_logits.ndim - 1)), -1
            ).expand_as(route_logits),
            "route_gate": route_gate,
            "query_route_gate": query_gate,
            "expert_residuals": expert_residuals,
            "route_delta": routed_residual,
            "applied_route_delta": residual,
        }
        if self.use_span_refine:
            diagnostics.update({
                "base_spans": base_spans,
                "expert_span_deltas": expert_span_deltas,
                "span_expert_scales": span_expert_scales.view(
                    *([1] * (route_weights.ndim - 1)), 3
                ).expand_as(route_weights),
                "routed_span_delta": routed_span_delta,
                "applied_span_delta": applied_span_delta,
            })
        return corrected_logits, corrected_spans, diagnostics

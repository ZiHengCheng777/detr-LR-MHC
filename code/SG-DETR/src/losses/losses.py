"""Module for computing losses."""

from typing import Any, Dict, List

import torch
from torch import nn
from torch.nn import functional as func

from src.losses.auxiliary_losses import AuxiliaryLosses
from src.losses.boundary_loss import BoundaryDiscriminationLoss
from src.losses.matcher import HungarianMatcher
from src.losses.mom2txt_losses import Moment2TextLosses
from src.losses.regression_losses.atss_losses import ATSSRetrievalLoss
from src.losses.regression_losses.aux_references_losses import AuxRefLosses
from src.losses.regression_losses.denoise_losses import DenoiseLosses
from src.losses.regression_losses.retrieval_losses import MainRegressionLosses
from src.losses.saliency_losses import SaliencyLosses
from src.losses.utils import get_src_permutation_idx
from src.utils.span_utils import span_cxw_to_xx, temporal_iou

EPS: float = 1e-7


def create_targets_k_repeats(targets: Dict[str, Any], times: int) -> Dict[str, Any]:
    """
    Duplicate target k times.

    Args:
        targets (Dict[str, Any]): target spans data
        times (int): number of replications

    Returns:
        Dict[str, Any]: Duplicated target data
    """
    targets_spans_k = []
    for item_targets in targets["span_labels"]:
        spans = item_targets["spans"]
        x_repeated = [spans] * times  # noqa: WPS435
        result = torch.cat(x_repeated, dim=0)
        targets_spans_k.append({"spans": result})
    return {"span_labels": targets_spans_k}


class SetCriterion(nn.Module):  # noqa: WPS230, WPS211
    """Compute the loss for DETR."""

    # pylint: disable=too-many-locals, too-many-arguments
    def __init__(  # noqa: WPS211
        self,
        matcher: HungarianMatcher,
        weight_dict: Dict[str, int],
        main_reg_losses: MainRegressionLosses,
        top_k_positive_anchors: int = 9,
        saliency_margin: float = 0.15,
        contrastive_reducer: float = 0.25,
        denoise_reducer: float = 0.5,
        colab_ref_reducer: float = 0.5,
        target_repeat: int = 3,
        one2one: bool = True,
        use_focal: bool = True,
        gamma: float = 2,
        local_saliency_loss_scale: float = 1.0,
        use_negative_losses: bool = True,
        route_target_beta: float = 2.0,
        route_target_centers: List[float] = [-3.1781, -2.1253, -1.3863],
        expert_neutral_weight: float = 0.05,
        expert_residual_clip: float = 4.0,
        expert_ranking_margin: float = 0.2,
        expert_hard_negatives: int = 8,
        expert_route_sharpen: float = 1.0,
        expert_responsibility_source: str = "predicted",
        mixed_ranking_weight: float = 0.0,
        expert_specialization_temperature: float = 0.5,
        route_safety_margin: float = 0.01,
        route_safety_width_power: float = 0.5,
        route_ap_safety_margin: float = 0.0,
        route_ap_safety_temperature: float = 0.01,
        route_ap_rank_temperature: float = 0.02,
        route_ap_direct_weight: float = 0.0,
        route_ap_thresholds: List[float] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        route_gate_budget: float = 0.2,
        route_gate_oracle_margin: float = 1e-3,
        route_correction_margin: float = 0.05,
        query_prior_safety_threshold: float = 10.0,
        nonlong_route_safety_threshold: float = 30.0,
        query_prior_target_mode: str = "normalized",
        query_prior_absolute_thresholds: List[float] = [10.0, 30.0],
    ) -> None:
        """
        Create the criterion.

        Args:
            matcher (HungarianMatcher): instance of the Matcher class used to match predictions and targets
            weight_dict (Dict[str, int]): Key is the name of the loss and Value its relative weight.
            main_reg_losses (MainRegressionLosses): main detr losses.
            top_k_positive_anchors (int): top k samples to select for ATSS loss.
            saliency_margin (float): margin for saliency loss
            contrastive_reducer (float): weight reducer for constrastive loss
            denoise_reducer (float): weight reducer for denoise loss
            colab_ref_reducer (float): weight reducer for colab ref loss
            target_repeat (int): number of times to repeat the target.
            one2one (bool): matching type
            use_focal: whether to use focal loss or not.
            gamma (float): Gamma factor for focal loss calculation. Defaults to 2.0.
            local_saliency_loss_scale (float): scale for local saliency loss.
            use_negative_losses (bool): whether to use negative losses or not.
        """
        super().__init__()
        self.matcher = matcher
        self.target_repeat = target_repeat
        self.one2one = one2one
        self.weight_dict = weight_dict
        self.route_target_beta = route_target_beta
        self.expert_neutral_weight = expert_neutral_weight
        self.expert_residual_clip = expert_residual_clip
        self.expert_ranking_margin = expert_ranking_margin
        self.expert_hard_negatives = expert_hard_negatives
        self.expert_route_sharpen = expert_route_sharpen
        if expert_responsibility_source not in {"predicted", "target"}:
            raise ValueError("expert_responsibility_source must be predicted or target")
        self.expert_responsibility_source = expert_responsibility_source
        self.mixed_ranking_weight = mixed_ranking_weight
        self.expert_specialization_temperature = expert_specialization_temperature
        self.route_safety_margin = route_safety_margin
        self.route_safety_width_power = route_safety_width_power
        self.route_ap_safety_margin = route_ap_safety_margin
        self.route_ap_safety_temperature = route_ap_safety_temperature
        self.route_ap_rank_temperature = route_ap_rank_temperature
        self.route_ap_direct_weight = route_ap_direct_weight
        self.route_ap_thresholds = tuple(route_ap_thresholds)
        self.route_gate_budget = route_gate_budget
        self.route_gate_oracle_margin = route_gate_oracle_margin
        self.route_correction_margin = route_correction_margin
        self.query_prior_safety_threshold = query_prior_safety_threshold
        self.nonlong_route_safety_threshold = nonlong_route_safety_threshold
        if query_prior_target_mode not in {"normalized", "absolute"}:
            raise ValueError("query_prior_target_mode must be normalized or absolute")
        self.query_prior_target_mode = query_prior_target_mode
        self.query_prior_absolute_thresholds = query_prior_absolute_thresholds
        self.register_buffer("route_target_centers", torch.tensor(route_target_centers, dtype=torch.float32))

        # span losses
        self.retrieval_losses = main_reg_losses
        self.denoise_losses = DenoiseLosses(use_focal=use_focal, gamma=gamma, denoise_reducer=denoise_reducer)
        self.aux_ref_losses = AuxRefLosses(use_focal=use_focal, gamma=gamma, colab_ref_reducer=colab_ref_reducer)
        self.aux_head_losses = ATSSRetrievalLoss(top_k_positive_anchors=top_k_positive_anchors)

        # saliency losses
        self.saliency_losses = SaliencyLosses(
            saliency_margin=saliency_margin,
            contrastive_reducer=contrastive_reducer,
            local_saliency_loss_scale=local_saliency_loss_scale,
            use_negative_losses=use_negative_losses,
        )
        #  other losses
        self.auxiliary_losses = AuxiliaryLosses()
        self.moment2text_losses = Moment2TextLosses()

        # boundary discrimination loss
        self.boundary_loss = BoundaryDiscriminationLoss(d_model=256)

    def _route_alignment_loss(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        indices: List[Any],
    ) -> Dict[str, torch.Tensor]:
        """Supervise matched queries with soft GT-length assignments."""
        if "route_logits" not in outputs:
            return {}

        batch_idx, query_idx = get_src_permutation_idx(indices)
        if query_idx.numel() == 0:
            return {"loss_route": outputs["route_logits"].sum() * 0}

        device = outputs["route_logits"].device
        batch_idx = batch_idx.to(device)
        query_idx = query_idx.to(device)
        matched_logits = outputs["route_logits"][batch_idx, query_idx]
        matched_widths = torch.cat(
            [target["spans"][target_idx, 1] for target, (_, target_idx) in zip(targets["span_labels"], indices)],
        ).to(device)
        gt_width_logits = torch.logit(matched_widths.clamp(min=1e-4, max=1 - 1e-4)).unsqueeze(-1)
        target_logits = -self.route_target_beta * (gt_width_logits - self.route_target_centers) ** 2
        target_probs = torch.softmax(target_logits, dim=-1)
        loss_route = -(target_probs * func.log_softmax(matched_logits, dim=-1)).sum(dim=-1).mean()
        return {"loss_route": loss_route}

    def _query_length_prior_loss(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        meta: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """Supervise the sample-level prior with the GT span-length distribution."""
        if "query_length_prior_logits" not in outputs:
            return {}

        prior_logits = outputs["query_length_prior_logits"]
        if self.query_prior_target_mode == "absolute":
            labels = []
            low, high = self.query_prior_absolute_thresholds
            for sample_meta in meta:
                durations = [end - start for start, end in sample_meta["relevant_windows"]]
                mean_duration = sum(durations) / len(durations)
                labels.append(0 if mean_duration < low else 1 if mean_duration < high else 2)
            labels_tensor = torch.tensor(labels, device=prior_logits.device)
            counts = torch.bincount(labels_tensor, minlength=3).clamp_min(1)
            class_weights = labels_tensor.numel() / (3.0 * counts.to(prior_logits.dtype))
            loss = func.cross_entropy(prior_logits, labels_tensor, weight=class_weights)
            return {"loss_query_length_prior": loss}

        target_distributions = []
        for target in targets["span_labels"]:
            widths = target["spans"][:, 1].to(prior_logits.device)
            width_logits = torch.logit(widths.clamp(min=1e-4, max=1 - 1e-4)).unsqueeze(-1)
            target_logits = -self.route_target_beta * (width_logits - self.route_target_centers) ** 2
            target_distributions.append(torch.softmax(target_logits, dim=-1).mean(dim=0))
        target_probs = torch.stack(target_distributions).detach()
        loss = -(target_probs * func.log_softmax(prior_logits, dim=-1)).sum(dim=-1).mean()
        return {"loss_query_length_prior": loss}

    def _query_prior_safety_loss(
        self,
        outputs: Dict[str, Any],
        meta: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """Calibrate whether the sample prior is safe for short-window metrics."""
        if "query_prior_safety_logits" not in outputs:
            return {}

        logits = outputs["query_prior_safety_logits"]
        mean_durations = []
        for sample_meta in meta:
            durations = [end - start for start, end in sample_meta["relevant_windows"]]
            mean_durations.append(sum(durations) / len(durations))
        targets = torch.tensor(mean_durations, device=logits.device) >= self.query_prior_safety_threshold
        targets = targets.to(logits.dtype)
        positive_fraction = targets.mean().clamp(min=1e-3, max=1 - 1e-3)
        weights = torch.where(
            targets.bool(),
            0.5 / positive_fraction,
            0.5 / (1.0 - positive_fraction),
        )
        loss = func.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return {"loss_query_prior_safety": (loss * weights).mean()}

    def _expert_quality_loss(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        indices: List[Any],
    ) -> Dict[str, torch.Tensor]:
        """Train experts to rank matched moments above hard negatives."""
        required = {
            "pred_logits",
            "pred_quality_scores",
            "pred_expert_quality_scores",
            "route_logits",
        }
        if not required.issubset(outputs):
            return {}

        batch_idx, query_idx = get_src_permutation_idx(indices)
        expert_logits = outputs["pred_expert_quality_scores"]
        if query_idx.numel() == 0:
            return {"loss_expert_quality": expert_logits.sum() * 0}

        # The deployed score is sqrt(class_probability * quality_probability).
        # Log space turns this into an average and avoids numerical square roots.
        class_scores = outputs.get("base_class_scores", outputs["pred_logits"])
        class_log_prob = func.logsigmoid(class_scores.squeeze(-1))
        expert_combo = 0.5 * (class_log_prob.detach().unsqueeze(-1) + func.logsigmoid(expert_logits))
        if {"base_class_scores", "base_quality_scores", "expert_quality_residuals"}.issubset(outputs):
            raw_residuals = outputs["expert_quality_residuals"]
            centered_residuals = raw_residuals - raw_residuals.mean(dim=-1, keepdim=True)
            auxiliary_route = torch.softmax(outputs["route_logits"], dim=-1)
            auxiliary_correction = (auxiliary_route * centered_residuals).sum(dim=-1)
            routed_class = outputs["base_class_scores"].squeeze(-1) + auxiliary_correction
            routed_quality = outputs["base_quality_scores"].squeeze(-1) + auxiliary_correction
            mixed_combo = 0.5 * (func.logsigmoid(routed_class) + func.logsigmoid(routed_quality))
        else:
            mixed_combo = 0.5 * (
                class_log_prob + func.logsigmoid(outputs["pred_quality_scores"].squeeze(-1))
            )
        responsibilities = torch.softmax(
            outputs["route_logits"].detach() * self.expert_route_sharpen,
            dim=-1,
        )

        numerators = expert_logits.new_zeros(expert_logits.shape[-1])
        denominators = expert_logits.new_zeros(expert_logits.shape[-1])
        mixed_numerator = expert_logits.new_zeros(())
        mixed_denominator = expert_logits.new_zeros(())
        for sample_idx, (src_idx, target_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            src_idx = src_idx.to(expert_logits.device)
            target_idx = target_idx.to(expert_logits.device)
            negative_mask = torch.ones(expert_logits.shape[1], dtype=torch.bool, device=expert_logits.device)
            negative_mask[src_idx] = False
            negative_idx = torch.nonzero(negative_mask, as_tuple=False).squeeze(-1)
            if negative_idx.numel() == 0:
                continue

            hard_count = min(self.expert_hard_negatives, negative_idx.numel())
            hard_order = torch.topk(mixed_combo[sample_idx, negative_idx].detach(), hard_count).indices
            hard_idx = negative_idx[hard_order]
            positive_scores = expert_combo[sample_idx, src_idx]
            negative_scores = expert_combo[sample_idx, hard_idx]
            pair_losses = func.softplus(
                self.expert_ranking_margin
                - positive_scores.unsqueeze(1)
                + negative_scores.unsqueeze(0),
            ).mean(dim=1)
            mixed_pair_losses = func.softplus(
                self.expert_ranking_margin
                - mixed_combo[sample_idx, src_idx].unsqueeze(1)
                + mixed_combo[sample_idx, hard_idx].unsqueeze(0),
            ).mean(dim=1)

            matched_pred = outputs["pred_spans"][sample_idx, src_idx]
            matched_gt = targets["span_labels"][sample_idx]["spans"][target_idx]
            ious = torch.diag(
                temporal_iou(span_cxw_to_xx(matched_pred), span_cxw_to_xx(matched_gt))[0],
            ).detach()
            if self.expert_responsibility_source == "target":
                gt_width_logits = torch.logit(
                    matched_gt[:, 1].clamp(min=1e-4, max=1 - 1e-4),
                ).unsqueeze(-1)
                target_logits = -self.route_target_beta * (
                    gt_width_logits - self.route_target_centers
                ) ** 2
                expert_responsibilities = torch.softmax(
                    target_logits * self.expert_route_sharpen,
                    dim=-1,
                ).detach()
            else:
                expert_responsibilities = responsibilities[sample_idx, src_idx]
            weights = expert_responsibilities * ious.clamp_min(0.1).unsqueeze(-1)
            numerators = numerators + (pair_losses * weights).sum(dim=0)
            denominators = denominators + weights.sum(dim=0)
            mixed_numerator = mixed_numerator + (mixed_pair_losses * ious.clamp_min(0.1)).sum()
            mixed_denominator = mixed_denominator + ious.clamp_min(0.1).sum()

        per_expert = numerators / denominators.clamp_min(1.0)
        mixed_loss = mixed_numerator / mixed_denominator.clamp_min(1.0)
        return {"loss_expert_quality": per_expert.mean() + self.mixed_ranking_weight * mixed_loss}

    def _expert_specialization_loss(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        indices: List[Any],
    ) -> Dict[str, torch.Tensor]:
        """Align relative expert responses with predicted-width responsibilities."""
        required = {"expert_quality_residuals", "route_logits"}
        if not required.issubset(outputs):
            return {}

        batch_idx, query_idx = get_src_permutation_idx(indices)
        residuals = outputs["expert_quality_residuals"]
        if query_idx.numel() == 0:
            return {"loss_expert_specialization": residuals.sum() * 0}
        batch_idx = batch_idx.to(residuals.device)
        query_idx = query_idx.to(residuals.device)
        matched_residuals = residuals[batch_idx, query_idx]
        matched_gt = torch.cat(
            [target["spans"][target_idx] for target, (_, target_idx) in zip(targets["span_labels"], indices)],
        ).to(residuals.device)
        gt_width_logits = torch.logit(matched_gt[:, 1].clamp(min=1e-4, max=1 - 1e-4)).unsqueeze(-1)
        target_logits = -self.route_target_beta * (gt_width_logits - self.route_target_centers) ** 2
        target_probs = torch.softmax(target_logits, dim=-1).detach()
        expert_log_probs = func.log_softmax(
            matched_residuals / self.expert_specialization_temperature,
            dim=-1,
        )
        loss = -(target_probs * expert_log_probs).sum(dim=-1).mean()
        return {"loss_expert_specialization": loss}

    def _route_safety_loss(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        indices: List[Any],
    ) -> Dict[str, torch.Tensor]:
        """Prevent routed scores from ranking worse than the uniform counterfactual."""
        required = {
            "pred_safety_logits",
            "pred_safety_quality_scores",
            "pred_uniform_logits",
            "pred_uniform_quality_scores",
        }
        if not required.issubset(outputs):
            return {}

        actual = 0.5 * (
            func.logsigmoid(outputs["pred_safety_logits"].squeeze(-1))
            + func.logsigmoid(outputs["pred_safety_quality_scores"].squeeze(-1))
        )
        uniform = 0.5 * (
            func.logsigmoid(outputs["pred_uniform_logits"].squeeze(-1))
            + func.logsigmoid(outputs["pred_uniform_quality_scores"].squeeze(-1))
        ).detach()
        penalties = []
        width_weights = []
        for sample_idx, (src_idx, target_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            src_idx = src_idx.to(actual.device)
            target_idx = target_idx.to(actual.device)
            negative_mask = torch.ones(actual.shape[1], dtype=torch.bool, device=actual.device)
            negative_mask[src_idx] = False
            negative_idx = torch.nonzero(negative_mask, as_tuple=False).squeeze(-1)
            if negative_idx.numel() == 0:
                continue

            hard_count = min(self.expert_hard_negatives, negative_idx.numel())
            hard_idx = negative_idx[torch.topk(uniform[sample_idx, negative_idx], hard_count).indices]
            actual_pair = func.softplus(
                self.expert_ranking_margin
                - actual[sample_idx, src_idx].unsqueeze(1)
                + actual[sample_idx, hard_idx].unsqueeze(0),
            ).mean(dim=1)
            uniform_pair = func.softplus(
                self.expert_ranking_margin
                - uniform[sample_idx, src_idx].unsqueeze(1)
                + uniform[sample_idx, hard_idx].unsqueeze(0),
            ).mean(dim=1)
            penalties.append(func.relu(actual_pair - uniform_pair + self.route_safety_margin))
            widths = targets["span_labels"][sample_idx]["spans"][target_idx, 1].to(actual.device)
            width_weights.append(widths.clamp_min(0.02).pow(-self.route_safety_width_power))

        if not penalties:
            return {"loss_route_safety": actual.sum() * 0}
        penalty = torch.cat(penalties)
        weights = torch.cat(width_weights)
        weights = weights / weights.mean().clamp_min(EPS)
        return {"loss_route_safety": (penalty * weights).mean()}

    def _route_ap_safety_loss(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """Keep routed all-query ranking competitive with the uniform path."""
        if self.weight_dict.get("loss_route_ap_safety", 0) <= 0:
            return {}
        required = {
            "pred_logits",
            "pred_quality_scores",
            "pred_uniform_logits",
            "pred_uniform_quality_scores",
            "pred_spans",
        }
        if not required.issubset(outputs):
            return {}

        routed = 0.5 * (
            func.logsigmoid(outputs["pred_logits"].squeeze(-1))
            + func.logsigmoid(outputs["pred_quality_scores"].squeeze(-1))
        )
        uniform = 0.5 * (
            func.logsigmoid(outputs["pred_uniform_logits"].squeeze(-1))
            + func.logsigmoid(outputs["pred_uniform_quality_scores"].squeeze(-1))
        ).detach()
        sample_losses = []
        for sample_idx, target in enumerate(targets["span_labels"]):
            gt_spans = target["spans"].to(routed.device)
            if gt_spans.numel() == 0:
                continue
            candidate_iou = temporal_iou(
                span_cxw_to_xx(outputs["pred_spans"][sample_idx].detach()),
                span_cxw_to_xx(gt_spans),
            )[0].amax(dim=1)
            threshold_losses = []
            for threshold in self.route_ap_thresholds:
                positive = candidate_iou >= threshold
                negative = ~positive
                if not positive.any() or not negative.any():
                    continue

                def smooth_ap(scores: torch.Tensor) -> torch.Tensor:
                    outranking = torch.sigmoid(
                        (
                            scores[negative].unsqueeze(0)
                            - scores[positive].unsqueeze(1)
                        )
                        / self.route_ap_rank_temperature,
                    ).sum(dim=1)
                    return (1.0 / (1.0 + outranking)).mean()

                routed_ap = smooth_ap(routed[sample_idx])
                uniform_ap = smooth_ap(uniform[sample_idx])
                deficit = uniform_ap - routed_ap + self.route_ap_safety_margin
                threshold_losses.append(
                    self.route_ap_safety_temperature
                    * func.softplus(deficit / self.route_ap_safety_temperature)
                    + self.route_ap_direct_weight * (1.0 - routed_ap),
                )
            if threshold_losses:
                sample_losses.append(torch.stack(threshold_losses).mean())

        if not sample_losses:
            return {"loss_route_ap_safety": routed.sum() * 0}
        return {"loss_route_ap_safety": torch.stack(sample_losses).mean()}

    def _nonlong_route_safety_loss(
        self,
        outputs: Dict[str, Any],
        indices: List[Any],
        meta: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """Prevent expert routing from degrading short and middle samples."""
        required = {
            "pred_logits",
            "pred_quality_scores",
            "pred_uniform_logits",
            "pred_uniform_quality_scores",
        }
        if not required.issubset(outputs):
            return {}

        actual = 0.5 * (
            func.logsigmoid(outputs["pred_logits"].squeeze(-1))
            + func.logsigmoid(outputs["pred_quality_scores"].squeeze(-1))
        )
        uniform = 0.5 * (
            func.logsigmoid(outputs["pred_uniform_logits"].squeeze(-1))
            + func.logsigmoid(outputs["pred_uniform_quality_scores"].squeeze(-1))
        ).detach()
        penalties = []
        for sample_idx, (src_idx, _) in enumerate(indices):
            durations = [end - start for start, end in meta[sample_idx]["relevant_windows"]]
            if sum(durations) / len(durations) >= self.nonlong_route_safety_threshold:
                continue
            if src_idx.numel() == 0:
                continue
            src_idx = src_idx.to(actual.device)
            negative_mask = torch.ones(actual.shape[1], dtype=torch.bool, device=actual.device)
            negative_mask[src_idx] = False
            negative_idx = torch.nonzero(negative_mask, as_tuple=False).squeeze(-1)
            if negative_idx.numel() == 0:
                continue
            hard_count = min(self.expert_hard_negatives, negative_idx.numel())
            hard_idx = negative_idx[torch.topk(uniform[sample_idx, negative_idx], hard_count).indices]
            actual_pair = func.softplus(
                self.expert_ranking_margin
                - actual[sample_idx, src_idx].unsqueeze(1)
                + actual[sample_idx, hard_idx].unsqueeze(0),
            ).mean()
            uniform_pair = func.softplus(
                self.expert_ranking_margin
                - uniform[sample_idx, src_idx].unsqueeze(1)
                + uniform[sample_idx, hard_idx].unsqueeze(0),
            ).mean()
            penalties.append(func.relu(actual_pair - uniform_pair + self.route_safety_margin))

        if not penalties:
            return {"loss_nonlong_route_safety": actual.sum() * 0}
        return {"loss_nonlong_route_safety": torch.stack(penalties).mean()}

    def _route_gate_budget_loss(self, outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Keep the learned safety gate selective instead of globally open."""
        if "route_gate_scores" not in outputs:
            return {}
        mean_gate = outputs["route_gate_scores"].mean()
        return {"loss_route_gate_budget": func.relu(mean_gate - self.route_gate_budget).square()}

    def _route_gate_oracle_loss(
        self,
        outputs: Dict[str, Any],
        indices: List[Any],
    ) -> Dict[str, torch.Tensor]:
        """Open the gate only when its detached correction improves query ranking."""
        required = {
            "route_gate_scores",
            "route_delta_scores",
            "pred_uniform_logits",
            "pred_uniform_quality_scores",
        }
        if not required.issubset(outputs):
            return {}

        gates = outputs["route_gate_scores"].squeeze(-1).clamp(min=EPS, max=1 - EPS)
        deltas = outputs["route_delta_scores"].squeeze(-1).detach()
        uniform = 0.5 * (
            func.logsigmoid(outputs["pred_uniform_logits"].squeeze(-1))
            + func.logsigmoid(outputs["pred_uniform_quality_scores"].squeeze(-1))
        ).detach()
        positive_terms = []
        negative_terms = []
        for sample_idx, (src_idx, _) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            src_idx = src_idx.to(gates.device)
            negative_mask = torch.ones(gates.shape[1], dtype=torch.bool, device=gates.device)
            negative_mask[src_idx] = False
            negative_idx = torch.nonzero(negative_mask, as_tuple=False).squeeze(-1)
            if negative_idx.numel() == 0:
                continue

            hard_count = min(self.expert_hard_negatives, negative_idx.numel())
            hard_idx = negative_idx[torch.topk(uniform[sample_idx, negative_idx], hard_count).indices]
            for query_idx, beneficial_sign, terms in (
                (src_idx, 1.0, positive_terms),
                (hard_idx, -1.0, negative_terms),
            ):
                delta = deltas[sample_idx, query_idx]
                informative = delta.abs() >= self.route_gate_oracle_margin
                if not informative.any():
                    continue
                delta = delta[informative]
                target = (beneficial_sign * delta > 0).to(gates.dtype)
                losses = func.binary_cross_entropy(
                    gates[sample_idx, query_idx][informative],
                    target,
                    reduction="none",
                )
                impact = delta.abs()
                impact = impact / impact.mean().clamp_min(EPS)
                terms.append((losses * impact.clamp(max=4.0)).mean())

        if not positive_terms and not negative_terms:
            return {"loss_route_gate_oracle": gates.sum() * 0}
        group_losses = []
        if positive_terms:
            group_losses.append(torch.stack(positive_terms).mean())
        if negative_terms:
            group_losses.append(torch.stack(negative_terms).mean())
        return {"loss_route_gate_oracle": torch.stack(group_losses).mean()}

    def _route_correction_polarity_loss(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        indices: List[Any],
    ) -> Dict[str, torch.Tensor]:
        """Calibrate routed corrections to matched IoU and suppress hard negatives."""
        required = {
            "route_delta_scores",
            "pred_uniform_logits",
            "pred_uniform_quality_scores",
        }
        if not required.issubset(outputs):
            return {}

        deltas = outputs["route_delta_scores"].squeeze(-1)
        uniform = 0.5 * (
            func.logsigmoid(outputs["pred_uniform_logits"].squeeze(-1))
            + func.logsigmoid(outputs["pred_uniform_quality_scores"].squeeze(-1))
        ).detach()
        positive_terms = []
        negative_terms = []
        for sample_idx, (src_idx, target_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            src_idx = src_idx.to(deltas.device)
            target_idx = target_idx.to(deltas.device)
            negative_mask = torch.ones(deltas.shape[1], dtype=torch.bool, device=deltas.device)
            negative_mask[src_idx] = False
            negative_idx = torch.nonzero(negative_mask, as_tuple=False).squeeze(-1)
            if negative_idx.numel() == 0:
                continue

            hard_count = min(self.expert_hard_negatives, negative_idx.numel())
            hard_idx = negative_idx[torch.topk(uniform[sample_idx, negative_idx], hard_count).indices]
            matched_pred = outputs["pred_spans"][sample_idx, src_idx]
            matched_gt = targets["span_labels"][sample_idx]["spans"][target_idx]
            ious = torch.diag(
                temporal_iou(span_cxw_to_xx(matched_pred), span_cxw_to_xx(matched_gt))[0],
            ).detach()
            positive_targets = self.route_correction_margin * (2.0 * ious - 1.0)
            negative_targets = torch.full_like(
                deltas[sample_idx, hard_idx],
                -self.route_correction_margin,
            )
            positive_terms.append(
                func.smooth_l1_loss(
                    deltas[sample_idx, src_idx],
                    positive_targets,
                    beta=0.01,
                ),
            )
            negative_terms.append(
                func.smooth_l1_loss(
                    deltas[sample_idx, hard_idx],
                    negative_targets,
                    beta=0.01,
                ),
            )

        if not positive_terms:
            return {"loss_route_correction_polarity": deltas.sum() * 0}
        positive_loss = torch.stack(positive_terms).mean()
        negative_loss = torch.stack(negative_terms).mean()
        return {"loss_route_correction_polarity": 0.5 * (positive_loss + negative_loss)}

    def compute_matches(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        pos_ref_points: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Compute matching for interim encoder, detector heads including auxiliary heads.

        Args:
            outputs (Dict[str, Any]): model outputs
            targets (Dict[str, Any]): targets
            pos_ref_points (torch.Tensor): reference points

        Returns:
            Dict[str, Any]: dict with matching indexes and corresponding costs.
        """
        matching = {}
        # Retrieve the matching
        if self.one2one:
            retrieval_targets = targets
        else:
            retrieval_targets = create_targets_k_repeats(targets, self.target_repeat)

        if "encoder_outputs" in outputs:
            ecnoder_outputs = outputs["encoder_outputs"]
            indices, enc_matcher_costs = self.matcher(ecnoder_outputs, retrieval_targets, pos_ref_points)
            matching["encoder"] = {"indices": indices, "costs": enc_matcher_costs}

        # outputs without aux
        outputs_without_aux = {key: value for key, value in outputs.items() if key != "aux_outputs"}  # noqa: WPS204
        indices, pos_matcher_costs = self.matcher(outputs_without_aux, retrieval_targets, pos_ref_points)
        matching["positive"] = {"indices": indices, "costs": pos_matcher_costs}

        # compute auxiliary head matching
        matching["positive_aux"] = {"indices": [], "costs": []}
        for aux_outputs in outputs.get("aux_outputs"):  # type: ignore
            indices, costs = self.matcher(aux_outputs, retrieval_targets, pos_ref_points)
            matching["positive_aux"]["indices"].append(indices)
            matching["positive_aux"]["costs"].append(costs)
        return matching

    def forward(  # noqa: WPS213,C901
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        meta: List[Dict[str, Any]],
        matching: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute the losses for the model during training.

        Args:
            outputs (Dict[str, Any]): dict of tensors, see the output specification of the model for the format
            targets (Dict[str, Any]): Targets to use.
            meta (List[Dict[str, Any]]): Meta information.
            matching (Dict[str, Any]): Matching results.

        Returns:
            Dict[str, Any]: dict of tensors, with the loss values.
        """
        # Compute main losses
        losses = {}

        # Retrieve the matching
        if self.one2one:
            retrieval_targets = targets
        else:
            retrieval_targets = create_targets_k_repeats(targets, self.target_repeat)

        # compute losses for main head
        outputs_without_aux = {key: value for key, value in outputs.items() if key != "aux_outputs"}  # noqa: WPS204
        indices = matching["positive"]["indices"]
        enc_indices = matching["encoder"]["indices"] if "encoder" in matching else None
        losses.update(self.saliency_losses(outputs_without_aux, targets))
        losses.update(self.retrieval_losses(outputs_without_aux, retrieval_targets, indices, enc_indices))
        losses.update(self.auxiliary_losses(outputs_without_aux))
        losses.update(self.moment2text_losses(outputs_without_aux, targets))
        losses.update(self.aux_head_losses(outputs_without_aux, targets, meta))
        losses.update(self._route_alignment_loss(outputs_without_aux, retrieval_targets, indices))
        losses.update(self._query_length_prior_loss(outputs_without_aux, targets, meta))
        losses.update(self._query_prior_safety_loss(outputs_without_aux, meta))
        losses.update(self._expert_quality_loss(outputs_without_aux, retrieval_targets, indices))
        losses.update(self._expert_specialization_loss(outputs_without_aux, retrieval_targets, indices))
        losses.update(self._route_safety_loss(outputs_without_aux, retrieval_targets, indices))
        losses.update(self._route_ap_safety_loss(outputs_without_aux, retrieval_targets))
        losses.update(self._nonlong_route_safety_loss(outputs_without_aux, indices, meta))
        losses.update(self._route_gate_budget_loss(outputs_without_aux))
        losses.update(self._route_gate_oracle_loss(outputs_without_aux, indices))
        losses.update(self._route_correction_polarity_loss(outputs_without_aux, retrieval_targets, indices))

        # boundary discrimination loss
        if "vid_features" in outputs:
            losses.update(self.boundary_loss(outputs["vid_features"], targets))

        if outputs["collab_ref_dict"] is not None:
            losses.update(self.aux_ref_losses(outputs["collab_ref_dict"], aux_num=-1))

        if outputs["denoise_ref_dict"] is not None:
            losses.update(self.denoise_losses(outputs["denoise_ref_dict"], targets, aux_num=-1))

        if "aux_outputs" not in outputs:
            return losses

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        for idx, aux_outputs in enumerate(outputs.get("aux_outputs")):  # type: ignore
            indices = matching["positive_aux"]["indices"][idx]
            loss_dict = self.retrieval_losses(aux_outputs, retrieval_targets, indices)

            # update weights dict
            weight_dict = {f"{key}_{idx}": self.weight_dict.get(key, 0) for key, _ in loss_dict.items()}  # noqa: WPS221
            self.weight_dict.update(weight_dict)

            # update loss dict
            loss_dict = {f"{key}_{idx}": value for key, value in loss_dict.items()}  # noqa: WPS221
            losses.update(loss_dict)

            # compute aux references losses if it is possible
            if outputs["collab_ref_dict"] is not None:
                loss_dict_aux_ref = self.aux_ref_losses(outputs["collab_ref_dict"], aux_num=idx)
                loss_dict_aux_ref = {f"{key}_{idx}": value for key, value in loss_dict_aux_ref.items()}  # noqa: WPS221
                losses.update(loss_dict_aux_ref)

            # compute denoise losses if it is possible
            if outputs["denoise_ref_dict"] is not None:
                loss_dict_denoise = self.denoise_losses(outputs["denoise_ref_dict"], targets, aux_num=idx)
                loss_dict_denoise = {f"{key}_{idx}": value for key, value in loss_dict_denoise.items()}  # noqa: WPS221
                losses.update(loss_dict_denoise)
        return losses

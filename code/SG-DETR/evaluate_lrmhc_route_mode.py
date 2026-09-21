#!/usr/bin/env python3
"""Evaluate one LR-MHC checkpoint with soft or forced-uniform routing."""

import argparse
import copy
import json
import math
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Saved Hydra config.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--non-strict-checkpoint", action="store_true")
    parser.add_argument("--mode", choices=("soft", "uniform"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override validation batch size to reduce evaluation-only GPU memory.",
    )
    parser.add_argument(
        "--ranking-attribute",
        choices=("probs", "iou_score", "combo"),
        default=None,
        help="Override the DETR candidate ranking score used by the postprocessor.",
    )
    parser.add_argument("--nms-threshold", type=float, default=None)
    parser.add_argument(
        "--nms-attribute",
        choices=("probs", "iou_score", "combo"),
        default=None,
    )
    parser.add_argument(
        "--route-gain",
        type=float,
        default=None,
        help="Override the positive deployed route gain for counterfactual evaluation.",
    )
    parser.add_argument(
        "--route-beta",
        type=float,
        default=None,
        help="Override the fixed inverse temperature used by prototype routing.",
    )
    parser.add_argument(
        "--route-rescore-target",
        choices=("class", "quality", "both"),
        default=None,
        help="Override which prediction branch receives the routed residual.",
    )
    parser.add_argument(
        "--route-alpha",
        type=float,
        nargs="+",
        default=None,
        help="Override the deployed expert residual scales (one shared value or three values).",
    )
    parser.add_argument(
        "--absolute-duration-routing",
        action="store_true",
        help="Route predicted absolute durations to duration-space expert prototypes.",
    )
    parser.add_argument(
        "--direct-route-gain",
        type=float,
        default=None,
        help="Scale the routed-minus-uniform residual when no route gate is used.",
    )
    parser.add_argument("--direct-route-topk", type=int, default=None)
    parser.add_argument("--direct-route-topk-temperature", type=float, default=0.05)
    parser.add_argument("--direct-route-duration-gains", type=float, nargs=3, default=None)
    parser.add_argument("--direct-route-confidence-threshold", type=float, default=None)
    parser.add_argument("--direct-route-confidence-temperature", type=float, default=0.05)
    parser.add_argument(
        "--route-gate-pivot",
        type=float,
        default=None,
        help="Override the monotonic gate pivot in route-signal logit space.",
    )
    parser.add_argument(
        "--force-expert",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="Diagnostic override selecting one expert for every query.",
    )
    parser.add_argument(
        "--disable-monotonic-gate",
        action="store_true",
        help="Use the shared soft expert gate instead of the Long-only monotonic gate.",
    )
    parser.add_argument(
        "--disable-route-gate",
        action="store_true",
        help="Apply the learned soft routing without any reliability gate.",
    )
    parser.add_argument(
        "--positive-route-correction",
        action="store_true",
        help="Apply only positive routed-over-uniform expert advantages.",
    )
    parser.add_argument(
        "--signed-route-gate",
        action="store_true",
        help="Gate positive advantages with g and negative advantages with 1-g.",
    )
    parser.add_argument(
        "--signed-route-gate-temperature",
        type=float,
        default=None,
        help="Calibrate signed reliability logits before applying routed corrections.",
    )
    parser.add_argument(
        "--signed-route-negative-gain",
        type=float,
        default=None,
        help="Scale negative signed corrections independently from positive corrections.",
    )
    parser.add_argument("--route-width-safety-threshold", type=float, default=None)
    parser.add_argument("--route-width-safety-temperature", type=float, default=0.05)
    parser.add_argument("--sample-width-safety-threshold", type=float, default=None)
    parser.add_argument("--sample-width-min-safety-threshold", type=float, default=None)
    parser.add_argument("--sample-width-safety-temperature", type=float, default=0.05)
    parser.add_argument("--sample-width-score-temperature", type=float, default=0.05)
    parser.add_argument(
        "--signed-route-expert-gains",
        type=float,
        nargs=3,
        default=None,
        help="Softly mix three expert-specific gains using each query's route probabilities.",
    )
    parser.add_argument(
        "--selective-short-threshold",
        type=float,
        default=None,
        help="Softly interpolate to expert 0 above this sample Short probability.",
    )
    parser.add_argument(
        "--selective-short-temperature",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--selective-middle-threshold",
        type=float,
        default=None,
        help="Softly interpolate to expert 1 above this sample Middle probability.",
    )
    parser.add_argument(
        "--selective-middle-temperature",
        type=float,
        default=0.05,
    )
    parser.add_argument("--selective-middle-hard-argmax", action="store_true")
    parser.add_argument("--selective-middle-blend", type=float, default=None)
    parser.add_argument("--selective-middle-full-expert", action="store_true")
    parser.add_argument(
        "--selective-long-threshold",
        type=float,
        default=None,
        help="Softly interpolate to expert 2 above this sample Long probability.",
    )
    parser.add_argument(
        "--selective-long-temperature",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--selective-uniform-base",
        action="store_true",
        help="Start selective expert interpolation from the uniform expert mixture.",
    )
    parser.add_argument("--selective-expert-topk", type=int, default=None)
    parser.add_argument(
        "--selective-expert-topk-temperature",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--query-prior-scale",
        type=float,
        default=None,
        help="Override the sample-level length-prior contribution at evaluation time.",
    )
    parser.add_argument(
        "--duration-direct-routing-blend",
        type=float,
        default=None,
        help="Convex blend from query routing to sample-duration expert probabilities.",
    )
    parser.add_argument(
        "--duration-direct-routing-temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--duration-safety-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint supplying a separately balanced duration safety head.",
    )
    parser.add_argument(
        "--duration-router-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint replacing the deployed sample-duration router.",
    )
    parser.add_argument("--duration-short-safety-threshold", type=float, default=None)
    parser.add_argument("--duration-short-safety-temperature", type=float, default=0.05)
    parser.add_argument("--duration-middle-safety-threshold", type=float, default=None)
    parser.add_argument("--duration-middle-safety-temperature", type=float, default=0.05)
    parser.add_argument(
        "--short-safe-query-prior",
        action="store_true",
        help="Attenuate the sample prior when it predicts a short target.",
    )
    parser.add_argument(
        "--hard-query-prior-safety",
        action="store_true",
        help="Use the learned binary safety decision instead of its soft probability.",
    )
    parser.add_argument("--query-route-gate-threshold", type=float, default=None)
    parser.add_argument("--query-route-gate-temperature", type=float, default=0.05)
    parser.add_argument("--sample-route-gate-threshold", type=float, default=None)
    parser.add_argument("--sample-route-gate-temperature", type=float, default=0.05)
    parser.add_argument(
        "--preserve-selective-short-from-sample-gate",
        action="store_true",
    )
    parser.add_argument(
        "--preserve-uniform-top1",
        action="store_true",
        help="Keep the uniform path's top-1 query and rerank only lower candidates.",
    )
    parser.add_argument("--preserve-uniform-top1-margin", type=float, default=0.05)
    parser.add_argument(
        "--preserve-uniform-top1-width-threshold",
        type=float,
        default=None,
        help="Softly enable top-1 preservation as predicted sample width increases.",
    )
    parser.add_argument(
        "--preserve-uniform-top1-width-temperature",
        type=float,
        default=0.03,
    )
    parser.add_argument("--calibrated-middle-checkpoint", type=Path, default=None)
    parser.add_argument("--middle-adapter-low", type=float, default=10.0)
    parser.add_argument("--middle-adapter-high", type=float, default=40.0)
    parser.add_argument("--middle-adapter-temperature", type=float, default=8.0)
    parser.add_argument("--middle-adapter-gain", type=float, default=1.25)
    parser.add_argument("--long-route-boost-start", type=float, default=75.0)
    parser.add_argument("--long-route-boost-temperature", type=float, default=2.0)
    parser.add_argument("--long-route-boost-gain", type=float, default=0.2)
    return parser.parse_args()


def install_calibrated_middle_adapter(runner, args: argparse.Namespace) -> None:
    """Blend one calibrated middle expert without changing fixed candidate order."""
    if args.mode != "soft":
        raise ValueError("the calibrated middle adapter is only defined for soft routing")
    if not 0 < args.middle_adapter_low < args.middle_adapter_high:
        raise ValueError("middle adapter bounds must satisfy 0 < low < high")
    if args.middle_adapter_temperature <= 0 or args.middle_adapter_gain <= 0:
        raise ValueError("middle adapter temperature and gain must be positive")
    if args.long_route_boost_start <= 0:
        raise ValueError("long route boost start must be positive")
    if args.long_route_boost_temperature <= 0 or args.long_route_boost_gain < 0:
        raise ValueError("long route boost temperature must be positive and gain non-negative")

    checkpoint_state = torch.load(
        args.calibrated_middle_checkpoint,
        map_location="cpu",
        weights_only=False,
    )["state_dict"]
    weight_key = "model.main_det_head.length_rescoring.weight"
    bias_key = "model.main_det_head.length_rescoring.bias"
    if weight_key not in checkpoint_state or bias_key not in checkpoint_state:
        raise ValueError("calibrated checkpoint does not contain the shared expert head")
    head = runner.model.main_det_head
    rescoring = head.length_rescoring
    if not isinstance(rescoring, torch.nn.Linear) or rescoring.out_features != 3:
        raise ValueError("calibrated middle adapter requires a three-output linear expert head")
    calibrated_weight = checkpoint_state[weight_key]
    calibrated_bias = checkpoint_state[bias_key]
    fixed_weight = rescoring.weight.detach().cpu()
    fixed_bias = rescoring.bias.detach().cpu()
    if not torch.equal(calibrated_weight[[0, 2]], fixed_weight[[0, 2]]) or not torch.equal(
        calibrated_bias[[0, 2]],
        fixed_bias[[0, 2]],
    ):
        raise ValueError("calibrated checkpoint changed the short or long expert")
    calibrated_middle_weight = calibrated_weight[1].clone()
    calibrated_middle_bias = calibrated_bias[1].clone()
    fixed_middle_weight = rescoring.weight[1].detach().clone()
    fixed_middle_bias = rescoring.bias[1].detach().clone()
    original_forward = runner.model.forward

    def adapter_forward(*forward_args, **forward_kwargs):
        fixed_outputs = original_forward(*forward_args, **forward_kwargs)
        with torch.no_grad():
            rescoring.weight[1].copy_(calibrated_middle_weight.to(rescoring.weight))
            rescoring.bias[1].copy_(calibrated_middle_bias.to(rescoring.bias))
        try:
            calibrated_outputs = original_forward(*forward_args, **forward_kwargs)
        finally:
            with torch.no_grad():
                rescoring.weight[1].copy_(fixed_middle_weight)
                rescoring.bias[1].copy_(fixed_middle_bias)

        required = {
            "pred_logits",
            "pred_quality_scores",
            "pred_spans",
            "pred_uniform_logits",
            "pred_uniform_quality_scores",
        }
        if not required.issubset(fixed_outputs) or not required.issubset(calibrated_outputs):
            raise RuntimeError("middle adapter requires routed and uniform counterfactual outputs")
        meta = forward_kwargs.get("meta")
        if meta is None:
            raise RuntimeError("middle adapter requires video durations from metadata")
        fixed_logits = fixed_outputs["pred_logits"]
        fixed_quality = fixed_outputs["pred_quality_scores"]
        calibrated_logits = calibrated_outputs["pred_logits"]
        calibrated_quality = calibrated_outputs["pred_quality_scores"]
        uniform_logits = fixed_outputs["pred_uniform_logits"]
        uniform_quality = fixed_outputs["pred_uniform_quality_scores"]
        durations = fixed_logits.new_tensor([sample["duration"] for sample in meta]).view(-1, 1, 1)
        spans = fixed_outputs["pred_spans"].detach()
        starts = (spans[..., 0:1] - spans[..., 1:2] / 2) * durations
        ends = (spans[..., 0:1] + spans[..., 1:2] / 2) * durations
        starts = torch.minimum(starts.clamp_min(0), durations).clamp(max=150)
        ends = torch.minimum(ends.clamp_min(0), durations).clamp(max=150)
        starts = torch.round(starts / 2) * 2
        ends = torch.round(ends / 2) * 2
        predicted_duration = (ends - starts).clamp(min=2, max=150)
        middle_gate = torch.sigmoid(
            (predicted_duration - args.middle_adapter_low) / args.middle_adapter_temperature,
        ) * torch.sigmoid(
            (args.middle_adapter_high - predicted_duration) / args.middle_adapter_temperature,
        )
        long_gate = torch.sigmoid(
            (predicted_duration - args.long_route_boost_start)
            / args.long_route_boost_temperature,
        )

        def rounded_probability(value):
            return torch.round(torch.sigmoid(value) * 10000) / 10000

        def combined_score(logits, quality):
            return (
                rounded_probability(logits) * rounded_probability(quality)
            ).clamp_min(1e-12).sqrt()

        fixed_score = combined_score(fixed_logits, fixed_quality)
        calibrated_score = combined_score(calibrated_logits, calibrated_quality)
        uniform_score = combined_score(uniform_logits, uniform_quality)
        desired_score = (
            fixed_score
            + args.middle_adapter_gain * middle_gate * (calibrated_score - fixed_score)
            + args.long_route_boost_gain * long_gate * (fixed_score - uniform_score)
        ).clamp(min=1e-6, max=1 - 1e-6)
        # Keep classification logits unchanged so the fixed branch's candidate
        # ordering and top-k set are exactly preserved.
        desired_quality_probability = (
            desired_score.square() / rounded_probability(fixed_logits).clamp_min(1e-8)
        ).clamp(min=1e-6, max=1 - 1e-6)
        outputs = dict(fixed_outputs)
        outputs["pred_quality_scores"] = torch.logit(desired_quality_probability)
        outputs["middle_adapter_gate_scores"] = middle_gate
        outputs["long_route_boost_gate_scores"] = long_gate
        outputs["fixed_branch_scores"] = fixed_score
        outputs["calibrated_middle_branch_scores"] = calibrated_score
        outputs["uniform_branch_scores"] = uniform_score
        outputs["desired_adapter_scores"] = desired_score
        return outputs

    runner.model.forward = adapter_forward


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config)
    seed_everything(int(config.get("seed", 0)), workers=True)
    config.data.num_workers = args.num_workers
    if args.ranking_attribute is not None:
        config.postprocessor.postprocessor.ranking_attribute = args.ranking_attribute
    if args.nms_threshold is not None:
        if not 0 <= args.nms_threshold <= 1:
            raise ValueError("--nms-threshold must be in [0, 1]")
        config.postprocessor.postprocessor.nms_threshold = args.nms_threshold
    if args.nms_attribute is not None:
        config.postprocessor.postprocessor.nms_attribute = args.nms_attribute
    # The target checkpoint below is self-contained. Avoid a redundant preload
    # from the training-time source path, which may have been a transient symlink.
    config.model.runner.checkpoint_path = None
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive")
        config.data.batch_size = args.batch_size
        config.data.val_batch_size = args.batch_size
        # Validation uses a separate fixed batch-size constant.
        import src.datamodule as datamodule_module

        datamodule_module.VAL_BATCH_SIZE = args.batch_size
    datamodule = instantiate(config.data)
    runner = instantiate(config.model.runner)
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    runner.load_state_dict(state_dict, strict=not args.non_strict_checkpoint)
    if args.duration_router_checkpoint is not None:
        router_state = torch.load(
            args.duration_router_checkpoint,
            map_location="cpu",
            weights_only=False,
        )["state_dict"]
        prefix = "model.main_det_head.duration_safety_prior."
        module_state = {
            key.removeprefix(prefix): value
            for key, value in router_state.items()
            if key.startswith(prefix)
        }
        if not module_state:
            raise ValueError("duration router checkpoint does not contain a duration prior")
        runner.model.main_det_head.duration_safety_prior.load_state_dict(module_state)
        runner.model.main_det_head.use_duration_safety_prior = True
    if args.duration_safety_checkpoint is not None:
        safety_state = torch.load(
            args.duration_safety_checkpoint,
            map_location="cpu",
            weights_only=False,
        )["state_dict"]
        prefix = "model.main_det_head.duration_safety_prior."
        module_state = {
            key.removeprefix(prefix): value
            for key, value in safety_state.items()
            if key.startswith(prefix)
        }
        if not module_state:
            raise ValueError("duration safety checkpoint does not contain a duration prior")
        head = runner.model.main_det_head
        head.use_duration_safety_prior = True
        head.duration_route_safety_prior = copy.deepcopy(head.duration_safety_prior)
        head.duration_route_safety_prior.load_state_dict(module_state)
        if args.duration_short_safety_threshold is not None:
            if not 0 < args.duration_short_safety_threshold < 1:
                raise ValueError("--duration-short-safety-threshold must be in (0, 1)")
            if args.duration_short_safety_temperature <= 0:
                raise ValueError("--duration-short-safety-temperature must be positive")
            head.duration_short_safety_threshold = args.duration_short_safety_threshold
            head.duration_short_safety_temperature = args.duration_short_safety_temperature
        if args.duration_middle_safety_threshold is not None:
            if not 0 < args.duration_middle_safety_threshold < 1:
                raise ValueError("--duration-middle-safety-threshold must be in (0, 1)")
            if args.duration_middle_safety_temperature <= 0:
                raise ValueError("--duration-middle-safety-temperature must be positive")
            head.duration_middle_safety_threshold = args.duration_middle_safety_threshold
            head.duration_middle_safety_temperature = args.duration_middle_safety_temperature
    runner.model.main_det_head.uniform_routing = args.mode == "uniform"
    if args.mode == "uniform":
        for loss_name in (
            "loss_route_pairwise_gain",
            "loss_route_utility",
            "loss_route_gate_relevance",
            "loss_route_gate_budget",
            "loss_route_gate_oracle",
            "loss_route_correction_polarity",
            "loss_route_safety",
        ):
            runner.losses.weight_dict[loss_name] = 0.0
    runner.model.main_det_head.forced_route_expert = args.force_expert
    if args.route_beta is not None:
        if args.route_beta <= 0:
            raise ValueError("--route-beta must be positive")
        runner.model.main_det_head.route_beta = args.route_beta
    if args.route_rescore_target is not None:
        if args.route_rescore_target == "quality" and not runner.model.main_det_head.predict_quality_score:
            raise ValueError("quality routing requires a quality prediction head")
        runner.model.main_det_head.route_rescore_target = args.route_rescore_target
    if args.absolute_duration_routing:
        runner.model.main_det_head.use_absolute_duration_routing = True
    if args.force_expert is not None:
        runner.model.main_det_head.use_route_gate = False
    if args.disable_route_gate:
        runner.model.main_det_head.use_route_gate = False
    if args.positive_route_correction:
        runner.model.main_det_head.positive_route_correction = True
    if args.signed_route_gate:
        runner.model.main_det_head.signed_route_gate = True
    if args.signed_route_gate_temperature is not None:
        if args.signed_route_gate_temperature <= 0:
            raise ValueError("--signed-route-gate-temperature must be positive")
        runner.model.main_det_head.signed_route_gate_temperature = (
            args.signed_route_gate_temperature
        )
    if args.signed_route_negative_gain is not None:
        if args.signed_route_negative_gain < 0:
            raise ValueError("--signed-route-negative-gain must be non-negative")
        runner.model.main_det_head.signed_route_negative_gain = (
            args.signed_route_negative_gain
        )
    if args.route_width_safety_threshold is not None:
        if not 0 < args.route_width_safety_threshold < 1:
            raise ValueError("--route-width-safety-threshold must be in (0, 1)")
        if args.route_width_safety_temperature <= 0:
            raise ValueError("--route-width-safety-temperature must be positive")
        runner.model.main_det_head.route_width_safety_threshold = (
            args.route_width_safety_threshold
        )
        runner.model.main_det_head.route_width_safety_temperature = (
            args.route_width_safety_temperature
        )
    if args.sample_width_safety_threshold is not None:
        if not 0 < args.sample_width_safety_threshold < 1:
            raise ValueError("--sample-width-safety-threshold must be in (0, 1)")
        if args.sample_width_safety_temperature <= 0 or args.sample_width_score_temperature <= 0:
            raise ValueError("sample-width temperatures must be positive")
        head = runner.model.main_det_head
        head.sample_width_safety_threshold = args.sample_width_safety_threshold
        head.sample_width_safety_temperature = args.sample_width_safety_temperature
        head.sample_width_score_temperature = args.sample_width_score_temperature
    if args.sample_width_min_safety_threshold is not None:
        if not 0 < args.sample_width_min_safety_threshold < 1:
            raise ValueError("--sample-width-min-safety-threshold must be in (0, 1)")
        if args.sample_width_safety_temperature <= 0 or args.sample_width_score_temperature <= 0:
            raise ValueError("sample-width temperatures must be positive")
        head = runner.model.main_det_head
        head.sample_width_min_safety_threshold = args.sample_width_min_safety_threshold
        head.sample_width_safety_temperature = args.sample_width_safety_temperature
        head.sample_width_score_temperature = args.sample_width_score_temperature
    if args.signed_route_expert_gains is not None:
        if any(value < 0 for value in args.signed_route_expert_gains):
            raise ValueError("--signed-route-expert-gains must be non-negative")
        runner.model.main_det_head.signed_route_expert_gains = tuple(
            args.signed_route_expert_gains,
        )
    if args.disable_monotonic_gate:
        runner.model.main_det_head.use_monotonic_route_gate = False
    if args.selective_short_threshold is not None:
        head = runner.model.main_det_head
        if not 0 < args.selective_short_threshold < 1:
            raise ValueError("--selective-short-threshold must be in (0, 1)")
        if args.selective_short_temperature <= 0:
            raise ValueError("--selective-short-temperature must be positive")
        head.use_selective_short_expert = True
        head.selective_short_threshold = args.selective_short_threshold
        head.selective_short_temperature = args.selective_short_temperature
    if args.selective_middle_threshold is not None:
        head = runner.model.main_det_head
        if not 0 < args.selective_middle_threshold < 1:
            raise ValueError("--selective-middle-threshold must be in (0, 1)")
        if args.selective_middle_temperature <= 0:
            raise ValueError("--selective-middle-temperature must be positive")
        head.use_selective_middle_expert = True
        head.selective_middle_threshold = args.selective_middle_threshold
        head.selective_middle_temperature = args.selective_middle_temperature
        head.selective_middle_hard_argmax = args.selective_middle_hard_argmax
        head.selective_middle_full_expert = args.selective_middle_full_expert
        if args.selective_middle_blend is not None:
            if not 0 < args.selective_middle_blend <= 1:
                raise ValueError("--selective-middle-blend must be in (0, 1]")
            head.selective_middle_blend = args.selective_middle_blend
    if args.selective_long_threshold is not None:
        head = runner.model.main_det_head
        if not 0 < args.selective_long_threshold < 1:
            raise ValueError("--selective-long-threshold must be in (0, 1)")
        if args.selective_long_temperature <= 0:
            raise ValueError("--selective-long-temperature must be positive")
        head.use_selective_long_expert = True
        head.selective_long_threshold = args.selective_long_threshold
        head.selective_long_temperature = args.selective_long_temperature
    if args.selective_uniform_base:
        runner.model.main_det_head.selective_uniform_base = True
    if args.selective_expert_topk is not None:
        if args.selective_expert_topk <= 0:
            raise ValueError("--selective-expert-topk must be positive")
        if args.selective_expert_topk_temperature <= 0:
            raise ValueError("--selective-expert-topk-temperature must be positive")
        head = runner.model.main_det_head
        head.selective_expert_topk = args.selective_expert_topk
        head.selective_expert_topk_temperature = args.selective_expert_topk_temperature
    if args.query_prior_scale is not None:
        if args.query_prior_scale < 0:
            raise ValueError("--query-prior-scale must be non-negative")
        runner.model.main_det_head.query_length_prior_scale = args.query_prior_scale
    if args.duration_direct_routing_blend is not None:
        if not 0 < args.duration_direct_routing_blend <= 1:
            raise ValueError("--duration-direct-routing-blend must be in (0, 1]")
        runner.model.main_det_head.duration_direct_routing_blend = (
            args.duration_direct_routing_blend
        )
        if args.duration_direct_routing_temperature <= 0:
            raise ValueError("--duration-direct-routing-temperature must be positive")
        runner.model.main_det_head.duration_direct_routing_temperature = (
            args.duration_direct_routing_temperature
        )
    if args.short_safe_query_prior:
        runner.model.main_det_head.use_short_safe_query_prior = True
    if args.hard_query_prior_safety:
        runner.model.main_det_head.hard_query_prior_safety = True
    if args.query_route_gate_threshold is not None:
        if not 0 < args.query_route_gate_threshold < 1:
            raise ValueError("--query-route-gate-threshold must be in (0, 1)")
        if args.query_route_gate_temperature <= 0:
            raise ValueError("--query-route-gate-temperature must be positive")
        head = runner.model.main_det_head
        head.query_route_gate_threshold = args.query_route_gate_threshold
        head.query_route_gate_temperature = args.query_route_gate_temperature
    if args.sample_route_gate_threshold is not None:
        if not 0 < args.sample_route_gate_threshold < 1:
            raise ValueError("--sample-route-gate-threshold must be in (0, 1)")
        if args.sample_route_gate_temperature <= 0:
            raise ValueError("--sample-route-gate-temperature must be positive")
        head = runner.model.main_det_head
        if not head.use_sample_route_gate:
            raise ValueError("--sample-route-gate-threshold requires a sample route gate")
        head.sample_route_gate_threshold = args.sample_route_gate_threshold
        head.sample_route_gate_temperature = args.sample_route_gate_temperature
    if args.preserve_selective_short_from_sample_gate:
        runner.model.main_det_head.preserve_selective_short_from_sample_gate = True
    if args.preserve_uniform_top1:
        if args.preserve_uniform_top1_margin <= 0:
            raise ValueError("--preserve-uniform-top1-margin must be positive")
        if args.preserve_uniform_top1_width_threshold is not None:
            if not 0 < args.preserve_uniform_top1_width_threshold < 1:
                raise ValueError("top-1 width threshold must be in (0, 1)")
            if args.preserve_uniform_top1_width_temperature <= 0:
                raise ValueError("top-1 width temperature must be positive")
        head = runner.model.main_det_head
        head.preserve_uniform_top1 = True
        head.preserve_uniform_top1_margin = args.preserve_uniform_top1_margin
        head.preserve_uniform_top1_width_threshold = (
            args.preserve_uniform_top1_width_threshold
        )
        head.preserve_uniform_top1_width_temperature = (
            args.preserve_uniform_top1_width_temperature
        )
    if args.route_gain is not None:
        if args.route_gain <= 0:
            raise ValueError("--route-gain must be positive")
        head = runner.model.main_det_head
        raw_gain = math.log(math.expm1(args.route_gain))
        if hasattr(head, "route_gain_raw"):
            head.route_gain_raw.data.fill_(raw_gain)
        else:
            head.register_parameter("route_gain_raw", torch.nn.Parameter(torch.tensor(raw_gain)))
            head.use_learnable_route_gain = True
    if args.route_alpha is not None:
        if len(args.route_alpha) not in (1, 3):
            raise ValueError("--route-alpha requires one shared value or three expert values")
        if any(value < 0 for value in args.route_alpha):
            raise ValueError("--route-alpha values must be non-negative")
        head = runner.model.main_det_head
        if not hasattr(head, "length_rescoring_alpha"):
            raise ValueError("--route-alpha requires the learnable alpha gate")
        values = args.route_alpha * 3 if len(args.route_alpha) == 1 else args.route_alpha
        head.length_rescoring_alpha.data.copy_(
            torch.tensor(values, device=head.length_rescoring_alpha.device),
        )
    if args.direct_route_gain is not None:
        if args.direct_route_gain <= 0:
            raise ValueError("--direct-route-gain must be positive")
        runner.model.main_det_head.direct_route_gain = args.direct_route_gain
    if args.direct_route_topk is not None:
        if args.direct_route_topk <= 0:
            raise ValueError("--direct-route-topk must be positive")
        if args.direct_route_topk_temperature <= 0:
            raise ValueError("--direct-route-topk-temperature must be positive")
        head = runner.model.main_det_head
        head.direct_route_topk = args.direct_route_topk
        head.direct_route_topk_temperature = args.direct_route_topk_temperature
    if args.direct_route_duration_gains is not None:
        if any(value <= 0 for value in args.direct_route_duration_gains):
            raise ValueError("--direct-route-duration-gains must be positive")
        runner.model.main_det_head.direct_route_duration_gains = tuple(
            args.direct_route_duration_gains,
        )
    if args.direct_route_confidence_threshold is not None:
        if not 0 < args.direct_route_confidence_threshold < 1:
            raise ValueError("--direct-route-confidence-threshold must be in (0, 1)")
        if args.direct_route_confidence_temperature <= 0:
            raise ValueError("--direct-route-confidence-temperature must be positive")
        head = runner.model.main_det_head
        head.direct_route_confidence_threshold = args.direct_route_confidence_threshold
        head.direct_route_confidence_temperature = args.direct_route_confidence_temperature
    if args.route_gate_pivot is not None:
        head = runner.model.main_det_head
        if not hasattr(head, "route_gate_pivot"):
            raise ValueError("--route-gate-pivot requires a monotonic route gate")
        head.route_gate_pivot.data.fill_(args.route_gate_pivot)
    if args.calibrated_middle_checkpoint is not None:
        install_calibrated_middle_adapter(runner, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = TensorBoardLogger(save_dir=str(args.output_dir), name=args.mode)
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        enable_checkpointing=False,
        inference_mode=True,
        precision=config.trainer.get("precision", "32-true"),
    )
    # Module construction consumes RNG differently across ablations. Resetting
    # here makes the validation-time query path identical across architectures.
    seed_everything(int(config.get("seed", 0)), workers=True)
    results = trainer.validate(model=runner, datamodule=datamodule, verbose=False)[0]
    metrics = {key: float(value) for key, value in results.items() if key.startswith("val/")}
    output_path = args.output_dir / f"metrics_{args.mode}.json"
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(output_path)
    for key in (
        "val/MR-mAP-Full_Avg",
        "val/MR-mAP-Short_Avg",
        "val/MR-mAP-Middle_Avg",
        "val/MR-mAP-Long_Avg",
    ):
        print(f"{key}: {metrics[key]:.6f}")


if __name__ == "__main__":
    main()

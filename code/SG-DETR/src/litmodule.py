# pylint:disable=arguments-differ,unused-argument
"""MomentRetrievalRunner Module."""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as func
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.optim.lr_scheduler import _LRScheduler  # noqa: WPS450

from src.dataset.collate import move_inputs_to_device
from src.losses.losses import SetCriterion
from src.losses.utils import fix_loss_name
from src.metrics.matching.metrics import MatchingMetric
from src.metrics.metrics_collection import (
    get_aux_metrics,
    get_charades_metrics,
    get_metrics,
    get_tvsum_metrics,
    get_youtube_metrics,
)
from src.model.model import MRDETR
from src.model.utils.params import get_params_by_name
from src.postprocessor.postprocessing import (
    OutputCombiner,
    PostProcessorDETR,
    Preparator,
)
from src.utils.rw_utils import save_jsonl
from src.utils.span_utils import span_cxw_to_xx

MetaTypes = List[Dict[str, Any]]
MAIN_METRICS_DICT: Tuple[str, ...] = (
    "HL-HIT@1-VeryGood",
    "HL-mAP-VeryGood",
    "MR-mAP-Full_0.5",
    "MR-mAP-Full_0.75",
    "MR-mAP-Full_Avg",
    "MR-mAP-Short_Avg",
    "MR-mAP-Middle_Avg",
    "MR-mAP-Long_Avg",
    "MR-R1-Full_0.3",
    "MR-R1-Full_0.5",
    "MR-R1-Full_0.7",
    "MR-R1-Full_mIoU",
)


def _fix_output(postprocessed_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    They fix the output of post-processing. This is necessary during training because the model might produce zero
    objects in the early epochs, which causes issues in calculating metrics.

    Args:
        postprocessed_outputs (List[Dict[str, Any]]): postprocessed outputs

    Returns:
        List[Dict[str, Any]]: fixes postprocessed outputs
    """
    new_outputs: List[Dict[str, Any]] = []
    for outputs in postprocessed_outputs:
        if len(outputs["pred_relevant_windows"]) == 0:
            outputs["pred_relevant_windows"] = torch.tensor([[0.0, 0.0, 0.0]])
        new_outputs.append(outputs)
    return new_outputs


# pylint: disable=too-many-instance-attributes
class MomentRetrievalRunner(LightningModule):  # noqa: WPS214,WPS230,E302
    """The main LightningModule for moment retrieval tasks."""

    def __init__(
        self,
        model: MRDETR,
        optimizer: torch.optim.Optimizer,
        scheduler: _LRScheduler,
        postprocessor: PostProcessorDETR,
        preparator: Preparator,
        combiner: OutputCombiner,
        losses: SetCriterion,
        metrics_mode: str = "qvhighlights",
        checkpoint_path: Optional[str] = None,
        check_train_every_n_epoch: int = 5,
        new_module_lr: Optional[float] = None,
        router_lr: Optional[float] = None,
        gate_lr: Optional[float] = None,
        freeze_base_epochs: int = 0,
        joint_base_lr: Optional[float] = None,
        joint_new_module_lr: Optional[float] = None,
        keep_base_frozen_after_warmup: bool = False,
        calibration_module_token: Optional[str] = None,
        calibration_expert_index: Optional[int] = None,
        center_expert_gradients: bool = False,
        route_beta_start: Optional[float] = None,
        route_beta_end: Optional[float] = None,
        route_beta_anneal_fraction: float = 1.0,
    ) -> None:
        """
        Initialize the MomentRetrievalRunner.

        Args:
            model (MRDETR): The model to train.
            optimizer (torch.optim.Optimizer): The optimizer to use.
            scheduler (_LRScheduler): The learning rate scheduler to use.
            preparator(Preparator): Convert predictions to format.
            postprocessor (PostProcessorDETR): The postprocessor to use.
            losses (SetCriterion): The loss function to use.
            checkpoint_path (Optional[str]): path to base checkpoint.
            check_train_every_n_epoch (int): compute train metrics every N epoch
        """
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.save_hyperparameters(ignore=["model", "losses", "postprocessor", "scheduler", "combiner"])
        self.check_train_every_n_epoch = check_train_every_n_epoch
        self.freeze_base_epochs = freeze_base_epochs
        self.joint_base_lr = joint_base_lr
        self.joint_new_module_lr = joint_new_module_lr
        self.keep_base_frozen_after_warmup = keep_base_frozen_after_warmup
        self.calibration_expert_index = calibration_expert_index
        self.center_expert_gradients = center_expert_gradients
        self.route_beta_start = route_beta_start
        self.route_beta_end = route_beta_end
        self.route_beta_anneal_fraction = route_beta_anneal_fraction
        if (route_beta_start is None) != (route_beta_end is None):
            raise ValueError("route_beta_start and route_beta_end must be set together")
        if route_beta_start is not None:
            if route_beta_start <= 0 or route_beta_end <= 0:  # type: ignore[operator]
                raise ValueError("route beta schedule values must be positive")
            if not 0 < route_beta_anneal_fraction <= 1:
                raise ValueError("route_beta_anneal_fraction must be in (0, 1]")
        self.model = model
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, weights_only=False)["state_dict"]
            state_dict = {key[6:]: value for key, value in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)

        if self.freeze_base_epochs > 0:
            self._set_base_trainable(False)
        if calibration_module_token is not None:
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad = calibration_module_token in name
        if calibration_expert_index is not None:
            if calibration_module_token != "length_rescoring":
                raise ValueError(
                    "calibration_expert_index requires calibration_module_token=length_rescoring",
                )
            if calibration_expert_index not in (0, 1, 2):
                raise ValueError("calibration_expert_index must be 0, 1, or 2")
            rescoring = self.model.main_det_head.length_rescoring
            if not isinstance(rescoring, nn.Linear) or rescoring.out_features != 3:
                raise ValueError("expert-row calibration requires a three-output linear head")
            def keep_expert_row(gradient: Tensor) -> Tensor:
                mask = torch.zeros_like(gradient)
                mask[calibration_expert_index] = 1
                return gradient * mask

            rescoring.weight.register_hook(keep_expert_row)
            rescoring.bias.register_hook(keep_expert_row)
        if center_expert_gradients:
            rescoring = self.model.main_det_head.length_rescoring
            if not isinstance(rescoring, nn.Linear) or rescoring.out_features != 3:
                raise ValueError("center_expert_gradients requires a three-output linear expert head")

            def center_rows(gradient: Tensor) -> Tensor:
                return gradient - gradient.mean(dim=0, keepdim=True)

            rescoring.weight.register_hook(center_rows)
            rescoring.bias.register_hook(center_rows)

        self.losses = losses
        if calibration_module_token is not None:
            for parameter in self.losses.parameters():
                parameter.requires_grad = False
        self.postprocessor = postprocessor
        self.preparator = preparator
        self.combiner = combiner
        self.scheduler = scheduler
        self._init_metrics(metrics_mode)
        self.metrics_mode = metrics_mode

    @staticmethod
    def _is_lrmhc_parameter(name: str) -> bool:
        """Return whether a parameter belongs to the new soft-routing module."""
        return any(
            token in name
            for token in (
                "main_det_head.length_rescoring",
                "main_det_head.length_class_embed",
                "main_det_head.route_center_start",
                "main_det_head.route_gap_raw",
                "main_det_head.route_gate_logits",
                "main_det_head.length_router",
                "main_det_head.query_route_gate",
                "main_det_head.route_gate_pivot",
                "main_det_head.route_gate_slope_raw",
                "main_det_head.route_gain_raw",
                "main_det_head.route_expert_gain_raw",
                "main_det_head.query_length_prior",
                "main_det_head.duration_safety_prior",
                "main_det_head.query_prior_safety_gate",
                "main_det_head.route_safety_adapter",
                "main_det_head.sample_route_gate",
            )
        )

    @staticmethod
    def _is_router_parameter(name: str) -> bool:
        """Return whether a parameter controls routing rather than expert prediction."""
        return any(
            token in name
            for token in (
                "main_det_head.route_center_start",
                "main_det_head.route_gap_raw",
                "main_det_head.route_gate_logits",
                "main_det_head.length_router",
                "main_det_head.query_route_gate",
                "main_det_head.route_gate_pivot",
                "main_det_head.route_gate_slope_raw",
                "main_det_head.route_gain_raw",
                "main_det_head.route_expert_gain_raw",
                "main_det_head.query_length_prior",
                "main_det_head.duration_safety_prior",
                "main_det_head.query_prior_safety_gate",
                "main_det_head.route_safety_adapter",
                "main_det_head.sample_route_gate",
            )
        )

    @staticmethod
    def _is_gate_parameter(name: str) -> bool:
        """Return whether a parameter belongs to a reliability gate."""
        return any(
            token in name
            for token in (
                "main_det_head.route_gate_logits",
                "main_det_head.query_route_gate",
                "main_det_head.route_gate_pivot",
                "main_det_head.route_gate_slope_raw",
                "main_det_head.sample_route_gate",
            )
        )

    def _set_base_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze the pretrained model while keeping LR-MHC trainable."""
        for name, parameter in self.model.named_parameters():
            if not self._is_lrmhc_parameter(name):
                parameter.requires_grad = trainable

    def _init_metrics(self, metrics_mode: str) -> None:  # noqa: C901
        """
        Initialize metrics for validation and testing.

        Args:
            metrics_mode (str): mode of metrics, define the metric list as well as the main metric
        """
        assert metrics_mode in {"qvhighlights", "youtube", "tvsum", "charades", "tacos"}
        aux_metrics = get_aux_metrics()
        self.aux_train_metrics = aux_metrics.clone(prefix="train/")
        self.aux_valid_metrics = aux_metrics.clone(prefix="val/")
        self.aux_test_metrics = aux_metrics.clone(prefix="test/")

        comb_metrics = get_aux_metrics()
        self.comb_train_metrics = comb_metrics.clone(prefix="train/")
        self.comb_valid_metrics = comb_metrics.clone(prefix="val/")
        self.comb_test_metrics = comb_metrics.clone(prefix="test/")

        if metrics_mode == "qvhighlights":
            metrics = get_metrics()
        elif metrics_mode == "tvsum":
            metrics = get_tvsum_metrics()
        elif metrics_mode in {"charades", "tacos"}:
            metrics = get_charades_metrics()
        else:
            metrics = get_youtube_metrics()
        self.train_metrics = metrics.clone(prefix="train/")
        self.valid_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.train_matching_metrics = MatchingMetric()
        self.valid_matching_metrics = MatchingMetric()
        self.test_matching_metrics = MatchingMetric()

        self.best_metric = 0
        self.submission: List[Dict[str, Any]] = []
        self.aux_submission: List[Dict[str, Any]] = []
        self.comb_submission: List[Dict[str, Any]] = []

        if metrics_mode == "qvhighlights":
            self.main_metric = "val/MR-mAP-Full_Avg"
        elif metrics_mode == "tvsum":
            self.main_metric = "val/HL-mAP-top5"
        elif metrics_mode == "charades":
            self.main_metric = "val/MR-R1-Full_0.5"
        elif metrics_mode == "tacos":
            self.main_metric = "val/MR-R1-Full_0.3"
        else:
            self.main_metric = "val/HL-mAP-Binary"

    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler.

        Returns:
            dict: A dictionary containing the optimizer and the learning rate scheduler.
        """
        # select 3 groups of parameters: anchors, local_sal_params and everything else
        local_sal_params = get_params_by_name(
            self.model,
            include_prefixes=["local_saliency_head"],
            exclude_prefixes=None,
        )

        reference_params = get_params_by_name(
            self.model,
            include_prefixes=["main_det_head.refpoint_embed"],
            exclude_prefixes=None,
        )
        other_params = get_params_by_name(
            self.model,
            include_prefixes=None,
            exclude_prefixes=["main_det_head.refpoint_embed", "local_saliency_head"],
        )

        lr = self.hparams.optimizer.keywords["lr"]  # type: ignore # noqa: WPS111

        # collect boundary loss params (in self.losses, not self.model)
        boundary_params = [p for n, p in self.losses.named_parameters() if p.requires_grad]

        # differential lr: new module params use higher lr if configured
        new_lr = getattr(self.hparams, "new_module_lr", None)
        gaussian_params = [p for n, p in self.model.named_parameters() if "gaussian_sigma" in n]
        router_params = [
            p
            for n, p in self.model.named_parameters()
            if self._is_router_parameter(n) and not self._is_gate_parameter(n)
        ]
        gate_params = [p for n, p in self.model.named_parameters() if self._is_gate_parameter(n)]
        expert_params = [
            p
            for n, p in self.model.named_parameters()
            if self._is_lrmhc_parameter(n) and not self._is_router_parameter(n)
        ]
        new_module_lr = new_lr if new_lr is not None else lr
        router_lr = getattr(self.hparams, "router_lr", None)
        router_lr = router_lr if router_lr is not None else new_module_lr
        gate_lr = getattr(self.hparams, "gate_lr", None)
        gate_lr = gate_lr if gate_lr is not None else router_lr

        param_groups = [
            {"params": local_sal_params, "lr": lr, "name": "local_sal", "weight_decay": 1e-1},
        ]
        if reference_params:
            param_groups.append({"params": reference_params, "lr": lr, "name": "reference"})

        # separate gaussian_sigma from other_params if differential lr
        if new_lr is not None:
            other_params_filtered = [
                p for n, p in self.model.named_parameters()
                if "gaussian_sigma" not in n
                and not self._is_lrmhc_parameter(n)
                and "local_saliency_head" not in n
                and "main_det_head.refpoint_embed" not in n
            ]
            param_groups.append({"params": other_params_filtered, "lr": lr, "name": "other_layers"})
            if gaussian_params:
                param_groups.append({"params": gaussian_params, "lr": new_module_lr, "name": "gaussian"})
            if expert_params:
                expert_group = {
                    "params": expert_params,
                    "lr": new_module_lr,
                    "name": "lrmhc_expert",
                }
                if self.calibration_expert_index is not None or self.center_expert_gradients:
                    # AdamW decay acts on the full shared expert tensor, including
                    # rows whose gradients are masked during one-expert calibration.
                    expert_group["weight_decay"] = 0.0
                param_groups.append(expert_group)
            if router_params:
                param_groups.append({
                    "params": router_params,
                    "lr": router_lr,
                    "name": "lrmhc_router",
                })
            if gate_params:
                param_groups.append({
                    "params": gate_params,
                    "lr": gate_lr,
                    "name": "lrmhc_gate",
                })
            if boundary_params:
                param_groups.append({"params": boundary_params, "lr": new_module_lr, "name": "boundary"})
        else:
            param_groups.append({"params": other_params, "lr": lr, "name": "other_layers"})
            if boundary_params:
                param_groups.append({"params": boundary_params, "lr": lr, "name": "boundary"})

        if reference_params:
            optimizer = self.hparams.optimizer(params=param_groups)  # type: ignore
        else:
            optimizer = self.hparams.optimizer(params=param_groups)  # type: ignore

        if self.scheduler is not None:  # type: ignore
            scheduler = self.scheduler(optimizer=optimizer)  # type: ignore

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # pylint: disable=attribute-defined-outside-init
    def save_submission(  # noqa: WPS234
        self,
        submission_batch: Optional[List[Dict[str, Any]]],
        prefix: str,
        metric: Optional[float] = None,
    ) -> None:
        """Accumulate and save submission.

        Args:
            submission_batch (Optional[List[Dict[str, Any]]]): postprocessed predictions.
            prefix (str): Prefix indicating the phase.
            metric (Optional[float]): current model score.
        """
        if submission_batch is not None:
            self.submission.extend(submission_batch)
            return

        for result in self.submission:
            for key, value in result.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value.tolist()
        submission_name = os.path.join(self.trainer.log_dir, f"submission_{prefix}_last.jsonl")  # type: ignore
        save_jsonl(self.submission, submission_name)
        if (metric is not None) and (metric > self.best_metric):
            save_jsonl(self.submission, submission_name.replace("_last", "_best"))
            self.best_metric = metric  # type: ignore
        self.submission = []

    def save_auxiliary_submissions(self, prefix: str) -> None:
        """Persist auxiliary and combined candidates for offline diagnostics."""
        for name, rows in (
            ("aux", self.aux_submission),
            ("comb", self.comb_submission),
        ):
            for result in rows:
                for key, value in result.items():
                    if isinstance(value, torch.Tensor):
                        result[key] = value.tolist()
            path = os.path.join(self.trainer.log_dir, f"submission_{prefix}_{name}.jsonl")
            save_jsonl(rows, path)
        self.aux_submission = []
        self.comb_submission = []

    def _get_ref_points(self) -> Optional[Tensor]:
        """
        Get reference points of the model.

        Returns:
            Optional[Tensor]: reference points
        """
        if self.model.main_det_head.use_rpn:
            return None
        ref_points = self.model.main_det_head.refpoint_embed.get_reference_points()  # type: ignore
        return span_cxw_to_xx(torch.sigmoid(ref_points))

    def _get_combiner_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Use the stable uniform branch for auxiliary fusion in top-1 safety mode."""
        if not getattr(self.model.main_det_head, "preserve_uniform_top1", False):
            return outputs
        required = {"pred_uniform_logits", "pred_uniform_quality_scores"}
        if not required.issubset(outputs):
            return outputs
        combiner_outputs = dict(outputs)
        combiner_outputs["pred_logits"] = outputs["pred_uniform_logits"]
        combiner_outputs["pred_quality_scores"] = outputs[
            "pred_uniform_quality_scores"
        ]
        return combiner_outputs

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Track whether routing receives meaningful gradients independently of experts."""
        del optimizer
        router_norm_sq = None
        gate_norm_sq = None
        expert_norm_sq = None
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None or not self._is_lrmhc_parameter(name):
                continue
            norm_sq = parameter.grad.detach().float().square().sum()
            if self._is_gate_parameter(name):
                gate_norm_sq = norm_sq if gate_norm_sq is None else gate_norm_sq + norm_sq
            elif self._is_router_parameter(name):
                router_norm_sq = norm_sq if router_norm_sq is None else router_norm_sq + norm_sq
            else:
                expert_norm_sq = norm_sq if expert_norm_sq is None else expert_norm_sq + norm_sq
        for metric_name, norm_sq in (
            ("train/router_grad_norm", router_norm_sq),
            ("train/gate_grad_norm", gate_norm_sq),
            ("train/expert_grad_norm", expert_norm_sq),
        ):
            if norm_sq is not None:
                self.log(metric_name, norm_sq.sqrt(), on_step=True, on_epoch=True)

    def _process_batch(  # noqa: WPS210,C901,WPS213
        self,
        data: Tuple[MetaTypes, Dict[str, Any]],
        prefix: str,
    ) -> Optional[Tensor]:
        """
        Process a batch of embeddings for either training, validation, or testing.

        Args:
            data (Tuple[MetaTypes, Dict[str, Any]]): A tuple containing seq and ground truth labels.
            prefix (str): Prefix indicating the phase.

        Returns:
            Optional[torch.Tensor]: Computed total loss for train step.
        """
        meta, batch = data
        batch, targets = move_inputs_to_device(batch, self.device, non_blocking=True)
        outputs = self.model(targets=targets, meta=meta, **batch)
        route_weights = getattr(self.model.main_det_head, "last_route_weights", None)
        if route_weights is not None:
            usage = route_weights.mean(dim=(0, 1))
            entropy = -(route_weights * route_weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
            for expert_idx, expert_usage in enumerate(usage):
                self.log(
                    f"{prefix}/route_usage_{expert_idx}",
                    expert_usage,
                    on_step=False,
                    on_epoch=True,
                    batch_size=self.model.batch_size,
                )
            self.log(
                f"{prefix}/route_entropy",
                entropy,
                on_step=False,
                on_epoch=True,
                batch_size=self.model.batch_size,
            )
            route_gate = getattr(self.model.main_det_head, "last_route_gate", None)
            if route_gate is not None:
                self.log(
                    f"{prefix}/route_gate",
                    route_gate.mean(),
                    on_step=False,
                    on_epoch=True,
                    batch_size=self.model.batch_size,
                )
        matching = self.losses.compute_matches(outputs, targets, self._get_ref_points())  # type: ignore
        route_gate_scores = outputs.get("route_gate_scores")
        if route_gate_scores is not None:
            positive_gates = []
            negative_gates = []
            hard_negative_gates = []
            uniform_class = outputs.get("pred_uniform_logits")
            uniform_quality = outputs.get("pred_uniform_quality_scores")
            uniform_scores = None
            if uniform_class is not None and uniform_quality is not None:
                uniform_scores = 0.5 * (
                    func.logsigmoid(uniform_class.squeeze(-1))
                    + func.logsigmoid(uniform_quality.squeeze(-1))
                )
            for sample_idx, (src_idx, _) in enumerate(matching["positive"]["indices"]):
                src_idx = src_idx.to(route_gate_scores.device)
                positive_gates.append(route_gate_scores[sample_idx, src_idx])
                negative_mask = torch.ones(
                    route_gate_scores.shape[1],
                    dtype=torch.bool,
                    device=route_gate_scores.device,
                )
                negative_mask[src_idx] = False
                negative_gates.append(route_gate_scores[sample_idx, negative_mask])
                if uniform_scores is not None:
                    negative_idx = torch.nonzero(negative_mask, as_tuple=False).squeeze(-1)
                    hard_count = min(self.losses.expert_hard_negatives, negative_idx.numel())
                    if hard_count > 0:
                        hard_order = torch.topk(
                            uniform_scores[sample_idx, negative_idx],
                            hard_count,
                        ).indices
                        hard_negative_gates.append(
                            route_gate_scores[sample_idx, negative_idx[hard_order]],
                        )
            if positive_gates:
                self.log(
                    f"{prefix}/route_gate_positive",
                    torch.cat(positive_gates).mean(),
                    on_step=False,
                    on_epoch=True,
                    batch_size=self.model.batch_size,
                )
            if negative_gates:
                self.log(
                    f"{prefix}/route_gate_negative",
                    torch.cat(negative_gates).mean(),
                    on_step=False,
                    on_epoch=True,
                    batch_size=self.model.batch_size,
                )
            if hard_negative_gates:
                self.log(
                    f"{prefix}/route_gate_hard_negative",
                    torch.cat(hard_negative_gates).mean(),
                    on_step=False,
                    on_epoch=True,
                    batch_size=self.model.batch_size,
                )
        # Add data to model in order to use in MatcherCallback
        self._current_meta = meta
        self._current_targets = targets
        self._current_outputs = outputs
        self._matching = matching
        # compute loss for all phases (train/val/test) for logging & comparison
        losses = self.losses(outputs, targets, meta, matching)
        total_loss = torch.Tensor([0]).to(self.device)
        weight_dict = self.losses.weight_dict
        for loss_name, loss_value in losses.items():
            self.log(
                f"{prefix}/{loss_name}",
                loss_value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.model.batch_size,
            )
            total_loss = total_loss + loss_value * weight_dict.get(fix_loss_name(loss_name), 0)

        self.log(
            f"{prefix}/total_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.model.batch_size,
        )

        if "train" in prefix:
            self.train_matching_metrics.update(matching)

            if self.current_epoch % self.check_train_every_n_epoch == 0 and self.current_epoch != 0:
                with torch.no_grad():
                    aux_outputs, detr_outputs = self.preparator(meta, batch, outputs)  # type: ignore
                    comb_outputs = self.combiner(
                        meta,
                        batch,
                        self._get_combiner_outputs(outputs),
                    )
                    postprocessed_detr_outputs = self.postprocessor(detr_outputs)  # type: ignore
                    postprocessed_aux_outputs = self.postprocessor(aux_outputs, aux_head=True)  # type: ignore
                    postprocessed_comb_outputs = self.postprocessor(comb_outputs, aux_head=True)
                    # plug ################
                    postprocessed_comb_outputs = _fix_output(postprocessed_comb_outputs)
                    postprocessed_aux_outputs = _fix_output(postprocessed_aux_outputs)
                    postprocessed_detr_outputs = _fix_output(postprocessed_detr_outputs)
                    # plug ################

                    self.train_metrics(submissions=postprocessed_detr_outputs, targets=meta)
                    self.aux_train_metrics(submissions=postprocessed_aux_outputs, targets=meta)
                    self.comb_train_metrics(submissions=postprocessed_comb_outputs, targets=meta)
            return total_loss

        with torch.autocast(dtype=torch.float32, device_type="cuda"):  # type: ignore
            aux_outputs, detr_outputs = self.preparator(meta, batch, outputs)  # type: ignore
            comb_outputs = self.combiner(
                meta,
                batch,
                self._get_combiner_outputs(outputs),
            )
            postprocessed_detr_outputs = self.postprocessor(detr_outputs)  # type: ignore
            postprocessed_aux_outputs = self.postprocessor(aux_outputs, aux_head=True)  # type: ignore
            postprocessed_comb_outputs = self.postprocessor(comb_outputs, aux_head=True)

            # plug ################
            postprocessed_comb_outputs = _fix_output(postprocessed_comb_outputs)
            postprocessed_aux_outputs = _fix_output(postprocessed_aux_outputs)
            postprocessed_detr_outputs = _fix_output(postprocessed_detr_outputs)
            # plug ################

            if "val" in prefix:
                self.valid_metrics(submissions=postprocessed_detr_outputs, targets=meta)
                self.aux_valid_metrics(submissions=postprocessed_aux_outputs, targets=meta)
                self.comb_valid_metrics(submissions=postprocessed_comb_outputs, targets=meta)
                self.valid_matching_metrics.update(matching)
            else:
                self.test_metrics(submissions=postprocessed_detr_outputs, targets=meta)
                self.aux_test_metrics(submissions=postprocessed_aux_outputs, targets=meta)
                self.comb_test_metrics(submissions=postprocessed_comb_outputs, targets=meta)
                self.test_matching_metrics.update(matching)
            self.aux_submission.extend(postprocessed_aux_outputs)
            self.comb_submission.extend(postprocessed_comb_outputs)
            self.save_submission(postprocessed_detr_outputs, prefix)
        return None

    def training_step(self, batch, batch_idx: int) -> Tensor:
        """
        Process a batch during training.

        Args:
            batch (tuple): A tuple containing seqs and labels.
            batch_idx (int): Index of the current batch.

        Returns:
            Tensor: Computed Loss

        """
        return self._process_batch(batch, "train")  # type: ignore

    def on_train_batch_start(self, batch, batch_idx: int) -> None:
        """Anneal the inverse temperature near the end of route training."""
        if self.route_beta_start is None:
            return
        total_steps = max(int(self.trainer.estimated_stepping_batches), 1)
        progress = min(self.global_step / max(total_steps - 1, 1), 1.0)
        anneal_start = 1.0 - self.route_beta_anneal_fraction
        ratio = max(progress - anneal_start, 0.0) / self.route_beta_anneal_fraction
        beta = self.route_beta_start * (self.route_beta_end / self.route_beta_start) ** ratio
        self.model.main_det_head.route_beta = float(beta)
        self.log("train/route_beta", beta, on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_idx: int) -> None:
        """
        Process a batch during validation.

        Args:
            batch (tuple): A tuple containing seqs and labels.
            batch_idx (int): Index of the current batch.
        """
        self._process_batch(batch, "val")

    def test_step(self, batch, batch_idx: int) -> None:
        """
        Process a batch during testing.

        Args:
            batch (tuple): A tuple containing seqs and labels.
            batch_idx (int): Index of the current batch.
        """
        self._process_batch(batch, "test")

    def on_validation_epoch_start(self) -> None:
        """Reset the validation metrics at the start of a validation epoch."""
        if self.route_beta_end is not None:
            self.model.main_det_head.route_beta = float(self.route_beta_end)
        self.valid_metrics.reset()
        self.aux_valid_metrics.reset()
        self.comb_valid_metrics.reset()
        self.valid_matching_metrics.reset()

    def on_validation_epoch_end(self):  # noqa: C901,WPS231
        """Log the computed validation metrics at the end of a validation epoch."""
        detr_val_metrics = self.valid_metrics.compute()
        aux_val_metrics = self.aux_valid_metrics.compute()
        comb_val_metrics = self.comb_valid_metrics.compute()
        valid_matching_metrics = self.valid_matching_metrics.compute()

        # log detr metrics
        if self.metrics_mode in {"charades", "tacos"}:
            metric = float(comb_val_metrics[self.main_metric])
        else:
            metric = float(detr_val_metrics[self.main_metric])
        self.save_submission(submission_batch=None, prefix="val", metric=metric)
        self.save_auxiliary_submissions(prefix="val")
        for name, value in detr_val_metrics.items():  # noqa: WPS204
            if "HL" in name:
                self.log(name, value, on_epoch=True, sync_dist=True, prog_bar=True)
            elif name[4:] in MAIN_METRICS_DICT:
                self.log(name, value, on_epoch=True, sync_dist=True, prog_bar=True)
            else:
                self.log(name, value, on_epoch=True, sync_dist=True)

        # log aux metrics
        for name, value in aux_val_metrics.items():
            if name[4:] in MAIN_METRICS_DICT:
                self.log(f"{name}-AUX", value, on_epoch=True, sync_dist=True)

        # log comb metrics
        for name, value in comb_val_metrics.items():
            if name[4:] in MAIN_METRICS_DICT:
                self.log(f"{name}-COMB", value, on_epoch=True, sync_dist=True)

        # log matching
        for name, value in valid_matching_metrics.items():
            self.log(f"val/{name}", value, on_epoch=True, sync_dist=True)

    def on_test_epoch_start(self) -> None:
        """Reset the validation metrics at the start of a test epoch."""
        self.test_metrics.reset()
        self.aux_test_metrics.reset()
        self.comb_test_metrics.reset()
        self.test_matching_metrics.reset()

    def on_test_epoch_end(self) -> None:
        """Log the computed validation metrics at the end of a test epoch."""
        detr_test_metrics = self.test_metrics.compute()
        aux_test_metrics = self.aux_test_metrics.compute()
        comb_test_metrics = self.comb_test_metrics.compute()
        matching_test_metrics = self.test_matching_metrics.compute()
        # log detr metrics
        self.log_dict(detr_test_metrics, on_epoch=True, sync_dist=True)
        for name, value in matching_test_metrics.items():
            self.log(f"test/{name}", value, on_epoch=True, sync_dist=True)

        # log aux metrics
        for name, value in aux_test_metrics.items():
            if name[5:] in MAIN_METRICS_DICT:
                self.log(f"{name}-AUX", value, on_epoch=True, sync_dist=True)

        # log comb metrics
        for name, value in comb_test_metrics.items():
            if name[5:] in MAIN_METRICS_DICT:
                self.log(f"{name}-COMB", value, on_epoch=True, sync_dist=True)

    def on_train_epoch_start(self) -> None:
        """Reset matching on train epoch end."""
        if self.freeze_base_epochs > 0 and self.current_epoch == self.freeze_base_epochs:
            if not self.keep_base_frozen_after_warmup:
                self._set_base_trainable(True)
            for param_group in self.trainer.optimizers[0].param_groups:
                is_new = param_group.get("name") in {
                    "lrmhc_expert", "lrmhc_router", "lrmhc_gate", "gaussian", "boundary",
                }
                target_lr = self.joint_new_module_lr if is_new else self.joint_base_lr
                if target_lr is not None:
                    param_group["lr"] = target_lr
        self.train_matching_metrics.reset()
        self.train_metrics.reset()
        self.aux_train_metrics.reset()
        self.comb_train_metrics.reset()

    def on_train_epoch_end(self) -> None:  # noqa: C901
        """Log metrics on train epoch end."""
        train_matching_metrics = self.train_matching_metrics.compute()

        # matching metrics
        for name, value in train_matching_metrics.items():
            self.log(f"train/{name}", value, on_epoch=True, sync_dist=True)

        if self.current_epoch % self.check_train_every_n_epoch == 0 and self.current_epoch != 0:
            detr_train_metrics = self.train_metrics.compute()
            aux_train_metrics = self.aux_train_metrics.compute()
            comb_train_metrics = self.comb_train_metrics.compute()

            # log detr metrics
            for name, value in detr_train_metrics.items():
                if name[6:] in MAIN_METRICS_DICT:
                    self.log(f"{name}", value, on_epoch=True, sync_dist=True)

            # log aux metrics
            for name, value in aux_train_metrics.items():
                if name[6:] in MAIN_METRICS_DICT:
                    self.log(f"{name}-AUX", value, on_epoch=True, sync_dist=True)
            # log comb metrics
            for name, value in comb_train_metrics.items():
                if name[6:] in MAIN_METRICS_DICT:
                    self.log(f"{name}-COMB", value, on_epoch=True, sync_dist=True)

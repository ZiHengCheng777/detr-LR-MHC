# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from qd_detr.span_utils import generalized_temporal_iou, span_cxw_to_xx

from qd_detr.matcher import build_matcher
from qd_detr.transformer import build_transformer
from qd_detr.position_encoding import build_position_encoding
from qd_detr.misc import accuracy
from qd_detr.adaptive_soft_lrmhc import (
    AdaptiveSoftLRMHC,
    balanced_expert_rank_loss,
    quality_aware_routed_rank_loss,
    query_gate_oracle_loss,
    routed_residual_rank_loss,
)
import numpy as np
def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

class QDDETR(nn.Module):
    """ QD DETR. """

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 num_queries, input_dropout, aux_loss=False,
                 contrastive_align_loss=False, contrastive_hdim=64,
                 max_v_l=75, span_loss_type="l1", use_txt_pos=False, n_input_proj=2, aud_dim=0,
                 args=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         QD-DETR can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            contrastive_align_loss: If true, perform span - tokens contrastive learning
            contrastive_hdim: dimension used for projecting the embeddings before computing contrastive loss
            max_v_l: int, maximum #clips in videos
            span_loss_type: str, one of [l1, ce]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
            # foreground_thd: float, intersection over prediction >= foreground_thd: labeled as foreground
            # background_thd: float, intersection over prediction <= background_thd: labeled background
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        span_pred_dim = 2 if span_loss_type == "l1" else max_v_l * 2
        self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground
        self.use_lrmhc = bool(getattr(args, "use_lrmhc", False))
        self.lrmhc_isolate_base_grad = bool(
            getattr(args, "lrmhc_isolate_base_grad", False)
        )
        self.clip_length = float(getattr(args, "clip_length", 2.0))
        if self.use_lrmhc:
            # Keep the base model's data/dropout RNG trajectory identical to A0.
            with torch.random.fork_rng(devices=[]):
                self.lrmhc = AdaptiveSoftLRMHC(
                    hidden_dim=hidden_dim,
                    foreground_index=0,
                    uniform_routing=getattr(args, "lrmhc_uniform", False),
                    content_routing=getattr(args, "lrmhc_content_routing", False),
                    hard_routing=getattr(args, "lrmhc_hard_routing", False),
                    learn_prototypes=not getattr(args, "lrmhc_fixed_prototypes", False),
                    use_gate=not getattr(args, "lrmhc_no_gate", False),
                    detach_width=not getattr(args, "lrmhc_width_no_detach", False),
                    route_beta=getattr(args, "lrmhc_route_beta", 1.0),
                    prototype_init=getattr(
                        args, "lrmhc_prototypes", (-3.1781, -2.1253, -1.3863)
                    ),
                    gate_init=getattr(args, "lrmhc_gate_init", 0.5),
                    router_hidden_dim=getattr(args, "lrmhc_router_hidden_dim", 64),
                    use_query_gate=getattr(args, "lrmhc_query_gate", False),
                    query_gate_hidden_dim=getattr(
                        args, "lrmhc_query_gate_hidden_dim", 32
                    ),
                    query_gate_init=getattr(args, "lrmhc_query_gate_init", 0.2),
                    use_route_confidence_gate=getattr(
                        args, "lrmhc_route_confidence_gate", False
                    ),
                    output_scale=getattr(args, "lrmhc_output_scale", 1.0),
                    use_span_refine=getattr(args, "lrmhc_span_refine", False),
                    span_output_scale=getattr(args, "lrmhc_span_output_scale", 0.1),
                    span_expert_scale_init=getattr(
                        args, "lrmhc_span_expert_scales", (1.0, 1.0, 1.0)
                    ),
                    use_span_short_guard=getattr(
                        args, "lrmhc_span_short_guard", False
                    ),
                    span_short_guard_init=getattr(
                        args, "lrmhc_span_short_guard_init", 0.9
                    ),
                    span_short_guard_boundary=getattr(
                        args, "lrmhc_span_short_guard_boundary", 10.0
                    ),
                    span_short_guard_softness=getattr(
                        args, "lrmhc_span_short_guard_softness", 2.0
                    ),
                    use_span_query_gate=getattr(
                        args, "lrmhc_span_query_gate", False
                    ),
                    span_query_gate_hidden_dim=getattr(
                        args, "lrmhc_span_query_gate_hidden_dim", 32
                    ),
                    span_query_gate_init=getattr(
                        args, "lrmhc_span_query_gate_init", 0.1
                    ),
                    isolate_base_grad=self.lrmhc_isolate_base_grad,
                    router_logit_limit=getattr(
                        args, "lrmhc_router_logit_limit", 0.0
                    ),
                    residual_limit=getattr(args, "lrmhc_residual_limit", 0.0),
                )
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        # self.foreground_thd = foreground_thd
        # self.background_thd = background_thd
        self.query_embed = nn.Embedding(num_queries, 2)
        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim + aud_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.contrastive_align_loss = contrastive_align_loss
        if contrastive_align_loss:
            self.contrastive_align_projection_query = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_txt = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_vid = nn.Linear(hidden_dim, contrastive_hdim)

        self.saliency_proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.saliency_proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.aux_loss = aux_loss

        self.hidden_dim = hidden_dim
        self.global_rep_token = torch.nn.Parameter(torch.randn(hidden_dim))
        self.global_rep_pos = torch.nn.Parameter(torch.randn(hidden_dim))

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, src_aud=None, src_aud_mask=None):
        """The forward expects two tensors:
               - src_txt: [batch_size, L_txt, D_txt]
               - src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer
               - src_vid: [batch_size, L_vid, D_vid]
               - src_vid_mask: [batch_size, L_vid], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer

            It returns a dict with the following elements:
               - "pred_spans": The normalized boxes coordinates for all queries, represented as
                               (center_x, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if src_aud is not None:
            src_vid = torch.cat([src_vid, src_aud], dim=2)
            
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)
        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)
        # TODO should we remove or use different positional embeddings to the src_txt?
        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        # pos_txt = torch.zeros_like(src_txt)
        # pad zeros for txt positions
        pos = torch.cat([pos_vid, pos_txt], dim=1)
        # (#layers, bsz, #queries, d), (bsz, L_vid+L_txt, d)

        # for global token
        mask_ = torch.tensor([[True]]).to(mask.device).repeat(mask.shape[0], 1)
        mask = torch.cat([mask_, mask], dim=1)
        src_ = self.global_rep_token.reshape([1, 1, self.hidden_dim]).repeat(src.shape[0], 1, 1)
        src = torch.cat([src_, src], dim=1)
        pos_ = self.global_rep_pos.reshape([1, 1, self.hidden_dim]).repeat(pos.shape[0], 1, 1)
        pos = torch.cat([pos_, pos], dim=1)

        video_length = src_vid.shape[1]
        
        hs, reference, memory, memory_global = self.transformer(src, ~mask, self.query_embed.weight, pos, video_length=video_length)
        outputs_class = self.class_embed(hs)  # (#layers, batch_size, #queries, #classes)
        reference_before_sigmoid = inverse_sigmoid(reference)
        tmp = self.span_embed(hs)
        outputs_coord = tmp + reference_before_sigmoid
        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()
        route_diagnostics = None
        if self.use_lrmhc:
            final_width = outputs_coord[-1:, ..., 1].expand_as(outputs_coord[..., 1])
            video_duration = (
                src_vid_mask.sum(dim=-1).to(final_width) * self.clip_length
            )
            predicted_duration = final_width * video_duration.view(1, -1, 1)
            outputs_class, outputs_coord, route_diagnostics = self.lrmhc(
                hs,
                outputs_class,
                final_width,
                outputs_coord,
                predicted_duration=predicted_duration,
            )
        out = {'pred_logits': outputs_class[-1], 'pred_spans': outputs_coord[-1]}
        if route_diagnostics is not None:
            out.update({key: value[-1] for key, value in route_diagnostics.items()})
            out['routed_logits'] = outputs_class[-1]
            out['routed_spans'] = outputs_coord[-1]

        txt_mem = memory[:, src_vid.shape[1]:]  # (bsz, L_txt, d)
        vid_mem = memory[:, :src_vid.shape[1]]  # (bsz, L_vid, d)
        if self.contrastive_align_loss:
            proj_queries = F.normalize(self.contrastive_align_projection_query(hs), p=2, dim=-1)
            proj_txt_mem = F.normalize(self.contrastive_align_projection_txt(txt_mem), p=2, dim=-1)
            proj_vid_mem = F.normalize(self.contrastive_align_projection_vid(vid_mem), p=2, dim=-1)
            out.update(dict(
                proj_queries=proj_queries[-1],
                proj_txt_mem=proj_txt_mem,
                proj_vid_mem=proj_vid_mem
            ))
            
            
        # !!! this is code for test
        if src_txt.shape[1] == 0:
            print("There is zero text query. You should change codes properly")
            exit(-1)

        ### Neg Pairs ###
        src_txt_neg = torch.cat([src_txt[1:], src_txt[0:1]], dim=0)
        src_txt_mask_neg = torch.cat([src_txt_mask[1:], src_txt_mask[0:1]], dim=0)
        src_neg = torch.cat([src_vid, src_txt_neg], dim=1)
        mask_neg = torch.cat([src_vid_mask, src_txt_mask_neg], dim=1).bool()

        mask_neg = torch.cat([mask_, mask_neg], dim=1)
        src_neg = torch.cat([src_, src_neg], dim=1)
        pos_neg = pos.clone()  # since it does not use actual content

        _, _, memory_neg, memory_global_neg = self.transformer(src_neg, ~mask_neg, self.query_embed.weight, pos_neg, video_length=video_length)
        vid_mem_neg = memory_neg[:, :src_vid.shape[1]]


        out["saliency_scores"] = (torch.sum(self.saliency_proj1(vid_mem) * self.saliency_proj2(memory_global).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim))

        out["saliency_scores_neg"] = (torch.sum(self.saliency_proj1(vid_mem_neg) * self.saliency_proj2(memory_global_neg).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim))

        # print(src_vid_mask.shape, src_vid.shape, vid_mem_neg.shape, vid_mem.shape)
        out["video_mask"] = src_vid_mask
        if self.aux_loss:
            # assert proj_queries and proj_txt_mem
            if self.use_lrmhc and self.lrmhc_isolate_base_grad:
                out['aux_outputs'] = [
                    {'pred_logits': a, 'pred_spans': b}
                    for a, b in zip(
                        route_diagnostics['base_logits'][:-1],
                        route_diagnostics['base_spans'][:-1],
                    )
                ]
            else:
                out['aux_outputs'] = [
                    {'pred_logits': a, 'pred_spans': b}
                    for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
                ]
            if self.contrastive_align_loss:
                assert proj_queries is not None
                for idx, d in enumerate(proj_queries[:-1]):
                    out['aux_outputs'][idx].update(dict(proj_queries=d, proj_txt_mem=proj_txt_mem))
        return out

    # @torch.jit.unused
    # def _set_aux_loss(self, outputs_class, outputs_coord):
    #     # this is a workaround to make torchscript happy, as torchscript
    #     # doesn't support dictionary with non-homogeneous values, such
    #     # as a dict having both a Tensor and a list.
    #     return [{'pred_logits': a, 'pred_spans': b}
    #             for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, eos_coef, losses, temperature, span_loss_type, max_v_l,
                 saliency_margin=1, use_matcher=True, clip_length=2.0,
                 group_robust_temperature=0.0,
                 span_improvement_temperature=0.05,
                 span_query_gate_temperature=0.05,
                 duration_boundaries=(10.0, 30.0), duration_softness=2.0,
                 prototype_responsibilities=False,
                 isolate_base_grad=False):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            temperature: float, temperature for NCE loss
            span_loss_type: str, [l1, ce]
            max_v_l: int,
            saliency_margin: float
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.temperature = temperature
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.saliency_margin = saliency_margin
        self.clip_length = float(clip_length)
        self.group_robust_temperature = float(group_robust_temperature)
        self.span_improvement_temperature = float(span_improvement_temperature)
        self.span_query_gate_temperature = float(span_query_gate_temperature)
        self.duration_boundaries = tuple(float(value) for value in duration_boundaries)
        self.duration_softness = float(duration_softness)
        self.prototype_responsibilities = bool(prototype_responsibilities)
        self.isolate_base_grad = bool(isolate_base_grad)
        if self.span_improvement_temperature <= 0:
            raise ValueError("span_improvement_temperature must be positive")
        if self.span_query_gate_temperature <= 0:
            raise ValueError("span_query_gate_temperature must be positive")
        if len(self.duration_boundaries) != 2 or not (
            0 < self.duration_boundaries[0] < self.duration_boundaries[1]
        ):
            raise ValueError("duration_boundaries must contain two increasing positive values")
        if self.duration_softness <= 0:
            raise ValueError("duration_softness must be positive")
        self.register_buffer('duration_group_loss_ema', torch.zeros(3))
        self.duration_group_ema_initialized = False

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)
        
        # for tvsum,
        self.use_matcher = use_matcher

    def _duration_responsibilities(self, outputs, idx, tgt_spans):
        """Return smooth memberships from prototypes or absolute durations."""
        if self.prototype_responsibilities:
            prototypes = outputs['route_prototypes'][idx].detach()
            target_width_logit = torch.logit(
                tgt_spans[:, 1].clamp(1e-4, 1.0 - 1e-4)
            ).unsqueeze(-1)
            return (
                -2.0 * (target_width_logit - prototypes).square()
            ).softmax(dim=-1)

        video_lengths = outputs['video_mask'].sum(dim=-1).to(dtype=tgt_spans.dtype)
        target_seconds = tgt_spans[:, 1] * video_lengths[idx[0]] * self.clip_length
        short_boundary, long_boundary = self.duration_boundaries
        short_weight = torch.sigmoid(
            (short_boundary - target_seconds) / self.duration_softness
        )
        long_weight = torch.sigmoid(
            (target_seconds - long_boundary) / self.duration_softness
        )
        middle_weight = (1.0 - short_weight) * (1.0 - long_weight)
        weights = torch.stack((short_weight, middle_weight, long_weight), dim=-1)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def loss_spans(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center_x, w), normalized by the image size.
        """
        assert 'pred_spans' in outputs
        targets = targets["span_labels"]
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx]  # (#spans, max_v_l * 2)
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, 2)
        if self.span_loss_type == "l1":
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        else:  # ce
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_v_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction='none')

            # giou
            # src_span_indices = src_spans.max(1)[1]  # (#spans, 2)
            # src_span_indices[:, 1] += 1  # ed non-inclusive [st, ed)
            #
            # tgt_span_indices = tgt_spans
            # tgt_span_indices[:, 1] += 1
            # loss_giou = 1 - torch.diag(generalized_temporal_iou(src_span_indices, tgt_span_indices))
            loss_giou = loss_span.new_zeros([1])

        losses = {}
        losses['loss_span'] = loss_span.mean()
        losses['loss_giou'] = loss_giou.mean()
        if (
            'loss_span_improvement' in self.weight_dict
            and 'base_spans' in outputs
            and 'video_mask' in outputs
            and self.span_loss_type == "l1"
            and src_spans.shape[0] > 0
        ):
            routed_spans = outputs.get('routed_spans', outputs['pred_spans'])
            corrected_spans = routed_spans[idx]
            corrected_l1 = F.l1_loss(
                corrected_spans, tgt_spans, reduction='none'
            )
            corrected_giou = 1 - torch.diag(generalized_temporal_iou(
                span_cxw_to_xx(corrected_spans), span_cxw_to_xx(tgt_spans)
            ))
            base_spans = outputs['base_spans'][idx].detach()
            base_l1 = F.l1_loss(base_spans, tgt_spans, reduction='none').mean(dim=-1)
            base_giou = 1 - torch.diag(generalized_temporal_iou(
                span_cxw_to_xx(base_spans), span_cxw_to_xx(tgt_spans)
            ))
            corrected_cost = corrected_l1.mean(dim=-1) + corrected_giou
            base_cost = base_l1 + base_giou
            improvement_gap = corrected_cost - base_cost
            safe_loss = self.span_improvement_temperature * F.softplus(
                improvement_gap / self.span_improvement_temperature
            )

            responsibilities = self._duration_responsibilities(
                outputs, idx, tgt_spans
            )
            group_safe_loss = (
                (safe_loss.unsqueeze(-1) * responsibilities).sum(dim=0)
                / responsibilities.sum(dim=0).clamp_min(1.0)
            )
            losses['loss_span_improvement'] = group_safe_loss.mean()
        if (
            'loss_span_query_gate' in self.weight_dict
            and 'ungated_routed_spans' in outputs
            and 'span_query_gate' in outputs
            and 'base_spans' in outputs
            and 'video_mask' in outputs
            and self.span_loss_type == "l1"
            and src_spans.shape[0] > 0
        ):
            candidate_spans = outputs['ungated_routed_spans'][idx]
            base_spans = outputs['base_spans'][idx].detach()
            candidate_cost = F.l1_loss(
                candidate_spans, tgt_spans, reduction='none'
            ).mean(dim=-1)
            candidate_cost = candidate_cost + 1 - torch.diag(
                generalized_temporal_iou(
                    span_cxw_to_xx(candidate_spans),
                    span_cxw_to_xx(tgt_spans),
                )
            )
            base_cost = F.l1_loss(
                base_spans, tgt_spans, reduction='none'
            ).mean(dim=-1)
            base_cost = base_cost + 1 - torch.diag(
                generalized_temporal_iou(
                    span_cxw_to_xx(base_spans),
                    span_cxw_to_xx(tgt_spans),
                )
            )
            oracle = torch.sigmoid(
                (base_cost - candidate_cost.detach())
                / self.span_query_gate_temperature
            )
            gate = outputs['span_query_gate'][idx].squeeze(-1)
            gate_bce = F.binary_cross_entropy(
                gate.clamp(1e-6, 1.0 - 1e-6), oracle, reduction='none'
            )
            responsibilities = self._duration_responsibilities(
                outputs, idx, tgt_spans
            )
            group_gate_loss = (
                (gate_bce.unsqueeze(-1) * responsibilities).sum(dim=0)
                / responsibilities.sum(dim=0).clamp_min(1.0)
            )
            losses['loss_span_query_gate'] = group_gate_loss.mean()
        if (
            'loss_duration_balance' in self.weight_dict
            and 'video_mask' in outputs
            and self.span_loss_type == "l1"
            and src_spans.shape[0] > 0
        ):
            responsibilities = self._duration_responsibilities(
                outputs, idx, tgt_spans
            )
            routed_spans = outputs.get('routed_spans', outputs['pred_spans'])
            corrected_spans = routed_spans[idx]
            corrected_l1 = F.l1_loss(
                corrected_spans, tgt_spans, reduction='none'
            ).mean(dim=-1)
            corrected_giou = 1 - torch.diag(generalized_temporal_iou(
                span_cxw_to_xx(corrected_spans), span_cxw_to_xx(tgt_spans)
            ))
            per_match_loss = corrected_l1 + corrected_giou
            expert_loss = (
                (per_match_loss.unsqueeze(-1) * responsibilities).sum(dim=0)
                / responsibilities.sum(dim=0).clamp_min(1.0)
            )
            losses['loss_duration_balance'] = expert_loss.mean()
        return losses

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # TODO add foreground and background classifier.  use all non-matched as background.
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (batch_size, #queries, #classes=2)
        # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #objects in batch
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                    dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries)
        target_classes[idx] = self.foreground_label

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")
        losses = {'loss_label': loss_ce.mean()}

        if (
            'loss_route' in self.weight_dict
            and 'route_weights' in outputs
            and idx[0].numel() > 0
        ):
            route_weights = outputs['route_weights'][idx].clamp_min(1e-8)
            span_targets = targets['span_labels']
            target_widths = torch.cat([
                target['spans'][target_idx, 1]
                for target, (_, target_idx) in zip(span_targets, indices)
            ]).clamp(1e-4, 1.0 - 1e-4)
            semantic_centers = src_logits.new_tensor((-3.1781, -2.1253, -1.3863))
            target_logits = -4.0 * (
                torch.logit(target_widths).unsqueeze(-1) - semantic_centers
            ).square()
            target_routes = target_logits.softmax(dim=-1)
            losses['loss_route'] = -(
                target_routes * route_weights.log()
            ).sum(dim=-1).mean()

        if (
            'loss_expert_rank' in self.weight_dict
            and 'expert_residuals' in outputs
            and idx[0].numel() > 0
        ):
            base_logits = outputs.get('base_logits', src_logits)
            base_margin = (
                base_logits[..., self.foreground_label]
                - base_logits[..., self.background_label]
            )
            losses['loss_expert_rank'] = balanced_expert_rank_loss(
                base_margin,
                outputs['expert_residuals'],
                targets,
                indices,
            )

        if (
            'loss_routed_rank' in self.weight_dict
            and 'applied_route_delta' in outputs
            and idx[0].numel() > 0
        ):
            base_logits = outputs.get('base_logits', src_logits)
            base_margin = (
                base_logits[..., self.foreground_label]
                - base_logits[..., self.background_label]
            )
            losses['loss_routed_rank'] = routed_residual_rank_loss(
                base_margin,
                outputs['applied_route_delta'],
                indices,
            )

        if (
            'loss_query_gate_oracle' in self.weight_dict
            and 'query_route_gate' in outputs
            and 'route_delta' in outputs
            and idx[0].numel() > 0
        ):
            base_logits = outputs.get('base_logits', src_logits)
            base_margin = (
                base_logits[..., self.foreground_label]
                - base_logits[..., self.background_label]
            )
            losses['loss_query_gate_oracle'] = query_gate_oracle_loss(
                base_margin,
                outputs['route_delta'],
                outputs['query_route_gate'],
                indices,
            )

        if (
            'loss_quality_rank' in self.weight_dict
            and 'applied_route_delta' in outputs
            and 'pred_spans' in outputs
        ):
            base_logits = outputs.get('base_logits', src_logits)
            base_margin = (
                base_logits[..., self.foreground_label]
                - base_logits[..., self.background_label]
            )
            losses['loss_quality_rank'] = quality_aware_routed_rank_loss(
                base_margin,
                outputs['applied_route_delta'],
                outputs['pred_spans'],
                targets,
            )

        if (
            'loss_balanced_quality_rank' in self.weight_dict
            and 'applied_route_delta' in outputs
            and 'pred_spans' in outputs
        ):
            base_logits = outputs.get('base_logits', src_logits)
            base_margin = (
                base_logits[..., self.foreground_label]
                - base_logits[..., self.background_label]
            )
            balanced_loss, group_losses = quality_aware_routed_rank_loss(
                base_margin,
                outputs['applied_route_delta'],
                outputs['pred_spans'],
                targets,
                balance_durations=True,
                video_durations=(
                    outputs['video_mask'].sum(dim=-1) * self.clip_length
                ),
                return_group_losses=True,
            )
            losses['loss_balanced_quality_rank'] = balanced_loss
            active_groups = group_losses.detach() > 0
            group_weights = torch.zeros_like(group_losses)
            if self.group_robust_temperature > 0 and active_groups.any():
                with torch.no_grad():
                    if self.training:
                        if not self.duration_group_ema_initialized:
                            self.duration_group_loss_ema[active_groups] = (
                                group_losses.detach()[active_groups]
                            )
                            self.duration_group_ema_initialized = True
                        else:
                            self.duration_group_loss_ema[active_groups] = (
                                0.95 * self.duration_group_loss_ema[active_groups]
                                + 0.05 * group_losses.detach()[active_groups]
                            )
                        robust_source = self.duration_group_loss_ema
                    else:
                        robust_source = group_losses.detach()
                    robust_logits = robust_source / self.group_robust_temperature
                    robust_logits = robust_logits.masked_fill(~active_groups, -torch.inf)
                    group_weights = robust_logits.softmax(dim=0)
                losses['loss_balanced_quality_rank'] = (
                    group_weights.detach() * group_losses
                ).sum()
            for group_name, group_loss in zip(
                ('short', 'middle', 'long'), group_losses
            ):
                losses[f'loss_balanced_quality_{group_name}'] = group_loss.detach()
            for group_name, group_weight in zip(
                ('short', 'middle', 'long'), group_weights
            ):
                losses[f'quality_weight_{group_name}'] = group_weight.detach()

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]
        return losses

    def loss_saliency(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}

        vid_token_mask = outputs["video_mask"]

        # Neg pair loss
        saliency_scores_neg = outputs["saliency_scores_neg"].clone()  # (N, L)
        # loss_neg_pair = torch.sigmoid(saliency_scores_neg).mean()
        
        loss_neg_pair = (- torch.log(1. - torch.sigmoid(saliency_scores_neg)) * vid_token_mask).sum(dim=1).mean()

        saliency_scores = outputs["saliency_scores"].clone()  # (N, L)
        saliency_contrast_label = targets["saliency_all_labels"]

        saliency_scores = torch.cat([saliency_scores, saliency_scores_neg], dim=1)
        saliency_contrast_label = torch.cat([saliency_contrast_label, torch.zeros_like(saliency_contrast_label)], dim=1)

        vid_token_mask = vid_token_mask.repeat([1, 2])
        saliency_scores = vid_token_mask * saliency_scores + (1. - vid_token_mask) * -1e+3

        tau = 0.5
        loss_rank_contrastive = 0.

        # for rand_idx in range(1, 13, 3):
        #     # 1, 4, 7, 10 --> 5 stages
        for rand_idx in range(1, 12):
            drop_mask = ~(saliency_contrast_label > 100)  # no drop
            pos_mask = (saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx

            if torch.sum(pos_mask) == 0:  # no positive sample
                continue
            else:
                batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

            # drop higher ranks
            cur_saliency_scores = saliency_scores * drop_mask / tau + ~drop_mask * -1e+3

            # numerical stability
            logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]

            # softmax
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

            mean_log_prob_pos = (pos_mask * log_prob * vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)

            loss = - mean_log_prob_pos * batch_drop_mask

            loss_rank_contrastive = loss_rank_contrastive + loss.mean()

        loss_rank_contrastive = loss_rank_contrastive / 12

        saliency_scores = outputs["saliency_scores"]  # (N, L)
        pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
        neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
        num_pairs = pos_indices.shape[1]  # typically 2 or 4
        batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
        pos_scores = torch.stack(
            [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        neg_scores = torch.stack(
            [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                        / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale

        # print(loss_saliency, loss_rank_contrastive)
        # loss_saliency = loss_saliency + loss_rank_contrastive
        loss_saliency = loss_saliency + loss_rank_contrastive + loss_neg_pair
        # loss_saliency = loss_rank_contrastive
        return {"loss_saliency": loss_saliency}

    def loss_contrastive_align(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d)  text tokens
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed)  # (bsz, #queries, #tokens)
        logits = logits.sum(2) / self.temperature  # (bsz, #queries)
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        pos_term = positive_logits.sum(1)  # (bsz, )
        num_pos = positive_map.sum(1)  # (bsz, )
        neg_term = logits.logsumexp(1)  # (bsz, )
        loss_nce = - pos_term / num_pos + neg_term  # (bsz, )
        losses = {"loss_contrastive_align": loss_nce.mean()}
        return losses

    def loss_contrastive_align_vid_txt(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        # TODO (1)  align vid_mem and txt_mem;
        # TODO (2) change L1 loss as CE loss on 75 labels, similar to soft token prediction in MDETR
        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d)  text tokens
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed)  # (bsz, #queries, #tokens)
        logits = logits.sum(2) / self.temperature  # (bsz, #queries)
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        pos_term = positive_logits.sum(1)  # (bsz, )
        num_pos = positive_map.sum(1)  # (bsz, )
        neg_term = logits.logsumexp(1)  # (bsz, )
        loss_nce = - pos_term / num_pos + neg_term  # (bsz, )
        losses = {"loss_contrastive_align": loss_nce.mean()}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "contrastive_align": self.loss_contrastive_align,
            "saliency": self.loss_saliency,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        if (
            self.isolate_base_grad
            and 'base_logits' in outputs_without_aux
            and 'base_spans' in outputs_without_aux
        ):
            loss_outputs = dict(outputs_without_aux)
            loss_outputs['pred_logits'] = outputs_without_aux['base_logits']
            loss_outputs['pred_spans'] = outputs_without_aux['base_spans']
        else:
            loss_outputs = outputs_without_aux

        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)

        # only for HL, do not use matcher
        if self.use_matcher:
            indices = self.matcher(loss_outputs, targets)
            losses_target = self.losses
        else:
            indices = None
            losses_target = ["saliency"]

        # Compute all the requested losses
        losses = {}
        # for loss in self.losses:
        for loss in losses_target:
            losses.update(self.get_loss(loss, loss_outputs, targets, indices))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                if self.use_matcher:
                    indices = self.matcher(aux_outputs, targets)
                    losses_target = self.losses
                else:
                    indices = None
                    losses_target = ["saliency"]    
                # for loss in self.losses:
                for loss in losses_target:
                    if "saliency" == loss:  # skip as it is only in the top layer
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_model(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/qd_detr/issues/108#issuecomment-650269223
    device = torch.device(args.device)

    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    if args.a_feat_dir is None:
        model = QDDETR(
            transformer,
            position_embedding,
            txt_position_embedding,
            txt_dim=args.t_feat_dim,
            vid_dim=args.v_feat_dim,
            num_queries=args.num_queries,
            input_dropout=args.input_dropout,
            aux_loss=args.aux_loss,
            contrastive_align_loss=args.contrastive_align_loss,
            contrastive_hdim=args.contrastive_hdim,
            span_loss_type=args.span_loss_type,
            use_txt_pos=args.use_txt_pos,
            n_input_proj=args.n_input_proj,
            args=args,
        )
    else:
        model = QDDETR(
            transformer,
            position_embedding,
            txt_position_embedding,
            txt_dim=args.t_feat_dim,
            vid_dim=args.v_feat_dim,
            aud_dim=args.a_feat_dim,
            num_queries=args.num_queries,
            input_dropout=args.input_dropout,
            aux_loss=args.aux_loss,
            contrastive_align_loss=args.contrastive_align_loss,
            contrastive_hdim=args.contrastive_hdim,
            span_loss_type=args.span_loss_type,
            use_txt_pos=args.use_txt_pos,
            n_input_proj=args.n_input_proj,
            args=args,
        )

    matcher = build_matcher(args)
    weight_dict = {"loss_span": args.span_loss_coef,
                   "loss_giou": args.giou_loss_coef,
                   "loss_label": args.label_loss_coef,
                   "loss_saliency": args.lw_saliency}
    if args.use_lrmhc and args.lrmhc_route_loss_coef > 0:
        weight_dict["loss_route"] = args.lrmhc_route_loss_coef
    if args.use_lrmhc and args.lrmhc_expert_rank_loss_coef > 0:
        weight_dict["loss_expert_rank"] = args.lrmhc_expert_rank_loss_coef
    if args.use_lrmhc and args.lrmhc_routed_rank_loss_coef > 0:
        weight_dict["loss_routed_rank"] = args.lrmhc_routed_rank_loss_coef
    if args.use_lrmhc and args.lrmhc_duration_balance_loss_coef > 0:
        weight_dict["loss_duration_balance"] = args.lrmhc_duration_balance_loss_coef
    if args.use_lrmhc and args.lrmhc_query_gate_oracle_loss_coef > 0:
        weight_dict["loss_query_gate_oracle"] = args.lrmhc_query_gate_oracle_loss_coef
    if args.use_lrmhc and args.lrmhc_quality_rank_loss_coef > 0:
        weight_dict["loss_quality_rank"] = args.lrmhc_quality_rank_loss_coef
    if args.use_lrmhc and args.lrmhc_balanced_quality_rank_loss_coef > 0:
        weight_dict["loss_balanced_quality_rank"] = args.lrmhc_balanced_quality_rank_loss_coef
    if args.use_lrmhc and args.lrmhc_span_improvement_loss_coef > 0:
        weight_dict["loss_span_improvement"] = args.lrmhc_span_improvement_loss_coef
    if args.use_lrmhc and args.lrmhc_span_query_gate_loss_coef > 0:
        weight_dict["loss_span_query_gate"] = args.lrmhc_span_query_gate_loss_coef
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({
                k + f'_{i}': v for k, v in weight_dict.items()
                if k not in (
                    "loss_saliency", "loss_route", "loss_expert_rank",
                    "loss_routed_rank", "loss_duration_balance",
                    "loss_query_gate_oracle", "loss_quality_rank",
                    "loss_balanced_quality_rank", "loss_span_improvement",
                    "loss_span_query_gate",
                )
            })
        weight_dict.update(aux_weight_dict)

    losses = ['spans', 'labels', 'saliency']
    if args.contrastive_align_loss:
        losses += ["contrastive_align"]
        
    # For tvsum dataset
    use_matcher = not (args.dset_name == 'tvsum')
        
    criterion = SetCriterion(
        matcher=matcher, weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, temperature=args.temperature,
        span_loss_type=args.span_loss_type, max_v_l=args.max_v_l,
        saliency_margin=args.saliency_margin, use_matcher=use_matcher,
        clip_length=args.clip_length,
        group_robust_temperature=args.lrmhc_group_robust_temperature,
        span_improvement_temperature=args.lrmhc_span_improvement_temperature,
        span_query_gate_temperature=args.lrmhc_span_query_gate_temperature,
        duration_boundaries=args.lrmhc_duration_boundaries,
        duration_softness=args.lrmhc_duration_softness,
        prototype_responsibilities=args.lrmhc_prototype_responsibilities,
        isolate_base_grad=args.lrmhc_isolate_base_grad,
    )
    criterion.to(device)
    return model, criterion

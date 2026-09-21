"""DETR Transformer class."""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as func

from src.model.blocks.decoder import TransformerDecoder
from src.model.blocks.encoder import TransformerEncoder
from src.model.blocks.feed_forward import SlimMLP
from src.model.blocks.layers import TransformerDecoderLayer, TransformerEncoderLayer
from src.model.utils.aux_anchors import prepare_anchors_codetr, prepare_anchors_dn
from src.model.utils.model_utils import gen_encoder_output_proposals, inverse_sigmoid
from src.model.utils.schemas import (
    DetectorOutput,
    DetEncoderOutput,
    QueryProposalsOutput,
)

EPS: float = 0.01
MIN_CONST: float = -1e7


class QuerySelector(nn.Module):  # noqa: WPS230
    """Class to select suqries for decoder."""

    def __init__(
        self,
        model_dim: int,
        num_queries: int,
        prior_prob: float = 0.35,
        default_widths: List[float] = [0.05, 0.2, 0.4, 0.85],
        init_spans_with_zeros: bool = True,
    ):
        """Initialize QuerySelector.

        Args:
            model_dim (int): Hidden dimension of the model.
            num_queries (int): number of queries to predict.
            prior_prob (float): prior foreground prob.
            default_widths (float): default width of encoder's anchors.
            init_spans_with_zeros (bool): whether to init last mlp layer with zeros or not
        """
        super().__init__()
        self.model_dim = model_dim
        self.num_queries = num_queries
        self.prior_prob = prior_prob
        self.default_widths = default_widths
        self.init_spans_with_zeros = init_spans_with_zeros

        self.enc_output = nn.Linear(model_dim, model_dim)
        self.enc_output_norm = nn.LayerNorm(model_dim)
        self.enc_out_span_embed = SlimMLP(input_dim=model_dim, hidden_dim=model_dim, output_dim=2, num_layers=3)
        self.enc_out_class_embed = nn.Linear(model_dim, 1)
        self.enc_out_iou_embed = nn.Linear(model_dim, 1)
        self._init_parameters(prior_prob)

    def _init_parameters(self, prior_prob: float) -> None:
        """Init parameters.

        Args:
            prior_prob (float): prior foreground prob.
        """
        # init cls layer with prior prob
        bias_value = math.log(prior_prob / (1 - prior_prob))
        torch.nn.init.normal_(self.enc_out_class_embed.weight, std=EPS)  # noqa: WPS432
        torch.nn.init.normal_(self.enc_out_iou_embed.weight, std=EPS)  # noqa: WPS432
        self.enc_out_class_embed.bias.data.fill_(bias_value)
        self.enc_out_iou_embed.bias.data.fill_(bias_value * 2)

        # init last reg layer with zeros
        for module in self.enc_out_span_embed.linear_mapper.modules():
            if isinstance(module, nn.Linear) and module.out_features == 2 and self.init_spans_with_zeros:
                nn.init.constant_(module.weight.data, 0)  # noqa: WPS219
                nn.init.constant_(module.bias.data, 0)  # noqa: WPS219

        # init mapper
        nn.init.xavier_uniform_(self.enc_output.weight.data)
        nn.init.constant_(self.enc_output.bias.data, 0)

    def get_query_proposals(
        self,
        memory_aux: Tensor,
        output_proposals: Tensor,
        mask: Tensor,
    ) -> QueryProposalsOutput:
        """Prepare query proposals.

        Args:
            memory_aux (Tensor): Query memory tensor from the encoder. Shape: [bs, seq, dim]
            output_proposals (Tensor): Tensor containing the initial proposals. Shape: [bs, seq, 2]
            mask (Tensor): mask for irrelevant embs.

        Returns:
            QueryProposalsOutput: Query proposal schema.
        """
        # map memory features
        memory_aux = self.enc_output_norm(self.enc_output(memory_aux))

        # predict objectness and iou scores
        enc_outputs_class_unselected = self.enc_out_class_embed(memory_aux)
        enc_outputs_iou_unselected = self.enc_out_iou_embed(memory_aux)

        # combine them
        enc_outputs_combo_unselected = torch.sqrt(
            enc_outputs_class_unselected.sigmoid() * enc_outputs_iou_unselected.sigmoid(),
        )
        enc_outputs_combo_unselected = enc_outputs_combo_unselected.masked_fill(~mask, float("-inf"))

        # predict reference points
        enc_outputs_offsets_unselected = self.enc_out_span_embed(memory_aux)
        enc_outputs_coord_unselected = output_proposals + enc_outputs_offsets_unselected

        # Find the most relevant indices
        topk_proposals = torch.topk(enc_outputs_combo_unselected[..., 0], self.num_queries, dim=1)[1]  # noqa: WPS221
        topk_proposals = topk_proposals.unsqueeze(-1)

        # gather features
        query_topk = topk_proposals.repeat(1, 1, self.model_dim)
        query_embs = torch.gather(memory_aux, 1, query_topk).detach()

        # gather spans
        spans_topk = topk_proposals.repeat(1, 1, 2)
        refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1, spans_topk)  # unsigmoid
        refpoint_embed_detach = refpoint_embed_undetach.detach()

        # gather logits
        class_logit_enc = torch.gather(enc_outputs_class_unselected, 1, topk_proposals[..., [0]])
        iou_logit_enc = torch.gather(enc_outputs_iou_unselected, 1, topk_proposals[..., [0]])

        return QueryProposalsOutput(
            query_embs=query_embs,
            refpoint_embed_detach=refpoint_embed_detach,
            refpoint_embed_enc=refpoint_embed_undetach.sigmoid(),
            class_logit_enc=class_logit_enc,
            iou_logit_enc=iou_logit_enc,
        )

    @staticmethod
    def _prepare_mask(fpn_features: List[Tensor], mask: Tensor) -> List[Tensor]:  # noqa: WPS602
        spatial_shapes = [seq.size(1) for seq in fpn_features]
        original_seq_length = mask.shape[1]

        updated_masks = []
        for seq_length in spatial_shapes:
            if seq_length > original_seq_length:
                scale = seq_length // original_seq_length
                updated_mask = mask.repeat_interleave(scale, dim=1)
                updated_masks.append(updated_mask)
            elif seq_length < original_seq_length:
                scale = original_seq_length // seq_length
                updated_mask = func.max_pool1d(mask.float(), kernel_size=scale, stride=scale)  # type: ignore
                updated_masks.append(updated_mask.bool())
            else:
                updated_masks.append(mask)
        return updated_masks

    def forward(self, multiscale: List[Tensor], vid_mask: Tensor) -> QueryProposalsOutput:
        """Forward pass of the QuerySelector.

        Args:
            multiscale (List[Tensor]): multiscale features.
            vid_mask (Tensor): mask for the source sequence. Shape: [batch_size, Lv]

        Returns:
            QueryProposalsOutput: query proposals.
        """
        masks = self._prepare_mask(multiscale, vid_mask)

        # compute reference points
        memory_aux, output_proposals, mask = gen_encoder_output_proposals(
            fpn_features=multiscale,
            memory_padding_masks=masks,
            default_widths=self.default_widths,
        )

        # get proposal outputs
        return self.get_query_proposals(memory_aux, output_proposals, mask)


class DetectorEncoder(nn.Module):
    """Stack of the encoder layers."""

    def __init__(
        self,
        model_dim: int,
        num_encoder_layers: int = 3,
        dropout: float = 0.1,
        droppath: float = 0.1,
    ) -> None:
        """Initialize Detector Encoder.

        Args:
            model_dim (int): Hidden dimension of the model
            num_encoder_layers (int): Number of encoder layers.
            dropout (float): Dropout rate
            droppath (float): Droppath rate
        """
        super().__init__()
        self.model_dim = model_dim
        self.enc_layers = num_encoder_layers

        # Init Encoder
        general_encoder_layer = TransformerEncoderLayer(model_dim, dropout=dropout, droppath=droppath)
        self.encoder = TransformerEncoder(general_encoder_layer, num_encoder_layers)

    def forward(
        self,
        src: Tensor,
        mask: Tensor,
        pos: Tensor,
        video_length: Tensor,
    ) -> DetEncoderOutput:
        """Forward pass of the Transformer.

        Args:
            src (Tensor): source sequence. Shape: [Lv, batch_size, dim]
            mask (Tensor): mask for the source sequence. Shape: [batch_size, Lv]
            pos (Tensor): positional embedding. Shape: [Lv, batch_size, dim]
            video_length (Tensor): length of the video. Shape: [batch_size]

        Returns:
            DetEncoderOutput: output of the encoder
        """
        vid_src = src[:video_length]  # (L_video, batch_size, dim)
        vid_mask = mask[:, :video_length]  # (batch_size, L_video)
        vid_pos = pos[:video_length]  # (L_video, batch_size, dim)

        # encoder forward pass
        memory = self.encoder(vid_src, vid_pos, src_key_padding_mask=~vid_mask)

        return DetEncoderOutput(memory=memory, vid_pos=vid_pos, vid_mask=vid_mask)


class MomentDetector(nn.Module):  # noqa: WPS230
    """Transformer module from DETR."""

    def __init__(  # noqa: WPS211
        self,
        reference: Optional[nn.Module],
        model_dim: int = 512,
        cont_pos_tradeoff: int = 0,
        num_queries: int = 25,
        use_rpn: bool = True,
        use_encoder_features: bool = True,
        num_decoder_layers: int = 3,
        dropout: float = 0.1,
        droppath: float = 0.1,
        temperature: int = 10000,
        prior_prob: float = 0.35,
        unique_content_queries: bool = True,
        init_spans_with_zeros: bool = True,
        return_intermediate_dec: bool = True,
        predict_quality_score: bool = True,
        num_groups: int = 5,
        span_noise_scale: float = 0.4,
        negative_offset: float = 1.0,
        look_at_target: bool = False,
        aux_anchors_type: Tuple[str, ...] = (),
        use_gaussian_bias: bool = True,
        use_length_aware: bool = False,
        use_soft_length_query_embedding: bool = False,
        length_thresholds: Tuple[float, float] = (0.15, 0.35),
        length_aware_gaussian: bool = False,
        use_length_routed_cls: bool = False,
        soft_routing: bool = True,
        route_beta: float = 4.0,
        route_prototype_init: Tuple[float, float, float] = (-3.1781, -2.1253, -1.3863),
        route_min_gap: float = 0.05,
        learn_route_prototypes: bool = True,
        detach_route_width: bool = True,
        route_width_source: str = "output",
        use_absolute_duration_routing: bool = False,
        route_duration_centers: Tuple[float, float, float] = (5.0, 17.320508, 60.0),
        route_alpha_init: float = 0.01,
        route_expert_init_std: float = 1e-4,
        route_expert_hidden_dim: int = 0,
        use_route_alpha: bool = True,
        uniform_routing: bool = False,
        route_rescore_target: str = "class",
        use_route_gate: bool = True,
        route_gate_init: float = 0.2,
        use_adaptive_router: bool = False,
        route_router_hidden_dim: int = 64,
        route_router_scale: float = 1.0,
        use_route_confidence_gate: bool = False,
        use_query_route_gate: bool = False,
        multiply_query_route_gate: bool = False,
        route_gate_hidden_dim: int = 64,
        route_gate_use_decoder_features: bool = True,
        use_monotonic_route_gate: bool = False,
        route_gate_pivot_init: float = -2.1253,
        route_gate_slope_init: float = 2.0,
        route_gate_min_slope: float = 0.1,
        route_gate_signal: str = "width",
        signed_route_gate: bool = False,
        center_expert_residuals: bool = False,
        use_learnable_route_gain: bool = False,
        route_gain_init: float = 1.0,
        use_expert_route_gain: bool = False,
        route_expert_gain_init: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        use_query_length_prior: bool = False,
        query_length_prior_hidden_dim: int = 64,
        query_length_prior_scale: float = 1.0,
        query_length_prior_use_width_stats: bool = False,
        query_length_prior_absolute_width_stats: bool = False,
        query_length_prior_score_weighted_width_stats: bool = False,
        use_duration_safety_prior: bool = False,
        duration_safety_prior_hidden_dim: int = 64,
        duration_safety_prior_dual_context: bool = False,
        use_short_safe_query_prior: bool = False,
        use_learned_query_prior_safety: bool = False,
        hard_query_prior_safety: bool = False,
        use_route_safety_adapter: bool = False,
        gate_query_prior_with_safety: bool = True,
        gate_safety_adapter: bool = False,
        query_prior_safety_use_context: bool = False,
        use_sample_route_gate: bool = False,
        sample_route_gate_hidden_dim: int = 64,
        sample_route_gate_init: float = 0.95,
        sample_route_gate_use_route_stats: bool = False,
        preserve_uniform_top1: bool = False,
        preserve_uniform_top1_margin: float = 0.05,
    ):
        """
        Initialize a Transformer module.

        Args:
            reference (nn.Module): anchors generator.
            model_dim (int): hidden dimension of the model
            cont_pos_tradeoff (int): Offset for the dim content/position dims.
            num_queries (int): number of queries
            use_rpn (bool): whether to use RPN as anchor generator or reference.
            use_encoder_features (bool): whether to use encoder features as content queries or not.
            num_decoder_layers (int): number of decoder layers
            dropout (float): dropout rate
            droppath (float): droppath rate
            temperature (int): temperature of the pos emb.
            prior_prob (float): prior foreground prob.
            unique_content_queries (bool): unique content embeddings.
            init_spans_with_zeros (bool): whether to init last mlp layer with zeros or not
            return_intermediate_dec (bool): whether to return intermediate results of the decoder
            predict_quality_score (bool): predict iou of the predicted interval
            num_groups (int): number of noised gt groups.
            span_noise_scale (float): noise scale for the bbox
            negative_offset (float): offset for negative samples
            look_at_target (bool): if true aux anchors can see main queries
            aux_anchors_type (Tuple[str]): aux anchors preparation method. Could be of {"collab", "denoise"}
        """
        super().__init__()
        self.model_dim = model_dim
        self.cont_pos_tradeoff = cont_pos_tradeoff
        self.dec_layers = num_decoder_layers
        self.num_groups = num_groups
        self.span_noise_scale = span_noise_scale
        self.use_rpn = use_rpn
        self.use_encoder_features = use_encoder_features
        self.look_at_target = look_at_target
        self.aux_anchors_type = aux_anchors_type
        self.unique_content_queries = unique_content_queries
        self.init_spans_with_zeros = init_spans_with_zeros
        self.predict_quality_score = predict_quality_score
        self.negative_offset = negative_offset

        # defince query tokens for decoder layers
        self.num_queries: int = num_queries
        self.refpoint_embed = None if use_rpn else reference

        # content query encoder
        self.init_content_queries(num_queries, model_dim, aux_anchors_type)

        decoder_layer = TransformerDecoderLayer(
            d_model=model_dim,
            cont_pos_tradeoff=cont_pos_tradeoff,
            dropout=dropout,
            droppath=droppath,
        )

        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            return_intermediate=return_intermediate_dec,
            d_model=model_dim,
            temperature=temperature,
            predict_quality_score=predict_quality_score,
            init_spans_with_zeros=init_spans_with_zeros,
            use_gaussian_bias=use_gaussian_bias,
            length_aware_gaussian=length_aware_gaussian,
            length_thresholds=length_thresholds,
        )
        # define heads for classification and box regression
        self.span_embed = SlimMLP(input_dim=model_dim, hidden_dim=model_dim, output_dim=2, num_layers=3)
        self.class_embed = nn.Linear(model_dim, 1)

        # Length-Aware class embeddings (LA-DETR)
        if use_soft_length_query_embedding and not use_length_routed_cls:
            raise ValueError("soft length query embedding requires length-routed classification")
        self.use_length_aware = use_length_aware
        self.use_soft_length_query_embedding = use_soft_length_query_embedding
        self.length_thresholds = length_thresholds
        if use_length_aware or use_soft_length_query_embedding:
            self.length_class_embed = nn.Embedding(3, model_dim)
            nn.init.normal_(self.length_class_embed.weight.data, std=1e-4)

        # Length-Routed Multi-Head Classification (LR-MHC)
        # Sews LA-DETR's length routing into FlashVTG's multi-head score refinement
        self.use_length_routed_cls = use_length_routed_cls
        self.soft_routing = soft_routing
        self.uniform_routing = uniform_routing
        self.detach_route_width = detach_route_width
        if route_width_source not in {"output", "reference"}:
            raise ValueError("route_width_source must be 'output' or 'reference'")
        self.route_width_source = route_width_source
        self.use_route_alpha = use_route_alpha
        self.use_route_gate = use_route_gate
        self.use_adaptive_router = use_adaptive_router
        self.route_router_scale = route_router_scale
        self.use_route_confidence_gate = use_route_confidence_gate
        self.use_query_route_gate = use_query_route_gate
        self.multiply_query_route_gate = multiply_query_route_gate
        self.route_gate_use_decoder_features = route_gate_use_decoder_features
        self.use_monotonic_route_gate = use_monotonic_route_gate
        self.route_gate_min_slope = route_gate_min_slope
        if route_gate_signal not in {"width", "long_probability"}:
            raise ValueError("route_gate_signal must be one of: width, long_probability")
        self.route_gate_signal = route_gate_signal
        self.signed_route_gate = signed_route_gate
        self.center_expert_residuals = center_expert_residuals
        self.use_learnable_route_gain = use_learnable_route_gain
        self.use_expert_route_gain = use_expert_route_gain
        self.use_query_length_prior = use_query_length_prior
        self.query_length_prior_scale = query_length_prior_scale
        self.query_length_prior_use_width_stats = query_length_prior_use_width_stats
        self.query_length_prior_absolute_width_stats = query_length_prior_absolute_width_stats
        self.query_length_prior_score_weighted_width_stats = query_length_prior_score_weighted_width_stats
        self.use_duration_safety_prior = use_duration_safety_prior
        self.duration_safety_prior_dual_context = duration_safety_prior_dual_context
        self.use_short_safe_query_prior = use_short_safe_query_prior
        self.use_learned_query_prior_safety = use_learned_query_prior_safety
        self.hard_query_prior_safety = hard_query_prior_safety
        self.use_route_safety_adapter = use_route_safety_adapter
        self.gate_query_prior_with_safety = gate_query_prior_with_safety
        self.gate_safety_adapter = gate_safety_adapter
        self.query_prior_safety_use_context = query_prior_safety_use_context
        self.use_sample_route_gate = use_sample_route_gate
        self.sample_route_gate_use_route_stats = sample_route_gate_use_route_stats
        self.preserve_uniform_top1 = preserve_uniform_top1
        self.preserve_uniform_top1_margin = preserve_uniform_top1_margin
        if gate_safety_adapter and not (use_route_safety_adapter and use_learned_query_prior_safety):
            raise ValueError(
                "gate_safety_adapter requires both route safety adapter and learned query-prior safety",
            )
        if route_rescore_target not in {"class", "quality", "both"}:
            raise ValueError("route_rescore_target must be one of: class, quality, both")
        if route_rescore_target in {"quality", "both"} and not predict_quality_score:
            raise ValueError("quality routing requires predict_quality_score=True")
        self.route_rescore_target = route_rescore_target
        if use_length_routed_cls:
            if route_expert_hidden_dim > 0:
                self.length_rescoring = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(model_dim, route_expert_hidden_dim),
                            nn.GELU(),
                            nn.Linear(route_expert_hidden_dim, 1),
                        )
                        for _ in range(3)
                    ],
                )
                for expert in self.length_rescoring:
                    nn.init.xavier_uniform_(expert[0].weight.data)
                    nn.init.zeros_(expert[0].bias.data)
                    nn.init.zeros_(expert[2].weight.data)
                    nn.init.zeros_(expert[2].bias.data)
            else:
                self.length_rescoring = nn.Linear(model_dim, 3)
                if route_expert_init_std == 0:
                    nn.init.zeros_(self.length_rescoring.weight.data)
                else:
                    nn.init.normal_(self.length_rescoring.weight.data, std=route_expert_init_std)
                nn.init.zeros_(self.length_rescoring.bias.data)
            if use_route_alpha:
                self.length_rescoring_alpha = nn.Parameter(torch.full((3,), route_alpha_init))
            if use_route_safety_adapter:
                self.route_safety_adapter = nn.Linear(model_dim, 3)
                nn.init.zeros_(self.route_safety_adapter.weight)
                nn.init.zeros_(self.route_safety_adapter.bias)

            # Ordered prototypes in logit-width space. Beta is fixed to avoid
            # the scale ambiguity caused by learning bandwidth and temperature together.
            initial_centers = torch.tensor(route_prototype_init, dtype=torch.float32)
            if not torch.all(initial_centers[1:] > initial_centers[:-1]):
                raise ValueError("route_prototype_init must be strictly increasing")
            initial_gaps = initial_centers[1:] - initial_centers[:-1] - route_min_gap
            if not torch.all(initial_gaps > 0):
                raise ValueError("route prototype gaps must be larger than route_min_gap")
            self.route_center_start = nn.Parameter(initial_centers[:1].clone())
            self.route_gap_raw = nn.Parameter(torch.log(torch.expm1(initial_gaps)))
            self.route_center_start.requires_grad_(learn_route_prototypes)
            self.route_gap_raw.requires_grad_(learn_route_prototypes)
            self.route_beta = float(route_beta)
            self.route_min_gap = float(route_min_gap)
            if len(route_duration_centers) != 3 or any(
                center <= 0 for center in route_duration_centers
            ):
                raise ValueError("route_duration_centers must contain three positive values")
            self.use_absolute_duration_routing = use_absolute_duration_routing
            self.route_duration_centers = tuple(float(center) for center in route_duration_centers)
            if not 0 < route_gate_init < 1:
                raise ValueError("route_gate_init must be in (0, 1)")
            if use_route_gate:
                gate_logit = math.log(route_gate_init / (1 - route_gate_init))
                self.route_gate_logits = nn.Parameter(torch.full((3,), gate_logit))
            if use_adaptive_router:
                self.length_router = nn.Sequential(
                    nn.Linear(model_dim + 1, route_router_hidden_dim),
                    nn.GELU(),
                    nn.Linear(route_router_hidden_dim, 3),
                )
                nn.init.xavier_uniform_(self.length_router[0].weight)
                nn.init.zeros_(self.length_router[0].bias)
                nn.init.zeros_(self.length_router[2].weight)
                nn.init.zeros_(self.length_router[2].bias)
            if use_query_length_prior:
                self.query_length_prior = nn.Sequential(
                    nn.Linear(
                        model_dim + (3 if query_length_prior_use_width_stats else 0),
                        query_length_prior_hidden_dim,
                    ),
                    nn.GELU(),
                    nn.Linear(query_length_prior_hidden_dim, 3),
                )
                nn.init.xavier_uniform_(self.query_length_prior[0].weight)
                nn.init.zeros_(self.query_length_prior[0].bias)
                nn.init.zeros_(self.query_length_prior[2].weight)
                nn.init.zeros_(self.query_length_prior[2].bias)
                if use_learned_query_prior_safety:
                    if query_prior_safety_use_context:
                        self.query_prior_safety_gate = nn.Sequential(
                            nn.Linear(model_dim, query_length_prior_hidden_dim),
                            nn.GELU(),
                            nn.Linear(query_length_prior_hidden_dim, 1),
                        )
                        nn.init.xavier_uniform_(self.query_prior_safety_gate[0].weight)
                        nn.init.zeros_(self.query_prior_safety_gate[0].bias)
                        nn.init.zeros_(self.query_prior_safety_gate[2].weight)
                        nn.init.zeros_(self.query_prior_safety_gate[2].bias)
                    else:
                        self.query_prior_safety_gate = nn.Linear(3, 1)
                        nn.init.zeros_(self.query_prior_safety_gate.weight)
                        nn.init.zeros_(self.query_prior_safety_gate.bias)
            if use_duration_safety_prior:
                duration_context_dim = model_dim * (
                    2 if duration_safety_prior_dual_context else 1
                )
                self.duration_safety_prior = nn.Sequential(
                    nn.Linear(duration_context_dim + 3, duration_safety_prior_hidden_dim),
                    nn.GELU(),
                    nn.Linear(duration_safety_prior_hidden_dim, 3),
                )
                nn.init.xavier_uniform_(self.duration_safety_prior[0].weight)
                nn.init.zeros_(self.duration_safety_prior[0].bias)
                nn.init.zeros_(self.duration_safety_prior[2].weight)
                nn.init.zeros_(self.duration_safety_prior[2].bias)
            if use_query_route_gate:
                # Features: decoder state, width, K route weights, route delta,
                # base class logit and base quality logit.
                gate_input_dim = (model_dim if route_gate_use_decoder_features else 0) + 1 + 3 + 1 + 2
                self.query_route_gate = nn.Sequential(
                    nn.Linear(gate_input_dim, route_gate_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(route_gate_hidden_dim, 1),
                )
                nn.init.xavier_uniform_(self.query_route_gate[0].weight)
                nn.init.zeros_(self.query_route_gate[0].bias)
                nn.init.zeros_(self.query_route_gate[2].weight)
                gate_logit = math.log(route_gate_init / (1 - route_gate_init))
                nn.init.constant_(self.query_route_gate[2].bias, gate_logit)
            if use_sample_route_gate:
                if not 0 < sample_route_gate_init < 1:
                    raise ValueError("sample_route_gate_init must be in (0, 1)")
                self.sample_route_gate = nn.Sequential(
                    nn.Linear(
                        model_dim + (6 if sample_route_gate_use_route_stats else 0),
                        sample_route_gate_hidden_dim,
                    ),
                    nn.GELU(),
                    nn.Linear(sample_route_gate_hidden_dim, 1),
                )
                nn.init.xavier_uniform_(self.sample_route_gate[0].weight)
                nn.init.zeros_(self.sample_route_gate[0].bias)
                nn.init.zeros_(self.sample_route_gate[2].weight)
                sample_gate_logit = math.log(sample_route_gate_init / (1 - sample_route_gate_init))
                nn.init.constant_(self.sample_route_gate[2].bias, sample_gate_logit)
            if use_monotonic_route_gate:
                if route_gate_slope_init <= route_gate_min_slope:
                    raise ValueError("route_gate_slope_init must exceed route_gate_min_slope")
                initial_raw_slope = math.log(
                    math.expm1(route_gate_slope_init - route_gate_min_slope),
                )
                self.route_gate_pivot = nn.Parameter(torch.tensor(route_gate_pivot_init))
                self.route_gate_slope_raw = nn.Parameter(torch.tensor(initial_raw_slope))
            if use_learnable_route_gain:
                if route_gain_init <= 0:
                    raise ValueError("route_gain_init must be positive")
                self.route_gain_raw = nn.Parameter(torch.tensor(math.log(math.expm1(route_gain_init))))
            if use_expert_route_gain:
                if len(route_expert_gain_init) != 3 or any(gain <= 0 for gain in route_expert_gain_init):
                    raise ValueError("route_expert_gain_init must contain three positive values")
                initial_gains = torch.tensor(route_expert_gain_init, dtype=torch.float32)
                self.route_expert_gain_raw = nn.Parameter(torch.log(torch.expm1(initial_gains)))

        self._init_parameters(prior_prob)

    def get_route_prototypes(self) -> Tensor:
        """Return strictly ordered length prototypes in logit-width space."""
        gaps = func.softplus(self.route_gap_raw) + self.route_min_gap
        return torch.cat((self.route_center_start, self.route_center_start + torch.cumsum(gaps, dim=0)))

    def apply_length_aware(self, query_label: Tensor, refpoint: Tensor) -> Tensor:
        """Add per-length-class embedding to content queries.

        Args:
            query_label: (Q, B, D) content queries
            refpoint: (Q, B, 2) reference points in (center, width) sigmoid space

        Returns:
            query_label with length-class embedding added
        """
        if not self.use_length_aware and not self.use_soft_length_query_embedding:
            return query_label
        widths = refpoint[..., 1].sigmoid()  # (Q, B)
        if self.use_soft_length_query_embedding:
            if self.detach_route_width:
                widths = widths.detach()
            width_logits = torch.logit(widths.clamp(min=1e-4, max=1 - 1e-4)).unsqueeze(-1)
            route_logits = -self.route_beta * (width_logits - self.get_route_prototypes()) ** 2
            route_weights = (
                torch.full_like(route_logits, 1.0 / route_logits.shape[-1])
                if self.uniform_routing
                else torch.softmax(route_logits, dim=-1)
            )
            embedding = torch.einsum("qbk,kd->qbd", route_weights, self.length_class_embed.weight)
            return query_label + embedding
        t1, t2 = self.length_thresholds
        cls = torch.zeros_like(widths, dtype=torch.long)
        cls[widths >= t1] = 1
        cls[widths >= t2] = 2
        emb = self.length_class_embed(cls)  # (Q, B, D)
        return query_label + emb

    def init_content_queries(self, num_queries: int, model_dim: int, aux_anchors_type: Tuple[str, ...]) -> None:
        """Init contnent queries.

        Args:
            num_queries (int): number of queries to use.
            model_dim (int): model dim.
            aux_anchors_type (Tuple[str, ...]): list of auxiliary anchors.
        """
        if self.use_rpn and self.use_encoder_features:
            self.label_enc = None
        else:
            if self.unique_content_queries:
                self.label_enc = nn.Embedding(num_queries, model_dim)
            else:
                self.label_enc = nn.Embedding(1, model_dim)
            nn.init.normal_(self.label_enc.weight.data, std=EPS)
        if "collab" in aux_anchors_type:
            self.co_mapper = nn.Sequential(nn.Linear(model_dim, model_dim), nn.LayerNorm(model_dim))
            nn.init.normal_(self.co_mapper[0].weight.data, std=EPS)
        if "denoise" in aux_anchors_type:
            self.dn_label_enc = nn.Embedding(1, model_dim)
            nn.init.normal_(self.dn_label_enc.weight.data, std=EPS)

    def _init_parameters(self, prior_prob: float) -> None:
        """Init parameters.

        Args:
            prior_prob (float): prior foreground prob.
        """
        # init cls layer with prior prob
        bias_value = math.log(prior_prob / (1 - prior_prob))
        nn.init.normal_(self.class_embed.weight, std=EPS)  # noqa: WPS432
        self.class_embed.bias.data.fill_(bias_value)

        # init last reg layer with zeros
        for module in self.span_embed.linear_mapper.modules():
            if isinstance(module, nn.Linear) and module.out_features == 2 and self.init_spans_with_zeros:
                nn.init.constant_(module.weight.data, 0)  # noqa: WPS219
                nn.init.constant_(module.bias.data, 0)  # noqa: WPS219

    def prepare_regular_detr(
        self,
        proposals: Optional[QueryProposalsOutput],
        refpoint_emb: Tensor,
        batch_size: int = 512,
    ) -> Tuple[Tensor, Tensor]:
        """
        Prepare reference points for detr decoder.

        Args:
            proposals (QueryProposalsOutput): query selector output.
            refpoint_emb (Tensor): positional queries as anchor points
            batch_size (int): batch size

        Returns:
            Tuple[Tensor, Tensor]:
                - input_query_label: label query embedding for detr decoder
                - input_query_spans: reference points for detr decoder
        """
        # prepare content queries
        if self.use_rpn and self.use_encoder_features:
            assert proposals is not None
            input_query_label = proposals.query_embs.transpose(0, 1)
        else:
            if self.label_enc is None:
                raise ValueError("label_enc cannot be None when using anchors")
            input_query_label = self.label_enc.weight[:, None, :]
            if self.unique_content_queries:
                input_query_label = input_query_label.repeat(1, batch_size, 1)
            else:
                input_query_label = input_query_label.repeat(self.num_queries, batch_size, 1)

        # prepare pos queries
        input_query_span = refpoint_emb if self.use_rpn else refpoint_emb.repeat(batch_size, 1, 1)
        input_query_span = input_query_span.transpose(0, 1)

        # add length-aware class embedding (LA-DETR)
        input_query_label = self.apply_length_aware(input_query_label, input_query_span)
        return input_query_label, input_query_span

    def get_collab_queries(  # noqa: WPS234
        self,
        matched_gts: Optional[Tensor],
        anchors_spans: Optional[Tensor],
        encoder_features: Optional[Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Dict[str, Any]]]:  # noqa: WPS221
        """Get coolab queries.

        Args:
            memory_local (Tensor): Output from the encoder. Shape: [Lv, batch_size, dim]
            matched_gts (Optional[Tensor]): gt spans mathced to selected anchors.
            anchors_spans (Optional[Tensor]): selected anchors.
            encoder_features (Optional[Tensor]): selected encoder features.

        Returns:
            Tuple[Optional[Tensor], Optional[Tensor], Optional[Dict[str, Any]]]: prepared collab info
        """
        if (
            "collab" in self.aux_anchors_type
            and self.training
            and matched_gts is not None
            and anchors_spans is not None
        ):
            co_query_label, co_query_span, co_info = prepare_anchors_codetr(
                linear_mapper=self.co_mapper,  # type: ignore
                matched_gts=matched_gts,  # type: ignore
                anchors_per_seq=anchors_spans,  # type: ignore
                encoder_features_per_seq=encoder_features,  # type: ignore
            )
        else:
            co_query_label = torch.empty((0, batch_size, self.model_dim), device=device)
            co_query_span = torch.empty((0, batch_size, 2), device=device)
            co_info = None
        return co_query_label, co_query_span, co_info

    def get_denoise_queries(  # noqa: WPS234
        self,
        targets: Optional[Dict[str, Any]],
        batch_size: int,
        device: torch.device,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Dict[str, Any]]]:  # noqa: WPS221
        """Get denoise queries.

        Args:
            targets (Optional[Dict[str, Any]]): gt spans mathced to selected anchors.
            batch_size (int): batch size
            device (torch.device): device

        Returns:
            Tuple[Optional[Tensor], Optional[Tensor], Optional[Dict[str, Any]]]: prepared denoise info
        """
        if "denoise" in self.aux_anchors_type and self.training and self.num_groups > 0 and targets is not None:
            dn_query_label, dn_query_span, dn_info = prepare_anchors_dn(
                label_enc=self.dn_label_enc,
                targets=targets,
                num_groups=self.num_groups,
                span_noise_scale=self.span_noise_scale,
                negative_offset=self.negative_offset,
                batch_size=batch_size,
            )
        else:
            dn_query_label = torch.empty((0, batch_size, self.model_dim), device=device)
            dn_query_span = torch.empty((0, batch_size, 2), device=device)
            dn_info = None
        return dn_query_label, dn_query_span, dn_info

    def _get_attention_mask(
        self,
        co_info: Optional[Dict[str, Any]],
        dn_info: Optional[Dict[str, Any]],
    ) -> Optional[Tensor]:
        """Compute attention mask.

        Args:
            co_info (Optional[Dict[str, Any]]): collaborative anchors info
            dn_info (Optional[Dict[str, Any]]): denoise anchors info

        Returns:
            Optional[Tensor]: computed attention mask
        """
        if co_info is None and dn_info is None:
            return None

        tgt_size = self.num_queries
        colab_size = co_info["pad_size"] if co_info is not None else 0
        denoise_size = dn_info["pad_size"] if dn_info is not None else 0
        attn_mask = torch.zeros(
            tgt_size + colab_size + denoise_size,
            tgt_size + colab_size + denoise_size,
            device=self.class_embed.weight.device,
        ).bool()

        total_mask = colab_size + denoise_size
        total_mask = total_mask if self.look_at_target else total_mask + tgt_size

        if dn_info is not None:
            num_groups = dn_info["num_groups"]
            double_pad = denoise_size / num_groups
            # match query cannot see the reconstruct
            attn_mask[denoise_size:, :denoise_size] = True
            # reconstruct cannot see each other
            for idx in range(num_groups):
                double_idx = int(double_pad * idx)
                double_idx_p = int(double_pad * (idx + 1))
                attn_mask[double_idx:double_idx_p, double_idx_p:total_mask] = True
                attn_mask[double_idx:double_idx_p, :double_idx] = True

        if co_info is not None:
            attn_mask[denoise_size + colab_size :, : denoise_size + colab_size] = True
            attn_mask[denoise_size : denoise_size + colab_size, denoise_size + colab_size : total_mask] = True

        return attn_mask

    def _predict_spans(self, output: Tensor, reference: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Predict spans.

        Args:
            output (Tensor): content vector from CA block
            reference (Tensor): anchor points

        Returns:
            Tuple[Tensor, Tensor]: predicter spans, shape [batch_size, quary_num, 2] and offsets
        """
        offset = self.span_embed(output)
        reference_before_sigmoid = inverse_sigmoid(reference)
        outputs_coord = offset + reference_before_sigmoid
        outputs_coord = outputs_coord.sigmoid()
        # due to the fact that the usual offset that is calculated above adds up before sigmoids
        final_offset = outputs_coord - reference
        return outputs_coord, final_offset

    # pylint: disable=too-many-locals
    def forward(  # noqa: R0913 C901
        self,
        memory_local: Tensor,
        vid_mask: Tensor,
        vid_pos: Tensor,
        matched_gts: Optional[List[Tensor]],
        anchors_spans: Optional[List[Tensor]],
        encoder_features: Optional[List[Tensor]],
        proposals: Optional[QueryProposalsOutput],
        targets: Optional[Dict[str, Any]] = None,
        video_durations: Optional[Tensor] = None,
    ) -> DetectorOutput:
        """Forward pass of the Transformer.

        Args:
            memory_local (Tensor): Output from the encoder. Shape: [Lv, batch_size, dim]
            vid_mask (Tensor): mask for the source sequence. Shape: [batch_size, Lv]
            vid_pos (Tensor): positional embedding. Shape: [Lv, batch_size, dim]
            matched_gts (Optional[List[Tensor]]): gt spans mathced to selected anchors.
            anchors_spans (Optional[List[Tensor]]): selected anchors.
            encoder_features (Optional[List[Tensor]]): selected encoder features.
            proposals (Optional[QueryProposalsOutput]): proposals info.
            targets (Optional[Dict[str, Any]]): target meta information. Defaults to None.
            video_durations (Optional[Tensor]): video durations in seconds, shape [batch_size].

        Returns:
            DetectorOutput: detector output schema.
        """
        batch_size = memory_local.size(1)
        device = memory_local.device
        if self.use_rpn:
            assert proposals is not None
            ref_points = proposals.refpoint_embed_detach
        else:
            ref_points = self.refpoint_embed.get_reference_points()  # type: ignore

        input_query_label, input_query_span = self.prepare_regular_detr(
            proposals=proposals,
            refpoint_emb=ref_points,
            batch_size=batch_size,
        )
        co_query_label, co_query_span, co_info = self.get_collab_queries(
            matched_gts,  # type: ignore
            anchors_spans,  # type: ignore
            encoder_features,  # type: ignore
            batch_size,
            device,
        )
        dn_query_label, dn_query_span, dn_info = self.get_denoise_queries(targets, batch_size, device)
        attn_mask = self._get_attention_mask(co_info, dn_info)
        input_query_label = torch.cat([dn_query_label, co_query_label, input_query_label], dim=0)  # type: ignore
        input_query_span = torch.cat([dn_query_span, co_query_span, input_query_span], dim=0)  # type: ignore

        hs, reference_points, quality_score = self.decoder(  # noqa: WPS111
            src=memory_local,
            src_key_padding_mask=~vid_mask,
            src_pos=vid_pos,
            content=input_query_label,
            content_mask=attn_mask,
            refpoints_unsigmoid=input_query_span,
        )  # (#layers, #queries, batch_size, dim)

        outputs_coord, offset = self._predict_spans(hs, reference_points)

        # get positive class and coords
        outputs_class: Tensor = self.class_embed(hs)  # (#layers, batch_size, #queries, 1)

        # Length-Routed Multi-Head Classification: rescore via length-conditional head
        route_logits_out = None
        query_length_prior_logits_out = None
        duration_safety_prior_logits_out = None
        duration_route_safety_logits_out = None
        query_prior_safety_logits_out = None
        expert_quality_scores = None
        base_quality_scores = None
        base_class_scores = None
        expert_quality_residuals = None
        uniform_class_scores = None
        uniform_quality_scores = None
        safety_class_scores = None
        safety_quality_scores = None
        route_gate_scores = None
        base_route_gate_scores = None
        sample_route_gate_logits_out = None
        route_delta_scores = None
        if self.use_length_routed_cls:
            base_class_scores = outputs_class.detach()
            # reference_points: (L, B, Q, 2) in sigmoid space
            widths = reference_points[..., 1]  # (L, B, Q)
            # K rescores from K heads (L, B, Q, K)
            if isinstance(self.length_rescoring, nn.ModuleList):
                all_rescores = torch.cat([expert(hs) for expert in self.length_rescoring], dim=-1)
            else:
                all_rescores = self.length_rescoring(hs)
            if self.use_route_safety_adapter:
                safety_adjustment = self.route_safety_adapter(hs.detach())
                safety_adjustment = safety_adjustment - safety_adjustment.mean(
                    dim=-1,
                    keepdim=True,
                )
                if not self.gate_safety_adapter:
                    all_rescores = all_rescores + safety_adjustment
            K = all_rescores.shape[-1]

            if getattr(self, "soft_routing", True):
                # All decoder layers use the final refined width. Detaching it
                # prevents the classification loss from changing span regression.
                if self.route_width_source == "reference":
                    final_widths = reference_points[-1:, ..., 1]
                else:
                    final_widths = outputs_coord[-1:, ..., 1]
                if self.detach_route_width:
                    final_widths = final_widths.detach()
                final_widths = final_widths.expand_as(widths)
                final_widths = final_widths.clamp(min=1e-4, max=1 - 1e-4)
                width_logits = torch.logit(final_widths).unsqueeze(-1)
                if self.use_absolute_duration_routing:
                    if video_durations is None:
                        raise ValueError("absolute duration routing requires video durations")
                    predicted_durations = (
                        final_widths
                        * video_durations.to(final_widths)[None, :, None]
                    ).clamp_min(1e-3)
                    route_coordinate = predicted_durations.log().unsqueeze(-1)
                    prototypes = route_coordinate.new_tensor(
                        self.route_duration_centers,
                    ).log()
                else:
                    route_coordinate = width_logits
                    prototypes = self.get_route_prototypes()
                route_logits = -self.route_beta * (route_coordinate - prototypes) ** 2
                if self.use_adaptive_router:
                    router_input = torch.cat((hs.detach(), width_logits), dim=-1)
                    route_logits = route_logits + self.route_router_scale * self.length_router(router_input)
                if self.use_query_length_prior:
                    co_pad_size = co_info["pad_size"] if co_info is not None else 0
                    dn_pad_size = dn_info["pad_size"] if dn_info is not None else 0
                    main_hs = hs[:, :, co_pad_size + dn_pad_size :, :]
                    # Excluding training-only auxiliary queries keeps this prior
                    # identical between training and inference.
                    query_context = main_hs.detach().mean(dim=2, keepdim=True)
                    if self.query_length_prior_use_width_stats:
                        main_widths = final_widths[
                            :, :, co_pad_size + dn_pad_size :
                        ].detach()
                        if self.query_length_prior_absolute_width_stats:
                            if video_durations is None:
                                raise ValueError(
                                    "video_durations is required for absolute width statistics",
                                )
                            main_widths = main_widths * video_durations.to(main_widths)[None, :, None]
                        if self.query_length_prior_score_weighted_width_stats:
                            if quality_score is None:
                                raise ValueError(
                                    "quality scores are required for score-weighted width statistics",
                                )
                            main_class = outputs_class[
                                :, :, co_pad_size + dn_pad_size :, 0
                            ].detach()
                            main_quality = quality_score[
                                :, :, co_pad_size + dn_pad_size :, 0
                            ].detach()
                            ranking_score = 0.5 * (
                                func.logsigmoid(main_class) + func.logsigmoid(main_quality)
                            )
                            weights = torch.softmax(ranking_score / 0.1, dim=2)
                            weighted_mean = (weights * main_widths).sum(dim=2)
                            weighted_var = (
                                weights * (main_widths - weighted_mean.unsqueeze(2)) ** 2
                            ).sum(dim=2)
                            top_idx = ranking_score.argmax(dim=2, keepdim=True)
                            top_width = main_widths.gather(2, top_idx).squeeze(2)
                            width_stats = torch.stack(
                                (weighted_mean, top_width, weighted_var.sqrt()),
                                dim=-1,
                            ).unsqueeze(2)
                        else:
                            width_stats = torch.stack(
                                (
                                    main_widths.mean(dim=2),
                                    main_widths.amax(dim=2),
                                    main_widths.std(dim=2),
                                ),
                                dim=-1,
                            ).unsqueeze(2)
                        query_context = torch.cat((query_context, width_stats), dim=-1)
                    query_length_prior_logits_out = self.query_length_prior(query_context)
                    prior_scale = self.query_length_prior_scale
                    if self.use_learned_query_prior_safety:
                        query_prior_safety_logits_out = self.query_prior_safety_gate(
                            query_context
                            if self.query_prior_safety_use_context
                            else query_length_prior_logits_out,
                        )
                        safety_probability = torch.sigmoid(query_prior_safety_logits_out)
                        if self.hard_query_prior_safety:
                            hard_safety = (safety_probability >= 0.5).to(safety_probability.dtype)
                            safety_probability = hard_safety + safety_probability - safety_probability.detach()
                        if self.gate_query_prior_with_safety:
                            prior_scale = prior_scale * safety_probability
                    elif self.use_short_safe_query_prior:
                        short_probability = torch.softmax(
                            query_length_prior_logits_out,
                            dim=-1,
                        )[..., :1]
                        prior_scale = prior_scale * (1.0 - short_probability)
                    route_logits = route_logits + (
                        prior_scale * query_length_prior_logits_out
                    )
                if self.use_duration_safety_prior:
                    if video_durations is None or quality_score is None:
                        raise ValueError(
                            "duration safety prior requires video durations and quality scores",
                        )
                    co_pad_size = co_info["pad_size"] if co_info is not None else 0
                    dn_pad_size = dn_info["pad_size"] if dn_info is not None else 0
                    main_hs = hs[:, :, co_pad_size + dn_pad_size :, :]
                    main_widths = final_widths[
                        :, :, co_pad_size + dn_pad_size :
                    ].detach() * video_durations.to(final_widths)[None, :, None]
                    main_class = outputs_class[
                        :, :, co_pad_size + dn_pad_size :, 0
                    ].detach()
                    main_quality = quality_score[
                        :, :, co_pad_size + dn_pad_size :, 0
                    ].detach()
                    ranking_score = 0.5 * (
                        func.logsigmoid(main_class) + func.logsigmoid(main_quality)
                    )
                    weights = torch.softmax(ranking_score / 0.1, dim=2)
                    mean_duration_context = main_hs.detach().mean(dim=2, keepdim=True)
                    weighted_duration_context = (
                        weights.unsqueeze(-1) * main_hs.detach()
                    ).sum(dim=2, keepdim=True)
                    duration_context = (
                        torch.cat(
                            (mean_duration_context, weighted_duration_context),
                            dim=-1,
                        )
                        if self.duration_safety_prior_dual_context
                        else mean_duration_context
                    )
                    weighted_mean = (weights * main_widths).sum(dim=2)
                    weighted_var = (
                        weights * (main_widths - weighted_mean.unsqueeze(2)) ** 2
                    ).sum(dim=2)
                    top_idx = ranking_score.argmax(dim=2, keepdim=True)
                    top_width = main_widths.gather(2, top_idx).squeeze(2)
                    duration_stats = torch.stack(
                        (weighted_mean, top_width, weighted_var.sqrt()),
                        dim=-1,
                    ).unsqueeze(2)
                    duration_context = torch.cat((duration_context, duration_stats), dim=-1)
                    duration_safety_prior_logits_out = self.duration_safety_prior(duration_context)
                    if hasattr(self, "duration_route_safety_prior"):
                        duration_route_safety_logits_out = self.duration_route_safety_prior(
                            duration_context,
                        )
                route_logits_out = route_logits
                if self.uniform_routing:
                    route_w = torch.full_like(route_logits, 1.0 / route_logits.shape[-1])
                else:
                    route_w = torch.softmax(route_logits, dim=-1)
                    duration_route_blend = getattr(self, "duration_direct_routing_blend", 0.0)
                    if duration_route_blend > 0:
                        if duration_safety_prior_logits_out is None or duration_route_blend > 1:
                            raise ValueError(
                                "duration direct routing requires logits and blend in (0, 1]",
                            )
                        duration_route_w = torch.softmax(
                            duration_safety_prior_logits_out
                            / getattr(self, "duration_direct_routing_temperature", 1.0),
                            dim=-1,
                        ).expand_as(route_w)
                        route_w = (
                            (1.0 - duration_route_blend) * route_w
                            + duration_route_blend * duration_route_w
                        )
                    if duration_route_safety_logits_out is not None:
                        safety_probabilities = torch.softmax(
                            duration_route_safety_logits_out,
                            dim=-1,
                        )
                        safety_gate = torch.ones_like(safety_probabilities[..., :1])
                        if hasattr(self, "duration_short_safety_threshold"):
                            safety_gate = safety_gate * torch.sigmoid(
                                (self.duration_short_safety_threshold - safety_probabilities[..., :1])
                                / getattr(self, "duration_short_safety_temperature", 0.05),
                            )
                        if hasattr(self, "duration_middle_safety_threshold"):
                            safety_gate = safety_gate * torch.sigmoid(
                                (
                                    self.duration_middle_safety_threshold
                                    - safety_probabilities[..., 1:2]
                                )
                                / getattr(self, "duration_middle_safety_temperature", 0.05),
                            )
                        uniform_route = torch.full_like(route_w, 1.0 / route_w.shape[-1])
                        route_w = uniform_route + safety_gate * (route_w - uniform_route)
                forced_expert = getattr(self, "forced_route_expert", None)
                if forced_expert is not None:
                    if not 0 <= forced_expert < route_w.shape[-1]:
                        raise ValueError("forced_route_expert is outside the expert range")
                    route_w = torch.zeros_like(route_w)
                    route_w[..., forced_expert] = 1.0
                if self.use_route_safety_adapter and self.gate_safety_adapter:
                    all_rescores = all_rescores + (1.0 - safety_probability) * safety_adjustment
                expert_scale = self.length_rescoring_alpha if self.use_route_alpha else 1.0
                raw_expert_rescores = expert_scale * all_rescores
                expert_rescores = raw_expert_rescores
                if self.center_expert_residuals:
                    expert_rescores = expert_rescores - expert_rescores.mean(dim=-1, keepdim=True)
                    uniform_residual = torch.zeros_like(expert_rescores[..., :1])
                else:
                    uniform_residual = expert_rescores.mean(dim=-1, keepdim=True)
                routed_residual = (route_w * expert_rescores).sum(dim=-1, keepdim=True)
                route_delta = routed_residual - uniform_residual
                if self.use_expert_route_gain:
                    expert_route_gains = func.softplus(self.route_expert_gain_raw)
                    route_gain = (route_w * expert_route_gains).sum(dim=-1, keepdim=True)
                elif self.use_learnable_route_gain:
                    route_gain = func.softplus(self.route_gain_raw)
                else:
                    route_gain = 1.0
                deployed_route_delta = (
                    route_delta.relu()
                    if getattr(self, "positive_route_correction", False)
                    else route_delta
                )
                sample_route_gate_score = None
                if self.use_sample_route_gate and not self.uniform_routing:
                    co_pad_size = co_info["pad_size"] if co_info is not None else 0
                    dn_pad_size = dn_info["pad_size"] if dn_info is not None else 0
                    main_hs = hs[:, :, co_pad_size + dn_pad_size :, :]
                    sample_context = main_hs.detach().mean(dim=2, keepdim=True)
                    if self.sample_route_gate_use_route_stats:
                        main_widths = final_widths[
                            :, :, co_pad_size + dn_pad_size :
                        ].detach()
                        main_long_probability = route_w[
                            :, :, co_pad_size + dn_pad_size :, -1
                        ].detach()
                        main_delta = route_delta[
                            :, :, co_pad_size + dn_pad_size :, 0
                        ].detach()
                        route_stats = torch.stack(
                            (
                                main_widths.mean(dim=2),
                                main_widths.amax(dim=2),
                                main_long_probability.mean(dim=2),
                                main_long_probability.amax(dim=2),
                                main_delta.mean(dim=2),
                                main_delta.std(dim=2),
                            ),
                            dim=-1,
                        ).unsqueeze(2)
                        sample_context = torch.cat((sample_context, route_stats), dim=-1)
                    sample_route_gate_logits_out = self.sample_route_gate(sample_context)
                    sample_route_gate_score = torch.sigmoid(sample_route_gate_logits_out)
                    sample_gate_threshold = getattr(
                        self,
                        "sample_route_gate_threshold",
                        None,
                    )
                    if sample_gate_threshold is not None:
                        sample_gate_temperature = getattr(
                            self,
                            "sample_route_gate_temperature",
                            0.05,
                        )
                        sample_route_gate_score = torch.sigmoid(
                            (sample_route_gate_score - sample_gate_threshold)
                            / sample_gate_temperature,
                        )
                base_route_gate = torch.zeros_like(uniform_residual)
                if self.uniform_routing:
                    residual = uniform_residual
                    route_gate = torch.zeros_like(uniform_residual)
                elif self.use_route_gate:
                    if self.use_monotonic_route_gate:
                        slope = func.softplus(self.route_gate_slope_raw) + self.route_gate_min_slope
                        if self.route_gate_signal == "long_probability":
                            long_probability = route_w[..., -1:].detach().clamp(min=1e-5, max=1 - 1e-5)
                            gate_signal = torch.logit(long_probability)
                        else:
                            gate_signal = width_logits
                        route_gate = torch.sigmoid(slope * (gate_signal - self.route_gate_pivot))
                    elif self.use_query_route_gate and not self.multiply_query_route_gate:
                        assert quality_score is not None
                        gate_features = [
                            width_logits,
                            route_w.detach(),
                            route_delta.detach(),
                            outputs_class.detach(),
                            quality_score.detach(),
                        ]
                        if self.route_gate_use_decoder_features:
                            gate_features.insert(0, hs.detach())
                        gate_input = torch.cat(gate_features, dim=-1)
                        route_gate = torch.sigmoid(self.query_route_gate(gate_input))
                    else:
                        gate_values = torch.sigmoid(self.route_gate_logits)
                        route_gate = (route_w * gate_values).sum(dim=-1, keepdim=True)
                    if self.use_route_confidence_gate:
                        route_entropy = -(route_w * route_w.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
                        route_confidence = (1.0 - route_entropy / math.log(route_w.shape[-1])).clamp(0, 1)
                        route_gate = (2.0 * route_gate * route_confidence.detach().sqrt()).clamp(max=1.0)
                    if self.use_query_route_gate and self.multiply_query_route_gate:
                        assert quality_score is not None
                        gate_features = [
                            width_logits,
                            route_w.detach(),
                            route_delta.detach(),
                            outputs_class.detach(),
                            quality_score.detach(),
                        ]
                        if self.route_gate_use_decoder_features:
                            gate_features.insert(0, hs.detach())
                        gate_input = torch.cat(gate_features, dim=-1)
                        route_gate = route_gate * torch.sigmoid(self.query_route_gate(gate_input))
                    query_gate_threshold = getattr(self, "query_route_gate_threshold", None)
                    if query_gate_threshold is not None:
                        query_gate_temperature = getattr(
                            self,
                            "query_route_gate_temperature",
                            0.05,
                        )
                        route_gate = torch.sigmoid(
                            (route_gate - query_gate_threshold) / query_gate_temperature,
                        )
                    base_route_gate = route_gate
                    if self.use_sample_route_gate:
                        assert sample_route_gate_score is not None
                        route_gate = route_gate * sample_route_gate_score
                    if getattr(self, "signed_route_gate", False):
                        signed_temperature = getattr(
                            self,
                            "signed_route_gate_temperature",
                            1.0,
                        )
                        signed_gate = torch.sigmoid(
                            torch.logit(route_gate.clamp(1e-5, 1 - 1e-5))
                            / signed_temperature,
                        )
                        negative_gain = getattr(self, "signed_route_negative_gain", 1.0)
                        gated_route_delta = (
                            signed_gate * deployed_route_delta.relu()
                            + negative_gain
                            * (1.0 - signed_gate)
                            * deployed_route_delta.clamp(max=0.0)
                        )
                    else:
                        gated_route_delta = route_gate * deployed_route_delta
                    width_safety_threshold = getattr(
                        self,
                        "route_width_safety_threshold",
                        None,
                    )
                    if width_safety_threshold is not None:
                        width_safety = torch.sigmoid(
                            (width_safety_threshold - final_widths.unsqueeze(-1))
                            / getattr(self, "route_width_safety_temperature", 0.05),
                        )
                        gated_route_delta = width_safety * gated_route_delta
                    sample_width_threshold = getattr(
                        self,
                        "sample_width_safety_threshold",
                        None,
                    )
                    sample_width_min_threshold = getattr(
                        self,
                        "sample_width_min_safety_threshold",
                        None,
                    )
                    if (
                        sample_width_threshold is not None
                        or sample_width_min_threshold is not None
                    ):
                        uniform_class = outputs_class + uniform_residual
                        uniform_quality = (
                            quality_score + uniform_residual
                            if quality_score is not None
                            else uniform_class
                        )
                        uniform_rank = 0.5 * (
                            func.logsigmoid(uniform_class.detach())
                            + func.logsigmoid(uniform_quality.detach())
                        )
                        score_weights = torch.softmax(
                            uniform_rank
                            / getattr(self, "sample_width_score_temperature", 0.05),
                            dim=2,
                        )
                        sample_width = (
                            score_weights * final_widths.detach().unsqueeze(-1)
                        ).sum(dim=2, keepdim=True)
                        sample_width_safety = torch.ones_like(sample_width)
                        safety_temperature = getattr(
                            self,
                            "sample_width_safety_temperature",
                            0.05,
                        )
                        if sample_width_threshold is not None:
                            sample_width_safety = sample_width_safety * torch.sigmoid(
                                (sample_width_threshold - sample_width)
                                / safety_temperature,
                            )
                        if sample_width_min_threshold is not None:
                            sample_width_safety = sample_width_safety * torch.sigmoid(
                                (sample_width - sample_width_min_threshold)
                                / safety_temperature,
                            )
                        gated_route_delta = sample_width_safety * gated_route_delta
                    signed_expert_gains = getattr(self, "signed_route_expert_gains", None)
                    if signed_expert_gains is not None:
                        gain_values = route_w.new_tensor(signed_expert_gains)
                        route_gain = route_gain * (route_w * gain_values).sum(
                            dim=-1,
                            keepdim=True,
                        )
                    residual = uniform_residual + route_gain * gated_route_delta
                else:
                    route_gate = torch.ones_like(uniform_residual)
                    base_route_gate = route_gate
                    # Use the learned positive gain by default. Evaluation may
                    # still set direct_route_gain for a controlled sweep.
                    direct_gain = getattr(self, "direct_route_gain", route_gain)
                    duration_gains = getattr(self, "direct_route_duration_gains", None)
                    if duration_gains is not None:
                        gain_values = route_delta.new_tensor(duration_gains)
                        direct_gain = (route_w * gain_values).sum(
                            dim=-1,
                            keepdim=True,
                        )
                    direct_topk = getattr(self, "direct_route_topk", None)
                    if direct_topk is not None:
                        if quality_score is None or not 0 < direct_topk <= route_delta.shape[2]:
                            raise ValueError("direct_route_topk must fit the query dimension")
                        uniform_candidate_score = 0.5 * (
                            func.logsigmoid((outputs_class + uniform_residual).detach())
                            + func.logsigmoid((quality_score + uniform_residual).detach())
                        )
                        kth_score = torch.topk(
                            uniform_candidate_score,
                            direct_topk,
                            dim=2,
                        ).values[..., -1:, :]
                        direct_temperature = getattr(self, "direct_route_topk_temperature", 0.05)
                        route_gate = torch.sigmoid(
                            (uniform_candidate_score - kth_score) / direct_temperature,
                        )
                    confidence_threshold = getattr(
                        self,
                        "direct_route_confidence_threshold",
                        None,
                    )
                    if confidence_threshold is not None:
                        confidence_temperature = getattr(
                            self,
                            "direct_route_confidence_temperature",
                            0.05,
                        )
                        route_confidence = route_w.amax(dim=-1, keepdim=True)
                        route_gate = route_gate * torch.sigmoid(
                            (route_confidence - confidence_threshold)
                            / confidence_temperature,
                        )
                    if self.use_sample_route_gate:
                        assert sample_route_gate_score is not None
                        route_gate = route_gate * sample_route_gate_score
                    residual = uniform_residual + direct_gain * route_gate * route_delta
                selector_logits_out = (
                    duration_route_safety_logits_out
                    if duration_route_safety_logits_out is not None
                    else duration_safety_prior_logits_out
                    if duration_safety_prior_logits_out is not None
                    else query_length_prior_logits_out
                )
                if getattr(self, "selective_uniform_base", False):
                    residual = uniform_residual
                selective_query_gate = 1.0
                selective_topk = getattr(self, "selective_expert_topk", None)
                if selective_topk is not None:
                    if quality_score is None or not 0 < selective_topk <= residual.shape[2]:
                        raise ValueError("selective_expert_topk must fit the query dimension")
                    uniform_candidate_score = 0.5 * (
                        func.logsigmoid((outputs_class + uniform_residual).detach())
                        + func.logsigmoid((quality_score + uniform_residual).detach())
                    )
                    kth_score = torch.topk(
                        uniform_candidate_score,
                        selective_topk,
                        dim=2,
                    ).values[..., -1:, :]
                    selective_temperature = getattr(
                        self,
                        "selective_expert_topk_temperature",
                        0.05,
                    )
                    selective_query_gate = torch.sigmoid(
                        (uniform_candidate_score - kth_score) / selective_temperature,
                    )
                if (
                    not self.uniform_routing
                    and getattr(self, "use_selective_short_expert", False)
                    and selector_logits_out is not None
                ):
                    short_probability = torch.softmax(
                        selector_logits_out,
                        dim=-1,
                    )[..., :1]
                    threshold = getattr(self, "selective_short_threshold", 0.5)
                    temperature = getattr(self, "selective_short_temperature", 0.05)
                    short_gate = torch.sigmoid(
                        (short_probability - threshold) / temperature,
                    ) * selective_query_gate
                    short_expert_residual = expert_rescores[..., :1]
                    short_route_gate = (
                        base_route_gate
                        if getattr(
                            self,
                            "preserve_selective_short_from_sample_gate",
                            False,
                        )
                        else route_gate
                    )
                    short_target_residual = uniform_residual + route_gain * short_route_gate * (
                        short_expert_residual - uniform_residual
                    )
                    residual = residual + short_gate * (short_target_residual - residual)
                if (
                    not self.uniform_routing
                    and getattr(self, "use_selective_middle_expert", False)
                    and selector_logits_out is not None
                ):
                    middle_probability = torch.softmax(
                        selector_logits_out,
                        dim=-1,
                    )[..., 1:2]
                    threshold = getattr(self, "selective_middle_threshold", 0.5)
                    temperature = getattr(self, "selective_middle_temperature", 0.05)
                    if getattr(self, "selective_middle_hard_argmax", False):
                        middle_is_argmax = (
                            selector_logits_out.argmax(dim=-1, keepdim=True) == 1
                        )
                        middle_gate = (
                            middle_is_argmax
                            & (middle_probability >= threshold)
                        ).to(middle_probability.dtype)
                    else:
                        middle_gate = torch.sigmoid(
                            (middle_probability - threshold) / temperature,
                        ) * selective_query_gate
                    middle_gate = middle_gate * getattr(
                        self,
                        "selective_middle_blend",
                        1.0,
                    )
                    middle_expert_residual = expert_rescores[..., 1:2]
                    if getattr(self, "selective_middle_full_expert", False):
                        middle_target_residual = middle_expert_residual
                    else:
                        middle_target_residual = uniform_residual + route_gain * route_gate * (
                            middle_expert_residual - uniform_residual
                        )
                    residual = residual + middle_gate * (middle_target_residual - residual)
                if (
                    not self.uniform_routing
                    and getattr(self, "use_selective_long_expert", False)
                    and selector_logits_out is not None
                ):
                    long_probability = torch.softmax(
                        selector_logits_out,
                        dim=-1,
                    )[..., 2:3]
                    threshold = getattr(self, "selective_long_threshold", 0.5)
                    temperature = getattr(self, "selective_long_temperature", 0.05)
                    long_gate = torch.sigmoid(
                        (long_probability - threshold) / temperature,
                    ) * selective_query_gate
                    long_expert_residual = expert_rescores[..., 2:3]
                    long_target_residual = uniform_residual + route_gain * route_gate * (
                        long_expert_residual - uniform_residual
                    )
                    residual = residual + long_gate * (long_target_residual - residual)
                if (
                    not self.uniform_routing
                    and getattr(self, "preserve_uniform_top1", False)
                ):
                    assert quality_score is not None
                    co_pad_size = co_info["pad_size"] if co_info is not None else 0
                    dn_pad_size = dn_info["pad_size"] if dn_info is not None else 0
                    main_start = co_pad_size + dn_pad_size
                    main_class = outputs_class[:, :, main_start:, :]
                    main_quality = quality_score[:, :, main_start:, :]
                    main_uniform_residual = uniform_residual[:, :, main_start:, :]
                    main_residual = residual[:, :, main_start:, :]
                    uniform_class = main_class + main_uniform_residual
                    uniform_quality = main_quality + main_uniform_residual
                    uniform_rank = 0.5 * (
                        func.logsigmoid(uniform_class)
                        + func.logsigmoid(uniform_quality)
                    )
                    top_idx = uniform_rank.argmax(dim=2, keepdim=True)
                    top_score = torch.gather(uniform_rank, 2, top_idx)
                    margin = getattr(
                        self,
                        "preserve_uniform_top1_margin",
                        1e-6,
                    )
                    ceiling = top_score - margin
                    uniform_class_score = func.logsigmoid(uniform_class)
                    class_top_idx = uniform_class_score.argmax(dim=2, keepdim=True)
                    top_class_score = torch.gather(
                        uniform_class_score,
                        2,
                        class_top_idx,
                    )
                    class_ceiling = top_class_score - margin

                    # Preserve the original winner while allowing safe reranking
                    # of lower-ranked candidates.
                    low = torch.zeros_like(main_residual)
                    high = torch.ones_like(main_residual)
                    delta = main_residual - main_uniform_residual
                    for _ in range(12):
                        midpoint = 0.5 * (low + high)
                        candidate_residual = main_uniform_residual + midpoint * delta
                        candidate_rank = 0.5 * (
                            func.logsigmoid(main_class + candidate_residual)
                            + func.logsigmoid(main_quality + candidate_residual)
                        )
                        candidate_class_score = func.logsigmoid(
                            main_class + candidate_residual,
                        )
                        safe = (
                            (candidate_rank <= ceiling)
                            & (candidate_class_score <= class_ceiling)
                        )
                        low = torch.where(safe, midpoint, low)
                        high = torch.where(safe, high, midpoint)
                    low.scatter_(2, top_idx, 0.0)
                    low.scatter_(2, class_top_idx, 0.0)
                    constrained = main_uniform_residual + low * delta
                    preserve_width_threshold = getattr(
                        self,
                        "preserve_uniform_top1_width_threshold",
                        None,
                    )
                    if preserve_width_threshold is not None:
                        width_weights = torch.softmax(
                            uniform_rank.detach() / 0.05,
                            dim=2,
                        )
                        sample_width = (
                            width_weights
                            * final_widths[:, :, main_start:].detach().unsqueeze(-1)
                        ).sum(dim=2, keepdim=True)
                        preserve_gate = torch.sigmoid(
                            (sample_width - preserve_width_threshold)
                            / getattr(
                                self,
                                "preserve_uniform_top1_width_temperature",
                                0.03,
                            ),
                        )
                        constrained = main_residual + preserve_gate * (
                            constrained - main_residual
                        )
                    residual = torch.cat(
                        (residual[:, :, :main_start, :], constrained),
                        dim=2,
                    )
                if self.route_rescore_target in {"quality", "both"}:
                    assert quality_score is not None
                    # The detached baseline lets this auxiliary output specialize
                    # experts without multiplying gradients into the original head.
                    base_quality_scores = quality_score.detach()
                    # Auxiliary expert objectives retain the unconstrained raw
                    # responses; centering only fixes the deployed mixture gauge.
                    expert_quality_residuals = raw_expert_rescores
                    expert_quality_scores = base_quality_scores + raw_expert_rescores
                    if not self.uniform_routing:
                        uniform_quality_scores = quality_score + uniform_residual
                        safety_quality_scores = (
                            uniform_quality_scores.detach()
                            + (residual - uniform_residual).detach()
                        )
                        if self.route_rescore_target == "quality":
                            uniform_class_scores = outputs_class
                            safety_class_scores = outputs_class.detach()
                    quality_score = quality_score + residual
                    if self.route_rescore_target == "both":
                        if not self.uniform_routing:
                            uniform_class_scores = outputs_class + uniform_residual
                            safety_class_scores = (
                                uniform_class_scores.detach()
                                + (residual - uniform_residual).detach()
                            )
                        outputs_class = outputs_class + residual
                else:
                    if not self.uniform_routing:
                        uniform_class_scores = outputs_class + uniform_residual
                    outputs_class = outputs_class + residual

                # Detached diagnostics are consumed by the Lightning module.
                self.last_route_weights = route_w[-1].detach()
                self.last_route_gate = route_gate[-1].detach()
                if not self.uniform_routing:
                    route_gate_scores = route_gate
                    base_route_gate_scores = base_route_gate
                    route_delta_scores = route_gain * route_delta
            else:
                # ---------- Hard Routing (original fallback) ----------
                t1, t2 = self.length_thresholds
                cls_idx = torch.zeros_like(widths, dtype=torch.long)
                cls_idx[widths >= t1] = 1
                cls_idx[widths >= t2] = 2
                rescore = torch.gather(all_rescores, -1, cls_idx.unsqueeze(-1))
                if self.use_route_alpha:
                    rescore = self.length_rescoring_alpha[cls_idx].unsqueeze(-1) * rescore
                outputs_class = outputs_class + rescore

        return DetectorOutput(
            outputs_class=outputs_class,
            outputs_coord=outputs_coord,
            offsets=offset,
            quality_scores=quality_score,
            route_logits=route_logits_out,
            query_length_prior_logits=query_length_prior_logits_out,
            duration_safety_prior_logits=duration_safety_prior_logits_out,
            query_prior_safety_logits=query_prior_safety_logits_out,
            expert_quality_scores=expert_quality_scores,
            base_quality_scores=base_quality_scores,
            base_class_scores=base_class_scores,
            expert_quality_residuals=expert_quality_residuals,
            uniform_class_scores=uniform_class_scores,
            uniform_quality_scores=uniform_quality_scores,
            safety_class_scores=safety_class_scores,
            safety_quality_scores=safety_quality_scores,
            route_gate_scores=route_gate_scores,
            base_route_gate_scores=base_route_gate_scores,
            sample_route_gate_logits=sample_route_gate_logits_out,
            route_delta_scores=route_delta_scores,
            co_info=co_info,
            dn_info=dn_info,
        )

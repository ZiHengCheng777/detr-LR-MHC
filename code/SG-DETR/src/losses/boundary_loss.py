"""Boundary Discrimination Loss (simplified from CVA's CBD loss).

Contrasts boundary features against adjacent background features
to improve temporal boundary precision.
"""

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class BoundaryDiscriminationLoss(nn.Module):
    """Contrastive loss on boundary vs background features."""

    def __init__(
        self,
        d_model: int = 256,
        proj_dim: int = 128,
        n_adj: int = 4,
        n_hard: int = 2,
        temperature: float = 0.07,
    ) -> None:
        """Initialize BoundaryDiscriminationLoss.

        Args:
            d_model: dimension of encoder features
            proj_dim: dimension of contrastive projection
            n_adj: number of adjacent background clips as negatives
            n_hard: number of hard negatives mined by similarity
            temperature: InfoNCE temperature
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim),
        )
        self.n_adj = n_adj
        self.n_hard = n_hard
        self.temperature = temperature

    def forward(self, vid_features: Tensor, targets: dict) -> dict:
        """Compute boundary discrimination loss.

        Args:
            vid_features: encoder features [L, B, D]
            targets: dict with 'span_labels' containing GT spans

        Returns:
            dict with 'loss_boundary' scalar
        """
        L, B, D = vid_features.shape
        device = vid_features.device

        # project all features: [L, B, D] -> [B, L, proj_dim]
        feat = self.proj(vid_features.permute(1, 0, 2))  # [B, L, proj_dim]
        feat = F.normalize(feat, dim=-1)

        total_loss = torch.tensor(0.0, device=device)
        count = 0

        for b in range(B):
            spans = targets["span_labels"][b]["spans"]  # [N_gt, 2] (center, width)
            for span in spans:
                center, width = span[0].item(), span[1].item()
                start_idx = int((center - width / 2) * L)
                end_idx = int((center + width / 2) * L)
                start_idx = max(0, min(start_idx, L - 1))
                end_idx = max(0, min(end_idx, L - 1))
                if start_idx == end_idx:
                    continue

                gt_indices = set(range(start_idx, end_idx + 1))
                bg_indices = [i for i in range(L) if i not in gt_indices]
                if len(bg_indices) < 2:
                    continue

                for boundary_idx in [start_idx, end_idx]:
                    anchor = feat[b, boundary_idx]  # [proj_dim]

                    # positive: the other boundary
                    pos_idx = end_idx if boundary_idx == start_idx else start_idx
                    pos = feat[b, pos_idx]  # [proj_dim]

                    # negatives: adjacent background
                    adj_negs = [
                        i for i in bg_indices
                        if abs(i - boundary_idx) <= self.n_adj
                    ]

                    # negatives: hard (most similar background)
                    remaining = [i for i in bg_indices if i not in adj_negs]
                    if remaining and self.n_hard > 0:
                        rem_feat = feat[b, remaining]  # [N_rem, proj_dim]
                        sims = torch.mv(rem_feat, anchor)
                        topk = min(self.n_hard, len(remaining))
                        _, hard_idx = sims.topk(topk)
                        hard_negs = [remaining[j] for j in hard_idx.tolist()]
                    else:
                        hard_negs = []

                    neg_indices = adj_negs + hard_negs
                    if not neg_indices:
                        continue

                    neg_feat = feat[b, neg_indices]  # [N_neg, proj_dim]

                    # InfoNCE
                    pos_sim = torch.dot(anchor, pos) / self.temperature
                    neg_sims = torch.mv(neg_feat, anchor) / self.temperature
                    logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])
                    labels = torch.zeros(1, dtype=torch.long, device=device)
                    total_loss = total_loss + F.cross_entropy(logits.unsqueeze(0), labels)
                    count += 1

        if count > 0:
            total_loss = total_loss / count
        return {"loss_boundary": total_loss}

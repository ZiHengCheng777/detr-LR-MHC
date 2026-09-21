"""Joint duration and expert-utility supervision for the sample router."""

from typing import Any, Dict, List

from src.losses.losses import SetCriterion
from src.losses.sample_expert_utility import sample_expert_utility_loss


class JointRoutePriorOnlySetCriterion(SetCriterion):
    """Calibrate a mutually exclusive soft prior from two complementary targets."""

    def __init__(
        self,
        *args: Any,
        sample_expert_utility_temperature: float = 0.02,
        sample_expert_utility_hard_negatives: int = 16,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sample_expert_utility_temperature = sample_expert_utility_temperature
        self.sample_expert_utility_hard_negatives = sample_expert_utility_hard_negatives

    def forward(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        meta: List[Dict[str, Any]],
        matching: Dict[str, Any],
    ) -> Dict[str, Any]:
        losses = self._query_length_prior_loss(outputs, targets, meta)
        required = {
            "query_length_prior_logits",
            "expert_quality_residuals",
            "base_class_scores",
            "base_quality_scores",
        }
        if required.issubset(outputs):
            losses["loss_sample_expert_utility"] = sample_expert_utility_loss(
                outputs,
                matching["positive"]["indices"],
                temperature=self.sample_expert_utility_temperature,
                hard_negatives=self.sample_expert_utility_hard_negatives,
            )
        return losses

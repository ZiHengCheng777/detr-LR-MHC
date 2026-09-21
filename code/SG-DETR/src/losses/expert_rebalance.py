"""Expert-only criterion for target-assigned LR-MHC calibration."""

from typing import Any, Dict, List

from src.losses.losses import SetCriterion, create_targets_k_repeats


class ExpertRebalanceOnlySetCriterion(SetCriterion):
    """Train expert heads without changing the baseline model or router."""

    def forward(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        meta: List[Dict[str, Any]],
        matching: Dict[str, Any],
    ) -> Dict[str, Any]:
        del meta
        retrieval_targets = (
            targets if self.one2one else create_targets_k_repeats(targets, self.target_repeat)
        )
        indices = matching["positive"]["indices"]
        losses = self._expert_quality_loss(outputs, retrieval_targets, indices)
        losses.update(self._expert_specialization_loss(outputs, retrieval_targets, indices))
        return losses

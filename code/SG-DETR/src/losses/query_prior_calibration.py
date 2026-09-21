"""Length-prior-only calibration criterion."""

from typing import Any, Dict, List

from src.losses.losses import SetCriterion


class QueryPriorOnlySetCriterion(SetCriterion):
    """Train only the sample-level soft route prior."""

    def forward(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        meta: List[Dict[str, Any]],
        matching: Dict[str, Any],
    ) -> Dict[str, Any]:
        del matching
        return self._query_length_prior_loss(outputs, targets, meta)

"""Isolated counterfactual safety calibration for LR-MHC experts."""

from typing import Any, Dict, List

from src.losses.losses import SetCriterion


class NonLongSafetyOnlySetCriterion(SetCriterion):
    """Return only the non-long routing safety objective during calibration."""

    def forward(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        meta: List[Dict[str, Any]],
        matching: Dict[str, Any],
    ) -> Dict[str, Any]:
        outputs_without_aux = {key: value for key, value in outputs.items() if key != "aux_outputs"}
        return self._nonlong_route_safety_loss(
            outputs_without_aux,
            matching["positive"]["indices"],
            meta,
        )

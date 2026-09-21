"""Candidate safety-gate-only calibration criterion."""

from typing import Any, Dict, List

from src.losses.losses import SetCriterion


class QueryGateOnlySetCriterion(SetCriterion):
    """Train only the candidate gate from detached correction utility labels."""

    def forward(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        meta: List[Dict[str, Any]],
        matching: Dict[str, Any],
    ) -> Dict[str, Any]:
        del targets, meta
        return self._route_gate_oracle_loss(
            outputs,
            matching["positive"]["indices"],
        )

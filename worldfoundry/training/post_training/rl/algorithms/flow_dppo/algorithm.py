"""Loss-only Flow-DPPO stage consumed by the shared flow-policy learner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import ClassVar

from ...contracts import FlowReplayResult
from ..stage import AnchorField, StageAnchor, StageLoss
from .objective import flow_dppo_policy_loss


@dataclass(frozen=True, slots=True)
class FlowDPPOStageAlgorithm:
    """KL-advantage masking math with explicit old-mean anchoring."""

    kl_mask_threshold: float = 1.0e-5
    add_kl_coefficient: bool = True

    name: ClassVar[str] = "flow-dppo"
    anchor_fields: ClassVar[frozenset[AnchorField]] = frozenset(
        {AnchorField.OLD_LOG_PROBS, AnchorField.OLD_TRANSITION_MEANS}
    )
    supports_multi_update: ClassVar[bool] = True
    supports_update_step_mask: ClassVar[bool] = False
    requires_reference_replay: ClassVar[bool] = False

    def __post_init__(self) -> None:
        threshold = float(self.kl_mask_threshold)
        if not isfinite(threshold) or threshold < 0:
            raise ValueError("kl_mask_threshold must be finite and non-negative")
        if not isinstance(self.add_kl_coefficient, bool):
            raise TypeError("add_kl_coefficient must be a bool")
        object.__setattr__(self, "kl_mask_threshold", threshold)

    @property
    def state_fields(self) -> Mapping[str, object]:
        return {
            "kl_mask_threshold": self.kl_mask_threshold,
            "add_kl_coefficient": self.add_kl_coefficient,
        }

    def loss(
        self,
        replay: FlowReplayResult,
        anchor: StageAnchor,
        reference: FlowReplayResult | None = None,
        *,
        optimizer_step: int = 0,
    ) -> StageLoss:
        del reference, optimizer_step
        if anchor.old_transition_means is None:
            raise ValueError("Flow-DPPO requires frozen old transition means")
        objective = flow_dppo_policy_loss(
            replay.log_probs,
            anchor.old_log_probs,
            replay.transition_means,
            anchor.old_transition_means,
            replay.transition_scales,
            anchor.advantages,
            kl_mask_threshold=self.kl_mask_threshold,
            add_kl_coefficient=self.add_kl_coefficient,
        )
        return StageLoss(
            loss=objective.loss,
            ratio=objective.ratio,
            metrics={
                "approx_kl": objective.approx_kl,
                "old_policy_kl": objective.old_policy_kl.mean(),
                "masked_fraction": objective.masked_fraction,
                "positive_masked_fraction": objective.positive_masked_fraction,
                "negative_masked_fraction": objective.negative_masked_fraction,
            },
        )


__all__ = ["FlowDPPOStageAlgorithm"]

"""Loss-only contract between flow-policy algorithms and the learner runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from ..contracts import FlowReplayResult


class AnchorField(str, Enum):
    """Frozen pre-update values an algorithm requires during learner replay."""

    OLD_LOG_PROBS = "old_log_probs"
    OLD_TRANSITION_MEANS = "old_transition_means"


@dataclass(frozen=True, slots=True)
class StageAnchor:
    """A microbatch view of immutable values captured before any update."""

    old_log_probs: object
    advantages: object
    old_transition_means: object | None = None
    update_step_mask: object | None = None


@dataclass(frozen=True, slots=True)
class StageLoss:
    """Algorithm loss and unreduced ratio returned to the learner runtime."""

    loss: object
    ratio: object
    metrics: Mapping[str, object]


@runtime_checkable
class StageAlgorithm(Protocol):
    """A stateless policy-loss stage with explicit frozen-anchor needs.

    Implementations may validate and retain loss hyperparameters, but must not
    own rollout, replay, optimizers, distributed collectives, or checkpoints.
    """

    name: str
    anchor_fields: frozenset[AnchorField]
    supports_multi_update: bool
    supports_update_step_mask: bool
    requires_reference_replay: bool

    @property
    def state_fields(self) -> Mapping[str, object]:
        """Stable loss configuration persisted by the optimizer state machine."""

    def loss(
        self,
        replay: FlowReplayResult,
        anchor: StageAnchor,
        reference: FlowReplayResult | None = None,
        *,
        optimizer_step: int = 0,
    ) -> StageLoss:
        """Compute the policy loss for one replay microbatch."""


__all__ = [
    "AnchorField",
    "StageAlgorithm",
    "StageAnchor",
    "StageLoss",
]

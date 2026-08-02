"""Loss-only Flow-GRPO stage consumed by the shared flow-policy learner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import ClassVar

from worldfoundry.training.recipes.post_training.common import (
    scheduled_clip_range,
    validate_clip_schedule,
)

from ...contracts import FlowReplayResult
from ..stage import AnchorField, StageAnchor, StageLoss
from .objective import clipped_policy_loss


@dataclass(frozen=True, slots=True)
class FlowGRPOStageAlgorithm:
    """Clipped grouped policy objective without execution responsibilities."""

    clip_range: float = 1.0e-4
    clip_schedule: str = "constant"
    clip_schedule_steps: int | None = None

    name: ClassVar[str] = "flow-grpo"
    anchor_fields: ClassVar[frozenset[AnchorField]] = frozenset({AnchorField.OLD_LOG_PROBS})
    supports_multi_update: ClassVar[bool] = True
    supports_update_step_mask: ClassVar[bool] = False
    requires_reference_replay: ClassVar[bool] = False

    def __post_init__(self) -> None:
        resolved = float(self.clip_range)
        if not isfinite(resolved) or not 0 < resolved < 1:
            raise ValueError("clip_range must be finite and in (0,1)")
        schedule, schedule_steps = validate_clip_schedule(
            self.clip_schedule,
            self.clip_schedule_steps,
        )
        object.__setattr__(self, "clip_range", resolved)
        object.__setattr__(self, "clip_schedule", schedule)
        object.__setattr__(self, "clip_schedule_steps", schedule_steps)

    @property
    def state_fields(self) -> Mapping[str, object]:
        return {
            "clip_range": self.clip_range,
            "clip_schedule": self.clip_schedule,
            "clip_schedule_steps": self.clip_schedule_steps,
        }

    def loss(
        self,
        replay: FlowReplayResult,
        anchor: StageAnchor,
        reference: FlowReplayResult | None = None,
        *,
        optimizer_step: int = 0,
    ) -> StageLoss:
        del reference
        active_clip_range = scheduled_clip_range(
            self.clip_range,
            schedule=self.clip_schedule,
            schedule_steps=self.clip_schedule_steps,
            optimizer_step=optimizer_step,
        )
        objective = clipped_policy_loss(
            replay.log_probs,
            anchor.old_log_probs,
            anchor.advantages,
            clip_range=active_clip_range,
            step_mask=anchor.update_step_mask,
        )
        return StageLoss(
            loss=objective.loss,
            ratio=objective.ratio,
            metrics={
                "approx_kl": objective.approx_kl,
                "clip_fraction": objective.clip_fraction,
                "lower_clip_fraction": objective.lower_clip_fraction,
                "upper_clip_fraction": objective.upper_clip_fraction,
                "clip_range": replay.log_probs.new_tensor(active_clip_range),
            },
        )


__all__ = ["FlowGRPOStageAlgorithm"]

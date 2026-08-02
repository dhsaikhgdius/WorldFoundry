"""Loss-only GRPO-Guard stage for the shared flow-policy learner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import ClassVar

from ...contracts import FlowReplayResult
from ..stage import AnchorField, StageAnchor, StageLoss
from .objective import grpo_guard_policy_loss


@dataclass(frozen=True, slots=True)
class GRPOGuardStageAlgorithm:
    """Reverse-SDE mean-bias objective without execution responsibilities."""

    clip_range: float = 1.0e-4
    advantage_clip_max: float = 5.0

    name: ClassVar[str] = "grpo-guard"
    anchor_fields: ClassVar[frozenset[AnchorField]] = frozenset(
        {AnchorField.OLD_LOG_PROBS, AnchorField.OLD_TRANSITION_MEANS}
    )
    supports_multi_update: ClassVar[bool] = True
    supports_update_step_mask: ClassVar[bool] = False
    requires_reference_replay: ClassVar[bool] = False

    def __post_init__(self) -> None:
        clip = float(self.clip_range)
        advantage_clip = float(self.advantage_clip_max)
        if not isfinite(clip) or not 0 < clip < 1:
            raise ValueError("clip_range must be finite and in (0,1)")
        if not isfinite(advantage_clip) or advantage_clip <= 0:
            raise ValueError("advantage_clip_max must be finite and positive")
        object.__setattr__(self, "clip_range", clip)
        object.__setattr__(self, "advantage_clip_max", advantage_clip)

    @property
    def state_fields(self) -> Mapping[str, object]:
        return {
            "clip_range": self.clip_range,
            "advantage_clip_max": self.advantage_clip_max,
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
            raise ValueError("GRPO-Guard requires frozen old transition means")
        if replay.std_dev_t is None or replay.sqrt_dt is None:
            raise ValueError("GRPO-Guard replay requires std_dev_t and sqrt_dt")
        objective = grpo_guard_policy_loss(
            replay.log_probs,
            anchor.old_log_probs,
            replay.transition_means,
            anchor.old_transition_means,
            replay.std_dev_t,
            replay.sqrt_dt,
            anchor.advantages,
            clip_range=self.clip_range,
            advantage_clip_max=self.advantage_clip_max,
        )
        return StageLoss(
            loss=objective.loss,
            ratio=objective.ratio,
            metrics={
                "ppo_kl": objective.ppo_kl,
                "approx_kl": objective.approx_kl,
                "ratio_mean_bias": objective.ratio_mean_bias.mean(),
                "scale": objective.scale,
                "sqrt_dt_mean": objective.sqrt_dt_mean,
                "clip_fraction": objective.clip_fraction,
                "lower_clip_fraction": objective.lower_clip_fraction,
                "upper_clip_fraction": objective.upper_clip_fraction,
            },
        )


__all__ = ["GRPOGuardStageAlgorithm"]

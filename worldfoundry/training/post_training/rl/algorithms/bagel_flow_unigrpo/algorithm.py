"""Bagel Flow-UniGRPO loss stage for the shared flow-policy learner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import ClassVar

from ...contracts import FlowReplayResult
from ..stage import AnchorField, StageAnchor, StageLoss
from .objective import bagel_flow_unigrpo_loss


@dataclass(frozen=True, slots=True)
class BagelFlowUniGRPOStageAlgorithm:
    clip_range: float = 1.0e-4
    velocity_mse_weight: float = 1.0
    ratio_norm: bool = False
    grad_reweight: bool = False

    name: ClassVar[str] = "bagel-flow-unigrpo"
    supports_multi_update: ClassVar[bool] = True
    supports_update_step_mask: ClassVar[bool] = False
    requires_reference_replay: ClassVar[bool] = True

    @property
    def anchor_fields(self) -> frozenset[AnchorField]:
        fields = {AnchorField.OLD_LOG_PROBS}
        if self.ratio_norm:
            fields.add(AnchorField.OLD_TRANSITION_MEANS)
        return frozenset(fields)

    def __post_init__(self) -> None:
        clip_range = float(self.clip_range)
        velocity_mse_weight = float(self.velocity_mse_weight)
        if not isfinite(clip_range) or not 0 < clip_range < 1:
            raise ValueError("clip_range must be finite and in (0,1)")
        if not isfinite(velocity_mse_weight) or velocity_mse_weight <= 0:
            raise ValueError("velocity_mse_weight must be finite and positive")
        if not isinstance(self.ratio_norm, bool) or not isinstance(
            self.grad_reweight,
            bool,
        ):
            raise TypeError("ratio_norm and grad_reweight must be bool values")
        if self.grad_reweight and not self.ratio_norm:
            raise ValueError("grad_reweight requires ratio_norm")
        object.__setattr__(self, "clip_range", clip_range)
        object.__setattr__(self, "velocity_mse_weight", velocity_mse_weight)

    @property
    def state_fields(self) -> Mapping[str, object]:
        return {
            "clip_range": self.clip_range,
            "velocity_mse_weight": self.velocity_mse_weight,
            "ratio_norm": self.ratio_norm,
            "grad_reweight": self.grad_reweight,
        }

    def loss(
        self,
        replay: FlowReplayResult,
        anchor: StageAnchor,
        reference: FlowReplayResult | None = None,
        *,
        optimizer_step: int = 0,
    ) -> StageLoss:
        del optimizer_step
        if reference is None:
            raise ValueError("Bagel Flow-UniGRPO requires a frozen reference replay")
        if replay.velocities is None or reference.velocities is None:
            raise ValueError("Bagel Flow-UniGRPO replay must retain velocity predictions")
        objective = bagel_flow_unigrpo_loss(
            replay.log_probs,
            anchor.old_log_probs,
            replay.transition_means,
            anchor.old_transition_means,
            replay.transition_scales,
            replay.sqrt_dt,
            anchor.advantages,
            replay.velocities,
            reference.velocities,
            clip_range=self.clip_range,
            velocity_mse_weight=self.velocity_mse_weight,
            ratio_norm=self.ratio_norm,
            grad_reweight=self.grad_reweight,
        )
        metrics = {
            "surrogate_loss": objective.surrogate_loss,
            "velocity_mse": objective.velocity_mse,
            "approx_kl": objective.approx_kl,
            "clip_fraction": objective.clip_fraction,
            "lower_clip_fraction": objective.lower_clip_fraction,
            "upper_clip_fraction": objective.upper_clip_fraction,
            "raw_ratio_mean": objective.raw_ratio.mean(),
        }
        if objective.ratio_mean_bias is not None:
            metrics["ratio_mean_bias"] = objective.ratio_mean_bias.mean()
        return StageLoss(
            loss=objective.loss,
            ratio=objective.ratio,
            metrics=metrics,
        )


__all__ = ["BagelFlowUniGRPOStageAlgorithm"]

"""Stateless algorithm stages for the shared token-policy learner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import ClassVar, Protocol, runtime_checkable

import torch

from worldfoundry.training.recipes.post_training.common import (
    scheduled_clip_range,
    validate_clip_schedule,
)

from .objectives import (
    TokenObjective,
    token_cppo_objective,
    token_dppo_objective,
    token_drpo_objective,
    token_grpo_objective,
    token_gspo_objective,
)
from .packing import expand_sequence_values
from .reduction import (
    SUM_REDUCTIONS,
    TOKEN_MEAN,
    TOKEN_REDUCTIONS,
    reduce_token_losses,
    reduction_weight,
    validate_reduction,
)


@dataclass(frozen=True, slots=True)
class TokenPolicyStageLoss:
    """One objective numerator with its exact averaging denominator."""

    numerator: torch.Tensor
    denominator: int
    ratio: torch.Tensor
    metrics: Mapping[str, torch.Tensor]

    @property
    def loss(self) -> torch.Tensor:
        if self.denominator <= 0:
            return self.numerator
        return self.numerator / float(self.denominator)


@runtime_checkable
class TokenPolicyStage(Protocol):
    name: str
    supports_multi_update: bool

    @property
    def state_fields(self) -> Mapping[str, object]: ...

    def loss_weight(self, lengths: torch.Tensor) -> int: ...

    def loss(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        lengths: torch.Tensor,
        *,
        optimizer_step: int = 0,
    ) -> TokenPolicyStageLoss: ...


def _token_stage_loss(
    objective: TokenObjective,
    lengths: torch.Tensor,
    *,
    reduction: str,
    horizon: int,
) -> TokenPolicyStageLoss:
    reduced = reduce_token_losses(
        objective.losses,
        lengths,
        mode=reduction,
        horizon=horizon,
    )
    return TokenPolicyStageLoss(
        numerator=reduced.numerator,
        denominator=reduced.denominator,
        ratio=objective.ratio,
        metrics=objective.metrics,
    )


def _expanded_advantages(
    advantages: torch.Tensor,
    lengths: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    return expand_sequence_values(
        advantages,
        lengths,
        dtype=reference.dtype,
        device=reference.device,
    )


@dataclass(frozen=True, slots=True)
class TokenGRPOStage:
    """Per-token clipped GRPO with token- or sequence-level averaging."""

    clip_range: float = 1.0e-4
    clip_range_high: float | None = None
    clip_schedule: str = "constant"
    clip_schedule_steps: int | None = None
    reduction: str = TOKEN_MEAN
    horizon: int = 8192

    name: ClassVar[str] = "token-grpo"
    supports_multi_update: ClassVar[bool] = True

    def __post_init__(self) -> None:
        reduction, horizon = validate_reduction(
            self.reduction,
            allowed=TOKEN_REDUCTIONS,
            horizon=self.horizon,
        )
        clip = float(self.clip_range)
        high = None if self.clip_range_high is None else float(self.clip_range_high)
        if not isfinite(clip) or not 0 < clip < 1:
            raise ValueError("clip_range must be finite and in (0,1)")
        if high is not None and (not isfinite(high) or high <= 0):
            raise ValueError("clip_range_high must be finite and positive")
        schedule, schedule_steps = validate_clip_schedule(
            self.clip_schedule,
            self.clip_schedule_steps,
        )
        object.__setattr__(self, "clip_range", clip)
        object.__setattr__(self, "clip_range_high", high)
        object.__setattr__(self, "clip_schedule", schedule)
        object.__setattr__(self, "clip_schedule_steps", schedule_steps)
        object.__setattr__(self, "reduction", reduction)
        object.__setattr__(self, "horizon", horizon)

    @property
    def state_fields(self) -> Mapping[str, object]:
        return {
            "clip_range": self.clip_range,
            "clip_range_high": self.clip_range_high,
            "clip_schedule": self.clip_schedule,
            "clip_schedule_steps": self.clip_schedule_steps,
            "reduction": self.reduction,
            "horizon": self.horizon,
        }

    def loss_weight(self, lengths: torch.Tensor) -> int:
        return reduction_weight(lengths, mode=self.reduction)

    def loss(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        lengths: torch.Tensor,
        *,
        optimizer_step: int = 0,
    ) -> TokenPolicyStageLoss:
        active_clip_range = scheduled_clip_range(
            self.clip_range,
            schedule=self.clip_schedule,
            schedule_steps=self.clip_schedule_steps,
            optimizer_step=optimizer_step,
        )
        active_clip_range_high = (
            None
            if self.clip_range_high is None
            else scheduled_clip_range(
                self.clip_range_high,
                schedule=self.clip_schedule,
                schedule_steps=self.clip_schedule_steps,
                optimizer_step=optimizer_step,
            )
        )
        objective = token_grpo_objective(
            new_log_probs,
            old_log_probs,
            _expanded_advantages(advantages, lengths, new_log_probs),
            clip_range=active_clip_range,
            clip_range_high=active_clip_range_high,
        )
        result = _token_stage_loss(
            objective,
            lengths,
            reduction=self.reduction,
            horizon=self.horizon,
        )
        return TokenPolicyStageLoss(
            numerator=result.numerator,
            denominator=result.denominator,
            ratio=result.ratio,
            metrics={
                **result.metrics,
                "clip_range": new_log_probs.new_tensor(active_clip_range),
            },
        )


@dataclass(frozen=True, slots=True)
class TokenGSPOStage:
    """Sequence-level clipped GSPO over mean packed-token log-ratios."""

    clip_range: float = 3.0e-4
    clip_range_high: float | None = None
    clip_schedule: str = "constant"
    clip_schedule_steps: int | None = None

    name: ClassVar[str] = "token-gspo"
    supports_multi_update: ClassVar[bool] = True

    def __post_init__(self) -> None:
        clip = float(self.clip_range)
        high = None if self.clip_range_high is None else float(self.clip_range_high)
        if not isfinite(clip) or not 0 < clip < 1:
            raise ValueError("clip_range must be finite and in (0,1)")
        if high is not None and (not isfinite(high) or high <= 0):
            raise ValueError("clip_range_high must be finite and positive")
        schedule, schedule_steps = validate_clip_schedule(
            self.clip_schedule,
            self.clip_schedule_steps,
        )
        object.__setattr__(self, "clip_range", clip)
        object.__setattr__(self, "clip_range_high", high)
        object.__setattr__(self, "clip_schedule", schedule)
        object.__setattr__(self, "clip_schedule_steps", schedule_steps)

    @property
    def state_fields(self) -> Mapping[str, object]:
        return {
            "clip_range": self.clip_range,
            "clip_range_high": self.clip_range_high,
            "clip_schedule": self.clip_schedule,
            "clip_schedule_steps": self.clip_schedule_steps,
        }

    def loss_weight(self, lengths: torch.Tensor) -> int:
        return int((lengths > 0).sum().item())

    def loss(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        lengths: torch.Tensor,
        *,
        optimizer_step: int = 0,
    ) -> TokenPolicyStageLoss:
        active_clip_range = scheduled_clip_range(
            self.clip_range,
            schedule=self.clip_schedule,
            schedule_steps=self.clip_schedule_steps,
            optimizer_step=optimizer_step,
        )
        active_clip_range_high = (
            None
            if self.clip_range_high is None
            else scheduled_clip_range(
                self.clip_range_high,
                schedule=self.clip_schedule,
                schedule_steps=self.clip_schedule_steps,
                optimizer_step=optimizer_step,
            )
        )
        objective = token_gspo_objective(
            new_log_probs,
            old_log_probs,
            advantages,
            lengths,
            clip_range=active_clip_range,
            clip_range_high=active_clip_range_high,
        )
        return TokenPolicyStageLoss(
            numerator=objective.losses.sum(),
            denominator=int(objective.losses.shape[0]),
            ratio=objective.ratio,
            metrics={
                **objective.metrics,
                "clip_range": new_log_probs.new_tensor(active_clip_range),
            },
        )


@dataclass(frozen=True, slots=True)
class TokenDPPOStage:
    """Uniform Binary-TV hard-mask stage."""

    delta: float = 0.15
    reduction: str = TOKEN_MEAN
    horizon: int = 8192

    name: ClassVar[str] = "token-dppo"
    supports_multi_update: ClassVar[bool] = True

    def __post_init__(self) -> None:
        delta = float(self.delta)
        if not isfinite(delta) or delta <= 0:
            raise ValueError("delta must be finite and positive")
        reduction, horizon = validate_reduction(
            self.reduction,
            allowed=SUM_REDUCTIONS,
            horizon=self.horizon,
        )
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "reduction", reduction)
        object.__setattr__(self, "horizon", horizon)

    @property
    def state_fields(self) -> Mapping[str, object]:
        return {
            "delta": self.delta,
            "reduction": self.reduction,
            "horizon": self.horizon,
        }

    def loss_weight(self, lengths: torch.Tensor) -> int:
        return reduction_weight(lengths, mode=self.reduction)

    def loss(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        lengths: torch.Tensor,
        *,
        optimizer_step: int = 0,
    ) -> TokenPolicyStageLoss:
        del optimizer_step
        objective = token_dppo_objective(
            new_log_probs,
            old_log_probs,
            _expanded_advantages(advantages, lengths, new_log_probs),
            delta=self.delta,
        )
        return _token_stage_loss(
            objective,
            lengths,
            reduction=self.reduction,
            horizon=self.horizon,
        )


@dataclass(frozen=True, slots=True)
class TokenDRPOStage:
    """Smooth advantage-weighted quadratic stage."""

    epsilon: float = 12.5
    mu_weighted: bool = True
    reduction: str = TOKEN_MEAN
    horizon: int = 8192

    name: ClassVar[str] = "token-drpo"
    supports_multi_update: ClassVar[bool] = True

    def __post_init__(self) -> None:
        epsilon = float(self.epsilon)
        if not isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if not isinstance(self.mu_weighted, bool):
            raise TypeError("mu_weighted must be a bool")
        reduction, horizon = validate_reduction(
            self.reduction,
            allowed=SUM_REDUCTIONS,
            horizon=self.horizon,
        )
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "reduction", reduction)
        object.__setattr__(self, "horizon", horizon)

    @property
    def state_fields(self) -> Mapping[str, object]:
        return {
            "epsilon": self.epsilon,
            "mu_weighted": self.mu_weighted,
            "reduction": self.reduction,
            "horizon": self.horizon,
        }

    def loss_weight(self, lengths: torch.Tensor) -> int:
        return reduction_weight(lengths, mode=self.reduction)

    def loss(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        lengths: torch.Tensor,
        *,
        optimizer_step: int = 0,
    ) -> TokenPolicyStageLoss:
        del optimizer_step
        objective = token_drpo_objective(
            new_log_probs,
            old_log_probs,
            _expanded_advantages(advantages, lengths, new_log_probs),
            epsilon=self.epsilon,
            mu_weighted=self.mu_weighted,
        )
        return _token_stage_loss(
            objective,
            lengths,
            reduction=self.reduction,
            horizon=self.horizon,
        )


@dataclass(frozen=True, slots=True)
class TokenCPPOStage:
    """Position-weighted cumulative-prefix Binary-TV stage."""

    delta: float = 0.2
    w_min: float = 0.8
    delta_b: float = 0.02
    reduction: str = TOKEN_MEAN
    horizon: int = 8192

    name: ClassVar[str] = "token-cppo"
    supports_multi_update: ClassVar[bool] = True

    def __post_init__(self) -> None:
        delta = float(self.delta)
        w_min = float(self.w_min)
        delta_b = float(self.delta_b)
        if not isfinite(delta) or delta <= 0:
            raise ValueError("delta must be finite and positive")
        if not isfinite(w_min) or not 0 < w_min <= 1:
            raise ValueError("w_min must be finite and in (0,1]")
        if not isfinite(delta_b) or delta_b < 0:
            raise ValueError("delta_b must be finite and non-negative")
        reduction, horizon = validate_reduction(
            self.reduction,
            allowed=SUM_REDUCTIONS,
            horizon=self.horizon,
        )
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "w_min", w_min)
        object.__setattr__(self, "delta_b", delta_b)
        object.__setattr__(self, "reduction", reduction)
        object.__setattr__(self, "horizon", horizon)

    @property
    def state_fields(self) -> Mapping[str, object]:
        return {
            "delta": self.delta,
            "w_min": self.w_min,
            "delta_b": self.delta_b,
            "reduction": self.reduction,
            "horizon": self.horizon,
        }

    def loss_weight(self, lengths: torch.Tensor) -> int:
        return reduction_weight(lengths, mode=self.reduction)

    def loss(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        lengths: torch.Tensor,
        *,
        optimizer_step: int = 0,
    ) -> TokenPolicyStageLoss:
        del optimizer_step
        objective = token_cppo_objective(
            new_log_probs,
            old_log_probs,
            _expanded_advantages(advantages, lengths, new_log_probs),
            lengths,
            delta=self.delta,
            w_min=self.w_min,
            delta_b=self.delta_b,
        )
        return _token_stage_loss(
            objective,
            lengths,
            reduction=self.reduction,
            horizon=self.horizon,
        )


__all__ = [
    "TokenCPPOStage",
    "TokenDPPOStage",
    "TokenDRPOStage",
    "TokenGRPOStage",
    "TokenGSPOStage",
    "TokenPolicyStage",
    "TokenPolicyStageLoss",
]

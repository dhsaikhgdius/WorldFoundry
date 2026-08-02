"""Strict recipe contracts for packed autoregressive policy optimization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import ClassVar

from ..common import (
    advantage_normalization_mode,
    frozen_float_mapping,
    mapping,
    strict_mapping,
    validate_clip_schedule,
)

TOKEN_MEAN = "token-mean"
SEQUENCE_MEAN_TOKEN_MEAN = "seq-mean-token-mean"
SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED = "seq-mean-token-sum-norm"

_TOKEN_REDUCTIONS = frozenset(
    {
        TOKEN_MEAN,
        SEQUENCE_MEAN_TOKEN_MEAN,
        SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED,
    }
)
_SUM_REDUCTIONS = frozenset({TOKEN_MEAN, SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED})

TOKEN_POLICY_COMMON_FIELDS = {
    "type",
    "reward_weights",
    "updates_per_trajectory",
    "group_size",
    "old_log_prob_source",
    "advantage_epsilon",
    "advantage_clip_max",
    "advantage_normalization",
    "sampling_temperature",
    "replay_microbatch_size",
    "first_update_log_ratio_tolerance",
}
TOKEN_GRPO_ALGORITHM_FIELDS = TOKEN_POLICY_COMMON_FIELDS | {
    "clip_range",
    "clip_range_high",
    "clip_schedule",
    "clip_schedule_steps",
    "reduction",
    "horizon",
}
TOKEN_GSPO_ALGORITHM_FIELDS = TOKEN_POLICY_COMMON_FIELDS | {
    "clip_range",
    "clip_range_high",
    "clip_schedule",
    "clip_schedule_steps",
}
TOKEN_DPPO_ALGORITHM_FIELDS = TOKEN_POLICY_COMMON_FIELDS | {
    "delta",
    "reduction",
    "horizon",
}
TOKEN_DRPO_ALGORITHM_FIELDS = TOKEN_POLICY_COMMON_FIELDS | {
    "epsilon",
    "mu_weighted",
    "reduction",
    "horizon",
}
TOKEN_CPPO_ALGORITHM_FIELDS = TOKEN_POLICY_COMMON_FIELDS | {
    "delta",
    "w_min",
    "delta_b",
    "reduction",
    "horizon",
}


def _finite_float(value: object, *, field_name: str) -> float:
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    return resolved


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer, not bool or float")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _reduction(
    value: object,
    *,
    allowed: frozenset[str],
) -> str:
    resolved = str(value).strip().lower().replace("_", "-")
    if resolved not in allowed:
        raise ValueError(f"reduction must be one of {sorted(allowed)}")
    return resolved


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenPolicyAlgorithmSpec:
    """Fields consumed by every native autoregressive policy learner.

    ``updates_per_trajectory`` is the number of balanced contiguous optimizer
    partitions, while ``replay_microbatch_size`` only accumulates gradients
    inside the active partition.
    """

    reward_weights: Mapping[str, float]
    updates_per_trajectory: int = 1
    group_size: int = 2
    old_log_prob_source: str = "rollout"
    advantage_epsilon: float = 1.0e-8
    advantage_clip_max: float | None = None
    advantage_normalization: str = "group-population-variance"
    sampling_temperature: float = 1.0
    replay_microbatch_size: int | None = None
    first_update_log_ratio_tolerance: float = 1.0e-5
    type: str = "token-policy"

    algorithm_type: ClassVar[str] = "token-policy"

    def __post_init__(self) -> None:
        resolved_type = str(self.type).strip().lower().replace("_", "-")
        if resolved_type != self.algorithm_type:
            raise ValueError(f"{type(self).__name__} algorithm type must be {self.algorithm_type!r}")
        if not isinstance(self.reward_weights, Mapping):
            raise TypeError("reward_weights must be a mapping")
        reward_weights = frozen_float_mapping(
            self.reward_weights,
            field_name="reward_weights",
        )
        if not any(weight != 0 for weight in reward_weights.values()):
            raise ValueError("reward_weights cannot all be zero")
        updates = _positive_int(
            self.updates_per_trajectory,
            field_name="updates_per_trajectory",
        )
        group_size = _positive_int(self.group_size, field_name="group_size")
        if group_size < 2:
            raise ValueError("group_size must be at least two")
        source = str(self.old_log_prob_source).strip().lower()
        if source not in {"rollout", "replay"}:
            raise ValueError("old_log_prob_source must be 'rollout' or 'replay'")
        advantage_epsilon = _finite_float(
            self.advantage_epsilon,
            field_name="advantage_epsilon",
        )
        if advantage_epsilon <= 0:
            raise ValueError("advantage_epsilon must be positive")
        advantage_clip_max = (
            None
            if self.advantage_clip_max is None
            else _finite_float(
                self.advantage_clip_max,
                field_name="advantage_clip_max",
            )
        )
        if advantage_clip_max is not None and advantage_clip_max <= 0:
            raise ValueError("advantage_clip_max must be positive")
        advantage_normalization = advantage_normalization_mode(
            self.advantage_normalization,
            field_name="algorithm.advantage_normalization",
        )
        sampling_temperature = _finite_float(
            self.sampling_temperature,
            field_name="sampling_temperature",
        )
        if sampling_temperature <= 0:
            raise ValueError("sampling_temperature must be positive")
        replay_microbatch_size = (
            None
            if self.replay_microbatch_size is None
            else _positive_int(
                self.replay_microbatch_size,
                field_name="replay_microbatch_size",
            )
        )
        anchor_tolerance = _finite_float(
            self.first_update_log_ratio_tolerance,
            field_name="first_update_log_ratio_tolerance",
        )
        if anchor_tolerance < 0:
            raise ValueError("first_update_log_ratio_tolerance must be non-negative")
        object.__setattr__(self, "type", resolved_type)
        object.__setattr__(self, "reward_weights", reward_weights)
        object.__setattr__(self, "updates_per_trajectory", updates)
        object.__setattr__(self, "group_size", group_size)
        object.__setattr__(self, "old_log_prob_source", source)
        object.__setattr__(self, "advantage_epsilon", advantage_epsilon)
        object.__setattr__(self, "advantage_clip_max", advantage_clip_max)
        object.__setattr__(self, "advantage_normalization", advantage_normalization)
        object.__setattr__(self, "sampling_temperature", sampling_temperature)
        object.__setattr__(self, "replay_microbatch_size", replay_microbatch_size)
        object.__setattr__(
            self,
            "first_update_log_ratio_tolerance",
            anchor_tolerance,
        )


def _validate_clipping(
    spec: object,
    *,
    prefix: str,
) -> None:
    clip_range = _finite_float(getattr(spec, "clip_range"), field_name="clip_range")
    if not 0 < clip_range < 1:
        raise ValueError(f"{prefix} clip_range must be in (0,1)")
    clip_range_high = getattr(spec, "clip_range_high")
    if clip_range_high is not None:
        clip_range_high = _finite_float(
            clip_range_high,
            field_name="clip_range_high",
        )
        if clip_range_high <= 0:
            raise ValueError(f"{prefix} clip_range_high must be positive")
    clip_schedule, clip_schedule_steps = validate_clip_schedule(
        getattr(spec, "clip_schedule"),
        getattr(spec, "clip_schedule_steps"),
    )
    object.__setattr__(spec, "clip_range", clip_range)
    object.__setattr__(spec, "clip_range_high", clip_range_high)
    object.__setattr__(spec, "clip_schedule", clip_schedule)
    object.__setattr__(spec, "clip_schedule_steps", clip_schedule_steps)


def _validate_reduction(
    spec: object,
    *,
    allowed: frozenset[str],
) -> None:
    reduction = _reduction(getattr(spec, "reduction"), allowed=allowed)
    horizon = _positive_int(getattr(spec, "horizon"), field_name="horizon")
    object.__setattr__(spec, "reduction", reduction)
    object.__setattr__(spec, "horizon", horizon)


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenGRPOAlgorithmSpec(TokenPolicyAlgorithmSpec):
    clip_range: float = 1.0e-4
    clip_range_high: float | None = None
    clip_schedule: str = "constant"
    clip_schedule_steps: int | None = None
    reduction: str = TOKEN_MEAN
    horizon: int = 8192
    type: str = "token-grpo"

    algorithm_type: ClassVar[str] = "token-grpo"

    def __post_init__(self) -> None:
        TokenPolicyAlgorithmSpec.__post_init__(self)
        if self.old_log_prob_source != "rollout":
            raise ValueError("Token-GRPO requires rollout old log probabilities")
        _validate_clipping(self, prefix="Token-GRPO")
        _validate_reduction(self, allowed=_TOKEN_REDUCTIONS)


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenGSPOAlgorithmSpec(TokenPolicyAlgorithmSpec):
    clip_range: float = 3.0e-4
    clip_range_high: float | None = None
    clip_schedule: str = "constant"
    clip_schedule_steps: int | None = None
    type: str = "token-gspo"

    algorithm_type: ClassVar[str] = "token-gspo"

    def __post_init__(self) -> None:
        TokenPolicyAlgorithmSpec.__post_init__(self)
        _validate_clipping(self, prefix="Token-GSPO")


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenDPPOAlgorithmSpec(TokenPolicyAlgorithmSpec):
    delta: float = 0.15
    reduction: str = TOKEN_MEAN
    horizon: int = 8192
    type: str = "token-dppo"

    algorithm_type: ClassVar[str] = "token-dppo"

    def __post_init__(self) -> None:
        TokenPolicyAlgorithmSpec.__post_init__(self)
        delta = _finite_float(self.delta, field_name="delta")
        if delta <= 0:
            raise ValueError("Token-DPPO delta must be positive")
        object.__setattr__(self, "delta", delta)
        _validate_reduction(self, allowed=_SUM_REDUCTIONS)


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenDRPOAlgorithmSpec(TokenPolicyAlgorithmSpec):
    epsilon: float = 12.5
    mu_weighted: bool = True
    reduction: str = TOKEN_MEAN
    horizon: int = 8192
    type: str = "token-drpo"

    algorithm_type: ClassVar[str] = "token-drpo"

    def __post_init__(self) -> None:
        TokenPolicyAlgorithmSpec.__post_init__(self)
        epsilon = _finite_float(self.epsilon, field_name="epsilon")
        if epsilon <= 0:
            raise ValueError("Token-DRPO epsilon must be positive")
        if not isinstance(self.mu_weighted, bool):
            raise TypeError("Token-DRPO mu_weighted must be a bool")
        object.__setattr__(self, "epsilon", epsilon)
        _validate_reduction(self, allowed=_SUM_REDUCTIONS)


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenCPPOAlgorithmSpec(TokenPolicyAlgorithmSpec):
    delta: float = 0.2
    w_min: float = 0.8
    delta_b: float = 0.02
    reduction: str = TOKEN_MEAN
    horizon: int = 8192
    type: str = "token-cppo"

    algorithm_type: ClassVar[str] = "token-cppo"

    def __post_init__(self) -> None:
        TokenPolicyAlgorithmSpec.__post_init__(self)
        delta = _finite_float(self.delta, field_name="delta")
        w_min = _finite_float(self.w_min, field_name="w_min")
        delta_b = _finite_float(self.delta_b, field_name="delta_b")
        if delta <= 0:
            raise ValueError("Token-CPPO delta must be positive")
        if not 0 < w_min <= 1:
            raise ValueError("Token-CPPO w_min must be in (0,1]")
        if delta_b < 0:
            raise ValueError("Token-CPPO delta_b must be non-negative")
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "w_min", w_min)
        object.__setattr__(self, "delta_b", delta_b)
        _validate_reduction(self, allowed=_SUM_REDUCTIONS)


_TOKEN_POLICY_TYPES = {
    "token-grpo": (TokenGRPOAlgorithmSpec, TOKEN_GRPO_ALGORITHM_FIELDS),
    "token-gspo": (TokenGSPOAlgorithmSpec, TOKEN_GSPO_ALGORITHM_FIELDS),
    "token-dppo": (TokenDPPOAlgorithmSpec, TOKEN_DPPO_ALGORITHM_FIELDS),
    "token-drpo": (TokenDRPOAlgorithmSpec, TOKEN_DRPO_ALGORITHM_FIELDS),
    "token-cppo": (TokenCPPOAlgorithmSpec, TOKEN_CPPO_ALGORITHM_FIELDS),
}


def parse_token_policy_algorithm(value: object) -> TokenPolicyAlgorithmSpec:
    """Parse one strict token-policy algorithm section."""

    raw = mapping(value, field_name="algorithm")
    algorithm_type = str(raw.get("type", "")).strip().lower().replace("_", "-")
    resolved = _TOKEN_POLICY_TYPES.get(algorithm_type)
    if resolved is None:
        raise ValueError(f"unsupported token-policy algorithm: {algorithm_type!r}")
    spec_type, allowed = resolved
    payload = strict_mapping(raw, field_name="algorithm", allowed=allowed)
    if "reward_weights" not in payload:
        raise ValueError("token-policy algorithm is missing reward_weights")
    return spec_type(**payload)


__all__ = [
    "TokenCPPOAlgorithmSpec",
    "TokenDPPOAlgorithmSpec",
    "TokenDRPOAlgorithmSpec",
    "TokenGRPOAlgorithmSpec",
    "TokenGSPOAlgorithmSpec",
    "TokenPolicyAlgorithmSpec",
    "parse_token_policy_algorithm",
]

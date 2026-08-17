"""Strict recipe contract for classic autoregressive PPO."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from ..common import (
    frozen_float_mapping,
    mapping,
    strict_mapping,
    validate_clip_schedule,
)

TOKEN_PPO_ALGORITHM_FIELDS = {
    "type",
    "reward_weights",
    "update_epochs",
    "update_partitions",
    "clip_range",
    "clip_range_high",
    "clip_schedule",
    "clip_schedule_steps",
    "value_clip_range",
    "vf_coef",
    "gamma",
    "gae_lambda",
    "reduction",
    "horizon",
    "sampling_temperature",
    "replay_microbatch_size",
}
TOKEN_PPO_REDUCTIONS = frozenset(
    {
        "token-mean",
        "seq-mean-token-mean",
        "seq-mean-token-sum-norm",
    }
)


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _finite_float(value: object, *, field_name: str) -> float:
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    return resolved


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenPPOAlgorithmSpec:
    """Settings for native actor-critic PPO.

    ``update_partitions`` splits each local rollout into equal, disjoint
    contiguous optimizer steps. ``update_epochs`` repeats that partition
    sweep while retaining the rollout's frozen old-policy/value/GAE anchor.
    """

    reward_weights: Mapping[str, float]
    update_epochs: int = 1
    update_partitions: int = 1
    clip_range: float = 0.2
    clip_range_high: float | None = None
    clip_schedule: str = "constant"
    clip_schedule_steps: int | None = None
    value_clip_range: float = 0.2
    vf_coef: float = 0.5
    gamma: float = 1.0
    gae_lambda: float = 0.95
    reduction: str = "token-mean"
    horizon: int = 8192
    sampling_temperature: float = 1.0
    replay_microbatch_size: int | None = None
    type: str = "token-ppo"

    def __post_init__(self) -> None:
        if str(self.type).strip().lower().replace("_", "-") != "token-ppo":
            raise ValueError("TokenPPOAlgorithmSpec type must be 'token-ppo'")
        if not isinstance(self.reward_weights, Mapping):
            raise TypeError("reward_weights must be a mapping")
        reward_weights = frozen_float_mapping(
            self.reward_weights,
            field_name="reward_weights",
        )
        if not any(weight != 0 for weight in reward_weights.values()):
            raise ValueError("reward_weights cannot all be zero")
        update_epochs = _positive_int(self.update_epochs, field_name="update_epochs")
        update_partitions = _positive_int(
            self.update_partitions,
            field_name="update_partitions",
        )
        clip_range = _finite_float(self.clip_range, field_name="clip_range")
        value_clip_range = _finite_float(
            self.value_clip_range,
            field_name="value_clip_range",
        )
        clip_range_high = (
            None if self.clip_range_high is None else _finite_float(self.clip_range_high, field_name="clip_range_high")
        )
        if clip_range < 0 or value_clip_range < 0:
            raise ValueError("policy and value clip ranges must be non-negative")
        if clip_range_high is not None and clip_range_high < 0:
            raise ValueError("clip_range_high must be non-negative")
        clip_schedule, clip_schedule_steps = validate_clip_schedule(
            self.clip_schedule,
            self.clip_schedule_steps,
        )
        vf_coef = _finite_float(self.vf_coef, field_name="vf_coef")
        if vf_coef < 0:
            raise ValueError("vf_coef must be non-negative")
        gamma = _finite_float(self.gamma, field_name="gamma")
        gae_lambda = _finite_float(self.gae_lambda, field_name="gae_lambda")
        if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in [0,1]")
        reduction = str(self.reduction).strip().lower().replace("_", "-")
        if reduction not in TOKEN_PPO_REDUCTIONS:
            raise ValueError(f"reduction must be one of {sorted(TOKEN_PPO_REDUCTIONS)}")
        horizon = _positive_int(self.horizon, field_name="horizon")
        temperature = _finite_float(
            self.sampling_temperature,
            field_name="sampling_temperature",
        )
        if temperature <= 0:
            raise ValueError("sampling_temperature must be positive")
        replay_microbatch_size = (
            None
            if self.replay_microbatch_size is None
            else _positive_int(
                self.replay_microbatch_size,
                field_name="replay_microbatch_size",
            )
        )
        object.__setattr__(self, "type", "token-ppo")
        object.__setattr__(self, "reward_weights", reward_weights)
        object.__setattr__(self, "update_epochs", update_epochs)
        object.__setattr__(self, "update_partitions", update_partitions)
        object.__setattr__(self, "clip_range", clip_range)
        object.__setattr__(self, "clip_range_high", clip_range_high)
        object.__setattr__(self, "clip_schedule", clip_schedule)
        object.__setattr__(self, "clip_schedule_steps", clip_schedule_steps)
        object.__setattr__(self, "value_clip_range", value_clip_range)
        object.__setattr__(self, "vf_coef", vf_coef)
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "gae_lambda", gae_lambda)
        object.__setattr__(self, "reduction", reduction)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "sampling_temperature", temperature)
        object.__setattr__(self, "replay_microbatch_size", replay_microbatch_size)


def parse_token_ppo_algorithm(value: object) -> TokenPPOAlgorithmSpec:
    """Parse a strict ``token-ppo`` algorithm section."""

    raw = mapping(value, field_name="algorithm")
    payload = strict_mapping(
        raw,
        field_name="algorithm",
        allowed=TOKEN_PPO_ALGORITHM_FIELDS,
    )
    if "reward_weights" not in payload:
        raise ValueError("token PPO algorithm is missing reward_weights")
    return TokenPPOAlgorithmSpec(**payload)


__all__ = ["TokenPPOAlgorithmSpec", "parse_token_ppo_algorithm"]

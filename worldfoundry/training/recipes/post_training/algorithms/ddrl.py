"""Strict recipe contract for selected-transition DDRL training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from ..common import (
    advantage_normalization_mode,
    frozen_float_mapping,
    strict_mapping,
)
from ..rewards.videoalign import VIDEOALIGN_REWARD_FIELDS, VideoAlignRewardSpec

DDRL_ALGORITHM_FIELDS = {
    "type",
    "train_on",
    "clip_range",
    "loss_scale",
    "advantage_epsilon",
    "advantage_normalization",
    "advantage_clip_min",
    "advantage_clip_max",
    "exponential_advantage",
    "kl_beta",
    "data_beta",
    "data_on_first_step_only",
    "reward_weights",
    "reward_model",
}


def _finite_float(value: object, *, field_name: str) -> float:
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"DDRL {field_name} must be finite")
    return resolved


@dataclass(frozen=True, slots=True)
class DDRLAlgorithmSpec:
    """Selected rollout steps and every tensor objective coefficient."""

    train_on: tuple[int, ...]
    reward_weights: Mapping[str, float]
    reward_model: VideoAlignRewardSpec
    clip_range: float = 1.0e-4
    loss_scale: float = 10.0
    advantage_epsilon: float = 1.0e-4
    advantage_normalization: str = "group-sample-std"
    advantage_clip_min: float | None = None
    advantage_clip_max: float | None = None
    exponential_advantage: bool = False
    kl_beta: float = 0.0
    data_beta: float = 0.0
    data_on_first_step_only: bool = False
    type: str = "ddrl"

    def __post_init__(self) -> None:
        resolved_type = str(self.type).strip().lower().replace("_", "-")
        if resolved_type != "ddrl":
            raise ValueError("DDRL algorithm type must be 'ddrl'")
        if not isinstance(self.train_on, (tuple, list)):
            raise TypeError("DDRL train_on must be a list or tuple of integer indices")
        if any(isinstance(step, bool) or not isinstance(step, int) for step in self.train_on):
            raise TypeError("DDRL train_on values must be integers, not bool or float")
        train_on = tuple(self.train_on)
        if not train_on or train_on[0] < 0 or train_on != tuple(sorted(set(train_on))):
            raise ValueError("DDRL train_on must be non-empty, non-negative, strictly increasing, and unique")

        clip_range = _finite_float(self.clip_range, field_name="clip_range")
        if not 0 < clip_range < 1:
            raise ValueError("DDRL clip_range must be in (0,1)")
        loss_scale = _finite_float(self.loss_scale, field_name="loss_scale")
        if loss_scale <= 0:
            raise ValueError("DDRL loss_scale must be positive")
        advantage_epsilon = _finite_float(
            self.advantage_epsilon,
            field_name="advantage_epsilon",
        )
        if advantage_epsilon <= 0:
            raise ValueError("DDRL advantage_epsilon must be positive")
        advantage_normalization = advantage_normalization_mode(
            self.advantage_normalization,
            field_name="algorithm.advantage_normalization",
        )
        lower = (
            None
            if self.advantage_clip_min is None
            else _finite_float(
                self.advantage_clip_min,
                field_name="advantage_clip_min",
            )
        )
        upper = (
            None
            if self.advantage_clip_max is None
            else _finite_float(
                self.advantage_clip_max,
                field_name="advantage_clip_max",
            )
        )
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError("DDRL advantage_clip_min must be smaller than advantage_clip_max")
        if not isinstance(self.exponential_advantage, bool):
            raise TypeError("DDRL exponential_advantage must be a bool")

        kl_beta = _finite_float(self.kl_beta, field_name="kl_beta")
        data_beta = _finite_float(self.data_beta, field_name="data_beta")
        if kl_beta < 0 or data_beta < 0:
            raise ValueError("DDRL kl_beta and data_beta must be non-negative")
        if not isinstance(self.data_on_first_step_only, bool):
            raise TypeError("DDRL data_on_first_step_only must be a bool")
        if data_beta == 0 and self.data_on_first_step_only:
            raise ValueError("DDRL data_on_first_step_only is unused when data_beta is zero")
        if not isinstance(self.reward_model, VideoAlignRewardSpec):
            raise TypeError("DDRL reward_model must be VideoAlignRewardSpec")
        reward_weights = frozen_float_mapping(
            self.reward_weights,
            field_name="reward_weights",
        )
        if set(reward_weights) != set(self.reward_model.reward_ids):
            raise ValueError("reward_weights must exactly match reward_model.reward_ids")
        if not any(weight != 0 for weight in reward_weights.values()):
            raise ValueError("DDRL reward_weights cannot all be zero")

        object.__setattr__(self, "type", resolved_type)
        object.__setattr__(self, "train_on", train_on)
        object.__setattr__(self, "clip_range", clip_range)
        object.__setattr__(self, "loss_scale", loss_scale)
        object.__setattr__(self, "advantage_epsilon", advantage_epsilon)
        object.__setattr__(self, "advantage_normalization", advantage_normalization)
        object.__setattr__(self, "advantage_clip_min", lower)
        object.__setattr__(self, "advantage_clip_max", upper)
        object.__setattr__(self, "kl_beta", kl_beta)
        object.__setattr__(self, "data_beta", data_beta)
        object.__setattr__(self, "reward_weights", reward_weights)


def parse_ddrl_algorithm(value: object) -> DDRLAlgorithmSpec:
    """Parse one strict DDRL section and its reward model."""

    payload = strict_mapping(
        value,
        field_name="algorithm",
        allowed=DDRL_ALGORITHM_FIELDS,
    )
    missing = sorted(name for name in ("train_on", "reward_weights", "reward_model") if name not in payload)
    if missing:
        raise ValueError(f"DDRL algorithm is missing required fields: {missing}")
    reward_payload = strict_mapping(
        payload.pop("reward_model"),
        field_name="algorithm.reward_model",
        allowed=VIDEOALIGN_REWARD_FIELDS,
    )
    return DDRLAlgorithmSpec(
        **payload,
        reward_model=VideoAlignRewardSpec(**reward_payload),
    )


__all__ = ["DDRLAlgorithmSpec", "parse_ddrl_algorithm"]

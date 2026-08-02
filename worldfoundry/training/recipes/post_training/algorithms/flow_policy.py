"""Shared recipe contract for stochastic flow-policy algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import ClassVar

from ..common import (
    advantage_normalization_mode,
    frozen_float_mapping,
    mapping,
    positive_int,
    strict_mapping,
)
from ..rewards.videoalign import VIDEOALIGN_REWARD_FIELDS, VideoAlignRewardSpec

FLOW_POLICY_ALGORITHM_FIELDS = {
    "type",
    "sigmas",
    "sde_step_indices",
    "sde_timestep_fraction",
    "num_sde_steps",
    "sde_window",
    "reward_weights",
    "num_train_timesteps",
    "guidance_scale",
    "init_same_noise",
    "eta",
    "sigma_max",
    "updates_per_trajectory",
    "group_size",
    "old_log_prob_source",
    "reference_kl_weight",
    "reference_checkpoint",
    "advantage_epsilon",
    "advantage_normalization",
    "advantage_clip_max",
    "trajectory_dtype",
    "transition_strategy",
    "reward_model",
}


@dataclass(frozen=True, slots=True)
class FlowSDEWindowSpec:
    """Pure sliding-window schedule parameters using an explicit start stride.

    An omitted stride selects a non-overlapping window, matching UniRL's
    ``overlap_size=0`` default.
    """

    window_size: int
    iterations_per_window: int
    stride: int | None = None
    initial_index: int = 0
    rollback: bool = False

    def __post_init__(self) -> None:
        window_size = positive_int(self.window_size, field_name="sde_window.window_size")
        iterations = positive_int(
            self.iterations_per_window,
            field_name="sde_window.iterations_per_window",
        )
        stride = window_size if self.stride is None else positive_int(
            self.stride,
            field_name="sde_window.stride",
        )
        if stride > window_size:
            raise ValueError("sde_window.stride cannot exceed window_size")
        if isinstance(self.initial_index, bool):
            raise TypeError("sde_window.initial_index must be an integer, not bool")
        initial_index = int(self.initial_index)
        if initial_index < 0:
            raise ValueError("sde_window.initial_index must be non-negative")
        if not isinstance(self.rollback, bool):
            raise TypeError("sde_window.rollback must be a bool")
        object.__setattr__(self, "window_size", window_size)
        object.__setattr__(self, "iterations_per_window", iterations)
        object.__setattr__(self, "stride", stride)
        object.__setattr__(self, "initial_index", initial_index)


@dataclass(frozen=True, slots=True)
class FlowPolicyAlgorithmSpec:
    """Rollout, reward, replay, and reference-policy configuration shared by flow RL.

    ``updates_per_trajectory`` partitions each rank-local rollout contiguously
    and evenly across optimizer steps; it is not a count of full-batch epochs.
    """

    sigmas: tuple[float, ...]
    reward_weights: Mapping[str, float]
    reward_model: VideoAlignRewardSpec
    sde_step_indices: tuple[int, ...] | None = None
    sde_timestep_fraction: tuple[float, float] | None = None
    num_sde_steps: int | None = None
    sde_window: FlowSDEWindowSpec | None = None
    num_train_timesteps: int = 1000
    guidance_scale: float = 1.0
    init_same_noise: bool = False
    eta: float = 0.7
    sigma_max: float | None = None
    updates_per_trajectory: int = 1
    group_size: int = 2
    old_log_prob_source: str = "rollout"
    reference_kl_weight: float = 0.0
    reference_checkpoint: str | None = None
    advantage_epsilon: float = 1.0e-8
    advantage_normalization: str = "group-population-variance"
    advantage_clip_max: float | None = None
    trajectory_dtype: str = "bfloat16"
    transition_strategy: str = "variance-preserving"
    type: str = "flow-policy"

    algorithm_type: ClassVar[str] = "flow-policy"

    @property
    def requires_reference_policy(self) -> bool:
        """Whether this objective consumes a frozen reference replay."""

        return float(self.reference_kl_weight) > 0

    def __post_init__(self) -> None:
        resolved_type = str(self.type).lower().replace("_", "-")
        if resolved_type != self.algorithm_type:
            raise ValueError(f"{type(self).__name__} algorithm type must be {self.algorithm_type!r}")
        sigmas = tuple(float(value) for value in self.sigmas)
        if len(sigmas) < 2 or any(not isfinite(value) or not 0 <= value <= 1 for value in sigmas):
            raise ValueError("flow-policy sigmas must contain at least two finite values in [0,1]")
        if any(left <= right for left, right in zip(sigmas, sigmas[1:])):
            raise ValueError("flow-policy sigmas must be strictly descending")
        indices = None if self.sde_step_indices is None else tuple(int(value) for value in self.sde_step_indices)
        fraction = (
            None if self.sde_timestep_fraction is None else tuple(float(value) for value in self.sde_timestep_fraction)
        )
        sparse_count = self.num_sde_steps
        window = self.sde_window
        if window is not None and not isinstance(window, FlowSDEWindowSpec):
            raise TypeError("sde_window must be FlowSDEWindowSpec")
        if indices is not None:
            if (
                not indices
                or indices != tuple(sorted(set(indices)))
                or indices[0] < 0
                or indices[-1] >= len(sigmas) - 1
            ):
                raise ValueError("sde_step_indices must be unique sorted transition indices")
            if fraction is not None or sparse_count is not None or window is not None:
                raise ValueError("static sde_step_indices cannot be combined with another SDE schedule")
        elif window is not None:
            if fraction is not None or sparse_count is not None:
                raise ValueError("sde_window cannot be combined with fractional SDE selection")
            if window.initial_index + window.window_size > len(sigmas) - 1:
                raise ValueError("sde_window exceeds the flow transition schedule")
        else:
            if fraction is None or sparse_count is None:
                raise ValueError(
                    "flow-policy algorithms require static sde_step_indices or both "
                    "sde_timestep_fraction and num_sde_steps"
                )
            if (
                len(fraction) != 2
                or any(not isfinite(value) or not 0 <= value <= 1 for value in fraction)
                or fraction[0] > fraction[1]
            ):
                raise ValueError("sde_timestep_fraction must be an ordered pair in [0,1]")
            start = int((len(sigmas) - 1) * fraction[0])
            end = int((len(sigmas) - 1) * fraction[1])
            if end <= start or isinstance(sparse_count, bool) or int(sparse_count) <= 0:
                raise ValueError("num_sde_steps requires a non-empty fractional SDE window")
            if int(sparse_count) > end - start:
                raise ValueError("num_sde_steps exceeds the fractional SDE timestep window")
            sparse_count = int(sparse_count)

        transition_strategy = str(self.transition_strategy).strip().lower().replace("_", "-")
        if transition_strategy not in {"variance-preserving", "constant-diffusion"}:
            raise ValueError("transition_strategy must be 'variance-preserving' or 'constant-diffusion'")
        if transition_strategy == "constant-diffusion":
            if self.sigma_max is not None:
                raise ValueError("sigma_max is unused by the constant-diffusion transition strategy")
            sigma_max = None
        else:
            sigma_max = sigmas[1] if self.sigma_max is None else float(self.sigma_max)
        for name, value in (
            ("eta", self.eta),
            ("guidance_scale", self.guidance_scale),
            ("reference_kl_weight", self.reference_kl_weight),
            ("advantage_epsilon", self.advantage_epsilon),
        ):
            if not isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if float(self.eta) <= 0:
            raise ValueError("eta must be positive")
        if sigma_max is not None and (not isfinite(sigma_max) or not 0 < sigma_max < 1):
            raise ValueError("sigma_max must be finite and in (0,1)")
        if float(self.guidance_scale) < 1:
            raise ValueError("guidance_scale must be at least one")
        if not isinstance(self.init_same_noise, bool):
            raise TypeError("init_same_noise must be a bool")
        if float(self.reference_kl_weight) < 0 or float(self.advantage_epsilon) <= 0:
            raise ValueError("reference_kl_weight must be non-negative and advantage_epsilon positive")
        advantage_normalization = advantage_normalization_mode(
            self.advantage_normalization,
            field_name="algorithm.advantage_normalization",
        )
        if self.advantage_clip_max is not None and float(self.advantage_clip_max) <= 0:
            raise ValueError("advantage_clip_max must be positive")
        source = str(self.old_log_prob_source).lower().strip()
        if source not in {"rollout", "replay"}:
            raise ValueError("old_log_prob_source must be 'rollout' or 'replay'")
        dtype = str(self.trajectory_dtype).lower().removeprefix("torch.")
        dtype = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(
            dtype,
            dtype,
        )
        if dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("trajectory_dtype must be bfloat16, float16, or float32")
        if not isinstance(self.reward_model, VideoAlignRewardSpec):
            raise TypeError("flow-policy reward_model must be VideoAlignRewardSpec")
        reward_weights = frozen_float_mapping(
            self.reward_weights,
            field_name="reward_weights",
        )
        if set(reward_weights) != set(self.reward_model.reward_ids):
            raise ValueError("reward_weights must exactly match reward_model.reward_ids")
        num_train_timesteps = positive_int(
            self.num_train_timesteps,
            field_name="algorithm.num_train_timesteps",
        )
        if num_train_timesteps < 2:
            raise ValueError("num_train_timesteps must be at least two")
        updates = positive_int(
            self.updates_per_trajectory,
            field_name="algorithm.updates_per_trajectory",
        )
        group_size = positive_int(self.group_size, field_name="algorithm.group_size")
        if group_size < 2:
            raise ValueError("flow-policy group_size must be at least two")
        reference_checkpoint = None if self.reference_checkpoint is None else str(self.reference_checkpoint).strip()
        if self.requires_reference_policy and not reference_checkpoint:
            raise ValueError(f"{self.algorithm_type} requires an explicit reference_checkpoint")
        if not self.requires_reference_policy and reference_checkpoint is not None:
            raise ValueError("reference_checkpoint is unused when reference_kl_weight is zero")

        object.__setattr__(self, "type", resolved_type)
        object.__setattr__(self, "sigmas", sigmas)
        object.__setattr__(self, "sde_step_indices", indices)
        object.__setattr__(self, "sde_timestep_fraction", fraction)
        object.__setattr__(self, "num_sde_steps", sparse_count)
        object.__setattr__(self, "sigma_max", sigma_max)
        object.__setattr__(self, "guidance_scale", float(self.guidance_scale))
        object.__setattr__(self, "reward_weights", reward_weights)
        object.__setattr__(self, "num_train_timesteps", num_train_timesteps)
        object.__setattr__(self, "old_log_prob_source", source)
        object.__setattr__(self, "trajectory_dtype", dtype)
        object.__setattr__(self, "transition_strategy", transition_strategy)
        object.__setattr__(self, "reference_checkpoint", reference_checkpoint)
        object.__setattr__(self, "advantage_normalization", advantage_normalization)
        object.__setattr__(self, "updates_per_trajectory", updates)
        object.__setattr__(self, "group_size", group_size)


def parse_flow_policy_fields(
    value: object,
    *,
    allowed: set[str],
) -> dict[str, object]:
    """Parse the common strict payload and nested reward model."""

    payload = mapping(value, field_name="algorithm")
    if "reward_model" not in payload:
        raise ValueError("flow-policy algorithms require an explicit algorithm.reward_model")
    algorithm_payload = strict_mapping(
        payload,
        field_name="algorithm",
        allowed=allowed,
    )
    if algorithm_payload.get("sde_window") is not None:
        window_payload = strict_mapping(
            algorithm_payload["sde_window"],
            field_name="algorithm.sde_window",
            allowed={
                "window_size",
                "iterations_per_window",
                "stride",
                "initial_index",
                "rollback",
            },
        )
        missing = sorted({"window_size", "iterations_per_window"} - set(window_payload))
        if missing:
            raise ValueError(f"algorithm.sde_window is missing fields: {missing}")
        algorithm_payload["sde_window"] = FlowSDEWindowSpec(**window_payload)
    reward_payload = strict_mapping(
        algorithm_payload.pop("reward_model"),
        field_name="algorithm.reward_model",
        allowed=VIDEOALIGN_REWARD_FIELDS,
    )
    return {
        **algorithm_payload,
        "reward_model": VideoAlignRewardSpec(**reward_payload),
    }


__all__ = [
    "FLOW_POLICY_ALGORITHM_FIELDS",
    "FlowPolicyAlgorithmSpec",
    "FlowSDEWindowSpec",
    "parse_flow_policy_fields",
]

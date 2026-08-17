"""Typed trajectory and replay contracts for native diffusion-policy RL."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..shared.contracts import (
    TensorLike,
    freeze_mapping,
    is_broadcastable,
    non_empty_ids,
    tensor_shape,
)


@dataclass(frozen=True, slots=True)
class RolloutPrompt:
    """One immutable prompt/condition identity before group expansion."""

    prompt_id: str
    prompt: str
    conditions: Mapping[str, object] = field(default_factory=dict)
    generation: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (("prompt_id", self.prompt_id), ("prompt", self.prompt)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(self, "conditions", freeze_mapping(self.conditions, field_name="conditions"))
        object.__setattr__(self, "generation", freeze_mapping(self.generation, field_name="generation"))


@dataclass(frozen=True, slots=True)
class FlowTrajectory:
    """An exact stochastic flow trajectory and its frozen old-policy anchor.

    ``latents`` is ``[B, S+1, ...]``.  Log-probs and means only cover
    ``step_indices`` because deterministic ODE steps have no Gaussian policy
    likelihood and must never be silently inserted into a policy objective.
    """

    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    policy_revision: str
    latents: TensorLike
    sigmas: TensorLike
    step_indices: tuple[int, ...]
    old_log_probs: TensorLike
    transition_means: TensorLike
    transition_scales: TensorLike
    update_step_mask: TensorLike | None = None
    conditioning: Mapping[str, object] = field(default_factory=dict)
    transition_identity: Mapping[str, object] = field(
        default_factory=lambda: {
            "kind": "variance-preserving",
            "eta": 1.0,
            "sigma_max": 0.99,
        }
    )
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sample_ids = non_empty_ids(self.sample_ids, field_name="sample_ids", unique=True)
        group_ids = non_empty_ids(self.group_ids, field_name="group_ids", unique=False)
        if len(group_ids) != len(sample_ids):
            raise ValueError("group_ids length must match sample_ids")
        incomplete = sorted(group for group, count in Counter(group_ids).items() if count < 2)
        if incomplete:
            raise ValueError(f"trajectory groups must contain at least two samples: {incomplete}")
        if not isinstance(self.policy_revision, str) or not self.policy_revision.strip():
            raise ValueError("policy_revision must be a non-empty string")
        latent_shape = tensor_shape(self.latents, field_name="latents")
        if len(latent_shape) < 3 or latent_shape[0] != len(sample_ids) or latent_shape[1] < 2:
            raise ValueError("latents must have shape [B,S+1,...]")
        transition_count = latent_shape[1] - 1
        sigma_shape = tensor_shape(self.sigmas, field_name="sigmas")
        if sigma_shape not in {(transition_count + 1,), (len(sample_ids), transition_count + 1)}:
            raise ValueError("sigmas must have shape [S+1] or [B,S+1]")

        indices = tuple(int(index) for index in self.step_indices)
        if not indices or indices != tuple(sorted(set(indices))):
            raise ValueError("step_indices must be non-empty, strictly increasing, and unique")
        if indices[0] < 0 or indices[-1] >= transition_count:
            raise ValueError("step_indices fall outside the trajectory")
        selected = len(indices)
        if tensor_shape(self.old_log_probs, field_name="old_log_probs") != (len(sample_ids), selected):
            raise ValueError("old_log_probs must have shape [B,K]")
        expected_mean = (len(sample_ids), selected, *latent_shape[2:])
        if tensor_shape(self.transition_means, field_name="transition_means") != expected_mean:
            raise ValueError("transition_means must have shape [B,K,...latent]")
        scale_shape = tensor_shape(self.transition_scales, field_name="transition_scales")
        if not is_broadcastable(scale_shape, expected_mean):
            raise ValueError("transition_scales must broadcast to transition_means")
        if self.update_step_mask is not None and tensor_shape(
            self.update_step_mask,
            field_name="update_step_mask",
        ) != (len(sample_ids), selected):
            raise ValueError("update_step_mask must have shape [B,K]")
        from .rollout_strategies.transition import (
            flow_transition_strategy_from_identity,
        )

        transition_identity = freeze_mapping(
            self.transition_identity,
            field_name="transition_identity",
        )
        flow_transition_strategy_from_identity(transition_identity)

        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "group_ids", group_ids)
        object.__setattr__(self, "step_indices", indices)
        object.__setattr__(self, "transition_identity", transition_identity)
        object.__setattr__(self, "conditioning", freeze_mapping(self.conditioning, field_name="conditioning"))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata, field_name="metadata"))

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)

    @property
    def transition_count(self) -> int:
        return int(self.latents.shape[1]) - 1


@dataclass(frozen=True, slots=True)
class FlowTrajectoryReplayBatch:
    """A contiguous replay-only view over one complete trajectory.

    Group completeness is a rollout/advantage invariant.  Once advantages are
    frozen, learner replay may use a single sample at a time.  Keeping that
    view distinct prevents an incomplete group from being accepted anywhere a
    complete :class:`FlowTrajectory` is required.
    """

    source: FlowTrajectory
    start: int
    end: int
    conditioning: Mapping[str, object]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source, FlowTrajectory):
            raise TypeError("replay batch source must be a complete FlowTrajectory")
        if (
            isinstance(self.start, bool)
            or isinstance(self.end, bool)
            or not 0 <= int(self.start) < int(self.end) <= self.source.batch_size
        ):
            raise ValueError("replay batch must be a non-empty contiguous interval")
        object.__setattr__(self, "start", int(self.start))
        object.__setattr__(self, "end", int(self.end))
        object.__setattr__(
            self,
            "conditioning",
            freeze_mapping(self.conditioning, field_name="conditioning"),
        )
        object.__setattr__(
            self,
            "metadata",
            freeze_mapping(self.metadata, field_name="metadata"),
        )

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return self.source.sample_ids[self.start : self.end]

    @property
    def group_ids(self) -> tuple[str, ...]:
        return self.source.group_ids[self.start : self.end]

    @property
    def policy_revision(self) -> str:
        return self.source.policy_revision

    @property
    def latents(self) -> TensorLike:
        return self.source.latents[self.start : self.end]

    @property
    def sigmas(self) -> TensorLike:
        if len(tensor_shape(self.source.sigmas, field_name="source.sigmas")) == 2:
            return self.source.sigmas[self.start : self.end]
        return self.source.sigmas

    @property
    def step_indices(self) -> tuple[int, ...]:
        return self.source.step_indices

    @property
    def old_log_probs(self) -> TensorLike:
        return self.source.old_log_probs[self.start : self.end]

    @property
    def transition_means(self) -> TensorLike:
        return self.source.transition_means[self.start : self.end]

    @property
    def transition_scales(self) -> TensorLike:
        shape = tensor_shape(self.source.transition_scales, field_name="source.transition_scales")
        if shape and shape[0] == self.source.batch_size:
            return self.source.transition_scales[self.start : self.end]
        return self.source.transition_scales

    @property
    def update_step_mask(self) -> TensorLike | None:
        if self.source.update_step_mask is None:
            return None
        return self.source.update_step_mask[self.start : self.end]

    @property
    def transition_identity(self) -> Mapping[str, object]:
        return self.source.transition_identity

    @property
    def batch_size(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class FlowRolloutBatch:
    """Initial latent population and schedule for one synchronous RL iteration."""

    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    policy_revision: str
    initial_latents: TensorLike
    sigmas: TensorLike
    conditioning: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sample_ids = non_empty_ids(self.sample_ids, field_name="sample_ids", unique=True)
        group_ids = non_empty_ids(self.group_ids, field_name="group_ids", unique=False)
        if len(group_ids) != len(sample_ids):
            raise ValueError("group_ids length must match sample_ids")
        incomplete = sorted(group for group, count in Counter(group_ids).items() if count < 2)
        if incomplete:
            raise ValueError(f"rollout groups must contain at least two samples: {incomplete}")
        latent_shape = tensor_shape(self.initial_latents, field_name="initial_latents")
        if len(latent_shape) < 2 or latent_shape[0] != len(sample_ids):
            raise ValueError("initial_latents must have shape [B,...]")
        sigma_shape = tensor_shape(self.sigmas, field_name="sigmas")
        if not (
            (len(sigma_shape) == 1 and sigma_shape[0] >= 2)
            or (len(sigma_shape) == 2 and sigma_shape[0] == len(sample_ids) and sigma_shape[1] >= 2)
        ):
            raise ValueError("sigmas must have shape [S+1] or [B,S+1]")
        if not isinstance(self.policy_revision, str) or not self.policy_revision.strip():
            raise ValueError("policy_revision must be a non-empty string")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "group_ids", group_ids)
        object.__setattr__(self, "conditioning", freeze_mapping(self.conditioning, field_name="conditioning"))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata, field_name="metadata"))

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)


@dataclass(frozen=True, slots=True)
class FlowReplayResult:
    """Differentiable policy replay for the selected stochastic steps."""

    log_probs: TensorLike
    transition_means: TensorLike
    transition_scales: TensorLike
    velocities: TensorLike | None = None
    std_dev_t: TensorLike | None = None
    sqrt_dt: TensorLike | None = None

    def __post_init__(self) -> None:
        log_shape = tensor_shape(self.log_probs, field_name="log_probs")
        if len(log_shape) != 2 or any(size == 0 for size in log_shape):
            raise ValueError("log_probs must have shape [B,K]")
        mean_shape = tensor_shape(self.transition_means, field_name="transition_means")
        if len(mean_shape) < 3 or mean_shape[:2] != log_shape:
            raise ValueError("transition_means must start with [B,K]")
        if not is_broadcastable(
            tensor_shape(self.transition_scales, field_name="transition_scales"),
            mean_shape,
        ):
            raise ValueError("transition_scales must broadcast to transition_means")
        if (
            self.velocities is not None
            and tensor_shape(
                self.velocities,
                field_name="velocities",
            )
            != mean_shape
        ):
            raise ValueError("velocities must share shape [B,K,...latent]")
        if (self.std_dev_t is None) != (self.sqrt_dt is None):
            raise ValueError("std_dev_t and sqrt_dt must be provided together")
        if self.std_dev_t is not None:
            if not is_broadcastable(
                tensor_shape(self.std_dev_t, field_name="std_dev_t"),
                mean_shape,
            ):
                raise ValueError("std_dev_t must broadcast to transition_means")
            sqrt_dt_shape = tensor_shape(self.sqrt_dt, field_name="sqrt_dt")
            if sqrt_dt_shape != log_shape and not is_broadcastable(
                sqrt_dt_shape,
                mean_shape,
            ):
                raise ValueError("sqrt_dt must have shape [B,K] or broadcast to transition_means")


@runtime_checkable
class FlowTrajectoryReplayAdapter(Protocol):
    """Policy-owned replay seam consumed by the native Flow-GRPO engine."""

    module: object

    def replay(
        self,
        trajectory: FlowTrajectory | FlowTrajectoryReplayBatch,
        *,
        training: bool,
    ) -> FlowReplayResult: ...


@runtime_checkable
class FlowTrajectorySamplingAdapter(Protocol):
    """Rollout seam shared by local and distributed flow samplers."""

    def sample(
        self,
        initial_latents: TensorLike,
        sigmas: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        group_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        policy_revision: str,
        sde_step_indices: tuple[int, ...] | None = None,
        generator: object | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> FlowTrajectory: ...


@runtime_checkable
class TrajectoryRewardAdapter(Protocol):
    """Score a completed native trajectory into named tensor components."""

    def score(self, trajectory: FlowTrajectory) -> Mapping[str, TensorLike]: ...


__all__ = [
    "FlowReplayResult",
    "FlowRolloutBatch",
    "FlowTrajectory",
    "FlowTrajectoryReplayBatch",
    "FlowTrajectoryReplayAdapter",
    "FlowTrajectorySamplingAdapter",
    "RolloutPrompt",
    "TrajectoryRewardAdapter",
    "TensorLike",
]

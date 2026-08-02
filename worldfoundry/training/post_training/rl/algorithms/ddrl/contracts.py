"""Runtime contracts for DDRL rollout replay and regularization."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch

from ....shared.contracts import TensorLike, freeze_mapping, non_empty_ids


def _frozen_floating_tensor(
    value: object,
    *,
    field_name: str,
    shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{field_name} must be a floating torch.Tensor")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{field_name} must have shape {shape}")
    if value.requires_grad:
        raise ValueError(f"{field_name} must be a frozen rollout tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{field_name} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class DDRLRolloutBatch:
    """One grouped model-specific input batch awaiting rollout collection."""

    batch_id: str
    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    model_inputs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.batch_id, str) or not self.batch_id.strip():
            raise ValueError("batch_id must be a non-empty string")
        sample_ids = non_empty_ids(self.sample_ids, field_name="sample_ids", unique=True)
        group_ids = non_empty_ids(self.group_ids, field_name="group_ids", unique=False)
        if len(group_ids) != len(sample_ids):
            raise ValueError("group_ids length must match sample_ids")
        incomplete = sorted(group for group, count in Counter(group_ids).items() if count < 2)
        if incomplete:
            raise ValueError(f"every DDRL group needs at least two samples: {incomplete}")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "group_ids", group_ids)
        object.__setattr__(
            self,
            "model_inputs",
            freeze_mapping(self.model_inputs, field_name="model_inputs"),
        )

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)


@dataclass(frozen=True, slots=True)
class DDRLTrajectory:
    """Selected rollout transitions and their immutable behavior anchors."""

    trajectory_id: str
    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    train_on: tuple[int, ...]
    next_latents: torch.Tensor
    old_means: torch.Tensor
    terminal_latents: torch.Tensor
    reference_means: torch.Tensor | None = None
    replay_inputs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory_id, str) or not self.trajectory_id.strip():
            raise ValueError("trajectory_id must be a non-empty string")
        sample_ids = non_empty_ids(self.sample_ids, field_name="sample_ids", unique=True)
        group_ids = non_empty_ids(self.group_ids, field_name="group_ids", unique=False)
        if len(group_ids) != len(sample_ids):
            raise ValueError("group_ids length must match sample_ids")
        incomplete = sorted(group for group, count in Counter(group_ids).items() if count < 2)
        if incomplete:
            raise ValueError(f"every DDRL group needs at least two samples: {incomplete}")
        if any(isinstance(step, bool) for step in self.train_on):
            raise TypeError("train_on values must be integers, not bool")
        train_on = tuple(int(step) for step in self.train_on)
        if not train_on or train_on != tuple(sorted(set(train_on))) or train_on[0] < 0:
            raise ValueError("train_on must be non-empty, non-negative, strictly increasing, and unique")
        expected_prefix = (len(sample_ids), len(train_on))
        next_latents = _frozen_floating_tensor(
            self.next_latents,
            field_name="next_latents",
        )
        if next_latents.ndim < 3 or tuple(next_latents.shape[:2]) != expected_prefix:
            raise ValueError("next_latents must have shape [B,K,...latent]")
        shape = tuple(next_latents.shape)
        old_means = _frozen_floating_tensor(
            self.old_means,
            field_name="old_means",
            shape=shape,
        )
        if old_means.device != next_latents.device:
            raise ValueError("next_latents and old_means must share a device")
        terminal_latents = _frozen_floating_tensor(
            self.terminal_latents,
            field_name="terminal_latents",
        )
        if terminal_latents.ndim < 2 or int(terminal_latents.shape[0]) != len(sample_ids):
            raise ValueError("terminal_latents must have shape [B,...latent]")
        if terminal_latents.device != next_latents.device:
            raise ValueError("terminal and transition latents must share a device")
        reference_means = self.reference_means
        if reference_means is not None:
            reference_means = _frozen_floating_tensor(
                reference_means,
                field_name="reference_means",
                shape=shape,
            )
            if reference_means.device != next_latents.device:
                raise ValueError("reference_means must share the trajectory device")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "group_ids", group_ids)
        object.__setattr__(self, "train_on", train_on)
        object.__setattr__(self, "reference_means", reference_means)
        object.__setattr__(
            self,
            "replay_inputs",
            freeze_mapping(self.replay_inputs, field_name="replay_inputs"),
        )

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)

    @property
    def step_count(self) -> int:
        return len(self.train_on)


@runtime_checkable
class DDRLRolloutAdapter(Protocol):
    """Model-specific collection seam producing selected DDRL transitions."""

    def collect(
        self,
        batch: DDRLRolloutBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> DDRLTrajectory: ...


@runtime_checkable
class DDRLReplayAdapter(Protocol):
    """Differentiably replay one selected transition into its current mean."""

    module: object

    def replay_mean(
        self,
        trajectory: DDRLTrajectory,
        train_on_position: int,
        *,
        training: bool,
    ) -> TensorLike: ...


@runtime_checkable
class DDRLDataRegularizerAdapter(Protocol):
    """Compute a model-specific diffusion loss on real training data."""

    module: object

    def loss(
        self,
        trajectory: DDRLTrajectory,
        train_on_position: int,
        *,
        generator: torch.Generator | None = None,
        training: bool,
    ) -> TensorLike: ...


@runtime_checkable
class DDRLRewardAdapter(Protocol):
    """Score terminal rollout latents into named per-sample components."""

    reward_ids: tuple[str, ...]

    def score(self, trajectory: DDRLTrajectory) -> Mapping[str, TensorLike]: ...


__all__ = [
    "DDRLDataRegularizerAdapter",
    "DDRLReplayAdapter",
    "DDRLRewardAdapter",
    "DDRLRolloutAdapter",
    "DDRLRolloutBatch",
    "DDRLTrajectory",
]

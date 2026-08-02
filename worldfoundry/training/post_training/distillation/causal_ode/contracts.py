"""Typed Causal ODE trajectory batches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ...shared.contracts import TensorLike, freeze_mapping, non_empty_ids, tensor_shape


@dataclass(frozen=True, slots=True)
class CausalODETrainingBatch:
    """One batch of paired ODE trajectories ordered from noisy to clean."""

    sample_ids: tuple[str, ...]
    ode_trajectories: TensorLike
    conditioning: Mapping[str, object]

    def __post_init__(self) -> None:
        sample_ids = non_empty_ids(self.sample_ids, field_name="sample_ids", unique=True)
        shape = tensor_shape(self.ode_trajectories, field_name="ode_trajectories")
        if len(shape) < 4 or shape[0] != len(sample_ids):
            raise ValueError("ode_trajectories must have shape [B,S,...] with at least one latent axis")
        if shape[1] < 2 or any(size <= 0 for size in shape[2:]):
            raise ValueError("ode_trajectories must contain noisy states and a non-empty clean state")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(
            self,
            "conditioning",
            freeze_mapping(self.conditioning, field_name="conditioning"),
        )

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)

    @property
    def trajectory_states(self) -> int:
        return int(self.ode_trajectories.shape[1])


__all__ = ["CausalODETrainingBatch"]

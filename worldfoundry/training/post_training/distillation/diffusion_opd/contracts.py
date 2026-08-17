"""Runtime contracts for native teacher-anchored on-policy distillation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from ...shared.contracts import (
    TensorLike,
    freeze_mapping,
    is_broadcastable,
    non_empty_ids,
    tensor_shape,
)


@dataclass(frozen=True, slots=True)
class DiffusionOPDRolloutBatch:
    """One homogeneous-domain batch of initial noise and conditioning."""

    sample_ids: tuple[str, ...]
    domain: str
    initial_latents: TensorLike
    sigmas: TensorLike
    conditioning: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sample_ids = non_empty_ids(self.sample_ids, field_name="sample_ids", unique=True)
        domain = str(self.domain).strip()
        if not domain:
            raise ValueError("DiffusionOPD batch domain must be non-empty")
        latent_shape = tensor_shape(self.initial_latents, field_name="initial_latents")
        if len(latent_shape) < 2 or latent_shape[0] != len(sample_ids):
            raise ValueError("initial_latents must have shape [B,...]")
        sigma_shape = tensor_shape(self.sigmas, field_name="sigmas")
        if len(sigma_shape) != 1 or sigma_shape[0] < 2:
            raise ValueError("DiffusionOPD sigmas must have shape [S+1]")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(
            self,
            "conditioning",
            freeze_mapping(self.conditioning, field_name="conditioning"),
        )

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)


@dataclass(frozen=True, slots=True)
class DiffusionOPDTrajectory:
    """The student's own trajectory and shared transition variances."""

    sample_ids: tuple[str, ...]
    domain: str
    latents: TensorLike
    sigmas: TensorLike
    step_indices: tuple[int, ...]
    transition_scales: TensorLike
    conditioning: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sample_ids = non_empty_ids(self.sample_ids, field_name="sample_ids", unique=True)
        domain = str(self.domain).strip()
        if not domain:
            raise ValueError("DiffusionOPD trajectory domain must be non-empty")
        latent_shape = tensor_shape(self.latents, field_name="latents")
        if len(latent_shape) < 3 or latent_shape[0] != len(sample_ids) or latent_shape[1] < 2:
            raise ValueError("DiffusionOPD latents must have shape [B,S+1,...]")
        transition_count = latent_shape[1] - 1
        if tensor_shape(self.sigmas, field_name="sigmas") != (transition_count + 1,):
            raise ValueError("DiffusionOPD sigmas must have shape [S+1]")
        indices = tuple(int(index) for index in self.step_indices)
        if not indices or indices != tuple(sorted(set(indices))) or indices[0] < 0 or indices[-1] >= transition_count:
            raise ValueError("DiffusionOPD step_indices must be non-empty, sorted, unique, and in range")
        mean_shape = (len(sample_ids), len(indices), *latent_shape[2:])
        if not is_broadcastable(
            tensor_shape(self.transition_scales, field_name="transition_scales"),
            mean_shape,
        ):
            raise ValueError("DiffusionOPD transition_scales must broadcast to replay means")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "step_indices", indices)
        object.__setattr__(
            self,
            "conditioning",
            freeze_mapping(self.conditioning, field_name="conditioning"),
        )

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)

    @property
    def selected_steps(self) -> int:
        return len(self.step_indices)


@dataclass(frozen=True, slots=True)
class DiffusionOPDReplayResult:
    """Per-step transition means recomputed on a fixed student trajectory."""

    transition_means: TensorLike
    transition_scales: TensorLike

    def __post_init__(self) -> None:
        mean_shape = tensor_shape(self.transition_means, field_name="transition_means")
        if len(mean_shape) < 3 or mean_shape[0] == 0 or mean_shape[1] == 0:
            raise ValueError("DiffusionOPD transition_means must have shape [B,K,...]")
        if not is_broadcastable(
            tensor_shape(self.transition_scales, field_name="transition_scales"),
            mean_shape,
        ):
            raise ValueError("DiffusionOPD replay scales must broadcast to means")


__all__ = [
    "DiffusionOPDReplayResult",
    "DiffusionOPDRolloutBatch",
    "DiffusionOPDTrajectory",
]

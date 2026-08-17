"""Typed execution seams for WorldFoundry-native AnyFlow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from torch import Tensor

from worldfoundry.core.attention.chunk_partition import TemporalChunkPartition

from ...shared.contracts import TensorLike, freeze_mapping, non_empty_ids, tensor_shape


@dataclass(frozen=True, slots=True)
class AnyFlowTrainingBatch:
    """Clean BCTHW video latents and both classifier-free text branches."""

    sample_ids: tuple[str, ...]
    clean_latents: TensorLike
    conditioning: Mapping[str, object]
    unconditional_conditioning: Mapping[str, object]

    def __post_init__(self) -> None:
        sample_ids = non_empty_ids(self.sample_ids, field_name="sample_ids", unique=True)
        shape = tensor_shape(self.clean_latents, field_name="clean_latents")
        if len(shape) != 5 or shape[0] != len(sample_ids) or any(size <= 0 for size in shape[1:]):
            raise ValueError("AnyFlow clean_latents must be non-empty BCTHW tensors")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(
            self,
            "conditioning",
            freeze_mapping(self.conditioning, field_name="conditioning"),
        )
        object.__setattr__(
            self,
            "unconditional_conditioning",
            freeze_mapping(
                self.unconditional_conditioning,
                field_name="unconditional_conditioning",
            ),
        )

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)

    def head(self, count: int) -> AnyFlowTrainingBatch:
        """Take the official DMD prefix while preserving shared conditioning."""

        if isinstance(count, bool) or not 0 < int(count) <= self.batch_size:
            raise ValueError("AnyFlow batch prefix must lie in [1,batch_size]")
        size = int(count)

        def sliced(values: Mapping[str, object]) -> dict[str, object]:
            result: dict[str, object] = {}
            for name, value in values.items():
                if isinstance(value, Tensor) and value.ndim > 0:
                    if int(value.shape[0]) == self.batch_size:
                        result[name] = value[:size]
                        continue
                result[name] = value
            return result

        return AnyFlowTrainingBatch(
            sample_ids=self.sample_ids[:size],
            clean_latents=self.clean_latents[:size],
            conditioning=sliced(self.conditioning),
            unconditional_conditioning=sliced(self.unconditional_conditioning),
        )


@runtime_checkable
class AnyFlowModuleAdapter(Protocol):
    module: object


@runtime_checkable
class AnyFlowFARAdapter(AnyFlowModuleAdapter, Protocol):
    """Causal flow-map model seam used by pretraining and differentiable rollout."""

    module: object

    def create_rollout_state(
        self,
        *,
        partition: TemporalChunkPartition,
        reference: TensorLike,
    ) -> object: ...

    def predict_flow_map(
        self,
        noisy_latents: TensorLike,
        timesteps: TensorLike,
        destination_timesteps: TensorLike,
        *,
        clean_latents: TensorLike,
        context_latents: TensorLike,
        partition: TemporalChunkPartition,
        sampled_chunk_count: int,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> TensorLike: ...

    def predict_bidirectional_velocity(
        self,
        noisy_latents: TensorLike,
        timesteps: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> TensorLike: ...

    def rollout_velocity(
        self,
        noisy_latents: TensorLike,
        timesteps: TensorLike,
        destination_timesteps: TensorLike,
        *,
        partition: TemporalChunkPartition,
        chunk_index: int,
        rollout_state: object,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
    ) -> TensorLike: ...

    def commit_rollout_chunk(
        self,
        clean_prefix: TensorLike,
        *,
        partition: TemporalChunkPartition,
        chunk_index: int,
        rollout_state: object,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
    ) -> None: ...


@runtime_checkable
class AnyFlowBidirectionalAdapter(AnyFlowModuleAdapter, Protocol):
    """Full-video FlowMap student seam with no FAR partition or KV state."""

    def predict_flow_map(
        self,
        noisy_latents: TensorLike,
        timesteps: TensorLike,
        destination_timesteps: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> TensorLike: ...

    def rollout_velocity(
        self,
        noisy_latents: TensorLike,
        timesteps: TensorLike,
        destination_timesteps: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
    ) -> TensorLike: ...


@runtime_checkable
class AnyFlowScoreAdapter(Protocol):
    """Bidirectional real/fake score seam predicting RF velocity."""

    module: object

    def predict_velocity(
        self,
        noisy_latents: TensorLike,
        timesteps: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> TensorLike: ...


@dataclass(frozen=True, slots=True)
class AnyFlowLossResult:
    loss: TensorLike
    metrics: Mapping[str, object]


@runtime_checkable
class AnyFlowPretrainLossAdapter(Protocol):
    config_state: Mapping[str, object]
    decision_draws_per_student_loss: int
    student: AnyFlowModuleAdapter
    decisions: object

    def loss_denominator(
        self,
        batch: AnyFlowTrainingBatch,
        *,
        role: Literal["student"],
    ) -> object: ...

    def student_loss(
        self,
        batch: AnyFlowTrainingBatch,
        *,
        generator: object | None = None,
    ) -> AnyFlowLossResult: ...


@runtime_checkable
class AnyFlowOnPolicyLossAdapter(Protocol):
    config_state: Mapping[str, object]
    generator_decision_draws: int
    fake_score_decision_draws: int
    discriminator_update_ratio: int
    student: AnyFlowModuleAdapter
    real_score: AnyFlowScoreAdapter
    fake_score: AnyFlowScoreAdapter
    decisions: object

    def loss_denominator(
        self,
        batch: AnyFlowTrainingBatch,
        *,
        role: Literal["generator", "fake-score"],
    ) -> object: ...

    def generator_loss(
        self,
        batch: AnyFlowTrainingBatch,
        *,
        generator: object | None = None,
    ) -> AnyFlowLossResult: ...

    def fake_score_loss(
        self,
        batch: AnyFlowTrainingBatch,
        *,
        generator: object | None = None,
    ) -> AnyFlowLossResult: ...


__all__ = [
    "AnyFlowBidirectionalAdapter",
    "AnyFlowFARAdapter",
    "AnyFlowLossResult",
    "AnyFlowModuleAdapter",
    "AnyFlowOnPolicyLossAdapter",
    "AnyFlowPretrainLossAdapter",
    "AnyFlowScoreAdapter",
    "AnyFlowTrainingBatch",
]

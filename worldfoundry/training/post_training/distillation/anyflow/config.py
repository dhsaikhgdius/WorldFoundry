"""Execution configuration for native AnyFlow training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite

from worldfoundry.core.attention.chunk_partition import TemporalChunkPartition
from worldfoundry.core.io.integrity import canonical_sha256


def _finite(value: object, *, field_name: str, positive: bool = False) -> float:
    result = float(value)
    if not isfinite(result) or (positive and result <= 0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{field_name} must be {qualifier}")
    return result


def _positive_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


@dataclass(frozen=True, slots=True)
class AnyFlowMapConfig:
    """Flow-map interpolation, weighting, and pretraining mixture."""

    num_train_timesteps: int = 1000
    timestep_shift: float = 5.0
    central_difference_epsilon: float = 5.0
    diffusion_ratio: float = 0.5
    consistency_ratio: float = 0.25
    fused_guidance_scale: float = 3.0

    def __post_init__(self) -> None:
        steps = _positive_integer(
            self.num_train_timesteps,
            field_name="num_train_timesteps",
        )
        if steps < 2:
            raise ValueError("num_train_timesteps must be at least two")
        shift = _finite(self.timestep_shift, field_name="timestep_shift", positive=True)
        epsilon = _finite(
            self.central_difference_epsilon,
            field_name="central_difference_epsilon",
            positive=True,
        )
        diffusion = _finite(self.diffusion_ratio, field_name="diffusion_ratio")
        consistency = _finite(self.consistency_ratio, field_name="consistency_ratio")
        guidance = _finite(
            self.fused_guidance_scale,
            field_name="fused_guidance_scale",
            positive=True,
        )
        if not 0 <= diffusion <= 1 or not 0 <= consistency <= 1:
            raise ValueError("flow-map mixture ratios must lie in [0,1]")
        if diffusion + consistency > 1:
            raise ValueError("diffusion_ratio + consistency_ratio cannot exceed one")
        if diffusion == 0:
            raise ValueError("cross-objective balancing requires diffusion_ratio > 0")
        object.__setattr__(self, "num_train_timesteps", steps)
        object.__setattr__(self, "timestep_shift", shift)
        object.__setattr__(self, "central_difference_epsilon", epsilon)
        object.__setattr__(self, "diffusion_ratio", diffusion)
        object.__setattr__(self, "consistency_ratio", consistency)
        object.__setattr__(self, "fused_guidance_scale", guidance)


@dataclass(frozen=True, slots=True)
class AnyFlowFARConfig:
    """Frame-autoregressive partition and long-context sampling."""

    chunk_partition: tuple[int, ...] = (1, 3, 3, 3, 3, 3, 3, 2)
    full_chunk_limit: int = 3
    patch_size: tuple[int, int, int] = (1, 2, 2)
    compressed_patch_size: tuple[int, int, int] = (1, 4, 4)
    long_context_training_ratio: float = 0.5

    def __post_init__(self) -> None:
        partition = TemporalChunkPartition(
            chunks=tuple(self.chunk_partition),
            full_chunk_limit=self.full_chunk_limit,
            patch_size=tuple(self.patch_size),
            compressed_patch_size=tuple(self.compressed_patch_size),
        )
        ratio = _finite(
            self.long_context_training_ratio,
            field_name="long_context_training_ratio",
        )
        if not 0 <= ratio <= 1:
            raise ValueError("long_context_training_ratio must lie in [0,1]")
        object.__setattr__(self, "chunk_partition", partition.chunks)
        object.__setattr__(self, "full_chunk_limit", partition.full_chunk_limit)
        object.__setattr__(self, "patch_size", partition.patch_size)
        object.__setattr__(
            self,
            "compressed_patch_size",
            partition.compressed_patch_size,
        )
        object.__setattr__(self, "long_context_training_ratio", ratio)

    @property
    def partition(self) -> TemporalChunkPartition:
        return TemporalChunkPartition(
            chunks=self.chunk_partition,
            full_chunk_limit=self.full_chunk_limit,
            patch_size=self.patch_size,
            compressed_patch_size=self.compressed_patch_size,
        )


@dataclass(frozen=True, slots=True)
class AnyFlowPretrainConfig:
    """Causal flow-map pretraining choices consumed by the objective."""

    flow_map: AnyFlowMapConfig = field(default_factory=AnyFlowMapConfig)
    far: AnyFlowFARConfig = field(default_factory=AnyFlowFARConfig)
    bidirectional_modeling_probability: float = 0.1
    conditioning_dropout_probability: float = 0.1

    def __post_init__(self) -> None:
        if not isinstance(self.flow_map, AnyFlowMapConfig):
            raise TypeError("flow_map must be AnyFlowMapConfig")
        if not isinstance(self.far, AnyFlowFARConfig):
            raise TypeError("far must be AnyFlowFARConfig")
        probability = _finite(
            self.bidirectional_modeling_probability,
            field_name="bidirectional_modeling_probability",
        )
        if not 0 <= probability <= 1:
            raise ValueError("bidirectional_modeling_probability must lie in [0,1]")
        dropout = _finite(
            self.conditioning_dropout_probability,
            field_name="conditioning_dropout_probability",
        )
        if not 0 <= dropout <= 1:
            raise ValueError("conditioning_dropout_probability must lie in [0,1]")
        object.__setattr__(
            self,
            "bidirectional_modeling_probability",
            probability,
        )
        object.__setattr__(self, "conditioning_dropout_probability", dropout)

    @property
    def digest(self) -> str:
        return canonical_sha256({"schema": "worldfoundry-anyflow-pretrain", **asdict(self)})


@dataclass(frozen=True, slots=True)
class AnyFlowOnPolicyConfig:
    """On-policy DMD, fresh fake-score, FAR cotraining, and cadence."""

    flow_map: AnyFlowMapConfig = field(default_factory=AnyFlowMapConfig)
    far: AnyFlowFARConfig = field(default_factory=AnyFlowFARConfig)
    inference_steps: tuple[int, ...] = (2, 4, 8, 16, 50)
    dmd_weight: float = 1.0
    real_guidance_scale: float = 3.0
    fake_score_logit_mean: float = 0.0
    fake_score_logit_std: float = 1.0
    dmd_batch_size: int = 2
    dmd_min_timestep: float = 0.0
    dmd_max_timestep: float | None = None
    bidirectional_modeling_probability: float = 0.1
    conditioning_dropout_probability: float = 0.1
    cotrain_flowmap: bool = True
    discriminator_update_ratio: int = 1
    ema_decay: float = 0.99
    ema_warmup_steps: int = 200
    synchronized_seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.flow_map, AnyFlowMapConfig):
            raise TypeError("flow_map must be AnyFlowMapConfig")
        if not isinstance(self.far, AnyFlowFARConfig):
            raise TypeError("far must be AnyFlowFARConfig")
        schedule = tuple(_positive_integer(value, field_name="inference_steps") for value in self.inference_steps)
        if not schedule or len(set(schedule)) != len(schedule):
            raise ValueError("inference_steps must be non-empty and unique")
        if tuple(sorted(schedule)) != schedule:
            raise ValueError("inference_steps must be strictly increasing")
        weight = _finite(self.dmd_weight, field_name="dmd_weight", positive=True)
        guidance = _finite(
            self.real_guidance_scale,
            field_name="real_guidance_scale",
            positive=True,
        )
        logit_mean = _finite(
            self.fake_score_logit_mean,
            field_name="fake_score_logit_mean",
        )
        logit_std = _finite(
            self.fake_score_logit_std,
            field_name="fake_score_logit_std",
            positive=True,
        )
        dmd_batch_size = _positive_integer(
            self.dmd_batch_size,
            field_name="dmd_batch_size",
        )
        minimum = _finite(
            self.dmd_min_timestep,
            field_name="dmd_min_timestep",
        )
        maximum = (
            float(self.flow_map.num_train_timesteps)
            if self.dmd_max_timestep is None
            else _finite(
                self.dmd_max_timestep,
                field_name="dmd_max_timestep",
            )
        )
        if not 0 <= minimum < maximum <= self.flow_map.num_train_timesteps:
            raise ValueError("DMD timestep bounds must satisfy 0 <= min < max <= num_train_timesteps")
        bidirectional_probability = _finite(
            self.bidirectional_modeling_probability,
            field_name="bidirectional_modeling_probability",
        )
        if not 0 <= bidirectional_probability <= 1:
            raise ValueError("bidirectional_modeling_probability must lie in [0,1]")
        dropout = _finite(
            self.conditioning_dropout_probability,
            field_name="conditioning_dropout_probability",
        )
        if not 0 <= dropout <= 1:
            raise ValueError("conditioning_dropout_probability must lie in [0,1]")
        if not isinstance(self.cotrain_flowmap, bool):
            raise TypeError("cotrain_flowmap must be bool")
        ratio = _positive_integer(
            self.discriminator_update_ratio,
            field_name="discriminator_update_ratio",
        )
        decay = _finite(self.ema_decay, field_name="ema_decay")
        if not 0 <= decay < 1:
            raise ValueError("ema_decay must lie in [0,1)")
        if isinstance(self.ema_warmup_steps, bool) or int(self.ema_warmup_steps) < 0:
            raise ValueError("ema_warmup_steps must be a non-negative integer")
        if isinstance(self.synchronized_seed, bool) or int(self.synchronized_seed) < 0:
            raise ValueError("synchronized_seed must be a non-negative integer")
        object.__setattr__(self, "inference_steps", schedule)
        object.__setattr__(self, "dmd_weight", weight)
        object.__setattr__(self, "real_guidance_scale", guidance)
        object.__setattr__(self, "fake_score_logit_mean", logit_mean)
        object.__setattr__(self, "fake_score_logit_std", logit_std)
        object.__setattr__(self, "dmd_batch_size", dmd_batch_size)
        object.__setattr__(self, "dmd_min_timestep", minimum)
        object.__setattr__(self, "dmd_max_timestep", maximum)
        object.__setattr__(
            self,
            "bidirectional_modeling_probability",
            bidirectional_probability,
        )
        object.__setattr__(self, "conditioning_dropout_probability", dropout)
        object.__setattr__(self, "discriminator_update_ratio", ratio)
        object.__setattr__(self, "ema_decay", decay)
        object.__setattr__(self, "ema_warmup_steps", int(self.ema_warmup_steps))
        object.__setattr__(self, "synchronized_seed", int(self.synchronized_seed))

    @property
    def digest(self) -> str:
        return canonical_sha256({"schema": "worldfoundry-anyflow-on-policy", **asdict(self)})


@dataclass(frozen=True, slots=True)
class AnyFlowBidirectionalPretrainConfig:
    """Full-video AnyFlow pretraining, with optional first-frame conditioning."""

    flow_map: AnyFlowMapConfig = field(default_factory=AnyFlowMapConfig)
    image_conditioning_probability: float = 0.0
    conditioning_dropout_probability: float = 0.1

    def __post_init__(self) -> None:
        if not isinstance(self.flow_map, AnyFlowMapConfig):
            raise TypeError("flow_map must be AnyFlowMapConfig")
        probability = _finite(
            self.image_conditioning_probability,
            field_name="image_conditioning_probability",
        )
        if not 0 <= probability <= 1:
            raise ValueError("image_conditioning_probability must lie in [0,1]")
        dropout = _finite(
            self.conditioning_dropout_probability,
            field_name="conditioning_dropout_probability",
        )
        if not 0 <= dropout <= 1:
            raise ValueError("conditioning_dropout_probability must lie in [0,1]")
        object.__setattr__(self, "image_conditioning_probability", probability)
        object.__setattr__(self, "conditioning_dropout_probability", dropout)

    @property
    def digest(self) -> str:
        return canonical_sha256({"schema": "worldfoundry-anyflow-bidirectional-pretrain", **asdict(self)})


@dataclass(frozen=True, slots=True)
class AnyFlowBidirectionalOnPolicyConfig:
    """Full-video AnyFlow rollout, DMD, fake score, and FlowMap cotraining."""

    flow_map: AnyFlowMapConfig = field(default_factory=AnyFlowMapConfig)
    inference_steps: tuple[int, ...] = (2, 4, 8, 16, 50)
    dmd_weight: float = 1.0
    real_guidance_scale: float = 3.0
    fake_score_logit_mean: float = 0.0
    fake_score_logit_std: float = 1.0
    dmd_batch_size: int = 2
    dmd_min_timestep: float = 0.0
    dmd_max_timestep: float | None = None
    image_conditioning_probability: float = 0.0
    conditioning_dropout_probability: float = 0.1
    cotrain_flowmap: bool = True
    discriminator_update_ratio: int = 1
    ema_decay: float = 0.99
    ema_warmup_steps: int = 200
    synchronized_seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.flow_map, AnyFlowMapConfig):
            raise TypeError("flow_map must be AnyFlowMapConfig")
        schedule = tuple(_positive_integer(value, field_name="inference_steps") for value in self.inference_steps)
        if not schedule or len(set(schedule)) != len(schedule):
            raise ValueError("inference_steps must be non-empty and unique")
        if tuple(sorted(schedule)) != schedule:
            raise ValueError("inference_steps must be strictly increasing")
        weight = _finite(self.dmd_weight, field_name="dmd_weight", positive=True)
        guidance = _finite(
            self.real_guidance_scale,
            field_name="real_guidance_scale",
            positive=True,
        )
        logit_mean = _finite(
            self.fake_score_logit_mean,
            field_name="fake_score_logit_mean",
        )
        logit_std = _finite(
            self.fake_score_logit_std,
            field_name="fake_score_logit_std",
            positive=True,
        )
        dmd_batch_size = _positive_integer(
            self.dmd_batch_size,
            field_name="dmd_batch_size",
        )
        minimum = _finite(
            self.dmd_min_timestep,
            field_name="dmd_min_timestep",
        )
        maximum = (
            float(self.flow_map.num_train_timesteps)
            if self.dmd_max_timestep is None
            else _finite(
                self.dmd_max_timestep,
                field_name="dmd_max_timestep",
            )
        )
        if not 0 <= minimum < maximum <= self.flow_map.num_train_timesteps:
            raise ValueError("DMD timestep bounds must satisfy 0 <= min < max <= num_train_timesteps")
        image_probability = _finite(
            self.image_conditioning_probability,
            field_name="image_conditioning_probability",
        )
        if not 0 <= image_probability <= 1:
            raise ValueError("image_conditioning_probability must lie in [0,1]")
        dropout = _finite(
            self.conditioning_dropout_probability,
            field_name="conditioning_dropout_probability",
        )
        if not 0 <= dropout <= 1:
            raise ValueError("conditioning_dropout_probability must lie in [0,1]")
        if not isinstance(self.cotrain_flowmap, bool):
            raise TypeError("cotrain_flowmap must be bool")
        ratio = _positive_integer(
            self.discriminator_update_ratio,
            field_name="discriminator_update_ratio",
        )
        decay = _finite(self.ema_decay, field_name="ema_decay")
        if not 0 <= decay < 1:
            raise ValueError("ema_decay must lie in [0,1)")
        if isinstance(self.ema_warmup_steps, bool) or int(self.ema_warmup_steps) < 0:
            raise ValueError("ema_warmup_steps must be a non-negative integer")
        if isinstance(self.synchronized_seed, bool) or int(self.synchronized_seed) < 0:
            raise ValueError("synchronized_seed must be a non-negative integer")
        object.__setattr__(self, "inference_steps", schedule)
        object.__setattr__(self, "dmd_weight", weight)
        object.__setattr__(self, "real_guidance_scale", guidance)
        object.__setattr__(self, "fake_score_logit_mean", logit_mean)
        object.__setattr__(self, "fake_score_logit_std", logit_std)
        object.__setattr__(self, "dmd_batch_size", dmd_batch_size)
        object.__setattr__(self, "dmd_min_timestep", minimum)
        object.__setattr__(self, "dmd_max_timestep", maximum)
        object.__setattr__(self, "image_conditioning_probability", image_probability)
        object.__setattr__(self, "conditioning_dropout_probability", dropout)
        object.__setattr__(self, "discriminator_update_ratio", ratio)
        object.__setattr__(self, "ema_decay", decay)
        object.__setattr__(self, "ema_warmup_steps", int(self.ema_warmup_steps))
        object.__setattr__(self, "synchronized_seed", int(self.synchronized_seed))

    @property
    def digest(self) -> str:
        return canonical_sha256({"schema": "worldfoundry-anyflow-bidirectional-on-policy", **asdict(self)})


__all__ = [
    "AnyFlowBidirectionalOnPolicyConfig",
    "AnyFlowBidirectionalPretrainConfig",
    "AnyFlowFARConfig",
    "AnyFlowMapConfig",
    "AnyFlowOnPolicyConfig",
    "AnyFlowPretrainConfig",
]

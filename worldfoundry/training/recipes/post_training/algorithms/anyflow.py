"""Strict recipe contracts for native AnyFlow training."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from ..common import positive_int, strict_mapping


def _finite(
    value: object,
    *,
    field_name: str,
    minimum: float | None = None,
) -> float:
    resolved = float(value)
    if not isfinite(resolved) or (minimum is not None and resolved < minimum):
        qualifier = "finite" if minimum is None else f"finite and at least {minimum}"
        raise ValueError(f"{field_name} must be {qualifier}")
    return resolved


def _probability(value: object, *, field_name: str) -> float:
    resolved = _finite(value, field_name=field_name)
    if not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{field_name} must lie in [0,1]")
    return resolved


def _checkpoint(value: object, *, field_name: str) -> str:
    resolved = str(value).strip()
    if not resolved:
        raise ValueError(f"{field_name} must be a non-empty checkpoint reference")
    return resolved


@dataclass(frozen=True, slots=True)
class AnyFlowMapSpec:
    """Flow-map schedule and objective mixture used by every AnyFlow mode."""

    num_train_timesteps: int = 1000
    timestep_shift: float = 5.0
    central_difference_epsilon: float = 5.0
    diffusion_ratio: float = 0.5
    consistency_ratio: float = 0.25
    fused_guidance_scale: float = 3.0

    def __post_init__(self) -> None:
        steps = positive_int(
            self.num_train_timesteps,
            field_name="algorithm.flow_map.num_train_timesteps",
        )
        if steps < 2:
            raise ValueError("num_train_timesteps must be at least two")
        shift = _finite(
            self.timestep_shift,
            field_name="algorithm.flow_map.timestep_shift",
            minimum=0.0,
        )
        epsilon = _finite(
            self.central_difference_epsilon,
            field_name="algorithm.flow_map.central_difference_epsilon",
            minimum=0.0,
        )
        guidance = _finite(
            self.fused_guidance_scale,
            field_name="algorithm.flow_map.fused_guidance_scale",
            minimum=0.0,
        )
        if shift == 0.0 or epsilon == 0.0 or guidance == 0.0:
            raise ValueError("AnyFlow timestep shift, central difference epsilon, and guidance must be positive")
        diffusion = _probability(
            self.diffusion_ratio,
            field_name="algorithm.flow_map.diffusion_ratio",
        )
        consistency = _probability(
            self.consistency_ratio,
            field_name="algorithm.flow_map.consistency_ratio",
        )
        if diffusion == 0.0:
            raise ValueError("AnyFlow cross-objective balancing requires diffusion_ratio > 0")
        if diffusion + consistency > 1.0:
            raise ValueError("diffusion_ratio + consistency_ratio cannot exceed one")
        object.__setattr__(self, "num_train_timesteps", steps)
        object.__setattr__(self, "timestep_shift", shift)
        object.__setattr__(self, "central_difference_epsilon", epsilon)
        object.__setattr__(self, "diffusion_ratio", diffusion)
        object.__setattr__(self, "consistency_ratio", consistency)
        object.__setattr__(self, "fused_guidance_scale", guidance)


@dataclass(frozen=True, slots=True)
class AnyFlowFARSpec:
    """Frame-autoregressive partition and dual-resolution patch geometry."""

    chunk_partition: tuple[int, ...] = (1, 3, 3, 3, 3, 3, 3, 2)
    full_chunk_limit: int = 3
    patch_size: tuple[int, int, int] = (1, 2, 2)
    compressed_patch_size: tuple[int, int, int] = (1, 4, 4)
    long_context_training_ratio: float = 0.5

    def __post_init__(self) -> None:
        chunks = tuple(
            positive_int(value, field_name="algorithm.far.chunk_partition") for value in self.chunk_partition
        )
        if not chunks:
            raise ValueError("AnyFlow FAR chunk_partition cannot be empty")
        limit = positive_int(
            self.full_chunk_limit,
            field_name="algorithm.far.full_chunk_limit",
        )
        if limit > len(chunks):
            raise ValueError("full_chunk_limit cannot exceed the FAR chunk count")

        def patch_geometry(value: object, *, field_name: str) -> tuple[int, int, int]:
            if not isinstance(value, (tuple, list)) or len(value) != 3:
                raise ValueError(f"{field_name} must contain exactly three dimensions")
            return tuple(positive_int(item, field_name=field_name) for item in value)  # type: ignore[return-value]

        patch = patch_geometry(self.patch_size, field_name="algorithm.far.patch_size")
        compressed = patch_geometry(
            self.compressed_patch_size,
            field_name="algorithm.far.compressed_patch_size",
        )
        if compressed[0] != patch[0]:
            raise ValueError(
                "full and compressed AnyFlow temporal patch sizes must agree"
            )
        if any(compressed[index] < patch[index] for index in (1, 2)):
            raise ValueError(
                "compressed_patch_size cannot be spatially finer than patch_size"
            )
        ratio = _probability(
            self.long_context_training_ratio,
            field_name="algorithm.far.long_context_training_ratio",
        )
        object.__setattr__(self, "chunk_partition", chunks)
        object.__setattr__(self, "full_chunk_limit", limit)
        object.__setattr__(self, "patch_size", patch)
        object.__setattr__(self, "compressed_patch_size", compressed)
        object.__setattr__(self, "long_context_training_ratio", ratio)


@dataclass(frozen=True, slots=True)
class AnyFlowFARPretrainAlgorithmSpec:
    flow_map: AnyFlowMapSpec = field(default_factory=AnyFlowMapSpec)
    far: AnyFlowFARSpec = field(default_factory=AnyFlowFARSpec)
    bidirectional_modeling_probability: float = 0.1
    conditioning_dropout_probability: float = 0.1
    type: str = "anyflow-far-pretrain"

    def __post_init__(self) -> None:
        algorithm_type = str(self.type).strip().lower().replace("_", "-")
        if algorithm_type != "anyflow-far-pretrain":
            raise ValueError("AnyFlow FAR pretrain type must be 'anyflow-far-pretrain'")
        if not isinstance(self.flow_map, AnyFlowMapSpec):
            raise TypeError("algorithm.flow_map must be AnyFlowMapSpec")
        if not isinstance(self.far, AnyFlowFARSpec):
            raise TypeError("algorithm.far must be AnyFlowFARSpec")
        object.__setattr__(
            self,
            "bidirectional_modeling_probability",
            _probability(
                self.bidirectional_modeling_probability,
                field_name="algorithm.bidirectional_modeling_probability",
            ),
        )
        object.__setattr__(
            self,
            "conditioning_dropout_probability",
            _probability(
                self.conditioning_dropout_probability,
                field_name="algorithm.conditioning_dropout_probability",
            ),
        )
        object.__setattr__(self, "type", algorithm_type)


@dataclass(frozen=True, slots=True)
class AnyFlowBidirectionalPretrainAlgorithmSpec:
    flow_map: AnyFlowMapSpec = field(default_factory=AnyFlowMapSpec)
    image_conditioning_probability: float = 0.0
    conditioning_dropout_probability: float = 0.1
    type: str = "anyflow-bidirectional-pretrain"

    def __post_init__(self) -> None:
        algorithm_type = str(self.type).strip().lower().replace("_", "-")
        if algorithm_type != "anyflow-bidirectional-pretrain":
            raise ValueError("AnyFlow bidirectional pretrain type must be 'anyflow-bidirectional-pretrain'")
        if not isinstance(self.flow_map, AnyFlowMapSpec):
            raise TypeError("algorithm.flow_map must be AnyFlowMapSpec")
        object.__setattr__(
            self,
            "image_conditioning_probability",
            _probability(
                self.image_conditioning_probability,
                field_name="algorithm.image_conditioning_probability",
            ),
        )
        object.__setattr__(
            self,
            "conditioning_dropout_probability",
            _probability(
                self.conditioning_dropout_probability,
                field_name="algorithm.conditioning_dropout_probability",
            ),
        )
        object.__setattr__(self, "type", algorithm_type)


def _validate_on_policy(instance: object, *, expected_type: str) -> None:
    algorithm_type = str(getattr(instance, "type")).strip().lower().replace("_", "-")
    if algorithm_type != expected_type:
        raise ValueError(f"AnyFlow on-policy type must be {expected_type!r}")
    flow_map = getattr(instance, "flow_map")
    if not isinstance(flow_map, AnyFlowMapSpec):
        raise TypeError("algorithm.flow_map must be AnyFlowMapSpec")
    object.__setattr__(
        instance,
        "real_score_checkpoint",
        _checkpoint(
            getattr(instance, "real_score_checkpoint"),
            field_name="algorithm.real_score_checkpoint",
        ),
    )
    object.__setattr__(
        instance,
        "fake_score_checkpoint",
        _checkpoint(
            getattr(instance, "fake_score_checkpoint"),
            field_name="algorithm.fake_score_checkpoint",
        ),
    )
    schedule = tuple(
        positive_int(value, field_name="algorithm.inference_steps") for value in getattr(instance, "inference_steps")
    )
    if not schedule or tuple(sorted(set(schedule))) != schedule:
        raise ValueError("inference_steps must be non-empty, unique, and increasing")
    object.__setattr__(instance, "inference_steps", schedule)
    object.__setattr__(
        instance,
        "dmd_weight",
        _finite(
            getattr(instance, "dmd_weight"),
            field_name="algorithm.dmd_weight",
            minimum=0.0,
        ),
    )
    object.__setattr__(
        instance,
        "real_guidance_scale",
        _finite(
            getattr(instance, "real_guidance_scale"),
            field_name="algorithm.real_guidance_scale",
            minimum=0.0,
        ),
    )
    if getattr(instance, "dmd_weight") == 0.0 or getattr(instance, "real_guidance_scale") == 0.0:
        raise ValueError("AnyFlow DMD weight and real guidance scale must be positive")
    object.__setattr__(
        instance,
        "fake_score_logit_mean",
        _finite(
            getattr(instance, "fake_score_logit_mean"),
            field_name="algorithm.fake_score_logit_mean",
        ),
    )
    logit_std = _finite(
        getattr(instance, "fake_score_logit_std"),
        field_name="algorithm.fake_score_logit_std",
        minimum=0.0,
    )
    if logit_std == 0.0:
        raise ValueError("fake_score_logit_std must be positive")
    object.__setattr__(instance, "fake_score_logit_std", logit_std)
    object.__setattr__(
        instance,
        "dmd_batch_size",
        positive_int(
            getattr(instance, "dmd_batch_size"),
            field_name="algorithm.dmd_batch_size",
        ),
    )
    minimum = _finite(
        getattr(instance, "dmd_min_timestep"),
        field_name="algorithm.dmd_min_timestep",
    )
    configured_maximum = getattr(instance, "dmd_max_timestep")
    maximum = (
        float(flow_map.num_train_timesteps)
        if configured_maximum is None
        else _finite(configured_maximum, field_name="algorithm.dmd_max_timestep")
    )
    if not 0.0 <= minimum < maximum <= flow_map.num_train_timesteps:
        raise ValueError("DMD timestep bounds must satisfy 0 <= min < max <= num_train_timesteps")
    object.__setattr__(instance, "dmd_min_timestep", minimum)
    object.__setattr__(instance, "dmd_max_timestep", maximum)
    dropout = _probability(
        getattr(instance, "conditioning_dropout_probability"),
        field_name="algorithm.conditioning_dropout_probability",
    )
    object.__setattr__(instance, "conditioning_dropout_probability", dropout)
    if not isinstance(getattr(instance, "cotrain_flowmap"), bool):
        raise TypeError("algorithm.cotrain_flowmap must be a bool")
    object.__setattr__(
        instance,
        "discriminator_update_ratio",
        positive_int(
            getattr(instance, "discriminator_update_ratio"),
            field_name="algorithm.discriminator_update_ratio",
        ),
    )
    decay = _probability(
        getattr(instance, "ema_decay"),
        field_name="algorithm.ema_decay",
    )
    if decay == 1.0:
        raise ValueError("ema_decay must be less than one")
    object.__setattr__(instance, "ema_decay", decay)
    warmup = getattr(instance, "ema_warmup_steps")
    seed = getattr(instance, "synchronized_seed")
    if isinstance(warmup, bool) or int(warmup) < 0:
        raise ValueError("ema_warmup_steps must be a non-negative integer")
    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("synchronized_seed must be a non-negative integer")
    object.__setattr__(instance, "ema_warmup_steps", int(warmup))
    object.__setattr__(instance, "synchronized_seed", int(seed))
    object.__setattr__(instance, "type", algorithm_type)


@dataclass(frozen=True, slots=True)
class AnyFlowFAROnPolicyAlgorithmSpec:
    real_score_checkpoint: str
    fake_score_checkpoint: str
    flow_map: AnyFlowMapSpec = field(default_factory=AnyFlowMapSpec)
    far: AnyFlowFARSpec = field(default_factory=AnyFlowFARSpec)
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
    type: str = "anyflow-far-on-policy"

    def __post_init__(self) -> None:
        if not isinstance(self.far, AnyFlowFARSpec):
            raise TypeError("algorithm.far must be AnyFlowFARSpec")
        _validate_on_policy(self, expected_type="anyflow-far-on-policy")
        object.__setattr__(
            self,
            "bidirectional_modeling_probability",
            _probability(
                self.bidirectional_modeling_probability,
                field_name="algorithm.bidirectional_modeling_probability",
            ),
        )


@dataclass(frozen=True, slots=True)
class AnyFlowBidirectionalOnPolicyAlgorithmSpec:
    real_score_checkpoint: str
    fake_score_checkpoint: str
    flow_map: AnyFlowMapSpec = field(default_factory=AnyFlowMapSpec)
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
    type: str = "anyflow-bidirectional-on-policy"

    def __post_init__(self) -> None:
        _validate_on_policy(self, expected_type="anyflow-bidirectional-on-policy")
        object.__setattr__(
            self,
            "image_conditioning_probability",
            _probability(
                self.image_conditioning_probability,
                field_name="algorithm.image_conditioning_probability",
            ),
        )


AnyFlowAlgorithmSpec = (
    AnyFlowFARPretrainAlgorithmSpec
    | AnyFlowFAROnPolicyAlgorithmSpec
    | AnyFlowBidirectionalPretrainAlgorithmSpec
    | AnyFlowBidirectionalOnPolicyAlgorithmSpec
)


_MAP_FIELDS = set(AnyFlowMapSpec.__dataclass_fields__)
_FAR_FIELDS = set(AnyFlowFARSpec.__dataclass_fields__)


def _nested_specs(payload: dict[str, object], *, far: bool) -> dict[str, object]:
    normalized = dict(payload)
    flow_map = normalized.get("flow_map", {})
    normalized["flow_map"] = (
        flow_map
        if isinstance(flow_map, AnyFlowMapSpec)
        else AnyFlowMapSpec(
            **strict_mapping(
                flow_map,
                field_name="algorithm.flow_map",
                allowed=_MAP_FIELDS,
            )
        )
    )
    if far:
        far_value = normalized.get("far", {})
        normalized["far"] = (
            far_value
            if isinstance(far_value, AnyFlowFARSpec)
            else AnyFlowFARSpec(
                **strict_mapping(
                    far_value,
                    field_name="algorithm.far",
                    allowed=_FAR_FIELDS,
                )
            )
        )
    return normalized


def _parse(value: object, algorithm_class: type, *, far: bool):
    payload = strict_mapping(
        value,
        field_name="algorithm",
        allowed=set(algorithm_class.__dataclass_fields__),
    )
    return algorithm_class(**_nested_specs(payload, far=far))


def parse_anyflow_far_pretrain_algorithm(value: object) -> AnyFlowFARPretrainAlgorithmSpec:
    return _parse(value, AnyFlowFARPretrainAlgorithmSpec, far=True)


def parse_anyflow_far_on_policy_algorithm(value: object) -> AnyFlowFAROnPolicyAlgorithmSpec:
    return _parse(value, AnyFlowFAROnPolicyAlgorithmSpec, far=True)


def parse_anyflow_bidirectional_pretrain_algorithm(
    value: object,
) -> AnyFlowBidirectionalPretrainAlgorithmSpec:
    return _parse(value, AnyFlowBidirectionalPretrainAlgorithmSpec, far=False)


def parse_anyflow_bidirectional_on_policy_algorithm(
    value: object,
) -> AnyFlowBidirectionalOnPolicyAlgorithmSpec:
    return _parse(value, AnyFlowBidirectionalOnPolicyAlgorithmSpec, far=False)


__all__ = [
    "AnyFlowAlgorithmSpec",
    "AnyFlowBidirectionalOnPolicyAlgorithmSpec",
    "AnyFlowBidirectionalPretrainAlgorithmSpec",
    "AnyFlowFAROnPolicyAlgorithmSpec",
    "AnyFlowFARPretrainAlgorithmSpec",
    "AnyFlowFARSpec",
    "AnyFlowMapSpec",
    "parse_anyflow_bidirectional_on_policy_algorithm",
    "parse_anyflow_bidirectional_pretrain_algorithm",
    "parse_anyflow_far_on_policy_algorithm",
    "parse_anyflow_far_pretrain_algorithm",
]

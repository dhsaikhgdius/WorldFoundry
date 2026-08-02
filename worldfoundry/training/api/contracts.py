"""Typed, model-independent contracts for native training.

This module intentionally does not import torch.  The base WorldFoundry package
can therefore inspect recipes and manifests without installing the training
extra, while real tensors are still validated through their public ``shape``
interface at the training boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, runtime_checkable

TensorLike: TypeAlias = Any
TensorTree: TypeAlias = TensorLike | Mapping[str, TensorLike]


def _frozen_mapping(value: Mapping[str, Any], *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    normalized = {str(key): item for key, item in value.items()}
    if any(not key.strip() for key in normalized):
        raise ValueError(f"{field_name} keys cannot be empty")
    return MappingProxyType(normalized)


def _shape(value: TensorLike, *, field_name: str) -> tuple[int, ...]:
    raw_shape = getattr(value, "shape", None)
    if raw_shape is None:
        raise TypeError(f"{field_name} must expose a tensor-like shape")
    try:
        shape = tuple(int(item) for item in raw_shape)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field_name} has an invalid shape: {raw_shape!r}") from error
    if any(item < 0 for item in shape):
        raise ValueError(f"{field_name} shape cannot contain negative dimensions: {shape}")
    return shape


def _validate_ids(sample_ids: tuple[str, ...], prompts: tuple[str, ...] | None = None) -> int:
    if not sample_ids:
        raise ValueError("sample_ids cannot be empty")
    if any(not sample_id.strip() for sample_id in sample_ids):
        raise ValueError("sample_ids cannot contain empty values")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_ids must be unique within a batch")
    if prompts is not None and len(prompts) != len(sample_ids):
        raise ValueError(f"prompts has {len(prompts)} items, expected {len(sample_ids)}")
    return len(sample_ids)


def _is_broadcastable(source: tuple[int, ...], target: tuple[int, ...]) -> bool:
    if len(source) > len(target):
        return False
    padded = (1,) * (len(target) - len(source)) + source
    return all(left in (1, right) for left, right in zip(padded, target))


def _mask_is_broadcastable(mask_shape: tuple[int, ...], target_shape: tuple[int, ...]) -> bool:
    if _is_broadcastable(mask_shape, target_shape):
        return True
    # A common channel-free visual mask is [B,T,H,W] for [B,C,T,H,W].
    if len(mask_shape) + 1 == len(target_shape) and mask_shape[:1] == target_shape[:1]:
        return _is_broadcastable((mask_shape[0], 1, *mask_shape[1:]), target_shape)
    return False


def _validate_batch_tensor(value: TensorLike, *, field_name: str, batch_size: int) -> tuple[int, ...]:
    shape = _shape(value, field_name=field_name)
    if not shape or shape[0] != batch_size:
        raise ValueError(f"{field_name} first dimension must be batch size {batch_size}; got {shape}")
    return shape


def _normalize_tensor_tree(
    value: TensorTree,
    *,
    field_name: str,
    batch_size: int,
) -> tuple[TensorTree, Mapping[str, tuple[int, ...]]]:
    if isinstance(value, Mapping):
        if not value:
            raise ValueError(f"{field_name} cannot be an empty mapping")
        normalized = _frozen_mapping(value, field_name=field_name)
        shapes = {
            key: _validate_batch_tensor(item, field_name=f"{field_name}.{key}", batch_size=batch_size)
            for key, item in normalized.items()
        }
        return normalized, MappingProxyType(shapes)
    shape = _validate_batch_tensor(value, field_name=field_name, batch_size=batch_size)
    return value, MappingProxyType({"main": shape})


def _validate_loss_mask(
    mask: TensorTree | None,
    *,
    targets: Mapping[str, tuple[int, ...]],
    batch_size: int,
) -> TensorTree | None:
    if mask is None:
        return None
    if isinstance(mask, Mapping):
        normalized = _frozen_mapping(mask, field_name="loss_mask")
        if set(normalized) != set(targets):
            raise ValueError(
                f"loss_mask keys must match tensor keys; got {sorted(normalized)}, expected {sorted(targets)}"
            )
        for key, item in normalized.items():
            mask_shape = _validate_batch_tensor(item, field_name=f"loss_mask.{key}", batch_size=batch_size)
            if not _mask_is_broadcastable(mask_shape, targets[key]):
                raise ValueError(f"loss_mask.{key} shape {mask_shape} cannot broadcast to {targets[key]}")
        return normalized

    mask_shape = _validate_batch_tensor(mask, field_name="loss_mask", batch_size=batch_size)
    for key, target_shape in targets.items():
        if not _mask_is_broadcastable(mask_shape, target_shape):
            raise ValueError(f"loss_mask shape {mask_shape} cannot broadcast to {key} target {target_shape}")
    return mask


def _validate_sample_weights(value: TensorLike | None, *, batch_size: int) -> None:
    if value is None:
        return
    shape = _shape(value, field_name="sample_weights")
    if shape != (batch_size,):
        raise ValueError(f"sample_weights must have shape ({batch_size},); got {shape}")


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """Raw, normalized media and conditions emitted by a collator.

    Visual media is channel-first ``[B,C,T,H,W]``; images use ``T=1``.
    Model-specific encoders and packing do not belong at this boundary.
    """

    sample_ids: tuple[str, ...]
    prompts: tuple[str, ...]
    pixel_values: TensorLike | None = None
    conditions: Mapping[str, object] = field(default_factory=dict)
    valid_mask: TensorLike | None = None
    sample_weights: TensorLike | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sample_ids = tuple(str(value) for value in self.sample_ids)
        prompts = tuple(str(value) for value in self.prompts)
        batch_size = _validate_ids(sample_ids, prompts)
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "prompts", prompts)
        object.__setattr__(self, "conditions", _frozen_mapping(self.conditions, field_name="conditions"))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata, field_name="metadata"))

        if self.pixel_values is not None:
            media_shape = _validate_batch_tensor(
                self.pixel_values,
                field_name="pixel_values",
                batch_size=batch_size,
            )
            if len(media_shape) != 5:
                raise ValueError(f"pixel_values must be [B,C,T,H,W]; got {media_shape}")
            if any(size == 0 for size in media_shape[1:]):
                raise ValueError(f"pixel_values dimensions must be non-zero; got {media_shape}")
            if self.valid_mask is not None:
                mask_shape = _validate_batch_tensor(
                    self.valid_mask,
                    field_name="valid_mask",
                    batch_size=batch_size,
                )
                if not _mask_is_broadcastable(mask_shape, media_shape):
                    raise ValueError(f"valid_mask shape {mask_shape} cannot broadcast to {media_shape}")
        elif self.valid_mask is not None:
            raise ValueError("valid_mask requires pixel_values")
        _validate_sample_weights(self.sample_weights, batch_size=batch_size)

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)


@dataclass(frozen=True, slots=True)
class PreparedBatch:
    """Model-ready clean latents and encoded conditioning."""

    sample_ids: tuple[str, ...]
    clean_latents: TensorTree
    conditioning: Mapping[str, object] = field(default_factory=dict)
    loss_mask: TensorTree | None = None
    sample_weights: TensorLike | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sample_ids = tuple(str(value) for value in self.sample_ids)
        batch_size = _validate_ids(sample_ids)
        latents, shapes = _normalize_tensor_tree(
            self.clean_latents,
            field_name="clean_latents",
            batch_size=batch_size,
        )
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "clean_latents", latents)
        object.__setattr__(
            self, "loss_mask", _validate_loss_mask(self.loss_mask, targets=shapes, batch_size=batch_size)
        )
        object.__setattr__(self, "conditioning", _frozen_mapping(self.conditioning, field_name="conditioning"))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata, field_name="metadata"))
        _validate_sample_weights(self.sample_weights, batch_size=batch_size)

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)


@dataclass(frozen=True, slots=True)
class ObjectiveBatch:
    """Corrupted latents, targets, and sampled noise levels for one objective."""

    sample_ids: tuple[str, ...]
    model_input: TensorTree
    target: TensorTree
    sigmas: TensorLike
    timesteps: TensorLike
    conditioning: Mapping[str, object] = field(default_factory=dict)
    noise: TensorTree | None = None
    loss_mask: TensorTree | None = None
    sample_weights: TensorLike | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sample_ids = tuple(str(value) for value in self.sample_ids)
        batch_size = _validate_ids(sample_ids)
        model_input, input_shapes = _normalize_tensor_tree(
            self.model_input,
            field_name="model_input",
            batch_size=batch_size,
        )
        target, target_shapes = _normalize_tensor_tree(self.target, field_name="target", batch_size=batch_size)
        if dict(input_shapes) != dict(target_shapes):
            raise ValueError(
                f"model_input shapes {dict(input_shapes)} do not match target shapes {dict(target_shapes)}"
            )
        if self.noise is not None:
            noise, noise_shapes = _normalize_tensor_tree(self.noise, field_name="noise", batch_size=batch_size)
            if dict(noise_shapes) != dict(target_shapes):
                raise ValueError(f"noise shapes {dict(noise_shapes)} do not match target shapes {dict(target_shapes)}")
            object.__setattr__(self, "noise", noise)
        for name, value in (("sigmas", self.sigmas), ("timesteps", self.timesteps)):
            shape = _shape(value, field_name=name)
            if shape and shape[0] != batch_size:
                raise ValueError(f"{name} must be scalar or start with batch size {batch_size}; got {shape}")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "model_input", model_input)
        object.__setattr__(self, "target", target)
        object.__setattr__(
            self,
            "loss_mask",
            _validate_loss_mask(self.loss_mask, targets=target_shapes, batch_size=batch_size),
        )
        object.__setattr__(self, "conditioning", _frozen_mapping(self.conditioning, field_name="conditioning"))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata, field_name="metadata"))
        _validate_sample_weights(self.sample_weights, batch_size=batch_size)

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)


@dataclass(frozen=True, slots=True)
class TrainStepResult:
    """One differentiable loss plus explicitly weighted reporting counts."""

    loss: TensorLike
    losses: Mapping[str, TensorLike]
    metrics: Mapping[str, TensorLike]
    sample_count: int
    latent_token_count: int
    skipped: bool = False
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        loss_shape = _shape(self.loss, field_name="loss")
        if loss_shape not in ((), (1,)):
            raise ValueError(f"loss must be scalar; got {loss_shape}")
        if self.sample_count < 0 or self.latent_token_count < 0:
            raise ValueError("sample_count and latent_token_count must be non-negative")
        object.__setattr__(self, "losses", _frozen_mapping(self.losses, field_name="losses"))
        object.__setattr__(self, "metrics", _frozen_mapping(self.metrics, field_name="metrics"))
        object.__setattr__(self, "diagnostics", _frozen_mapping(self.diagnostics, field_name="diagnostics"))


@runtime_checkable
class TrainModelAdapter(Protocol):
    """Model-owned conditioning/packing seam; it does not own loss math."""

    prediction_type: str
    trainable_module: object
    lora_target_preset: str | None
    fsdp_block_classes: tuple[type, ...]

    def prepare_batch(self, batch: TrainingBatch) -> PreparedBatch: ...

    def forward_train(self, batch: ObjectiveBatch) -> TensorTree: ...


@runtime_checkable
class TrainingObjective(Protocol):
    """Objective-owned corruption and reduction seam."""

    prediction_type: str

    def corrupt(self, batch: PreparedBatch, *, generator: object | None = None) -> ObjectiveBatch: ...

    def compute_loss(self, prediction: TensorTree, batch: ObjectiveBatch) -> TrainStepResult: ...


__all__ = [
    "ObjectiveBatch",
    "PreparedBatch",
    "TensorLike",
    "TensorTree",
    "TrainModelAdapter",
    "TrainStepResult",
    "TrainingBatch",
    "TrainingObjective",
]

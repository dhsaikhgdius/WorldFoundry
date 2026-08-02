"""Tensor-light contracts shared across native post-training algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

TensorLike = Any


def tensor_shape(value: object, *, field_name: str) -> tuple[int, ...]:
    """Return a validated public tensor-like shape without importing PyTorch."""

    raw = getattr(value, "shape", None)
    if raw is None:
        raise TypeError(f"{field_name} must expose a tensor-like shape")
    try:
        shape = tuple(int(item) for item in raw)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field_name} has an invalid shape: {raw!r}") from error
    if any(item < 0 for item in shape):
        raise ValueError(f"{field_name} shape cannot contain negative dimensions")
    return shape


def freeze_mapping(value: Mapping[str, object], *, field_name: str) -> Mapping[str, object]:
    """Normalize string keys and expose an immutable mapping view."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    normalized = {str(key): item for key, item in value.items()}
    if any(not key.strip() for key in normalized):
        raise ValueError(f"{field_name} keys cannot be empty")
    return MappingProxyType(normalized)


def non_empty_ids(
    values: tuple[str, ...],
    *,
    field_name: str,
    unique: bool,
) -> tuple[str, ...]:
    """Validate and normalize an immutable identifier sequence."""

    normalized = tuple(str(value) for value in values)
    if not normalized or any(not value.strip() for value in normalized):
        raise ValueError(f"{field_name} must contain non-empty strings")
    if unique and len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must be unique")
    return normalized


def is_broadcastable(source: tuple[int, ...], target: tuple[int, ...]) -> bool:
    """Return whether ``source`` follows PyTorch trailing-dimension broadcast rules."""

    if len(source) > len(target):
        return False
    padded = (1,) * (len(target) - len(source)) + source
    return all(left in (1, right) for left, right in zip(padded, target))


@runtime_checkable
class FlowPredictionAdapter(Protocol):
    """Model seam shared by distillation and flow-policy replay."""

    module: object

    def predict_velocity(
        self,
        noisy_latents: TensorLike,
        sigmas: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> TensorLike: ...

    def predict_clean(
        self,
        noisy_latents: TensorLike,
        sigmas: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> TensorLike: ...


__all__ = [
    "FlowPredictionAdapter",
    "TensorLike",
    "freeze_mapping",
    "is_broadcastable",
    "non_empty_ids",
    "tensor_shape",
]

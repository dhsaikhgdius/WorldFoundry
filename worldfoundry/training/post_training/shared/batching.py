"""Conditioning batching primitives shared by post-training data paths."""

from __future__ import annotations

from collections.abc import Mapping
from math import prod

import torch


def latent_token_count(tensor: torch.Tensor) -> int:
    """Count per-batch latent tokens for a ``[B, C, ...]`` latent tensor."""

    if tensor.ndim < 2:
        raise ValueError("latent tensor must include batch and channel/feature dimensions")
    return int(tensor.shape[0]) * prod(int(size) for size in tensor.shape[2:])


def batch_shared_conditioning(
    shared: Mapping[str, object],
    positive: Mapping[str, object],
    *,
    batch_size: int,
) -> dict[str, object]:
    """Broadcast one shared conditioning branch to a positive batch."""

    if not isinstance(shared, Mapping):
        raise TypeError("shared unconditional conditioning must be a mapping")
    values = {str(key): value for key, value in shared.items()}
    if set(values) != set(positive):
        raise ValueError(
            "positive and unconditional conditioning keys must match exactly; "
            f"positive={sorted(positive)}, unconditional={sorted(values)}"
        )
    result: dict[str, object] = {}
    for key, value in values.items():
        reference = positive[key]
        if isinstance(reference, torch.Tensor):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"unconditional conditioning {key!r} must be a torch.Tensor")
            tensor = value.detach()
            expected = tuple(int(size) for size in reference.shape)
            if not expected or expected[0] != batch_size:
                raise ValueError(f"positive conditioning {key!r} must start with batch size {batch_size}")
            if tuple(tensor.shape) == expected[1:]:
                tensor = tensor.unsqueeze(0)
            if tuple(tensor.shape) == (1, *expected[1:]):
                tensor = tensor.expand(expected)
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"unconditional conditioning {key!r} shape {tuple(value.shape)} "
                    f"cannot form positive shape {expected}"
                )
            dtype = reference.dtype if tensor.is_floating_point() else tensor.dtype
            tensor = tensor.to(device=reference.device, dtype=dtype)
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"unconditional conditioning {key!r} contains NaN or infinity")
            result[key] = tensor
        else:
            if isinstance(value, torch.Tensor):
                raise TypeError(f"unconditional conditioning {key!r} differs in value kind")
            result[key] = value
    return result


__all__ = ["batch_shared_conditioning", "latent_token_count"]

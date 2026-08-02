"""Reusable selective activation-checkpoint configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import torch


class CheckpointMode(str, Enum):
    """Supported activation-checkpoint policies."""

    NONE = "none"
    MM_ONLY = "mm_only"
    BLOCK_WISE = "block_wise"

    def __str__(self) -> str:
        return self.value


_MATRIX_OPS = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten.addmm.default,
}


def matrix_ops_checkpoint_policy(context: Any, function: Any, *args: Any, **kwargs: Any) -> Any:
    """Save matrix/attention kernels and recompute inexpensive surrounding ops."""

    del context, args, kwargs
    from torch.utils.checkpoint import CheckpointPolicy

    save = function in _MATRIX_OPS or "flash_attn" in str(function)
    return CheckpointPolicy.MUST_SAVE if save else CheckpointPolicy.PREFER_RECOMPUTE


def matrix_ops_context_fn() -> tuple[Any, Any]:
    from torch.utils.checkpoint import create_selective_checkpoint_contexts

    return create_selective_checkpoint_contexts(matrix_ops_checkpoint_policy)


def block_context_fn() -> tuple[Any, Any]:
    from torch.utils.checkpoint import noop_context_fn

    return noop_context_fn()


@dataclass(frozen=True, slots=True)
class SACConfig:
    """Configuration consumed by models that optionally wrap whole blocks."""

    mode: CheckpointMode | str = CheckpointMode.MM_ONLY
    every_n_blocks: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", CheckpointMode(self.mode))
        if self.every_n_blocks <= 0:
            raise ValueError("every_n_blocks must be positive")

    def get_context_fn(self) -> Callable[[], tuple[Any, Any]]:
        if self.mode is CheckpointMode.MM_ONLY:
            return matrix_ops_context_fn
        if self.mode is CheckpointMode.BLOCK_WISE:
            return block_context_fn
        raise ValueError("CheckpointMode.NONE does not define a checkpoint context")


__all__ = [
    "CheckpointMode",
    "SACConfig",
    "block_context_fn",
    "matrix_ops_checkpoint_policy",
    "matrix_ops_context_fn",
]

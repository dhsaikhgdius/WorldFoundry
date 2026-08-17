"""Shared conditioning artifacts written by Wan cache materializers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import torch

from ..shared_conditioning import SharedConditioningArtifact, SharedConditioningStore


class _CacheRoot(Protocol):
    root: Path


def write_wan_unconditional_conditioning(
    *,
    store: _CacheRoot,
    context: torch.Tensor,
    model_recipe: str,
    conditioner: Mapping[str, object],
    tokenizer: Mapping[str, object],
) -> SharedConditioningArtifact:
    """Bind the empty-prompt conditioning branch once per Wan cache."""

    return SharedConditioningStore(store.root).write(
        branch="unconditional",
        prompt="",
        model_recipe=model_recipe,
        conditioner=conditioner,
        tokenizer=tokenizer,
        tensors={"context": context},
        layouts={"context": "sequence-features"},
    )


__all__ = ["write_wan_unconditional_conditioning"]

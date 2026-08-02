"""Shared conditioning artifacts written by Wan cache materializers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch

from ..sana_cache import text_sha256
from ..shared_conditioning import SharedConditioningArtifact, SharedConditioningStore
from .contracts import wan_cache_contract_digest


class _CacheRoot(Protocol):
    root: Path


def write_wan_unconditional_conditioning(
    *,
    store: _CacheRoot,
    context: torch.Tensor,
    model_recipe: str,
    conditioner_digest: str,
    tokenizer_digest: str,
) -> SharedConditioningArtifact:
    """Bind the empty-prompt conditioning branch once per Wan cache."""

    return SharedConditioningStore(store.root).write(
        branch="unconditional",
        prompt_sha256=text_sha256(""),
        model_recipe_digest=wan_cache_contract_digest(model_recipe),
        conditioner_digest=conditioner_digest,
        tokenizer_digest=tokenizer_digest,
        tensors={"context": context},
        layouts={"context": "sequence-features"},
    )


__all__ = ["write_wan_unconditional_conditioning"]

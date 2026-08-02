"""Diffusion-facing exports of WorldFoundry core runtime policies."""

from __future__ import annotations

import torch

from worldfoundry.core.model_loading.policy import (
    AttentionBackend,
    OffloadMode,
    OffloadPolicy,
    QuantizationMode,
    QuantizationPolicy,
    RuntimePolicy,
)


def parse_torch_dtype(value: object, *, owner: str = "diffusion model") -> torch.dtype:
    """Normalize public dtype spellings at the shared diffusion boundary."""

    normalized = str(value or "bfloat16").lower().removeprefix("torch.")
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError(f"unsupported {owner} dtype: {value!r}") from error


def parse_offload_policy(value: object, *, allow_disk: bool = True, owner: str = "diffusion model") -> OffloadPolicy:
    """Build the framework-owned offload policy from a public option."""

    normalized = str(value or "block").lower()
    if normalized in {"none", "false", "0"}:
        return OffloadPolicy()
    if normalized in {"block", "cpu", "layer"}:
        return OffloadPolicy(mode=OffloadMode.BLOCK, target="cpu", pin_memory=True)
    if normalized == "component":
        return OffloadPolicy(mode=OffloadMode.COMPONENT, target="cpu", pin_memory=True)
    if normalized == "disk" and allow_disk:
        return OffloadPolicy(mode=OffloadMode.DISK, target="disk")
    raise ValueError(f"unsupported {owner} offload mode: {value!r}")


__all__ = [
    "AttentionBackend",
    "OffloadMode",
    "OffloadPolicy",
    "QuantizationMode",
    "QuantizationPolicy",
    "RuntimePolicy",
    "parse_offload_policy",
    "parse_torch_dtype",
]

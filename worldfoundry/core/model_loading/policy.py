"""Model-independent runtime placement and optimization policy values.

Policies live in core so diffusion, autoregressive, perception, and other
model families can use the same loading and execution vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import torch


class OffloadMode(str, Enum):
    NONE = "none"
    COMPONENT = "component"
    BLOCK = "block"
    DISK = "disk"


class QuantizationMode(str, Enum):
    NONE = "none"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"
    NVFP4 = "nvfp4"
    GGUF = "gguf"


class AttentionBackend(str, Enum):
    AUTO = "auto"
    TORCH = "torch"
    SDPA = "sdpa"
    FLASH = "flash"
    SAGE = "sage"


@dataclass(frozen=True, slots=True)
class OffloadPolicy:
    mode: OffloadMode = OffloadMode.NONE
    target: str = "cpu"
    pin_memory: bool = False
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", OffloadMode(self.mode))
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))
        if self.mode is OffloadMode.DISK and self.target == "cpu":
            raise ValueError("disk offload requires a concrete disk target")


@dataclass(frozen=True, slots=True)
class QuantizationPolicy:
    mode: QuantizationMode = QuantizationMode.NONE
    compute_dtype: torch.dtype | None = None
    exclude: tuple[str, ...] = ()
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", QuantizationMode(self.mode))
        object.__setattr__(self, "exclude", tuple(self.exclude))
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


@dataclass(frozen=True, slots=True)
class RuntimePolicy:
    """One model-independent placement and optimization policy."""

    device: str | torch.device = "cpu"
    dtype: torch.dtype = torch.float32
    attention: AttentionBackend = AttentionBackend.AUTO
    offload: OffloadPolicy = OffloadPolicy()
    quantization: QuantizationPolicy = QuantizationPolicy()
    compile: bool = False
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", torch.device(self.device))
        object.__setattr__(self, "attention", AttentionBackend(self.attention))
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))
        if not self.dtype.is_floating_point:
            raise ValueError("runtime dtype must be floating point")


__all__ = [
    "AttentionBackend",
    "OffloadMode",
    "OffloadPolicy",
    "QuantizationMode",
    "QuantizationPolicy",
    "RuntimePolicy",
]

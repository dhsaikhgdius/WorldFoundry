"""Common shape of trainable adapter applications and exported artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from torch import nn


class AdapterApplication(Protocol):
    model: nn.Module
    targeted_module_names: tuple[str, ...]
    trainable_parameter_names: tuple[str, ...]
    trainable_parameter_count: int


class AdapterArtifact(Protocol):
    path: Path
    file_size_bytes: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class ExportedAdapterArtifact:
    path: Path
    file_size_bytes: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(
            self,
            "file_size_bytes",
            MappingProxyType({str(name): int(size) for name, size in self.file_size_bytes.items()}),
        )


__all__ = ["AdapterApplication", "AdapterArtifact", "ExportedAdapterArtifact"]

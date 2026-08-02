"""Independent Echo-Memory synthesis models with lazy Wan runtime imports."""

from __future__ import annotations

from typing import Any

from .echo_memory_synthesis import (
    EchoMemoryBlockSSMSynthesis,
    EchoMemoryContextK1Synthesis,
    EchoMemoryContextK20Synthesis,
    EchoMemorySpatialConcatTextSynthesis,
    EchoMemorySpatialCrossAttnT32Synthesis,
    EchoMemorySpatialNoInjectionSynthesis,
    EchoMemorySpatialSynthesis,
    EchoMemorySSMCtx1Every4Hint21Synthesis,
    EchoMemorySSMCtx5Every1Hint21Synthesis,
    EchoMemorySSMCtx5Every4Hint81Synthesis,
    EchoMemorySynthesis,
    EchoMemoryVideoSSMHybridSynthesis,
)
from .memory import EchoRolloutMemory

__all__ = [
    "EchoMemoryBlockSSMSynthesis",
    "EchoMemoryContextK1Synthesis",
    "EchoMemoryContextK20Synthesis",
    "EchoMemoryRuntime",
    "EchoMemorySSMCtx1Every4Hint21Synthesis",
    "EchoMemorySSMCtx5Every1Hint21Synthesis",
    "EchoMemorySSMCtx5Every4Hint81Synthesis",
    "EchoMemorySpatialConcatTextSynthesis",
    "EchoMemorySpatialCrossAttnT32Synthesis",
    "EchoMemorySpatialNoInjectionSynthesis",
    "EchoMemorySpatialSynthesis",
    "EchoMemorySynthesis",
    "EchoMemoryVideoSSMHybridSynthesis",
    "EchoRolloutMemory",
]


def __getattr__(name: str) -> Any:
    if name == "EchoMemoryRuntime":
        from .runtime import EchoMemoryRuntime

        return EchoMemoryRuntime
    raise AttributeError(name)

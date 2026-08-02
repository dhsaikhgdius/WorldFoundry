"""WorldFoundry memory infrastructure.

This package exposes one canonical memory stack:
bounded records, deterministic retrieval, media artifact memories, and
MosaicMem-inspired spatial patch memories.
"""

from .base import BaseMemory
from .conditioning import (
    DenoisingLayout,
    DenoisingMemoryAdapter,
    GuidanceMode,
    MemoryCondition,
    SequenceRole,
    SequenceSegment,
    SequenceUpdate,
)
from .mosaic import (
    CameraIntrinsics,
    CameraPose,
    LatentCanvas,
    MemoryRetriever,
    MosaicFrame,
    MosaicMemoryConfig,
    MosaicMemoryStore,
    Patch3D,
    RetrievedPatch,
)
from .store import MemoryQuery, MemoryRecord, MemorySelection, MemoryStore

__all__ = [
    "BaseMemory",
    "CameraIntrinsics",
    "CameraPose",
    "DenoisingLayout",
    "DenoisingMemoryAdapter",
    "GuidanceMode",
    "LatentCanvas",
    "MemoryCondition",
    "MemoryQuery",
    "MemoryRecord",
    "MemoryRetriever",
    "MemorySelection",
    "MemoryStore",
    "MosaicFrame",
    "MosaicMemoryConfig",
    "MosaicMemoryStore",
    "Patch3D",
    "RetrievedPatch",
    "SequenceRole",
    "SequenceSegment",
    "SequenceUpdate",
]

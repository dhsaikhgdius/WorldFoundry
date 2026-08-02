"""Validated checkpoint artifacts and their persistent staging contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

SYNCHRONOUS_DCP_STAGING = "synchronous-dcp"
IMMUTABLE_DTENSOR_ASYNC_STAGING = "immutable-dtensor-local-shard-snapshot"
CHECKPOINT_STAGING_STRATEGIES = frozenset({SYNCHRONOUS_DCP_STAGING, IMMUTABLE_DTENSOR_ASYNC_STAGING})
OPTIONAL_TRAINING_STATE_NAMES = (
    "lr_scheduler",
    "ema",
    "grad_scaler",
    "algorithm_state",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def normalize_non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool")
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return resolved


@dataclass(frozen=True, slots=True)
class TrainingCheckpointArtifact:
    """One checksum-verified, atomically committed training checkpoint."""

    path: Path
    global_step: int
    staging_strategy: str
    optional_state_presence: Mapping[str, bool]
    manifest_sha256: str
    identity_digest: str
    file_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(
            self,
            "global_step",
            normalize_non_negative_int(self.global_step, field_name="global_step"),
        )
        if self.staging_strategy not in CHECKPOINT_STAGING_STRATEGIES:
            raise ValueError(f"unsupported checkpoint staging strategy: {self.staging_strategy!r}")
        optional_presence = dict(self.optional_state_presence)
        if set(optional_presence) != set(OPTIONAL_TRAINING_STATE_NAMES) or any(
            not isinstance(value, bool) for value in optional_presence.values()
        ):
            raise ValueError("optional_state_presence is invalid")
        object.__setattr__(
            self,
            "optional_state_presence",
            MappingProxyType(optional_presence),
        )
        for name, value in (
            ("manifest_sha256", self.manifest_sha256),
            ("identity_digest", self.identity_digest),
        ):
            if SHA256_PATTERN.fullmatch(str(value)) is None:
                raise ValueError(f"{name} must be a SHA-256 digest")
        object.__setattr__(self, "file_sha256", MappingProxyType(dict(self.file_sha256)))


__all__ = [
    "IMMUTABLE_DTENSOR_ASYNC_STAGING",
    "SYNCHRONOUS_DCP_STAGING",
    "TrainingCheckpointArtifact",
]

"""Durable native-training manifests and metric streams."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from worldfoundry.core.io.integrity import canonical_json, replace_json_atomic

TRAINING_RUN_SCHEMA = "worldfoundry-training-run"
TRAINING_METRIC_SCHEMA = "worldfoundry-training-metric"

# Default cadence for LG-07: avoid fsync-per-line while staying durable enough.
_DEFAULT_FSYNC_EVERY_N = 32
_DEFAULT_FSYNC_EVERY_SECONDS = 2.0


def json_value(value: object) -> object:
    """Normalize runtime values into finite, canonical JSON-compatible data."""

    if isinstance(value, torch.Tensor):
        detached = value.detach()
        if detached.numel() == 1:
            return json_value(detached.item())
        return {
            "shape": [int(size) for size in detached.shape],
            "dtype": str(detached.dtype),
            "device": str(detached.device),
        }
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("training run metadata cannot contain NaN or infinity")
        return value
    return str(value)


def write_json_atomic(path: Path, value: object) -> None:
    """Replace one manifest atomically within its run directory."""

    replace_json_atomic(path, json_value(value), root=path.parent)


def training_log_run_id() -> str | None:
    """Return the logging-pipeline run id when one is bound or exported."""

    try:
        from worldfoundry.core.logging_setup import get_log_context
    except Exception:  # pragma: no cover - logging optional in minimal installs
        get_log_context = None  # type: ignore[assignment]
    if get_log_context is not None:
        bound = get_log_context().get("run_id")
        if bound is not None and str(bound).strip():
            return str(bound)
    env_run_id = os.environ.get("WORLDFOUNDRY_RUN_ID", "").strip()
    return env_run_id or None


class MetricWriter:
    """Append canonical JSON metric records with batched durability (LG-07)."""

    def __init__(
        self,
        path: Path,
        *,
        fsync_every_n: int = _DEFAULT_FSYNC_EVERY_N,
        fsync_every_seconds: float = _DEFAULT_FSYNC_EVERY_SECONDS,
    ) -> None:
        if fsync_every_n < 1:
            raise ValueError("fsync_every_n must be >= 1")
        if fsync_every_seconds < 0:
            raise ValueError("fsync_every_seconds must be >= 0")
        self.path = path
        self.fsync_every_n = fsync_every_n
        self.fsync_every_seconds = fsync_every_seconds
        self._handle = path.open("x", encoding="utf-8")
        self._writes_since_fsync = 0
        self._last_fsync_at = time.monotonic()

    def write(self, value: Mapping[str, object]) -> None:
        self._handle.write(canonical_json(json_value(value)) + "\n")
        self._handle.flush()
        self._writes_since_fsync += 1
        due_by_count = self._writes_since_fsync >= self.fsync_every_n
        due_by_time = (
            self.fsync_every_seconds > 0
            and (time.monotonic() - self._last_fsync_at) >= self.fsync_every_seconds
        )
        if due_by_count or due_by_time:
            self.force_fsync()

    def force_fsync(self) -> None:
        """Flush and fsync; call before checkpoints / run completion."""

        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._writes_since_fsync = 0
        self._last_fsync_at = time.monotonic()

    def close(self) -> None:
        try:
            self.force_fsync()
        finally:
            self._handle.close()


class NullMetricWriter:
    """Metric sink used by non-coordinator ranks."""

    def write(self, value: Mapping[str, object]) -> None:
        del value

    def force_fsync(self) -> None:
        return None

    def close(self) -> None:
        return None


__all__ = [
    "MetricWriter",
    "NullMetricWriter",
    "TRAINING_METRIC_SCHEMA",
    "TRAINING_RUN_SCHEMA",
    "json_value",
    "training_log_run_id",
    "write_json_atomic",
]

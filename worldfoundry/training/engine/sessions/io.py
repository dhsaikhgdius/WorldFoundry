"""Durable native-training manifests and metric streams."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from worldfoundry.core.io.integrity import canonical_json, replace_json_atomic

TRAINING_RUN_SCHEMA = "worldfoundry-training-run"
TRAINING_METRIC_SCHEMA = "worldfoundry-training-metric"


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


class MetricWriter:
    """Durably append canonical JSON metric records to a new stream."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = path.open("x", encoding="utf-8")

    def write(self, value: Mapping[str, object]) -> None:
        self._handle.write(canonical_json(json_value(value)) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.close()


class NullMetricWriter:
    """Metric sink used by non-coordinator ranks."""

    def write(self, value: Mapping[str, object]) -> None:
        del value

    def close(self) -> None:
        return None


__all__ = [
    "MetricWriter",
    "NullMetricWriter",
    "TRAINING_METRIC_SCHEMA",
    "TRAINING_RUN_SCHEMA",
    "json_value",
    "write_json_atomic",
]

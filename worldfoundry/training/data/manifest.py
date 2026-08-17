"""Strict, accelerator-free sample manifests for native training."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import unquote, urlsplit

TRAINING_SAMPLE_SCHEMA = "worldfoundry-training-sample"
TRAINING_MANIFEST_REPORT_SCHEMA = "worldfoundry-training-manifest-report"
_IDENTIFIER_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]*")


def _nonempty(value: object, *, field_name: str) -> str:
    resolved = str(value).strip()
    if not resolved:
        raise ValueError(f"{field_name} cannot be empty")
    return resolved


def _identifier(value: object, *, field_name: str, underscores: bool = False) -> str:
    resolved = _nonempty(value, field_name=field_name).lower()
    resolved = resolved.replace("-", "_") if underscores else resolved.replace("_", "-")
    pattern_value = resolved.replace("_", "-")
    if _IDENTIFIER_PATTERN.fullmatch(pattern_value) is None:
        raise ValueError(f"{field_name} contains unsupported characters: {value!r}")
    return resolved


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field_name} must be positive; got {resolved}")
    return resolved


def _positive_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric, not bool")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{field_name} must be finite and positive; got {resolved}")
    return resolved


def _freeze_json(value: object, *, field_name: str) -> object:
    """Validate and recursively freeze one JSON-compatible value."""

    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            resolved_key = str(key)
            if not resolved_key.strip():
                raise ValueError(f"{field_name} keys cannot be empty")
            normalized[resolved_key] = _freeze_json(item, field_name=f"{field_name}.{resolved_key}")
        return MappingProxyType(normalized)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item, field_name=field_name) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{field_name} cannot contain NaN or infinity")
        return value
    raise TypeError(f"{field_name} must contain only JSON-compatible values; got {type(value).__name__}")


def _frozen_mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    frozen = _freeze_json(value, field_name=field_name)
    assert isinstance(frozen, Mapping)
    return frozen


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _strict_mapping(
    value: object,
    *,
    field_name: str,
    allowed: set[str],
    required: set[str] = frozenset(),
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    payload = {str(key): item for key, item in value.items()}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"{field_name} contains unknown fields: {unknown}")
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{field_name} is missing required fields: {missing}")
    return payload


@dataclass(frozen=True, slots=True)
class MediaReference:
    """Media referenced by one training sample."""

    uri: str
    mime_type: str | None = None
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        uri = _nonempty(self.uri, field_name="media.uri")
        mime_type = None if self.mime_type is None else _nonempty(self.mime_type, field_name="media.mime_type")
        size_bytes = self.size_bytes
        if size_bytes is not None:
            size_bytes = _positive_int(size_bytes, field_name="media.size_bytes")
        object.__setattr__(self, "uri", uri)
        object.__setattr__(self, "mime_type", mime_type)
        object.__setattr__(self, "size_bytes", size_bytes)

    @classmethod
    def from_mapping(cls, value: object) -> "MediaReference":
        payload = _strict_mapping(
            value,
            field_name="media",
            allowed={"uri", "mime_type", "size_bytes"},
            required={"uri"},
        )
        return cls(**payload)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"uri": self.uri}
        if self.mime_type is not None:
            result["mime_type"] = self.mime_type
        if self.size_bytes is not None:
            result["size_bytes"] = self.size_bytes
        return result


@dataclass(frozen=True, slots=True)
class TrainingSample:
    """One canonical image/video/world-model training record."""

    sample_id: str
    task: str
    prompt: str
    media: MediaReference
    width: int
    height: int
    num_frames: int
    fps: float
    conditions: Mapping[str, object]
    split: str
    safety: Mapping[str, object]
    schema: str = TRAINING_SAMPLE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TRAINING_SAMPLE_SCHEMA:
            raise ValueError(f"unsupported training sample schema {self.schema!r}; expected {TRAINING_SAMPLE_SCHEMA!r}")
        object.__setattr__(self, "sample_id", _nonempty(self.sample_id, field_name="sample_id"))
        object.__setattr__(self, "task", _identifier(self.task, field_name="task", underscores=True))
        object.__setattr__(self, "prompt", _nonempty(self.prompt, field_name="prompt"))
        if not isinstance(self.media, MediaReference):
            raise TypeError("media must be a MediaReference")
        object.__setattr__(self, "width", _positive_int(self.width, field_name="width"))
        object.__setattr__(self, "height", _positive_int(self.height, field_name="height"))
        object.__setattr__(self, "num_frames", _positive_int(self.num_frames, field_name="num_frames"))
        object.__setattr__(self, "fps", _positive_float(self.fps, field_name="fps"))
        object.__setattr__(self, "conditions", _frozen_mapping(self.conditions, field_name="conditions"))
        object.__setattr__(self, "split", _identifier(self.split, field_name="split"))
        object.__setattr__(self, "safety", _frozen_mapping(self.safety, field_name="safety"))

    @classmethod
    def from_mapping(cls, value: object) -> "TrainingSample":
        payload = _strict_mapping(
            value,
            field_name="training sample",
            allowed={
                "schema",
                "sample_id",
                "task",
                "prompt",
                "media",
                "width",
                "height",
                "num_frames",
                "fps",
                "conditions",
                "split",
                "safety",
            },
            required={
                "schema",
                "sample_id",
                "task",
                "prompt",
                "media",
                "width",
                "height",
                "num_frames",
                "fps",
                "conditions",
                "split",
                "safety",
            },
        )
        payload["media"] = MediaReference.from_mapping(payload["media"])
        return cls(**payload)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "sample_id": self.sample_id,
            "task": self.task,
            "prompt": self.prompt,
            "media": self.media.to_dict(),
            "width": self.width,
            "height": self.height,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "conditions": _plain_json(self.conditions),
            "split": self.split,
            "safety": _plain_json(self.safety),
        }


@dataclass(frozen=True, slots=True)
class ManifestIssue:
    severity: str
    code: str
    message: str
    row_number: int | None = None
    sample_id: str | None = None

    def __post_init__(self) -> None:
        severity = str(self.severity).lower()
        if severity not in {"error", "warning"}:
            raise ValueError(f"unsupported manifest issue severity: {severity!r}")
        object.__setattr__(self, "severity", severity)

    def to_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "row_number": self.row_number,
            "sample_id": self.sample_id,
        }


@dataclass(frozen=True, slots=True)
class TrainingManifestReport:
    path: str
    requested_split: str | None
    row_count: int
    valid_sample_count: int
    selected_sample_count: int
    task_counts: Mapping[str, int]
    split_counts: Mapping[str, int]
    issues: tuple[ManifestIssue, ...]
    schema: str = TRAINING_MANIFEST_REPORT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_counts", MappingProxyType(dict(self.task_counts)))
        object.__setattr__(self, "split_counts", MappingProxyType(dict(self.split_counts)))
        object.__setattr__(self, "issues", tuple(self.issues))

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "path": self.path,
            "requested_split": self.requested_split,
            "ok": self.ok,
            "row_count": self.row_count,
            "valid_sample_count": self.valid_sample_count,
            "selected_sample_count": self.selected_sample_count,
            "task_counts": dict(self.task_counts),
            "split_counts": dict(self.split_counts),
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [issue.to_dict() for issue in self.issues],
        }


class TrainingManifestError(ValueError):
    """Raised when a manifest is invalid."""

    def __init__(self, report: TrainingManifestReport) -> None:
        self.report = report
        first = next((issue for issue in report.issues if issue.severity == "error"), None)
        detail = "unknown validation error" if first is None else f"{first.code}: {first.message}"
        super().__init__(f"training manifest validation failed with {report.error_count} error(s): {detail}")


@dataclass(frozen=True, slots=True)
class LoadedTrainingManifest:
    path: Path
    samples: tuple[TrainingSample, ...]
    report: TrainingManifestReport

def resolve_local_media_path(media: MediaReference, *, manifest_path: str | Path) -> Path | None:
    """Resolve local/file media; return ``None`` for remote URI schemes."""

    parsed = urlsplit(media.uri)
    if parsed.scheme and parsed.scheme != "file":
        return None
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"}:
            return None
        path = Path(unquote(parsed.path))
    else:
        path = Path(unquote(media.uri))
    if not path.is_absolute():
        path = Path(manifest_path).parent / path
    return path.resolve()


def _read_manifest_rows(path: Path) -> list[tuple[int, object]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        rows: list[tuple[int, object]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append((line_number, json.loads(line)))
                except json.JSONDecodeError as error:
                    rows.append((line_number, error))
        return rows
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            payload = payload.get("samples")
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
            raise TypeError("JSON training manifest must be a list or an object with a 'samples' list")
        return [(index, row) for index, row in enumerate(payload, start=1)]
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ModuleNotFoundError as error:
            raise RuntimeError("Parquet training manifests require the 'train-core' pyarrow dependency") from error
        return [(index, row) for index, row in enumerate(parquet.read_table(path).to_pylist(), start=1)]
    raise ValueError(f"training manifest must be .jsonl, .ndjson, .json, or .parquet: {path}")


def _inspect_training_manifest(
    path: str | Path,
    *,
    split: str | None,
    verify_files: bool,
) -> tuple[tuple[TrainingSample, ...], TrainingManifestReport]:
    source = Path(path).resolve()
    requested_split = None if split is None else _identifier(split, field_name="split")
    issues: list[ManifestIssue] = []
    samples: list[TrainingSample] = []
    row_count = 0
    try:
        if not source.is_file():
            raise FileNotFoundError(f"training manifest not found: {source}")
        rows = _read_manifest_rows(source)
        row_count = len(rows)
    except Exception as error:  # noqa: BLE001 - inspector returns structured diagnostics.
        issues.append(ManifestIssue("error", "manifest-read", f"{type(error).__name__}: {error}"))
        report = TrainingManifestReport(
            path=str(source),
            requested_split=requested_split,
            row_count=row_count,
            valid_sample_count=0,
            selected_sample_count=0,
            task_counts={},
            split_counts={},
            issues=tuple(issues),
        )
        return (), report

    sample_rows: dict[str, int] = {}
    for row_number, row in rows:
        if isinstance(row, json.JSONDecodeError):
            issues.append(
                ManifestIssue(
                    "error",
                    "invalid-json",
                    f"{row.msg} at column {row.colno}",
                    row_number=row_number,
                )
            )
            continue
        try:
            sample = TrainingSample.from_mapping(row)
        except Exception as error:  # noqa: BLE001 - preserve all row diagnostics.
            sample_id = str(row.get("sample_id")) if isinstance(row, Mapping) and row.get("sample_id") else None
            issues.append(
                ManifestIssue(
                    "error",
                    "invalid-sample",
                    f"{type(error).__name__}: {error}",
                    row_number=row_number,
                    sample_id=sample_id,
                )
            )
            continue

        previous_row = sample_rows.get(sample.sample_id)
        if previous_row is not None:
            issues.append(
                ManifestIssue(
                    "error",
                    "duplicate-sample-id",
                    f"sample_id first appeared at row {previous_row}",
                    row_number=row_number,
                    sample_id=sample.sample_id,
                )
            )
        else:
            sample_rows[sample.sample_id] = row_number
        samples.append(sample)

        local_path = resolve_local_media_path(sample.media, manifest_path=source)
        if verify_files and local_path is not None:
            if not local_path.is_file():
                issues.append(
                    ManifestIssue(
                        "error",
                        "media-not-found",
                        str(local_path),
                        row_number=row_number,
                        sample_id=sample.sample_id,
                    )
                )
                continue
            if sample.media.size_bytes is not None and local_path.stat().st_size != sample.media.size_bytes:
                issues.append(
                    ManifestIssue(
                        "error",
                        "media-size-mismatch",
                        f"expected {sample.media.size_bytes}, got {local_path.stat().st_size}",
                        row_number=row_number,
                        sample_id=sample.sample_id,
                    )
                )

    selected = tuple(sample for sample in samples if requested_split is None or sample.split == requested_split)
    if requested_split is not None and not selected:
        issues.append(
            ManifestIssue("error", "empty-split", f"manifest contains no valid samples for split {requested_split!r}")
        )

    task_counts = Counter(sample.task for sample in samples)
    split_counts = Counter(sample.split for sample in samples)
    report = TrainingManifestReport(
        path=str(source),
        requested_split=requested_split,
        row_count=row_count,
        valid_sample_count=len(samples),
        selected_sample_count=len(selected),
        task_counts=dict(sorted(task_counts.items())),
        split_counts=dict(sorted(split_counts.items())),
        issues=tuple(issues),
    )
    return selected, report


def inspect_training_manifest(
    path: str | Path,
    *,
    split: str | None = None,
    verify_files: bool = True,
) -> TrainingManifestReport:
    """Return all manifest diagnostics without raising on malformed rows."""

    _, report = _inspect_training_manifest(
        path,
        split=split,
        verify_files=verify_files,
    )
    return report


def load_training_manifest(
    path: str | Path,
    *,
    split: str | None = None,
    verify_files: bool = True,
) -> LoadedTrainingManifest:
    """Load a valid manifest or raise with a structured report."""

    samples, report = _inspect_training_manifest(
        path,
        split=split,
        verify_files=verify_files,
    )
    if not report.ok:
        raise TrainingManifestError(report)
    return LoadedTrainingManifest(path=Path(report.path), samples=samples, report=report)


__all__ = [
    "LoadedTrainingManifest",
    "ManifestIssue",
    "MediaReference",
    "TRAINING_MANIFEST_REPORT_SCHEMA",
    "TRAINING_SAMPLE_SCHEMA",
    "TrainingManifestError",
    "TrainingManifestReport",
    "TrainingSample",
    "inspect_training_manifest",
    "load_training_manifest",
    "resolve_local_media_path",
]

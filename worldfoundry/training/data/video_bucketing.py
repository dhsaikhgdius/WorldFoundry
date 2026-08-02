"""Video latent geometry, bucket contracts, and deterministic assignment."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from worldfoundry.core.io.integrity import canonical_sha256 as _core_canonical_sha256

from .manifest import TrainingSample

VIDEO_BUCKET_ASSIGNMENT_SCHEMA = "worldfoundry-video-bucket-assignment"
_TEMPORAL_ALIGNMENTS = frozenset({"first-frame", "uniform"})


def _canonical_sha256(value: object) -> str:
    try:
        return _core_canonical_sha256(value)
    except (TypeError, ValueError) as error:
        raise TypeError("bucket metadata must be JSON serializable without NaN or infinity") from error


def _nonempty(value: object, *, field_name: str) -> str:
    resolved = str(value).strip()
    if not resolved:
        raise ValueError(f"{field_name} cannot be empty")
    return resolved


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool")
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return resolved


def _positive_int(value: object, *, field_name: str) -> int:
    resolved = _non_negative_int(value, field_name=field_name)
    if resolved == 0:
        raise ValueError(f"{field_name} must be positive")
    return resolved


def _positive_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric, not bool")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return resolved


def _strict_mapping(
    value: object,
    *,
    field_name: str,
    allowed: set[str] | frozenset[str],
    required: set[str] | frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    payload = {str(key): item for key, item in value.items()}
    unknown = sorted(set(payload) - set(allowed))
    missing = sorted(set(required) - set(payload))
    if unknown or missing:
        raise ValueError(f"{field_name} fields mismatch; missing={missing}, unknown={unknown}")
    return payload


@dataclass(frozen=True, slots=True)
class VideoLatentShape:
    frames: int
    height: int
    width: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "frames", _positive_int(self.frames, field_name="latent frames"))
        object.__setattr__(self, "height", _positive_int(self.height, field_name="latent height"))
        object.__setattr__(self, "width", _positive_int(self.width, field_name="latent width"))

    @property
    def token_count(self) -> int:
        return self.frames * self.height * self.width

    def to_dict(self) -> dict[str, int]:
        return {"frames": self.frames, "height": self.height, "width": self.width}

    @classmethod
    def from_mapping(cls, value: object) -> VideoLatentShape:
        fields = set(cls.__dataclass_fields__)
        return cls(**_strict_mapping(value, field_name="latent shape", allowed=fields, required=fields))


@dataclass(frozen=True, slots=True)
class VideoLatentGeometry:
    """Exact pixel-to-latent shape rule for one codec contract.

    ``first-frame`` means ``latent_frames = 1 + (frames - 1) / factor`` and
    requires ``(frames - 1)`` to divide exactly.  ``uniform`` means
    ``latent_frames = frames / factor`` and also requires exact divisibility.
    Explicit alignment prevents silent floor/ceil behavior from changing the
    token budget or cached tensor shape.
    """

    spatial_compression_height: int
    spatial_compression_width: int
    temporal_compression: int
    temporal_alignment: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "spatial_compression_height",
            _positive_int(self.spatial_compression_height, field_name="spatial_compression_height"),
        )
        object.__setattr__(
            self,
            "spatial_compression_width",
            _positive_int(self.spatial_compression_width, field_name="spatial_compression_width"),
        )
        object.__setattr__(
            self,
            "temporal_compression",
            _positive_int(self.temporal_compression, field_name="temporal_compression"),
        )
        alignment = _nonempty(self.temporal_alignment, field_name="temporal_alignment").lower().replace("_", "-")
        if alignment not in _TEMPORAL_ALIGNMENTS:
            raise ValueError(f"temporal_alignment must be one of {sorted(_TEMPORAL_ALIGNMENTS)}")
        object.__setattr__(self, "temporal_alignment", alignment)

    def latent_shape(self, *, num_frames: int, height: int, width: int) -> VideoLatentShape:
        frames = _positive_int(num_frames, field_name="num_frames")
        pixel_height = _positive_int(height, field_name="height")
        pixel_width = _positive_int(width, field_name="width")
        if pixel_height % self.spatial_compression_height:
            raise ValueError(
                f"height {pixel_height} is not divisible by spatial compression {self.spatial_compression_height}"
            )
        if pixel_width % self.spatial_compression_width:
            raise ValueError(
                f"width {pixel_width} is not divisible by spatial compression {self.spatial_compression_width}"
            )
        if self.temporal_alignment == "first-frame":
            if (frames - 1) % self.temporal_compression:
                raise ValueError(
                    f"first-frame temporal geometry requires (num_frames - 1) divisible by "
                    f"{self.temporal_compression}; got {frames}"
                )
            latent_frames = 1 + (frames - 1) // self.temporal_compression
        else:
            if frames % self.temporal_compression:
                raise ValueError(
                    f"uniform temporal geometry requires num_frames divisible by "
                    f"{self.temporal_compression}; got {frames}"
                )
            latent_frames = frames // self.temporal_compression
        return VideoLatentShape(
            frames=latent_frames,
            height=pixel_height // self.spatial_compression_height,
            width=pixel_width // self.spatial_compression_width,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "spatial_compression_height": self.spatial_compression_height,
            "spatial_compression_width": self.spatial_compression_width,
            "temporal_compression": self.temporal_compression,
            "temporal_alignment": self.temporal_alignment,
        }

    @classmethod
    def from_mapping(cls, value: object) -> VideoLatentGeometry:
        fields = set(cls.__dataclass_fields__)
        return cls(**_strict_mapping(value, field_name="latent geometry", allowed=fields, required=fields))


@dataclass(frozen=True, slots=True)
class VideoBucketKey:
    task: str
    latent_frames: int
    latent_height: int
    latent_width: int
    aspect_bin: str
    conditioning_layout: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", _nonempty(self.task, field_name="bucket task").lower().replace("-", "_"))
        object.__setattr__(
            self,
            "latent_frames",
            _positive_int(self.latent_frames, field_name="bucket latent_frames"),
        )
        object.__setattr__(
            self,
            "latent_height",
            _positive_int(self.latent_height, field_name="bucket latent_height"),
        )
        object.__setattr__(
            self,
            "latent_width",
            _positive_int(self.latent_width, field_name="bucket latent_width"),
        )
        object.__setattr__(self, "aspect_bin", _nonempty(self.aspect_bin, field_name="bucket aspect_bin"))
        object.__setattr__(
            self,
            "conditioning_layout",
            _nonempty(self.conditioning_layout, field_name="bucket conditioning_layout").lower().replace("_", "-"),
        )

    @property
    def latent_shape(self) -> VideoLatentShape:
        return VideoLatentShape(self.latent_frames, self.latent_height, self.latent_width)

    @property
    def token_count(self) -> int:
        return self.latent_frames * self.latent_height * self.latent_width

    @property
    def digest(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "latent_frames": self.latent_frames,
            "latent_height": self.latent_height,
            "latent_width": self.latent_width,
            "aspect_bin": self.aspect_bin,
            "conditioning_layout": self.conditioning_layout,
        }

    @classmethod
    def from_mapping(cls, value: object) -> VideoBucketKey:
        fields = set(cls.__dataclass_fields__)
        return cls(**_strict_mapping(value, field_name="video bucket key", allowed=fields, required=fields))


@dataclass(frozen=True, slots=True)
class VideoResolutionBucket:
    num_frames: int
    height: int
    width: int
    conditioning_layout: str
    tasks: tuple[str, ...] = ()
    aspect_bin: str | None = None

    def __post_init__(self) -> None:
        frames = _positive_int(self.num_frames, field_name="bucket num_frames")
        height = _positive_int(self.height, field_name="bucket height")
        width = _positive_int(self.width, field_name="bucket width")
        layout = _nonempty(self.conditioning_layout, field_name="conditioning_layout").lower().replace("_", "-")
        tasks = tuple(_nonempty(task, field_name="bucket task").lower().replace("-", "_") for task in self.tasks)
        if len(tasks) != len(set(tasks)):
            raise ValueError("bucket tasks must be unique")
        divisor = math.gcd(width, height)
        aspect = self.aspect_bin
        if aspect is None:
            aspect = f"{width // divisor}:{height // divisor}"
        aspect = _nonempty(aspect, field_name="aspect_bin")
        object.__setattr__(self, "num_frames", frames)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "conditioning_layout", layout)
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "aspect_bin", aspect)

    def key(self, *, task: str, geometry: VideoLatentGeometry) -> VideoBucketKey:
        normalized_task = _nonempty(task, field_name="task").lower().replace("-", "_")
        if self.tasks and normalized_task not in self.tasks:
            raise ValueError(f"task {normalized_task!r} is not enabled for this bucket")
        shape = geometry.latent_shape(num_frames=self.num_frames, height=self.height, width=self.width)
        assert self.aspect_bin is not None
        return VideoBucketKey(
            task=normalized_task,
            latent_frames=shape.frames,
            latent_height=shape.height,
            latent_width=shape.width,
            aspect_bin=self.aspect_bin,
            conditioning_layout=self.conditioning_layout,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "num_frames": self.num_frames,
            "height": self.height,
            "width": self.width,
            "conditioning_layout": self.conditioning_layout,
            "tasks": list(self.tasks),
            "aspect_bin": self.aspect_bin,
        }


@dataclass(frozen=True, slots=True)
class VideoBucketSelectionPolicy:
    aspect_weight: float = 2.0
    spatial_weight: float = 1.0
    temporal_weight: float = 1.0
    allow_spatial_upscale: bool = False
    allow_temporal_padding: bool = False

    def __post_init__(self) -> None:
        for name in ("aspect_weight", "spatial_weight", "temporal_weight"):
            object.__setattr__(self, name, _positive_float(getattr(self, name), field_name=name))
        for name in ("allow_spatial_upscale", "allow_temporal_padding"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")

    def to_dict(self) -> dict[str, object]:
        return {
            "aspect_weight": self.aspect_weight,
            "spatial_weight": self.spatial_weight,
            "temporal_weight": self.temporal_weight,
            "allow_spatial_upscale": self.allow_spatial_upscale,
            "allow_temporal_padding": self.allow_temporal_padding,
        }


@dataclass(frozen=True, slots=True)
class VideoBucketAssignment:
    sample_index: int
    sample_id: str
    source_num_frames: int
    source_height: int
    source_width: int
    target_num_frames: int
    target_height: int
    target_width: int
    bucket_key: VideoBucketKey
    selection_score: float
    schema: str = VIDEO_BUCKET_ASSIGNMENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != VIDEO_BUCKET_ASSIGNMENT_SCHEMA:
            raise ValueError(f"unsupported video bucket assignment schema: {self.schema!r}")
        object.__setattr__(self, "sample_index", _non_negative_int(self.sample_index, field_name="sample_index"))
        object.__setattr__(self, "sample_id", _nonempty(self.sample_id, field_name="sample_id"))
        for name in (
            "source_num_frames",
            "source_height",
            "source_width",
            "target_num_frames",
            "target_height",
            "target_width",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), field_name=name))
        if not isinstance(self.bucket_key, VideoBucketKey):
            raise TypeError("bucket_key must be a VideoBucketKey")
        score = float(self.selection_score)
        if not math.isfinite(score) or score < 0:
            raise ValueError("selection_score must be finite and non-negative")
        object.__setattr__(self, "selection_score", score)

    @property
    def latent_tokens(self) -> int:
        return self.bucket_key.token_count

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "sample_index": self.sample_index,
            "sample_id": self.sample_id,
            "source_num_frames": self.source_num_frames,
            "source_height": self.source_height,
            "source_width": self.source_width,
            "target_num_frames": self.target_num_frames,
            "target_height": self.target_height,
            "target_width": self.target_width,
            "bucket_key": self.bucket_key.to_dict(),
            "selection_score": self.selection_score,
        }


def assign_video_buckets(
    samples: Sequence[TrainingSample],
    *,
    buckets: Sequence[VideoResolutionBucket],
    geometry: VideoLatentGeometry,
    conditioning_layout: str | Sequence[str] | Callable[[TrainingSample], str],
    policy: VideoBucketSelectionPolicy | None = None,
) -> tuple[VideoBucketAssignment, ...]:
    """Assign every manifest sample to one deterministic preprocessing bucket."""

    values = tuple(samples)
    bucket_values = tuple(buckets)
    if not values or not all(isinstance(sample, TrainingSample) for sample in values):
        raise ValueError("samples must be a non-empty sequence of TrainingSample")
    if not bucket_values or not all(isinstance(bucket, VideoResolutionBucket) for bucket in bucket_values):
        raise ValueError("buckets must be a non-empty sequence of VideoResolutionBucket")
    if not isinstance(geometry, VideoLatentGeometry):
        raise TypeError("geometry must be a VideoLatentGeometry")
    resolved_policy = VideoBucketSelectionPolicy() if policy is None else policy
    if not isinstance(resolved_policy, VideoBucketSelectionPolicy):
        raise TypeError("policy must be a VideoBucketSelectionPolicy")

    if isinstance(conditioning_layout, str):
        layouts = (_nonempty(conditioning_layout, field_name="conditioning_layout"),) * len(values)
    elif callable(conditioning_layout):
        layouts = tuple(
            _nonempty(conditioning_layout(sample), field_name=f"conditioning layout for {sample.sample_id!r}")
            for sample in values
        )
    else:
        layouts = tuple(str(item) for item in conditioning_layout)
        if len(layouts) != len(values):
            raise ValueError("conditioning_layout sequence must contain one value per sample")
        layouts = tuple(_nonempty(item, field_name="conditioning_layout") for item in layouts)
    layouts = tuple(layout.lower().replace("_", "-") for layout in layouts)

    assignments: list[VideoBucketAssignment] = []
    for sample_index, (sample, layout) in enumerate(zip(values, layouts)):
        candidates: list[tuple[float, int, VideoResolutionBucket, VideoBucketKey]] = []
        source_aspect = sample.width / sample.height
        source_area = sample.width * sample.height
        for declaration_index, bucket in enumerate(bucket_values):
            if bucket.conditioning_layout != layout:
                continue
            if bucket.tasks and sample.task not in bucket.tasks:
                continue
            if not resolved_policy.allow_temporal_padding and bucket.num_frames > sample.num_frames:
                continue
            cover_scale = max(bucket.width / sample.width, bucket.height / sample.height)
            if not resolved_policy.allow_spatial_upscale and cover_scale > 1.0 + 1e-12:
                continue
            try:
                key = bucket.key(task=sample.task, geometry=geometry)
            except ValueError:
                continue
            target_aspect = bucket.width / bucket.height
            target_area = bucket.width * bucket.height
            score = (
                resolved_policy.aspect_weight * abs(math.log(target_aspect / source_aspect))
                + resolved_policy.spatial_weight * abs(math.log(target_area / source_area))
                + resolved_policy.temporal_weight * abs(math.log(bucket.num_frames / sample.num_frames))
            )
            candidates.append((score, declaration_index, bucket, key))
        if not candidates:
            raise ValueError(
                f"sample {sample.sample_id!r} has no eligible bucket for task={sample.task!r}, "
                f"layout={layout!r}, shape=({sample.num_frames},{sample.height},{sample.width})"
            )
        score, _, bucket, key = min(candidates, key=lambda item: (item[0], item[1]))
        assignments.append(
            VideoBucketAssignment(
                sample_index=sample_index,
                sample_id=sample.sample_id,
                source_num_frames=sample.num_frames,
                source_height=sample.height,
                source_width=sample.width,
                target_num_frames=bucket.num_frames,
                target_height=bucket.height,
                target_width=bucket.width,
                bucket_key=key,
                selection_score=score,
            )
        )
    return tuple(assignments)


__all__ = [
    "VIDEO_BUCKET_ASSIGNMENT_SCHEMA",
    "VideoBucketAssignment",
    "VideoBucketKey",
    "VideoBucketSelectionPolicy",
    "VideoLatentGeometry",
    "VideoLatentShape",
    "VideoResolutionBucket",
    "assign_video_buckets",
]

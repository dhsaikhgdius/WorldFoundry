"""Deterministic local-video decoding and bucket-aligned pixel datasets."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import overload

import torch
import torch.nn.functional as torch_functional

from worldfoundry.training.api.contracts import TrainingBatch

from .dataset import TrainingManifestDataset
from .manifest import TrainingSample, resolve_local_media_path
from .video_bucketing import VideoBucketAssignment, VideoBucketKey

VIDEO_DECODE_TRANSFORM_SCHEMA = "worldfoundry-video-decode-transform"
_FRAME_SAMPLING_MODES = frozenset({"head", "seeded-random-contiguous", "uniform-full"})
_INTERPOLATION_MODES = frozenset({"bilinear", "bicubic"})
_RESIZE_ROUNDING_MODES = frozenset({"ceil", "floor"})
_SPATIAL_TRANSFORM_MODES = frozenset({"cover-resize-center-crop", "direct-resize"})
_VALUE_RANGES = frozenset({"zero-one", "minus-one-one"})
_THREAD_TYPES = frozenset({"auto", "slice"})


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field_name} must be positive")
    return resolved


def _non_negative_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric, not bool")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return resolved


def video_frame_indices(
    source_num_frames: int,
    target_num_frames: int,
    *,
    mode: str,
    seed: int = 0,
    sample_index: int = 0,
) -> tuple[int, ...]:
    """Return exact decoded-frame ordinals without floating-point rounding."""

    source = _positive_int(source_num_frames, field_name="source_num_frames")
    target = _positive_int(target_num_frames, field_name="target_num_frames")
    if target > source:
        raise ValueError(f"cannot select {target} frames from a {source}-frame source without padding")
    resolved_mode = str(mode).strip().lower().replace("_", "-")
    if resolved_mode not in _FRAME_SAMPLING_MODES:
        raise ValueError(f"frame sampling mode must be one of {sorted(_FRAME_SAMPLING_MODES)}")
    if resolved_mode == "head":
        return tuple(range(target))
    if resolved_mode == "seeded-random-contiguous":
        start = random.Random(int(seed) + int(sample_index)).randint(0, source - target)
        return tuple(range(start, start + target))
    if target == 1:
        return ((source - 1) // 2,)
    denominator = target - 1
    # Round each rational i*(source-1)/(target-1) to nearest, with exact
    # integer arithmetic.  source>=target guarantees strictly increasing ids.
    indices = tuple((2 * index * (source - 1) + denominator) // (2 * denominator) for index in range(target))
    if len(indices) != len(set(indices)) or indices[0] != 0 or indices[-1] != source - 1:
        raise RuntimeError("uniform frame-index construction violated its invariants")
    return indices


@dataclass(frozen=True, slots=True)
class VideoDecodeConfig:
    frame_sampling: str = "head"
    frame_sampling_seed: int = 0
    interpolation: str = "bicubic"
    resize_rounding: str = "ceil"
    spatial_transform: str = "cover-resize-center-crop"
    value_range: str = "minus-one-one"
    decoder_thread_type: str = "auto"
    verify_manifest_frame_count: bool = True
    verify_manifest_geometry: bool = True
    fps_tolerance: float = 0.05
    schema: str = VIDEO_DECODE_TRANSFORM_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != VIDEO_DECODE_TRANSFORM_SCHEMA:
            raise ValueError(f"unsupported video decode transform schema: {self.schema!r}")
        for name, supported in (
            ("frame_sampling", _FRAME_SAMPLING_MODES),
            ("interpolation", _INTERPOLATION_MODES),
            ("resize_rounding", _RESIZE_ROUNDING_MODES),
            ("spatial_transform", _SPATIAL_TRANSFORM_MODES),
            ("value_range", _VALUE_RANGES),
            ("decoder_thread_type", _THREAD_TYPES),
        ):
            value = str(getattr(self, name)).strip().lower().replace("_", "-")
            if value not in supported:
                raise ValueError(f"{name} must be one of {sorted(supported)}")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "frame_sampling_seed", int(self.frame_sampling_seed))
        for name in (
            "verify_manifest_frame_count",
            "verify_manifest_geometry",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        object.__setattr__(
            self,
            "fps_tolerance",
            _non_negative_float(self.fps_tolerance, field_name="fps_tolerance"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "frame_sampling": self.frame_sampling,
            "frame_sampling_seed": self.frame_sampling_seed,
            "interpolation": self.interpolation,
            "resize_rounding": self.resize_rounding,
            "spatial_transform": self.spatial_transform,
            "value_range": self.value_range,
            "decoder_thread_type": self.decoder_thread_type,
            "verify_manifest_frame_count": self.verify_manifest_frame_count,
            "verify_manifest_geometry": self.verify_manifest_geometry,
            "fps_tolerance": self.fps_tolerance,
        }


@dataclass(frozen=True, slots=True)
class DecodedVideoSample:
    sample_id: str
    prompt: str
    pixel_values: torch.Tensor
    valid_mask: torch.Tensor
    assignment: VideoBucketAssignment
    selected_frame_indices: tuple[int, ...]
    decoded_frame_count: int
    decoded_fps: float | None
    frame_sampling: Mapping[str, object]
    spatial_transform: Mapping[str, object]

    def __post_init__(self) -> None:
        if not str(self.sample_id).strip() or not str(self.prompt).strip():
            raise ValueError("decoded video sample id and prompt cannot be empty")
        if not isinstance(self.assignment, VideoBucketAssignment):
            raise TypeError("assignment must be a VideoBucketAssignment")
        if self.assignment.sample_id != self.sample_id:
            raise ValueError("decoded sample id differs from its bucket assignment")
        if not isinstance(self.pixel_values, torch.Tensor) or self.pixel_values.ndim != 4:
            raise TypeError("pixel_values must be an unbatched [C,T,H,W] tensor")
        expected = (
            3,
            self.assignment.target_num_frames,
            self.assignment.target_height,
            self.assignment.target_width,
        )
        if tuple(self.pixel_values.shape) != expected or not self.pixel_values.is_floating_point():
            raise ValueError(f"pixel_values must have floating shape {expected}")
        if not bool(torch.isfinite(self.pixel_values).all()):
            raise ValueError("decoded pixel_values contain NaN or infinity")
        if not isinstance(self.valid_mask, torch.Tensor) or tuple(self.valid_mask.shape) != (1, *expected[1:]):
            raise ValueError("valid_mask must be [1,T,H,W] matching pixel_values")
        if self.valid_mask.dtype is not torch.bool or not bool(self.valid_mask.all()):
            raise ValueError("decoded video valid_mask must be all-true bool")
        indices = tuple(int(index) for index in self.selected_frame_indices)
        if len(indices) != expected[1] or any(index < 0 for index in indices):
            raise ValueError("selected_frame_indices do not match the target frame count")
        if tuple(sorted(indices)) != indices or len(indices) != len(set(indices)):
            raise ValueError("selected_frame_indices must be strictly increasing")
        object.__setattr__(self, "selected_frame_indices", indices)
        object.__setattr__(
            self,
            "decoded_frame_count",
            _positive_int(self.decoded_frame_count, field_name="decoded_frame_count"),
        )
        if self.decoded_fps is not None:
            fps = float(self.decoded_fps)
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("decoded_fps must be finite and positive")
            object.__setattr__(self, "decoded_fps", fps)
        for name in ("frame_sampling", "spatial_transform"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            object.__setattr__(self, name, dict(value))


def _stream_fps(stream: object) -> float | None:
    for name in ("average_rate", "guessed_rate", "base_rate"):
        value = getattr(stream, name, None)
        if value is None:
            continue
        try:
            resolved = float(value)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if math.isfinite(resolved) and resolved > 0:
            return resolved
    return None


def _decode_selected_rgb_frames(
    path: Path,
    *,
    sample: TrainingSample,
    selected_indices: tuple[int, ...],
    config: VideoDecodeConfig,
) -> tuple[torch.Tensor, int, float | None]:
    try:
        import av
    except ModuleNotFoundError as error:
        raise RuntimeError("video decoding requires the 'train-video' PyAV dependency") from error
    selected = set(selected_indices)
    frames: dict[int, torch.Tensor] = {}
    decoded_count = 0
    decoded_fps: float | None = None
    try:
        with av.open(str(path), mode="r") as container:
            if not container.streams.video:
                raise ValueError(f"media contains no video stream: {path}")
            stream = container.streams.video[0]
            stream.thread_type = config.decoder_thread_type.upper()
            decoded_fps = _stream_fps(stream)
            for frame in container.decode(stream):
                ordinal = decoded_count
                decoded_count += 1
                if config.verify_manifest_geometry and (
                    int(frame.height) != sample.height or int(frame.width) != sample.width
                ):
                    raise ValueError(
                        f"decoded frame geometry differs from manifest for {sample.sample_id!r}: "
                        f"{frame.width}x{frame.height} vs {sample.width}x{sample.height}"
                    )
                if ordinal in selected:
                    array = frame.to_ndarray(format="rgb24")
                    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()
                    frames[ordinal] = tensor
                if not config.verify_manifest_frame_count and ordinal >= selected_indices[-1]:
                    break
    except (ValueError, RuntimeError):
        raise
    except Exception as error:
        raise RuntimeError(f"failed to decode video sample {sample.sample_id!r} from {path}") from error

    if config.verify_manifest_frame_count and decoded_count != sample.num_frames:
        raise ValueError(
            f"decoded frame count differs from manifest for {sample.sample_id!r}: "
            f"{decoded_count} vs {sample.num_frames}"
        )
    missing = [index for index in selected_indices if index not in frames]
    if missing:
        raise ValueError(f"video ended before selected frame indices were decoded: {missing}")
    if decoded_fps is not None and abs(decoded_fps - sample.fps) > config.fps_tolerance:
        raise ValueError(
            f"decoded fps differs from manifest for {sample.sample_id!r}: "
            f"{decoded_fps} vs {sample.fps} (tolerance {config.fps_tolerance})"
        )
    return torch.stack([frames[index] for index in selected_indices]), decoded_count, decoded_fps


def _cover_resize_center_crop(
    frames: torch.Tensor,
    *,
    target_height: int,
    target_width: int,
    interpolation: str,
    resize_rounding: str,
) -> tuple[torch.Tensor, dict[str, int]]:
    if frames.ndim != 4 or int(frames.shape[1]) != 3:
        raise ValueError("decoded frames must be [T,3,H,W]")
    source_height, source_width = (int(value) for value in frames.shape[-2:])
    if resize_rounding == "floor":
        if source_width * target_height > target_width * source_height:
            resized_height = target_height
            resized_width = max(target_width, target_height * source_width // source_height)
        elif source_width * target_height < target_width * source_height:
            resized_width = target_width
            resized_height = max(target_height, target_width * source_height // source_width)
        else:
            resized_height = target_height
            resized_width = target_width
    else:
        scale = max(target_height / source_height, target_width / source_width)
        resized_height = max(target_height, math.ceil(source_height * scale))
        resized_width = max(target_width, math.ceil(source_width * scale))
    pixels = frames.to(dtype=torch.float32).div_(255.0)
    if (resized_height, resized_width) != (source_height, source_width):
        pixels = torch_functional.interpolate(
            pixels,
            size=(resized_height, resized_width),
            mode=interpolation,
            align_corners=False,
            antialias=True,
        )
    crop_top = (resized_height - target_height) // 2
    crop_left = (resized_width - target_width) // 2
    pixels = (
        pixels[
            :,
            :,
            crop_top : crop_top + target_height,
            crop_left : crop_left + target_width,
        ]
        .clamp_(0.0, 1.0)
        .contiguous()
    )
    if tuple(pixels.shape[-2:]) != (target_height, target_width):
        raise RuntimeError("cover resize and center crop produced an invalid target shape")
    return pixels, {
        "source_height": source_height,
        "source_width": source_width,
        "resized_height": resized_height,
        "resized_width": resized_width,
        "crop_top": crop_top,
        "crop_left": crop_left,
        "target_height": target_height,
        "target_width": target_width,
    }


def _direct_resize(
    frames: torch.Tensor,
    *,
    target_height: int,
    target_width: int,
    interpolation: str,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Resize every frame to the exact target geometry without preserving aspect ratio."""

    if frames.ndim != 4 or int(frames.shape[1]) != 3:
        raise ValueError("decoded frames must be [T,3,H,W]")
    source_height, source_width = (int(value) for value in frames.shape[-2:])
    pixels = frames.to(dtype=torch.float32).div_(255.0)
    if (source_height, source_width) != (target_height, target_width):
        pixels = torch_functional.interpolate(
            pixels,
            size=(target_height, target_width),
            mode=interpolation,
            align_corners=False,
            antialias=True,
        )
    return pixels.clamp_(0.0, 1.0).contiguous(), {
        "source_height": source_height,
        "source_width": source_width,
        "target_height": target_height,
        "target_width": target_width,
    }


def decode_video_sample(
    sample: TrainingSample,
    assignment: VideoBucketAssignment,
    *,
    media_path: str | Path,
    config: VideoDecodeConfig | None = None,
) -> DecodedVideoSample:
    if not isinstance(sample, TrainingSample):
        raise TypeError("sample must be a TrainingSample")
    if not isinstance(assignment, VideoBucketAssignment):
        raise TypeError("assignment must be a VideoBucketAssignment")
    if assignment.sample_id != sample.sample_id:
        raise ValueError("sample id differs from bucket assignment")
    if (assignment.source_num_frames, assignment.source_height, assignment.source_width) != (
        sample.num_frames,
        sample.height,
        sample.width,
    ):
        raise ValueError("bucket assignment source geometry differs from the manifest sample")
    resolved_config = VideoDecodeConfig() if config is None else config
    if not isinstance(resolved_config, VideoDecodeConfig):
        raise TypeError("config must be a VideoDecodeConfig")
    path = Path(media_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"local video media is missing: {path}")

    indices = video_frame_indices(
        sample.num_frames,
        assignment.target_num_frames,
        mode=resolved_config.frame_sampling,
        seed=resolved_config.frame_sampling_seed,
        sample_index=assignment.sample_index,
    )
    frames, decoded_count, decoded_fps = _decode_selected_rgb_frames(
        path,
        sample=sample,
        selected_indices=indices,
        config=resolved_config,
    )
    if resolved_config.spatial_transform == "direct-resize":
        pixels, spatial_parameters = _direct_resize(
            frames,
            target_height=assignment.target_height,
            target_width=assignment.target_width,
            interpolation=resolved_config.interpolation,
        )
    else:
        pixels, spatial_parameters = _cover_resize_center_crop(
            frames,
            target_height=assignment.target_height,
            target_width=assignment.target_width,
            interpolation=resolved_config.interpolation,
            resize_rounding=resolved_config.resize_rounding,
        )
    if resolved_config.value_range == "minus-one-one":
        pixels = pixels.mul(2.0).sub_(1.0)
    pixel_values = pixels.permute(1, 0, 2, 3).contiguous()
    valid_mask = torch.ones(
        (1, assignment.target_num_frames, assignment.target_height, assignment.target_width),
        dtype=torch.bool,
    )
    frame_sampling = {
        "source_num_frames": sample.num_frames,
        "source_fps": sample.fps,
        "mode": resolved_config.frame_sampling,
        "selected_frame_indices": list(indices),
    }
    if resolved_config.frame_sampling == "seeded-random-contiguous":
        frame_sampling.update(
            {
                "seed": resolved_config.frame_sampling_seed,
                "sample_index": assignment.sample_index,
            }
        )
    spatial_transform = {
        "mode": resolved_config.spatial_transform,
        "interpolation": resolved_config.interpolation,
        "value_range": resolved_config.value_range,
        "parameters": spatial_parameters,
    }
    if resolved_config.spatial_transform == "cover-resize-center-crop":
        spatial_transform["resize_rounding"] = resolved_config.resize_rounding
    return DecodedVideoSample(
        sample_id=sample.sample_id,
        prompt=sample.prompt,
        pixel_values=pixel_values,
        valid_mask=valid_mask,
        assignment=assignment,
        selected_frame_indices=indices,
        decoded_frame_count=decoded_count,
        decoded_fps=decoded_fps,
        frame_sampling=frame_sampling,
        spatial_transform=spatial_transform,
    )


class VideoDecodingDataset(Sequence[DecodedVideoSample]):
    """Map-style decoded view over a validated manifest and bucket plan."""

    def __init__(
        self,
        manifest_dataset: TrainingManifestDataset,
        assignments: Sequence[VideoBucketAssignment],
        *,
        config: VideoDecodeConfig | None = None,
    ) -> None:
        if not isinstance(manifest_dataset, TrainingManifestDataset):
            raise TypeError("manifest_dataset must be a TrainingManifestDataset")
        values = tuple(assignments)
        if len(values) != len(manifest_dataset) or not all(
            isinstance(assignment, VideoBucketAssignment) for assignment in values
        ):
            raise ValueError("assignments must contain one VideoBucketAssignment per manifest sample")
        for index, (sample, assignment) in enumerate(zip(manifest_dataset, values)):
            if assignment.sample_index != index or assignment.sample_id != sample.sample_id:
                raise ValueError("bucket assignments must be ordered exactly like the manifest dataset")
        resolved_config = VideoDecodeConfig() if config is None else config
        if not isinstance(resolved_config, VideoDecodeConfig):
            raise TypeError("config must be a VideoDecodeConfig")
        self.manifest_dataset = manifest_dataset
        self.assignments = values
        self.config = resolved_config

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return self.manifest_dataset.sample_ids

    @property
    def bucket_keys(self) -> tuple[VideoBucketKey, ...]:
        return tuple(assignment.bucket_key for assignment in self.assignments)

    def __len__(self) -> int:
        return len(self.manifest_dataset)

    @overload
    def __getitem__(self, index: int) -> DecodedVideoSample: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[DecodedVideoSample, ...]: ...

    def __getitem__(self, index: int | slice) -> DecodedVideoSample | tuple[DecodedVideoSample, ...]:
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(len(self))))
        sample = self.manifest_dataset[index]
        path = resolve_local_media_path(
            sample.media,
            manifest_path=self.manifest_dataset.manifest_path,
        )
        if path is None:
            raise ValueError(f"native video decoder currently requires a local URI: {sample.media.uri!r}")
        return decode_video_sample(
            sample,
            self.assignments[index],
            media_path=path,
            config=self.config,
        )

    def __iter__(self) -> Iterator[DecodedVideoSample]:
        for index in range(len(self)):
            yield self[index]


def collate_decoded_video_samples(samples: Sequence[DecodedVideoSample]) -> TrainingBatch:
    """Stack one decoded video bucket into the public raw-media contract."""

    values = tuple(samples)
    if not values:
        raise ValueError("cannot collate an empty decoded video batch")
    if not all(isinstance(sample, DecodedVideoSample) for sample in values):
        raise TypeError("all values must be DecodedVideoSample")
    bucket_key = values[0].assignment.bucket_key
    pixel_shape = tuple(values[0].pixel_values.shape)
    pixel_dtype = values[0].pixel_values.dtype
    for sample in values[1:]:
        if sample.assignment.bucket_key != bucket_key:
            raise ValueError("decoded video batch cannot mix bucket keys")
        if tuple(sample.pixel_values.shape) != pixel_shape or sample.pixel_values.dtype != pixel_dtype:
            raise ValueError("decoded video batch cannot mix pixel shapes or dtypes")
    return TrainingBatch(
        sample_ids=tuple(sample.sample_id for sample in values),
        prompts=tuple(sample.prompt for sample in values),
        pixel_values=torch.stack([sample.pixel_values for sample in values]),
        valid_mask=torch.stack([sample.valid_mask for sample in values]),
        metadata={
            "decode_schema": VIDEO_DECODE_TRANSFORM_SCHEMA,
            "bucket_key": bucket_key.to_dict(),
            "samples_per_microbatch": len(values),
            "latent_tokens_per_sample": bucket_key.token_count,
            "latent_tokens_per_microbatch": len(values) * bucket_key.token_count,
            "selected_frame_indices": tuple(sample.selected_frame_indices for sample in values),
            "frame_sampling": tuple(dict(sample.frame_sampling) for sample in values),
            "spatial_transform": tuple(dict(sample.spatial_transform) for sample in values),
        },
    )


__all__ = [
    "VIDEO_DECODE_TRANSFORM_SCHEMA",
    "DecodedVideoSample",
    "VideoDecodeConfig",
    "VideoDecodingDataset",
    "collate_decoded_video_samples",
    "decode_video_sample",
    "video_frame_indices",
]

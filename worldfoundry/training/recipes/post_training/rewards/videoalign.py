"""Strict, immutable VideoAlign reward recipe contract."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from types import MappingProxyType

from ..common import frozen_float_mapping, positive_int

VIDEOALIGN_REWARD_IDS = (
    "video_quality",
    "motion_quality",
    "text_alignment",
)
VIDEOALIGN_BASE_MODEL_REPOSITORY = "Qwen/Qwen2-VL-2B-Instruct"
VIDEOALIGN_BASE_MODEL_REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"
VIDEOALIGN_CHECKPOINT_REPOSITORY = "KlingTeam/VideoReward"
VIDEOALIGN_CHECKPOINT_REVISION = "b8e421fe21aec3dde5f61fdd1dc44e1d603b9727"
VIDEOALIGN_CHECKPOINT_FILE = "checkpoint-11352/model.pth"
VIDEOALIGN_CHECKPOINT_SHA256 = "48375908e6112de9f0248402db156a23b480709a6960b091c598c6f4c88d21b9"
VIDEOALIGN_CHECKPOINT_SIZE_BYTES = 5_031_072_529
VIDEOALIGN_CALIBRATION_MEAN = MappingProxyType(
    {
        "video_quality": 3.6757,
        "motion_quality": 1.1646,
        "text_alignment": 2.8105,
    }
)
VIDEOALIGN_CALIBRATION_STD = MappingProxyType(
    {
        "video_quality": 2.2476,
        "motion_quality": 1.3811,
        "text_alignment": 2.5121,
    }
)

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class VideoAlignRewardSpec:
    """Immutable VideoAlign model and in-memory preprocessing contract."""

    base_model_repository: str = VIDEOALIGN_BASE_MODEL_REPOSITORY
    base_model_revision: str = VIDEOALIGN_BASE_MODEL_REVISION
    checkpoint_repository: str = VIDEOALIGN_CHECKPOINT_REPOSITORY
    checkpoint_revision: str = VIDEOALIGN_CHECKPOINT_REVISION
    checkpoint_file: str = VIDEOALIGN_CHECKPOINT_FILE
    checkpoint_sha256: str = VIDEOALIGN_CHECKPOINT_SHA256
    checkpoint_size_bytes: int = VIDEOALIGN_CHECKPOINT_SIZE_BYTES
    source_fps: float = 24.0
    target_fps: float = 2.0
    min_frames: int = 4
    max_frames: int = 768
    frame_factor: int = 2
    max_frame_pixels: int = 200_704
    batch_size: int = 1
    dtype: str = "bfloat16"
    quantize_to_uint8: bool = True
    input_range: str = "minus-one-to-one"
    calibration_mean: Mapping[str, float] = field(default_factory=lambda: dict(VIDEOALIGN_CALIBRATION_MEAN))
    calibration_std: Mapping[str, float] = field(default_factory=lambda: dict(VIDEOALIGN_CALIBRATION_STD))
    normalization_epsilon: float = 0.0
    type: str = "videoalign"

    def __post_init__(self) -> None:
        if str(self.type).lower().replace("_", "-") != "videoalign":
            raise ValueError("reward_model.type must be 'videoalign'")
        for name, value in (
            ("base_model_repository", self.base_model_repository),
            ("checkpoint_repository", self.checkpoint_repository),
        ):
            resolved = str(value).strip()
            if "/" not in resolved or resolved.startswith("/") or resolved.endswith("/"):
                raise ValueError(f"{name} must be a Hub repository id")
            object.__setattr__(self, name, resolved)
        for name, value in (
            ("base_model_revision", self.base_model_revision),
            ("checkpoint_revision", self.checkpoint_revision),
        ):
            resolved = str(value).strip().lower()
            if _COMMIT_PATTERN.fullmatch(resolved) is None:
                raise ValueError(f"{name} must be an immutable 40-hex commit")
            object.__setattr__(self, name, resolved)
        checkpoint_file = str(self.checkpoint_file).strip()
        path = Path(checkpoint_file)
        if not checkpoint_file or path.is_absolute() or ".." in path.parts:
            raise ValueError("checkpoint_file must be a safe relative path")
        object.__setattr__(self, "checkpoint_file", path.as_posix())
        checkpoint_sha256 = str(self.checkpoint_sha256).strip().lower()
        if _SHA256_PATTERN.fullmatch(checkpoint_sha256) is None:
            raise ValueError("checkpoint_sha256 must be a 64-hex digest")
        object.__setattr__(self, "checkpoint_sha256", checkpoint_sha256)
        object.__setattr__(
            self,
            "checkpoint_size_bytes",
            positive_int(
                self.checkpoint_size_bytes,
                field_name="reward_model.checkpoint_size_bytes",
            ),
        )
        for name, value in (
            ("source_fps", self.source_fps),
            ("target_fps", self.target_fps),
        ):
            resolved = float(value)
            if not isfinite(resolved) or resolved <= 0:
                raise ValueError(f"reward_model.{name} must be finite and positive")
            object.__setattr__(self, name, resolved)
        for name in (
            "min_frames",
            "max_frames",
            "frame_factor",
            "max_frame_pixels",
            "batch_size",
        ):
            object.__setattr__(
                self,
                name,
                positive_int(getattr(self, name), field_name=f"reward_model.{name}"),
            )
        if self.min_frames > self.max_frames:
            raise ValueError("reward_model frame bounds must satisfy min_frames <= max_frames")
        if self.min_frames % self.frame_factor or self.max_frames % self.frame_factor:
            raise ValueError("reward_model frame bounds must be multiples of frame_factor")
        if self.max_frame_pixels % (28 * 28):
            raise ValueError("reward_model.max_frame_pixels must preserve Qwen2-VL's 28-pixel grid")
        dtype = str(self.dtype).lower().removeprefix("torch.")
        dtype = {"bf16": "bfloat16", "fp16": "float16"}.get(dtype, dtype)
        if dtype not in {"bfloat16", "float16"}:
            raise ValueError("reward_model.dtype must be bfloat16 or float16")
        if not isinstance(self.quantize_to_uint8, bool):
            raise TypeError("reward_model.quantize_to_uint8 must be a bool")
        if self.input_range not in {"minus-one-to-one", "zero-to-one"}:
            raise ValueError("reward_model.input_range must be 'minus-one-to-one' or 'zero-to-one'")
        mean = frozen_float_mapping(
            self.calibration_mean,
            field_name="reward_model.calibration_mean",
        )
        std = frozen_float_mapping(
            self.calibration_std,
            field_name="reward_model.calibration_std",
        )
        expected = set(VIDEOALIGN_REWARD_IDS)
        if set(mean) != expected or set(std) != expected:
            raise ValueError("VideoAlign calibration keys must exactly match its three reward ids")
        if any(value <= 0 for value in std.values()):
            raise ValueError("VideoAlign calibration standard deviations must be positive")
        epsilon = float(self.normalization_epsilon)
        if not isfinite(epsilon) or epsilon < 0:
            raise ValueError("reward_model.normalization_epsilon must be finite and non-negative")
        object.__setattr__(self, "calibration_mean", mean)
        object.__setattr__(self, "calibration_std", std)
        object.__setattr__(self, "normalization_epsilon", epsilon)

    @property
    def reward_ids(self) -> tuple[str, ...]:
        return VIDEOALIGN_REWARD_IDS


VIDEOALIGN_REWARD_FIELDS = {
    "type",
    "base_model_repository",
    "base_model_revision",
    "checkpoint_repository",
    "checkpoint_revision",
    "checkpoint_file",
    "checkpoint_sha256",
    "checkpoint_size_bytes",
    "source_fps",
    "target_fps",
    "min_frames",
    "max_frames",
    "frame_factor",
    "max_frame_pixels",
    "batch_size",
    "dtype",
    "quantize_to_uint8",
    "input_range",
    "calibration_mean",
    "calibration_std",
    "normalization_epsilon",
}


__all__ = [
    "VIDEOALIGN_BASE_MODEL_REPOSITORY",
    "VIDEOALIGN_BASE_MODEL_REVISION",
    "VIDEOALIGN_CALIBRATION_MEAN",
    "VIDEOALIGN_CALIBRATION_STD",
    "VIDEOALIGN_CHECKPOINT_FILE",
    "VIDEOALIGN_CHECKPOINT_REPOSITORY",
    "VIDEOALIGN_CHECKPOINT_REVISION",
    "VIDEOALIGN_CHECKPOINT_SHA256",
    "VIDEOALIGN_CHECKPOINT_SIZE_BYTES",
    "VIDEOALIGN_REWARD_IDS",
    "VideoAlignRewardSpec",
]

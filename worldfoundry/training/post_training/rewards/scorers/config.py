"""Typed configuration for native audio-video reward scorers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class VideoPickScoreConfig:
    """PickScore model and batching settings for first-frame video scoring."""

    batch_size: int = 8
    device: str = "auto"
    processor_id: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_id: str = "yuvalkirstain/PickScore_v1"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("VideoPickScore batch_size must be positive")
        if not self.device.strip() or not self.processor_id.strip() or not self.model_id.strip():
            raise ValueError("VideoPickScore device and model identifiers must be non-empty")


@dataclass(frozen=True, slots=True)
class CLAPConfig:
    """CLAP model and batching settings for audio-text scoring."""

    batch_size: int = 8
    device: str = "auto"
    model_id: str = "laion/larger_clap_general"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("CLAP batch_size must be positive")
        if not self.device.strip() or not self.model_id.strip():
            raise ValueError("CLAP device and model identifier must be non-empty")


@dataclass(frozen=True, slots=True)
class AVRewardScorersConfig:
    """Configuration for the LTX audio-video reward sidecar."""

    videopickscore: VideoPickScoreConfig = field(default_factory=VideoPickScoreConfig)
    clap: CLAPConfig = field(default_factory=CLAPConfig)


__all__ = ["AVRewardScorersConfig", "CLAPConfig", "VideoPickScoreConfig"]

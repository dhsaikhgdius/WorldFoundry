"""Data-side descriptions for supported precomputed video tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .video_bucketing import VideoLatentGeometry

LVDM_SHORT_MODEL_RECIPE = "lvdm-short-unconditional"
DYNAMICRAFTER_MODEL_RECIPES = frozenset({"dynamicrafter-512-i2v", "dynamicrafter-1024-i2v"})


def lvdm_short_latent_normalization() -> dict[str, object]:
    return {
        "posterior": "sample",
        "operation": "scale*(sample+shift)",
        "scale": 0.220142075,
        "shift": 0.5837740898,
    }


def dynamicrafter_latent_normalization() -> dict[str, object]:
    return {"posterior": "sample", "operation": "sample*scale", "scale": 0.18215}


def t2v_turbo_latent_normalization() -> dict[str, object]:
    return {"posterior": "sample", "operation": "sample*scale", "scale": 0.18215}


@dataclass(frozen=True, slots=True)
class PrecomputedVideoTensorContract:
    geometry: VideoLatentGeometry
    conditioning_layout: str
    latent_normalization: Mapping[str, object]
    tensor_layouts: Mapping[str, str]


def precomputed_video_tensor_contract(model_recipe: str) -> PrecomputedVideoTensorContract:
    if model_recipe == LVDM_SHORT_MODEL_RECIPE:
        return PrecomputedVideoTensorContract(
            VideoLatentGeometry(8, 8, 4, "uniform"),
            "none",
            lvdm_short_latent_normalization(),
            {},
        )
    if model_recipe in DYNAMICRAFTER_MODEL_RECIPES:
        return PrecomputedVideoTensorContract(
            VideoLatentGeometry(8, 8, 1, "uniform"),
            "dynamicrafter-hybrid",
            dynamicrafter_latent_normalization(),
            {
                "text_context": "sequence-features",
                "empty_text_context": "sequence-features",
                "image_features_by_frame": "frames-sequence-features",
                "zero_image_features": "sequence-features",
                "fps": "scalar",
            },
        )
    if model_recipe == "t2v-turbo":
        return PrecomputedVideoTensorContract(
            VideoLatentGeometry(8, 8, 1, "uniform"),
            "videocrafter-text",
            t2v_turbo_latent_normalization(),
            {
                "context": "sequence-features",
                "unconditional_context": "sequence-features",
                "fps": "scalar",
            },
        )
    raise ValueError(f"precomputed video tensor import does not support {model_recipe!r}")


__all__ = [
    "DYNAMICRAFTER_MODEL_RECIPES",
    "LVDM_SHORT_MODEL_RECIPE",
    "PrecomputedVideoTensorContract",
    "dynamicrafter_latent_normalization",
    "lvdm_short_latent_normalization",
    "precomputed_video_tensor_contract",
    "t2v_turbo_latent_normalization",
]

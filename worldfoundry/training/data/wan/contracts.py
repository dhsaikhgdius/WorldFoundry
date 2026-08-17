"""Stable Wan cache constants and explicit identities."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..checkpoint_assets import checkpoint_asset_identity

WAN_CONDITIONING_LAYOUT = "umt5-sequence"
WAN_LATENT_MEAN = (
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
)
WAN_LATENT_STD = (
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
)


def require_positive_int(value: object, *, field_name: str) -> int:
    """Resolve a positive integer without accepting booleans."""

    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field_name} must be positive")
    return resolved


def wan_cache_contract(
    model_recipe: str,
    *,
    latent_channels: int = 16,
    temporal_compression: int = 4,
    spatial_compression: int = 8,
    text_length: int = 512,
    context_features: int = 4096,
    latent_patch_size: tuple[int, int, int] = (1, 2, 2),
) -> dict[str, object]:
    """Return every denoiser-facing Wan cache shape convention."""

    recipe = str(model_recipe).strip().lower().replace("_", "-")
    if not recipe:
        raise ValueError("model_recipe cannot be empty")
    patch = tuple(require_positive_int(value, field_name="latent_patch_size") for value in latent_patch_size)
    if len(patch) != 3:
        raise ValueError("latent_patch_size must contain temporal, height, and width")
    return {
            "model_recipe": recipe,
            "latent_channels": require_positive_int(
                latent_channels,
                field_name="latent_channels",
            ),
            "temporal_compression": require_positive_int(
                temporal_compression,
                field_name="temporal_compression",
            ),
            "temporal_alignment": "first-frame",
            "spatial_compression": require_positive_int(
                spatial_compression,
                field_name="spatial_compression",
            ),
            "text_length": require_positive_int(text_length, field_name="text_length"),
            "context_features": require_positive_int(
                context_features,
                field_name="context_features",
            ),
            "latent_patch_size": list(patch),
            "conditioning": WAN_CONDITIONING_LAYOUT,
    }


def wan_latent_normalization(
    mean: Sequence[float] = WAN_LATENT_MEAN,
    std: Sequence[float] = WAN_LATENT_STD,
) -> dict[str, object]:
    """Return the official deterministic VAE mean and per-channel affine."""

    resolved_mean = tuple(float(value) for value in mean)
    resolved_std = tuple(float(value) for value in std)
    if len(resolved_mean) != 16 or len(resolved_std) != 16:
        raise ValueError("Wan2.1 latent normalization requires 16 mean/std values")
    if any(not math.isfinite(value) for value in (*resolved_mean, *resolved_std)):
        raise ValueError("Wan latent normalization must be finite")
    if any(value <= 0 for value in resolved_std):
        raise ValueError("Wan latent standard deviations must be positive")
    return {
            "posterior": "deterministic-mean",
            "operation": "(mean-latent-channel-mean)/channel-std",
            "channel_mean": list(resolved_mean),
            "channel_std": list(resolved_std),
    }


def wan_checkpoint_asset_identity(spec: object) -> dict[str, object]:
    """Describe one Wan component by repository, revision, files, and sizes."""

    repository = getattr(spec, "repo_id", None) or "local-explicit"
    revision = getattr(spec, "revision", None) or "local-explicit"
    sources = tuple(str(source) for source in getattr(spec, "sources", ()))
    files = tuple(str(name) for name in getattr(spec, "files", ()))
    if not files:
        files = tuple(str(name) for name in getattr(spec, "allow_patterns", ()))
    return checkpoint_asset_identity(
        repo_id=str(repository),
        revision=str(revision),
        files=files,
        file_size_bytes=dict(getattr(spec, "file_size_bytes", {})),
        sources=sources,
    )


__all__ = [
    "WAN_CONDITIONING_LAYOUT",
    "WAN_LATENT_MEAN",
    "WAN_LATENT_STD",
    "wan_cache_contract",
    "wan_checkpoint_asset_identity",
    "wan_latent_normalization",
]

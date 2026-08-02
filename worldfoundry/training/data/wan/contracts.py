"""Stable Wan cache constants and identity digests."""

from __future__ import annotations

import math
from collections.abc import Sequence

from worldfoundry.core.io.integrity import canonical_sha256

from ..checkpoint_assets import checkpoint_asset_digest

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


def wan_cache_contract_digest(
    model_recipe: str,
    *,
    latent_channels: int = 16,
    temporal_compression: int = 4,
    spatial_compression: int = 8,
    text_length: int = 512,
    context_features: int = 4096,
    latent_patch_size: tuple[int, int, int] = (1, 2, 2),
) -> str:
    """Digest every denoiser-facing Wan cache shape convention."""

    recipe = str(model_recipe).strip().lower().replace("_", "-")
    if not recipe:
        raise ValueError("model_recipe cannot be empty")
    patch = tuple(require_positive_int(value, field_name="latent_patch_size") for value in latent_patch_size)
    if len(patch) != 3:
        raise ValueError("latent_patch_size must contain temporal, height, and width")
    return canonical_sha256(
        {
            "schema": "worldfoundry-wan-training-cache-contract",
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
    )


def wan_latent_normalization_digest(
    mean: Sequence[float] = WAN_LATENT_MEAN,
    std: Sequence[float] = WAN_LATENT_STD,
) -> str:
    """Bind the official deterministic VAE mean and per-channel affine."""

    resolved_mean = tuple(float(value) for value in mean)
    resolved_std = tuple(float(value) for value in std)
    if len(resolved_mean) != 16 or len(resolved_std) != 16:
        raise ValueError("Wan2.1 latent normalization requires 16 mean/std values")
    if any(not math.isfinite(value) for value in (*resolved_mean, *resolved_std)):
        raise ValueError("Wan latent normalization must be finite")
    if any(value <= 0 for value in resolved_std):
        raise ValueError("Wan latent standard deviations must be positive")
    return canonical_sha256(
        {
            "schema": "worldfoundry-wan-latent-normalization",
            "posterior": "deterministic-mean",
            "operation": "(mean-latent-channel-mean)/channel-std",
            "channel_mean": list(resolved_mean),
            "channel_std": list(resolved_std),
        }
    )


def wan_checkpoint_asset_digest(spec: object) -> str:
    """Bind one Wan component to its repository, revision, and byte audits."""

    repository = getattr(spec, "repo_id", None) or str(getattr(spec, "source", "local-explicit"))
    revision = getattr(spec, "revision", None) or "local-explicit"
    files = {f"file:{name}": digest for name, digest in dict(getattr(spec, "file_sha256", {})).items()}
    files.update({f"resource:{name}": digest for name, digest in dict(getattr(spec, "resource_sha256", {})).items()})
    if not files:
        raise ValueError(
            "Wan cache assets must carry SHA-256 integrity metadata; use an audited CheckpointSpec for local overrides"
        )
    return checkpoint_asset_digest(
        repository=str(repository),
        revision=str(revision),
        file_sha256=files,
    )


__all__ = [
    "WAN_CONDITIONING_LAYOUT",
    "WAN_LATENT_MEAN",
    "WAN_LATENT_STD",
    "wan_cache_contract_digest",
    "wan_checkpoint_asset_digest",
    "wan_latent_normalization_digest",
]

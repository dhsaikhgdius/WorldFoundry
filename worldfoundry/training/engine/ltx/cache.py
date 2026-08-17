"""Denoiser-facing cache contract for native LTX video training."""

from __future__ import annotations

from worldfoundry.training.data.video_cache import VideoCachedDataset
from worldfoundry.training.models.ltx import LTXTrainAdapter
from worldfoundry.training.recipes.spec import TrainingRecipe

LTX_MODEL_RECIPES = frozenset({"ltx-video-i2v", "ltx-2-i2v", "ltx-2.3-i2v"})


def ltx_latent_normalization(model_recipe: str) -> dict[str, object]:
    """Describe the latent tensor emitted by the matching official VAE encoder."""

    recipe = str(model_recipe).strip().lower().replace("_", "-")
    if recipe == "ltx-video-i2v":
        return {
            "posterior": "sample",
            "operation": "(sample-channel-mean)/channel-std",
            "statistics": "checkpoint-per-channel",
        }
    if recipe in {"ltx-2-i2v", "ltx-2.3-i2v"}:
        return {
            "posterior": "deterministic-mean",
            "operation": "checkpoint-per-channel-normalize",
            "statistics": "checkpoint-per-channel",
        }
    raise ValueError(f"unsupported LTX model recipe: {model_recipe!r}")


def ltx_cache_contract(model_recipe: str, adapter: LTXTrainAdapter) -> dict[str, object]:
    """Return the cache shapes and preprocessing consumed by the LTX transformer."""

    recipe = str(model_recipe).strip().lower().replace("_", "-")
    if recipe not in LTX_MODEL_RECIPES:
        raise ValueError(f"unsupported LTX model recipe: {model_recipe!r}")
    conditioning = "t5-sequence" if recipe == "ltx-video-i2v" else "gemma-sequence"
    return {
        "model_recipe": recipe,
        "latent_channels": adapter.expected_latent_channels,
        "temporal_compression": adapter.temporal_compression,
        "temporal_alignment": "first-frame",
        "spatial_compression": adapter.spatial_compression,
        "latent_patch_size": [1, 1, 1],
        "context_features": adapter.expected_context_features,
        "conditioning": conditioning,
        "latent_normalization": ltx_latent_normalization(recipe),
    }


def validate_ltx_cache_contract(
    recipe: TrainingRecipe,
    adapter: LTXTrainAdapter,
    dataset: VideoCachedDataset,
) -> dict[str, object]:
    """Validate cached latents and prompt embeddings before optimization starts."""

    expected = ltx_cache_contract(recipe.model.recipe, adapter)
    encoder_assets: tuple[object, object, object] | None = None
    for entry in dataset.index.entries:
        source = entry.provenance
        latents = entry.tensors["clean_latents"]
        context = entry.tensors.get("condition.video_context")
        context_mask = entry.tensors.get("condition.context_mask")
        if "condition.audio_context" in entry.tensors:
            raise ValueError("LTX video training does not consume cached audio context")

        if source.model_recipe != expected["model_recipe"]:
            raise ValueError(f"cache entry {entry.sample_id!r} was created for another LTX model")
        if dict(source.latent_normalization) != expected["latent_normalization"]:
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible LTX latent normalization")
        geometry = source.latent_geometry
        if geometry.temporal_alignment != "first-frame" or (
            geometry.temporal_compression,
            geometry.spatial_compression_height,
            geometry.spatial_compression_width,
        ) != (
            adapter.temporal_compression,
            adapter.spatial_compression,
            adapter.spatial_compression,
        ):
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible LTX VAE geometry")
        if latents.shape[0] != adapter.expected_latent_channels:
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible LTX latent channels")
        if context is None or len(context.shape) != 2 or context.shape[-1] != adapter.expected_context_features:
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible LTX text context")
        if context.layout != "sequence-features":
            raise ValueError(f"cache entry {entry.sample_id!r} has an unknown LTX context layout")
        if context_mask is None or context_mask.shape != context.shape[:1] or context_mask.layout != "sequence":
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible LTX context mask")
        if source.conditioning_layout != expected["conditioning"]:
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible LTX conditioning layout")

        assets = (source.codec, source.conditioner, source.tokenizer)
        if encoder_assets is None:
            encoder_assets = assets
        elif assets != encoder_assets:
            raise ValueError("one LTX run cannot mix caches from different VAE or text encoders")
    return expected


__all__ = [
    "LTX_MODEL_RECIPES",
    "ltx_cache_contract",
    "ltx_latent_normalization",
    "validate_ltx_cache_contract",
]

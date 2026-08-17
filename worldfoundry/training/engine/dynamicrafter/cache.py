"""Cached hybrid-conditioning contract for DynamiCrafter training."""

from __future__ import annotations

from worldfoundry.training.data.video_cache import VideoCachedDataset
from worldfoundry.training.data.video_tensor_contracts import (
    DYNAMICRAFTER_MODEL_RECIPES,
    dynamicrafter_latent_normalization,
)
from worldfoundry.training.models.dynamicrafter import DynamiCrafterTrainAdapter
from worldfoundry.training.recipes.spec import TrainingRecipe


def validate_dynamicrafter_cache_contract(
    recipe: TrainingRecipe,
    adapter: DynamiCrafterTrainAdapter,
    dataset: VideoCachedDataset,
) -> dict[str, object]:
    expected = {
        "model_recipe": recipe.model.recipe,
        "latent_channels": adapter.expected_latent_channels,
        "conditioning": "dynamicrafter-hybrid",
        "latent_normalization": dynamicrafter_latent_normalization(),
    }
    assets: tuple[object, object, object] | None = None
    for entry in dataset.index.entries:
        source = entry.provenance
        tensors = entry.tensors
        if source.model_recipe != recipe.model.recipe:
            raise ValueError(f"cache entry {entry.sample_id!r} was created for another model")
        if source.conditioning_layout != "dynamicrafter-hybrid":
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible conditioning layout")
        if dict(source.latent_normalization) != expected["latent_normalization"]:
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible latent normalization")
        if tensors["clean_latents"].shape[0] != adapter.expected_latent_channels:
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible latent channels")
        context = tensors.get("condition.text_context")
        empty = tensors.get("condition.empty_text_context")
        features = tensors.get("condition.image_features_by_frame")
        zero_features = tensors.get("condition.zero_image_features")
        fps = tensors.get("condition.fps")
        if context is None or empty is None or context.layout != "sequence-features":
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible text context")
        if empty.layout != "sequence-features" or empty.shape != context.shape:
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible empty text context")
        if features is None or len(features.shape) != 3 or features.shape[0] != tensors["clean_latents"].shape[1]:
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible per-frame image features")
        if features.layout != "frames-sequence-features":
            raise ValueError(f"cache entry {entry.sample_id!r} has unknown image feature layout")
        if zero_features is None or zero_features.shape != features.shape[1:]:
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible zero image features")
        if fps is None or fps.layout != "scalar" or fps.shape != (1,):
            raise ValueError(f"cache entry {entry.sample_id!r} must preserve its sampled FPS")
        current_assets = (source.codec, source.conditioner, source.tokenizer)
        if assets is None:
            assets = current_assets
        elif current_assets != assets:
            raise ValueError("one DynamiCrafter run cannot mix encoder assets")
    return expected


__all__ = [
    "DYNAMICRAFTER_MODEL_RECIPES",
    "dynamicrafter_latent_normalization",
    "validate_dynamicrafter_cache_contract",
]

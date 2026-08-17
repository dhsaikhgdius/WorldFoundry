"""Cached latent/context contract for T2V-Turbo distillation."""

from __future__ import annotations

import torch

from worldfoundry.training.data.video_cache import VideoCachedDataset
from worldfoundry.training.data.video_tensor_contracts import (
    t2v_turbo_latent_normalization,
)
from worldfoundry.training.post_training.distillation.t2v_turbo.objective import (
    T2VTurboTrainAdapter,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe


def validate_t2v_turbo_cache_contract(
    recipe: PostTrainingRecipe,
    adapter: T2VTurboTrainAdapter,
    dataset: VideoCachedDataset,
) -> dict[str, object]:
    expected = {
        "model_recipe": "t2v-turbo",
        "latent_channels": adapter.expected_latent_channels,
        "conditioning": "videocrafter-text",
        "context_features": 1024,
        "target_fps": 16.0,
        "latent_normalization": t2v_turbo_latent_normalization(),
    }
    assets: tuple[object, object, object] | None = None
    for entry in dataset.index.entries:
        source = entry.provenance
        tensors = entry.tensors
        if source.model_recipe != "t2v-turbo":
            raise ValueError(f"cache entry {entry.sample_id!r} was created for another model")
        if source.conditioning_layout != "videocrafter-text":
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible conditioning layout")
        if dict(source.latent_normalization) != expected["latent_normalization"]:
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible latent normalization")
        if source.target_fps != expected["target_fps"]:
            raise ValueError(f"cache entry {entry.sample_id!r} must target 16 FPS")
        if tensors["clean_latents"].shape[0] != adapter.expected_latent_channels:
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible latent channels")
        context = tensors.get("condition.context")
        unconditional = tensors.get("condition.unconditional_context")
        if context is None or context.layout != "sequence-features" or context.shape[-1] != 1024:
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible text context")
        if unconditional is None or unconditional.layout != "sequence-features":
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible unconditional context")
        fps = tensors.get("condition.fps")
        if fps is not None:
            if fps.layout != "scalar" or fps.shape != (1,):
                raise ValueError(f"cache entry {entry.sample_id!r} has incompatible FPS conditioning")
            from safetensors import safe_open

            with safe_open(dataset.store.root / entry.object_path, framework="pt", device="cpu") as handle:
                fps_value = handle.get_tensor("condition.fps")
            expected_fps = torch.full_like(fps_value, 16.0 if fps_value.dtype.is_floating_point else 16)
            if not torch.equal(fps_value, expected_fps):
                raise ValueError(f"cache entry {entry.sample_id!r} must condition on 16 FPS")
        current_assets = (source.codec, source.conditioner, source.tokenizer)
        if assets is None:
            assets = current_assets
        elif current_assets != assets:
            raise ValueError("one T2V-Turbo run cannot mix encoder assets")
    return expected


__all__ = ["t2v_turbo_latent_normalization", "validate_t2v_turbo_cache_contract"]

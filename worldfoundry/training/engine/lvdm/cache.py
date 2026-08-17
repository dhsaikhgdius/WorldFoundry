"""Cached-latent contract for the released LVDM short trainer."""

from __future__ import annotations

from worldfoundry.training.data.video_cache import VideoCachedDataset
from worldfoundry.training.data.video_tensor_contracts import (
    LVDM_SHORT_MODEL_RECIPE,
    lvdm_short_latent_normalization,
)
from worldfoundry.training.models.lvdm import LVDMUnconditionalTrainAdapter
from worldfoundry.training.recipes.spec import TrainingRecipe


def validate_lvdm_short_cache_contract(
    recipe: TrainingRecipe,
    adapter: LVDMUnconditionalTrainAdapter,
    dataset: VideoCachedDataset,
) -> dict[str, object]:
    """Validate tensors actually consumed by unconditional short training."""

    expected = {
        "model_recipe": LVDM_SHORT_MODEL_RECIPE,
        "latent_channels": adapter.expected_latent_channels,
        "conditioning": "none",
        "latent_normalization": lvdm_short_latent_normalization(),
    }
    for entry in dataset.index.entries:
        source = entry.provenance
        if source.model_recipe != LVDM_SHORT_MODEL_RECIPE:
            raise ValueError(f"cache entry {entry.sample_id!r} was created for another model")
        if source.conditioning_layout != "none":
            raise ValueError(f"cache entry {entry.sample_id!r} is not unconditional")
        if dict(source.latent_normalization) != expected["latent_normalization"]:
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible latent normalization")
        if entry.tensors["clean_latents"].shape[0] != adapter.expected_latent_channels:
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible latent channels")
        if any(name.startswith("condition.") for name in entry.tensors):
            raise ValueError(f"cache entry {entry.sample_id!r} contains unused conditioning tensors")
    return expected


__all__ = [
    "LVDM_SHORT_MODEL_RECIPE",
    "lvdm_short_latent_normalization",
    "validate_lvdm_short_cache_contract",
]

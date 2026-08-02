"""Wan immutable-cache validation and deterministic token-budget loading."""

from __future__ import annotations

import math

from worldfoundry.training.data.dataset import TrainingManifestDataset
from worldfoundry.training.data.latent_token_sampler import LatentTokenBatchSampler
from worldfoundry.training.data.loader import build_stateful_dataloader
from worldfoundry.training.data.sana_cache import text_sha256
from worldfoundry.training.data.video_cache import (
    VideoCachedDataset,
    collate_video_cached_samples,
)
from worldfoundry.training.data.wan.contracts import (
    wan_cache_contract_digest,
    wan_latent_normalization_digest,
)
from worldfoundry.training.models.wan import WanTrainAdapter
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
from worldfoundry.training.recipes.spec import TrainingRecipe

WanCacheRecipe = TrainingRecipe | PostTrainingRecipe

_CACHE_LOADER_OPTIONS = frozenset(
    {
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "snapshot_every_n_steps",
        "video_buckets",
        "bucket_policy",
        "decode",
    }
)


def _strict_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool")
    return value


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field_name} must be positive")
    return resolved


def _loader_options(recipe: WanCacheRecipe) -> dict[str, object]:
    options = dict(recipe.data.options)
    unknown = sorted(set(options) - _CACHE_LOADER_OPTIONS)
    if unknown:
        raise ValueError(f"unsupported cached Wan data options: {unknown}")
    options.pop("video_buckets", None)
    options.pop("bucket_policy", None)
    options.pop("decode", None)
    return options


def validate_wan_cache_contract(
    recipe: WanCacheRecipe,
    adapter: WanTrainAdapter,
    dataset: VideoCachedDataset,
) -> str:
    """Audit tensor geometry, preprocessing, and encoder identity for one run."""

    expected_contract = wan_cache_contract_digest(
        recipe.model.recipe,
        latent_channels=adapter.expected_latent_channels,
        temporal_compression=adapter.temporal_compression,
        spatial_compression=adapter.spatial_compression,
        text_length=adapter.expected_text_length,
        context_features=adapter.expected_context_features,
        latent_patch_size=adapter.patch_size,
    )
    expected_normalization = wan_latent_normalization_digest()
    asset_identities: set[tuple[str, ...]] = set()
    for entry in dataset.index.entries:
        provenance = entry.provenance
        latents = entry.tensors["clean_latents"]
        context = entry.tensors.get("condition.context")
        loss_mask = entry.tensors.get("latent_loss_mask")
        valid_mask = entry.tensors.get("valid_latent_mask")
        if provenance.model_recipe_digest != expected_contract:
            raise ValueError(f"cache entry {entry.sample_id!r} was created for a different Wan contract")
        if provenance.latent_normalization_digest != expected_normalization:
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible latent normalization")
        if provenance.latent_geometry.temporal_alignment != "first-frame" or (
            provenance.latent_geometry.temporal_compression,
            provenance.latent_geometry.spatial_compression_height,
            provenance.latent_geometry.spatial_compression_width,
        ) != (
            adapter.temporal_compression,
            adapter.spatial_compression,
            adapter.spatial_compression,
        ):
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible codec geometry")
        if latents.shape[0] != adapter.expected_latent_channels:
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible latent channels")
        if any(int(axis) % patch for axis, patch in zip(latents.shape[-3:], adapter.patch_size)):
            raise ValueError(f"cache entry {entry.sample_id!r} is not divisible by Wan patch geometry")
        if context is None or context.shape != (
            adapter.expected_text_length,
            adapter.expected_context_features,
        ):
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible UMT5 context")
        if context.layout != "sequence-features":
            raise ValueError(f"cache entry {entry.sample_id!r} has an unknown context layout")
        if loss_mask is None or valid_mask is None:
            raise ValueError(f"cache entry {entry.sample_id!r} lacks latent validity masks")
        if provenance.conditioning_layout != "umt5-sequence":
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible conditioning layout")
        asset_identities.add(
            (
                provenance.codec_digest,
                provenance.conditioner_digest,
                provenance.tokenizer_digest,
                provenance.latent_normalization_digest,
            )
        )
    if len(asset_identities) != 1:
        raise ValueError("one Wan run cannot mix cache objects from different encoder assets")
    return expected_contract


def build_wan_cache_loader(
    *,
    recipe: WanCacheRecipe,
    dataset: VideoCachedDataset,
    rank: int,
    world_size: int,
    default_pin_memory: bool,
) -> tuple[object, LatentTokenBatchSampler]:
    """Build the stateful loader shared by Wan SFT and cache-backed distillation."""

    options = _loader_options(recipe)
    token_budget = recipe.data.max_latent_tokens_per_microbatch
    assert token_budget is not None
    sampler = LatentTokenBatchSampler(
        dataset,
        max_latent_tokens=token_budget,
        seed=recipe.data.shuffle_seed,
        shuffle=recipe.data.shuffle,
        rank=rank,
        world_size=world_size,
        tail_policy=recipe.data.tail_policy,
    )
    workers = int(options.pop("num_workers", 0))
    pin_memory = _strict_bool(
        options.pop("pin_memory", default_pin_memory),
        field_name="data.options.pin_memory",
    )
    persistent_workers = _strict_bool(
        options.pop("persistent_workers", False),
        field_name="data.options.persistent_workers",
    )
    prefetch_factor = options.pop("prefetch_factor", None)
    snapshot_every = _positive_int(
        options.pop("snapshot_every_n_steps", 1),
        field_name="data.options.snapshot_every_n_steps",
    )
    if options:
        raise RuntimeError(f"unconsumed Wan loader options: {sorted(options)}")
    loader = build_stateful_dataloader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_video_cached_samples,
        num_workers=workers,
        worker_seed=recipe.data.shuffle_seed + rank,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=None if prefetch_factor is None else int(prefetch_factor),
        snapshot_every_n_steps=snapshot_every,
    )
    return loader, sampler


def audit_wan_cache_against_manifest(
    cache: VideoCachedDataset,
    manifest: TrainingManifestDataset,
) -> None:
    """Bind each immutable cache entry to its selected source manifest sample."""

    if cache.sample_ids != manifest.sample_ids:
        raise ValueError("Wan cache sample order/identity differs from the selected manifest")
    for entry, sample in zip(cache.index.entries, manifest):
        provenance = entry.provenance
        if provenance.media_sha256 != sample.media.sha256:
            raise ValueError(f"Wan cache media digest differs for sample {sample.sample_id!r}")
        if provenance.prompt_sha256 != text_sha256(sample.prompt):
            raise ValueError(f"Wan cache prompt digest differs for sample {sample.sample_id!r}")
        if (
            provenance.source_num_frames,
            provenance.source_height,
            provenance.source_width,
        ) != (sample.num_frames, sample.height, sample.width):
            raise ValueError(f"Wan cache source geometry differs for sample {sample.sample_id!r}")
        if not math.isclose(
            provenance.source_fps,
            sample.fps,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"Wan cache source fps differs for sample {sample.sample_id!r}")
        audit_digest = sample.safety.get("prompt_audit_digest")
        if audit_digest is None:
            raise ValueError(f"manifest sample {sample.sample_id!r} lacks safety.prompt_audit_digest")
        if provenance.safety_audit_digest != str(audit_digest).lower():
            raise ValueError(f"Wan cache safety audit digest differs for sample {sample.sample_id!r}")


__all__ = [
    "audit_wan_cache_against_manifest",
    "build_wan_cache_loader",
    "validate_wan_cache_contract",
]

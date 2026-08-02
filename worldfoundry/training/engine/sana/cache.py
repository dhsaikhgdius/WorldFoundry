"""SANA immutable-cache contract and source-manifest validation."""

from __future__ import annotations

from worldfoundry.training.data.dataset import TrainingManifestDataset
from worldfoundry.training.data.sana_cache import (
    SanaCachedDataset,
    sana_cache_contract_digest,
    text_sha256,
)
from worldfoundry.training.models.sana import SanaTrainAdapter
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
from worldfoundry.training.recipes.spec import TrainingRecipe


def validate_sana_cache_contract(
    recipe: TrainingRecipe | PostTrainingRecipe,
    adapter: SanaTrainAdapter,
    dataset: SanaCachedDataset,
    *,
    microbatch_size: int,
) -> str:
    """Audit cache geometry, preprocessing identity, and batch compatibility."""

    entries = dataset.index.entries
    reference = entries[0]
    context_shape = reference.tensors["context"].shape
    expected_contract = sana_cache_contract_digest(
        recipe.model.recipe,
        latent_channels=adapter.expected_latent_channels,
        spatial_compression=adapter.spatial_compression,
        max_text_length=reference.provenance.max_text_length,
        context_features=context_shape[-1],
    )
    preprocessing_identities = set()
    tensor_buckets = set()
    for entry in entries:
        provenance = entry.provenance
        latent = entry.tensors["clean_latents"]
        context = entry.tensors["context"]
        if provenance.model_recipe_digest != expected_contract:
            raise ValueError(f"cache entry {entry.sample_id!r} was created for a different SANA contract")
        if latent.shape[0] != adapter.expected_latent_channels:
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible latent channels")
        if provenance.spatial_compression != adapter.spatial_compression:
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible spatial compression")
        if context.shape[1] != provenance.max_text_length:
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible context length")
        preprocessing_identities.add(
            (
                provenance.codec_digest,
                provenance.conditioner_digest,
                provenance.tokenizer_digest,
                provenance.pixel_transform_digest,
                provenance.prompt_enhancement_digest,
                provenance.latent_scaling_factor,
                provenance.max_text_length,
                context.shape[-1],
            )
        )
        tensor_buckets.add(
            tuple(
                (name, descriptor.dtype, descriptor.shape, descriptor.layout)
                for name, descriptor in sorted(entry.tensors.items())
            )
        )
    if len(preprocessing_identities) != 1:
        raise ValueError("one training run cannot mix incompatible cache preprocessing identities")
    if microbatch_size > 1 and len(tensor_buckets) != 1:
        raise ValueError("microbatch_size > 1 requires a bucket sampler when cache tensor shapes differ")
    token_limit = recipe.data.max_latent_tokens_per_microbatch
    if token_limit is not None:
        largest = max(
            entry.tensors["clean_latents"].shape[-2] * entry.tensors["clean_latents"].shape[-1] for entry in entries
        )
        if largest * microbatch_size > token_limit:
            raise ValueError(
                "cached microbatch exceeds data.max_latent_tokens_per_microbatch: "
                f"{largest * microbatch_size} > {token_limit}"
            )
    return expected_contract


def audit_sana_cache_against_manifest(
    cache: SanaCachedDataset,
    manifest: TrainingManifestDataset,
) -> None:
    """Bind each cached tensor object to its selected manifest sample."""

    if cache.sample_ids != manifest.sample_ids:
        raise ValueError("cache index sample order/identity differs from the selected manifest")
    for entry, sample in zip(cache.index.entries, manifest):
        provenance = entry.provenance
        if provenance.media_sha256 != sample.media.sha256:
            raise ValueError(f"cache media digest differs for sample {sample.sample_id!r}")
        if provenance.prompt_sha256 != text_sha256(sample.prompt):
            raise ValueError(f"cache prompt digest differs for sample {sample.sample_id!r}")
        if (provenance.image_width, provenance.image_height) != (sample.width, sample.height):
            raise ValueError(f"cache image dimensions differ for sample {sample.sample_id!r}")
        audit_digest = sample.safety.get("prompt_audit_digest")
        if audit_digest is None:
            raise ValueError(f"manifest sample {sample.sample_id!r} lacks safety.prompt_audit_digest")
        if provenance.safety_audit_digest != str(audit_digest).lower():
            raise ValueError(f"cache safety audit digest differs for sample {sample.sample_id!r}")


__all__ = ["audit_sana_cache_against_manifest", "validate_sana_cache_contract"]

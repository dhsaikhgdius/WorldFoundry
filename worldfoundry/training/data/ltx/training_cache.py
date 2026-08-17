"""Native LTX cache preparation with separate text and VAE residency phases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import torch

from worldfoundry.training.engine.ltx.cache import LTX_MODEL_RECIPES, ltx_latent_normalization
from worldfoundry.training.recipes.spec import TrainingRecipe
from worldfoundry.training.safety.shieldgemma import PromptSafetyAudit

from ..dataset import TrainingManifestDataset
from ..video_bucketing import VideoLatentGeometry
from ..video_cache import VideoCacheEntry, VideoCacheStore
from ..video_dataset import VideoDecodingDataset
from ..video_precompute import (
    VideoCachePreparationResult,
    audit_video_prompts,
    build_video_decoding_dataset,
    checkpoint_spec_identity,
    release_accelerator_memory,
    staged_video_conditioning,
    validate_video_prompt_audits,
    write_video_cache_entry,
)
from .encoding import LTXTextFeatureEncoder, LTXVideoFeatureEncoder

LTX_VIDEO_GEOMETRY = VideoLatentGeometry(32, 32, 8, "first-frame")
_LTX_CONDITIONING_LAYOUTS = {
    "ltx-video-i2v": "t5-sequence",
    "ltx-2-i2v": "gemma-sequence",
    "ltx-2.3-i2v": "gemma-sequence",
}
_LTX_CONDITIONING_TENSOR_LAYOUTS = {
    "video_context": "sequence-features",
    "context_mask": "sequence",
}


def build_ltx_video_decoding_dataset(
    recipe: TrainingRecipe,
    manifest: TrainingManifestDataset,
) -> VideoDecodingDataset:
    try:
        layout = _LTX_CONDITIONING_LAYOUTS[recipe.model.recipe]
    except KeyError as error:
        raise ValueError(f"unsupported LTX cache recipe: {recipe.model.recipe!r}") from error
    return build_video_decoding_dataset(
        recipe,
        manifest,
        geometry=LTX_VIDEO_GEOMETRY,
        conditioning_layout=layout,
    )


def _write_ltx_entry(
    *,
    store: VideoCacheStore,
    dataset: VideoDecodingDataset,
    index: int,
    audit: PromptSafetyAudit,
    model_recipe: str,
    codec_identity: Mapping[str, object],
    conditioner_identity: Mapping[str, object],
    tokenizer_identity: Mapping[str, object],
    conditioning: Mapping[str, torch.Tensor],
    video_encoder: LTXVideoFeatureEncoder,
) -> VideoCacheEntry:
    decoded = dataset[index]
    latents, loss_mask, valid_mask = video_encoder.encode(decoded)
    return write_video_cache_entry(
        store=store,
        dataset=dataset,
        index=index,
        decoded=decoded,
        audit=audit,
        model_recipe=model_recipe,
        latent_geometry=LTX_VIDEO_GEOMETRY,
        latent_normalization=ltx_latent_normalization(model_recipe),
        codec=codec_identity,
        conditioner=conditioner_identity,
        tokenizer=tokenizer_identity,
        clean_latents=latents,
        conditioning=conditioning,
        conditioning_layouts=_LTX_CONDITIONING_TENSOR_LAYOUTS,
        latent_loss_mask=loss_mask,
        valid_latent_mask=valid_mask,
    )


def prepare_ltx_training_cache_from_audits(
    *,
    dataset: VideoDecodingDataset,
    store: VideoCacheStore,
    text_encoder: LTXTextFeatureEncoder,
    video_encoder: LTXVideoFeatureEncoder,
    safety_audits: Sequence[PromptSafetyAudit],
    model_recipe: str,
    codec: Mapping[str, object],
    conditioner: Mapping[str, object],
    tokenizer: Mapping[str, object],
) -> VideoCachePreparationResult:
    """Prepare a cache from caller-owned tiny or already-materialized components."""

    audits = validate_video_prompt_audits(dataset.manifest_dataset, safety_audits)
    entries: list[VideoCacheEntry] = []
    for index, (source, assignment, audit) in enumerate(
        zip(dataset.manifest_dataset, dataset.assignments, audits, strict=True)
    ):
        conditioning = text_encoder.encode(
            sample_id=source.sample_id,
            prompt=source.prompt,
            frames=assignment.target_num_frames,
            height=assignment.target_height,
            width=assignment.target_width,
            fps=source.fps,
        )
        entries.append(
            _write_ltx_entry(
                store=store,
                dataset=dataset,
                index=index,
                audit=audit,
                model_recipe=model_recipe,
                codec_identity=codec,
                conditioner_identity=conditioner,
                tokenizer_identity=tokenizer,
                conditioning=conditioning,
                video_encoder=video_encoder,
            )
        )
    index = store.write_index(entries=entries)
    return VideoCachePreparationResult(index, tuple(entries), audits)


def _component_options(recipe: TrainingRecipe) -> dict[str, object]:
    options: dict[str, object] = {}
    for source, destination in (
        ("vae_tiled", "tiled"),
        ("vae_spatial_tile_size", "spatial_tile_size"),
        ("vae_spatial_overlap", "spatial_overlap"),
        ("vae_temporal_tile_size", "temporal_tile_size"),
        ("vae_temporal_overlap", "temporal_overlap"),
    ):
        if source in recipe.model.options:
            options[destination] = recipe.model.options[source]
    return options


def _encoder_identities(
    model_recipe: str,
    checkpoints: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    codec = checkpoint_spec_identity(checkpoints["model"])
    tokenizer = checkpoint_spec_identity(checkpoints["tokenizer"])
    if model_recipe == "ltx-video-i2v":
        conditioner = checkpoint_spec_identity(checkpoints["text_encoder"])
    else:
        conditioner = {
            "backbone": checkpoint_spec_identity(checkpoints["gemma"]),
            "projection": checkpoint_spec_identity(checkpoints["model"]),
        }
    return codec, conditioner, tokenizer


def materialize_ltx_training_cache(
    recipe: TrainingRecipe,
    *,
    manifest_path: str | Path,
    cache_dir: str | Path,
    base_dir: str | Path = ".",
    device: str | torch.device = "cuda",
    checkpoint_overrides: Mapping[str, object] | None = None,
    shieldgemma_checkpoint: object | None = None,
    safety_audits: Sequence[PromptSafetyAudit] | None = None,
    verify_media_files: bool = True,
    safety_batch_size: int = 4,
) -> VideoCachePreparationResult:
    """Build LTX prompt features, release them, then build native VAE latents."""

    from worldfoundry.base_models.diffusion_model.assembly import NativeDiffusionAssembler
    from worldfoundry.base_models.diffusion_model.components import (
        BuildPurpose,
        ComponentKey,
        ComponentKind,
        ComponentSpec,
    )
    from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx import (
        build_ltx_tensor_video_codec,
    )
    from worldfoundry.base_models.diffusion_model.optimizations import AttentionBackend, RuntimePolicy
    from worldfoundry.base_models.diffusion_model.recipes.registry import default_native_diffusion_registry
    from worldfoundry.training.engine.video_flow import torch_dtype
    from worldfoundry.training.safety.shieldgemma import build_shieldgemma_prompt_filter

    if recipe.model.recipe not in LTX_MODEL_RECIPES:
        raise ValueError(f"LTX cache materialization does not support {recipe.model.recipe!r}")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    manifest = TrainingManifestDataset.from_file(
        manifest_path,
        split=recipe.data.split,
        verify_files=verify_media_files,
    )
    dataset = build_ltx_video_decoding_dataset(recipe, manifest)
    store = VideoCacheStore(cache_dir)
    if (store.root / "index.json").exists():
        raise FileExistsError("LTX cache index already exists; materialization will not overwrite it")

    if safety_audits is None:
        prompt_filter = build_shieldgemma_prompt_filter(
            shieldgemma_checkpoint,
            device=resolved_device,
            dtype=torch.bfloat16,
        )
        audits = audit_video_prompts(manifest, prompt_filter, batch_size=safety_batch_size)
        del prompt_filter
        release_accelerator_memory(resolved_device)
    else:
        audits = tuple(safety_audits)
    audits = validate_video_prompt_audits(manifest, audits)

    root = Path(base_dir).expanduser().resolve()
    native_recipe = default_native_diffusion_registry().resolve(recipe.model.recipe)
    overrides = dict(checkpoint_overrides or {})
    if recipe.model.checkpoint != "default":
        if "model" in overrides:
            raise ValueError("model.checkpoint and checkpoint_overrides['model'] cannot both be set")
        checkpoint = Path(recipe.model.checkpoint).expanduser()
        overrides["model"] = str(checkpoint if checkpoint.is_absolute() else root / checkpoint)
    assembler = NativeDiffusionAssembler()
    checkpoints = assembler.resolve_checkpoints(native_recipe, overrides)
    codec_identity, conditioner_identity, tokenizer_identity = _encoder_identities(
        recipe.model.recipe,
        checkpoints,
    )
    policy = RuntimePolicy(
        device=resolved_device,
        dtype=torch_dtype(recipe.runtime.param_dtype),
        attention=AttentionBackend.TORCH,
    )
    conditioner_key = ComponentKey(ComponentKind.CONDITIONER)
    codec_key = ComponentKey(ComponentKind.LATENT_ENCODER, "codec")
    codec_recipe = replace(
        native_recipe,
        components=(
            *native_recipe.components,
            ComponentSpec(codec_key, build_ltx_tensor_video_codec, {"weights": "model"}),
        ),
    )

    with staged_video_conditioning(store.root, family="ltx") as stage:
        text_components = assembler.build_components(
            native_recipe,
            purpose=BuildPurpose.TRAINING,
            policy=policy,
            checkpoint_overrides=overrides,
            component_keys=(conditioner_key,),
        )
        text_encoder = LTXTextFeatureEncoder(
            text_components[conditioner_key],
            device=resolved_device,
            dtype=policy.dtype,
            include_audio=False,
        )
        for index, (source, assignment) in enumerate(zip(manifest, dataset.assignments, strict=True)):
            stage.write(
                index,
                text_encoder.encode(
                    sample_id=source.sample_id,
                    prompt=source.prompt,
                    frames=assignment.target_num_frames,
                    height=assignment.target_height,
                    width=assignment.target_width,
                    fps=source.fps,
                ),
            )
        del text_encoder, text_components
        release_accelerator_memory(resolved_device)

        video_components = assembler.build_components(
            codec_recipe,
            purpose=BuildPurpose.TRAINING,
            policy=policy,
            checkpoint_overrides=overrides,
            component_options={codec_key: _component_options(recipe)},
            component_keys=(codec_key,),
        )
        video_encoder = LTXVideoFeatureEncoder(
            video_components[codec_key],
            sample_posterior=recipe.model.recipe == "ltx-video-i2v",
            latent_channels=int(native_recipe.options["latent_channels"]),
            temporal_compression=int(native_recipe.options["temporal_compression"]),
            spatial_compression=int(native_recipe.options["spatial_compression"]),
        )
        entries = [
            _write_ltx_entry(
                store=store,
                dataset=dataset,
                index=index,
                audit=audit,
                model_recipe=recipe.model.recipe,
                codec_identity=codec_identity,
                conditioner_identity=conditioner_identity,
                tokenizer_identity=tokenizer_identity,
                conditioning=stage.read(index),
                video_encoder=video_encoder,
            )
            for index, audit in enumerate(audits)
        ]
        index = store.write_index(entries=entries)
    return VideoCachePreparationResult(index, tuple(entries), audits)


__all__ = [
    "LTX_VIDEO_GEOMETRY",
    "build_ltx_video_decoding_dataset",
    "materialize_ltx_training_cache",
    "prepare_ltx_training_cache_from_audits",
]

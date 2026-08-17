"""Wan training-cache preparation and two-phase materialization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from worldfoundry.training.safety.shieldgemma import PromptSafetyAudit, ShieldGemmaPromptFilter

from ..dataset import TrainingManifestDataset
from ..shared_conditioning import SharedConditioningArtifact
from ..video_bucketing import VideoLatentGeometry
from ..video_cache import (
    VideoCacheEntry,
    VideoCacheIndex,
    VideoCacheProvenance,
    VideoCacheStore,
)
from ..video_dataset import (
    DecodedVideoSample,
    VideoDecodingDataset,
)
from ..video_precompute import (
    audit_video_prompts,
    build_video_decoding_dataset,
    release_accelerator_memory,
    staged_video_conditioning,
    validate_video_prompt_audits,
)
from .artifacts import write_wan_unconditional_conditioning
from .contracts import (
    WAN_CONDITIONING_LAYOUT,
    wan_checkpoint_asset_identity,
    wan_latent_normalization,
)
from .encoding import WanFeatureEncoder, WanTextFeatureEncoder, WanVideoFeatureEncoder


@dataclass(frozen=True, slots=True)
class WanCachePreparationResult:
    index: VideoCacheIndex
    entries: tuple[VideoCacheEntry, ...]
    safety_audits: tuple[PromptSafetyAudit, ...]
    unconditional_conditioning: SharedConditioningArtifact


def _audit_prompts(
    manifest: TrainingManifestDataset,
    prompt_filter: ShieldGemmaPromptFilter,
    *,
    batch_size: int,
) -> tuple[PromptSafetyAudit, ...]:
    return audit_video_prompts(manifest, prompt_filter, batch_size=batch_size)


def _validate_audits(
    dataset: VideoDecodingDataset,
    safety_audits: Sequence[PromptSafetyAudit],
) -> tuple[PromptSafetyAudit, ...]:
    return validate_video_prompt_audits(dataset.manifest_dataset, safety_audits)


def _write_cache_entry(
    *,
    store: VideoCacheStore,
    dataset: VideoDecodingDataset,
    index: int,
    decoded: DecodedVideoSample,
    audit: PromptSafetyAudit,
    latents: torch.Tensor,
    context: torch.Tensor,
    latent_loss_mask: torch.Tensor,
    valid_latent_mask: torch.Tensor,
    model_recipe: str,
    codec: Mapping[str, object],
    conditioner: Mapping[str, object],
    tokenizer: Mapping[str, object],
    latent_geometry: VideoLatentGeometry,
) -> VideoCacheEntry:
    source = dataset.manifest_dataset[index]
    assignment = decoded.assignment
    source_payload = source.to_dict()
    provenance = VideoCacheProvenance(
        media_uri=source.media.uri,
        prompt=source.prompt,
        model_recipe=model_recipe,
        codec=codec,
        conditioner=conditioner,
        tokenizer=tokenizer,
        conditioning_inputs={"task": source.task, "conditions": source_payload["conditions"]},
        safety_audit=audit.to_dict(),
        frame_sampling=decoded.frame_sampling,
        spatial_transform=decoded.spatial_transform,
        latent_normalization=wan_latent_normalization(),
        task=source.task,
        conditioning_layout=assignment.bucket_key.conditioning_layout,
        aspect_bin=assignment.bucket_key.aspect_bin,
        source_num_frames=source.num_frames,
        source_height=source.height,
        source_width=source.width,
        source_fps=source.fps,
        target_num_frames=assignment.target_num_frames,
        target_height=assignment.target_height,
        target_width=assignment.target_width,
        target_fps=source.fps,
        latent_geometry=latent_geometry,
    )
    return store.write_sample(
        sample_id=source.sample_id,
        provenance=provenance,
        clean_latents=latents,
        conditioning={"context": context},
        conditioning_layouts={"context": "sequence-features"},
        latent_loss_mask=latent_loss_mask,
        valid_latent_mask=valid_latent_mask,
    )


def prepare_wan_training_cache_from_audits(
    *,
    dataset: VideoDecodingDataset,
    store: VideoCacheStore,
    feature_encoder: WanFeatureEncoder,
    safety_audits: Sequence[PromptSafetyAudit],
    model_recipe: str,
    codec: Mapping[str, object],
    conditioner: Mapping[str, object],
    tokenizer: Mapping[str, object],
) -> WanCachePreparationResult:
    """Decode and encode one Wan cache after caller-owned safety auditing."""

    if not isinstance(dataset, VideoDecodingDataset):
        raise TypeError("dataset must be a VideoDecodingDataset")
    if not isinstance(store, VideoCacheStore):
        raise TypeError("store must be a VideoCacheStore")
    if not isinstance(feature_encoder, WanFeatureEncoder):
        raise TypeError("feature_encoder must be a WanFeatureEncoder")
    if (store.root / "index.json").exists():
        raise FileExistsError("Wan cache index already exists; preparation will not overwrite it")
    audits = _validate_audits(dataset, safety_audits)
    if not dataset.assignments:
        raise ValueError("Wan cache preparation requires at least one assigned sample")
    first_assignment = dataset.assignments[0]
    unconditional_context = feature_encoder.text.encode(
        sample_id="shared-unconditional",
        prompt="",
        frames=first_assignment.target_num_frames,
        height=first_assignment.target_height,
        width=first_assignment.target_width,
    )
    geometry = VideoLatentGeometry(8, 8, 4, "first-frame")
    entries: list[VideoCacheEntry] = []
    for index, (decoded, audit) in enumerate(zip(dataset, audits)):
        latents, context, loss_mask, valid_mask = feature_encoder.encode(decoded)
        entries.append(
            _write_cache_entry(
                store=store,
                dataset=dataset,
                index=index,
                decoded=decoded,
                audit=audit,
                latents=latents,
                context=context,
                latent_loss_mask=loss_mask,
                valid_latent_mask=valid_mask,
                model_recipe=model_recipe,
                codec=codec,
                conditioner=conditioner,
                tokenizer=tokenizer,
                latent_geometry=geometry,
            )
        )
    unconditional = write_wan_unconditional_conditioning(
        store=store,
        context=unconditional_context,
        model_recipe=model_recipe,
        conditioner=conditioner,
        tokenizer=tokenizer,
    )
    cache_index = store.write_index(
        entries=entries,
    )
    return WanCachePreparationResult(
        cache_index,
        tuple(entries),
        audits,
        unconditional,
    )


def build_wan_video_decoding_dataset(
    recipe: object,
    manifest: TrainingManifestDataset,
) -> VideoDecodingDataset:
    """Resolve strict bucket/decode declarations from a training recipe."""

    from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
    from worldfoundry.training.recipes.spec import TrainingRecipe

    if not isinstance(recipe, (TrainingRecipe, PostTrainingRecipe)):
        raise TypeError("recipe must be TrainingRecipe or PostTrainingRecipe")
    if not isinstance(manifest, TrainingManifestDataset):
        raise TypeError("manifest must be TrainingManifestDataset")
    geometry = VideoLatentGeometry(8, 8, 4, "first-frame")
    return build_video_decoding_dataset(
        recipe,
        manifest,
        geometry=geometry,
        conditioning_layout=WAN_CONDITIONING_LAYOUT,
    )


def materialize_wan_training_cache(
    recipe: object,
    *,
    manifest_path: str | Path,
    cache_dir: str | Path,
    device: str | torch.device = "cuda",
    checkpoint_overrides: Mapping[str, object] | None = None,
    shieldgemma_checkpoint: object | None = None,
    safety_audits: Sequence[PromptSafetyAudit] | None = None,
    verify_media_files: bool = True,
    safety_batch_size: int = 4,
) -> WanCachePreparationResult:
    """Build UMT5 and VAE features in separate GPU-residency phases."""

    from worldfoundry.base_models.diffusion_model.assembly import NativeDiffusionAssembler
    from worldfoundry.base_models.diffusion_model.components import (
        BuildPurpose,
        ComponentKey,
        ComponentKind,
    )
    from worldfoundry.base_models.diffusion_model.optimizations import (
        AttentionBackend,
        RuntimePolicy,
    )
    from worldfoundry.base_models.diffusion_model.recipes.registry import (
        default_native_diffusion_registry,
    )
    from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
    from worldfoundry.training.recipes.spec import TrainingRecipe
    from worldfoundry.training.safety.shieldgemma import build_shieldgemma_prompt_filter

    if not isinstance(recipe, (TrainingRecipe, PostTrainingRecipe)):
        raise TypeError("recipe must be TrainingRecipe or PostTrainingRecipe")
    if recipe.model.recipe != "wan2.1-t2v-1.3b":
        raise ValueError("Wan cache materialization currently requires wan2.1-t2v-1.3b")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    manifest = TrainingManifestDataset.from_file(
        manifest_path,
        split=recipe.data.split,
        verify_files=verify_media_files,
    )
    decoded_dataset = build_wan_video_decoding_dataset(recipe, manifest)
    store = VideoCacheStore(cache_dir)
    if (store.root / "index.json").exists():
        raise FileExistsError("Wan cache index already exists; materialization will not overwrite it")

    if safety_audits is not None and shieldgemma_checkpoint is not None:
        raise ValueError("safety_audits and shieldgemma_checkpoint are mutually exclusive")
    if safety_audits is None:
        prompt_filter = build_shieldgemma_prompt_filter(
            shieldgemma_checkpoint,
            device=resolved_device,
            dtype=torch.bfloat16,
        )
        audits = _audit_prompts(manifest, prompt_filter, batch_size=safety_batch_size)
        del prompt_filter
        release_accelerator_memory(resolved_device)
    else:
        audits = tuple(safety_audits)
    audits = _validate_audits(decoded_dataset, audits)

    native_recipe = default_native_diffusion_registry().resolve(recipe.model.recipe)
    assembler = NativeDiffusionAssembler()
    overrides = dict(checkpoint_overrides or {})
    resolved_checkpoints = assembler.resolve_checkpoints(native_recipe, overrides)
    policy = RuntimePolicy(
        device=resolved_device,
        dtype={
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[recipe.runtime.param_dtype],
        attention=AttentionBackend.TORCH,
    )
    conditioner_key = ComponentKey(ComponentKind.CONDITIONER)
    codec_key = ComponentKey(ComponentKind.LATENT_ENCODER, "codec")
    with staged_video_conditioning(store.root, family="wan") as text_stage:
        text_components = assembler.build_components(
            native_recipe,
            purpose=BuildPurpose.TRAINING,
            policy=policy,
            checkpoint_overrides=overrides,
            component_keys=(conditioner_key,),
        )
        text_encoder = WanTextFeatureEncoder(text_components[conditioner_key])
        if not decoded_dataset.assignments:
            raise ValueError("Wan cache materialization requires at least one assigned sample")
        first_assignment = decoded_dataset.assignments[0]
        unconditional_context = text_encoder.encode(
            sample_id="shared-unconditional",
            prompt="",
            frames=first_assignment.target_num_frames,
            height=first_assignment.target_height,
            width=first_assignment.target_width,
        )
        for index, (source, assignment) in enumerate(zip(manifest, decoded_dataset.assignments)):
            context = text_encoder.encode(
                sample_id=source.sample_id,
                prompt=source.prompt,
                frames=assignment.target_num_frames,
                height=assignment.target_height,
                width=assignment.target_width,
            )
            text_stage.write(index, {"context": context})
        del text_encoder, text_components
        release_accelerator_memory(resolved_device)

        codec_options: dict[str, object] = {}
        for source_name, destination_name in (
            ("vae_tiled", "tiled"),
            ("vae_tile_size", "tile_size"),
            ("vae_tile_stride", "tile_stride"),
        ):
            if source_name in recipe.model.options:
                codec_options[destination_name] = recipe.model.options[source_name]
        video_components = assembler.build_components(
            native_recipe,
            purpose=BuildPurpose.TRAINING,
            policy=policy,
            checkpoint_overrides=overrides,
            component_options={codec_key: codec_options},
            component_keys=(codec_key,),
        )
        video_encoder = WanVideoFeatureEncoder(video_components[codec_key])
        geometry = VideoLatentGeometry(8, 8, 4, "first-frame")
        entries: list[VideoCacheEntry] = []
        for index, (decoded, audit) in enumerate(zip(decoded_dataset, audits)):
            context = text_stage.read(index)["context"]
            latents, loss_mask, valid_mask = video_encoder.encode(decoded)
            entries.append(
                _write_cache_entry(
                    store=store,
                    dataset=decoded_dataset,
                    index=index,
                    decoded=decoded,
                    audit=audit,
                    latents=latents,
                    context=context,
                    latent_loss_mask=loss_mask,
                    valid_latent_mask=valid_mask,
                    model_recipe=recipe.model.recipe,
                    codec=wan_checkpoint_asset_identity(resolved_checkpoints["vae"]),
                    conditioner=wan_checkpoint_asset_identity(resolved_checkpoints["text-encoder"]),
                    tokenizer=wan_checkpoint_asset_identity(resolved_checkpoints["tokenizer"]),
                    latent_geometry=geometry,
                )
            )
        conditioner = wan_checkpoint_asset_identity(resolved_checkpoints["text-encoder"])
        tokenizer = wan_checkpoint_asset_identity(resolved_checkpoints["tokenizer"])
        unconditional = write_wan_unconditional_conditioning(
            store=store,
            context=unconditional_context,
            model_recipe=recipe.model.recipe,
            conditioner=conditioner,
            tokenizer=tokenizer,
        )
        cache_index = store.write_index(
            entries=entries,
        )
        return WanCachePreparationResult(
            cache_index,
            tuple(entries),
            audits,
            unconditional,
        )


__all__ = [
    "WanCachePreparationResult",
    "build_wan_video_decoding_dataset",
    "materialize_wan_training_cache",
    "prepare_wan_training_cache_from_audits",
]

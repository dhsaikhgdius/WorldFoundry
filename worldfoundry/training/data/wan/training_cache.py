"""Wan training-cache preparation and two-phase materialization."""

from __future__ import annotations

import gc
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from worldfoundry.core.io.integrity import canonical_sha256, text_sha256
from worldfoundry.training.safety.shieldgemma import PromptSafetyAudit, ShieldGemmaPromptFilter

from ..dataset import TrainingManifestDataset
from ..shared_conditioning import SharedConditioningArtifact
from ..video_bucketing import (
    VideoBucketSelectionPolicy,
    VideoLatentGeometry,
    VideoResolutionBucket,
    assign_video_buckets,
)
from ..video_cache import (
    VideoCacheEntry,
    VideoCacheIndex,
    VideoCacheProvenance,
    VideoCacheStore,
)
from ..video_dataset import (
    DecodedVideoSample,
    VideoDecodeConfig,
    VideoDecodingDataset,
)
from .artifacts import write_wan_unconditional_conditioning
from .contracts import (
    WAN_CONDITIONING_LAYOUT,
    require_positive_int,
    wan_cache_contract_digest,
    wan_checkpoint_asset_digest,
    wan_latent_normalization_digest,
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
    size = require_positive_int(batch_size, field_name="safety_batch_size")
    audits: list[PromptSafetyAudit] = []
    for offset in range(0, len(manifest), size):
        prompts = tuple(sample.prompt for sample in manifest[offset : offset + size])
        audits.extend(prompt_filter.require_safe(prompts))
    return tuple(audits)


def _validate_audits(
    dataset: VideoDecodingDataset,
    safety_audits: Sequence[PromptSafetyAudit],
) -> tuple[PromptSafetyAudit, ...]:
    audits = tuple(safety_audits)
    manifest = dataset.manifest_dataset
    if len(audits) != len(manifest) or not all(isinstance(audit, PromptSafetyAudit) for audit in audits):
        raise ValueError("one PromptSafetyAudit is required per Wan manifest sample")
    if any(not audit.safe for audit in audits):
        raise ValueError("unsafe prompt audits cannot be used to prepare a Wan cache")
    for sample, audit in zip(manifest, audits):
        if audit.prompt_sha256 != text_sha256(sample.prompt):
            raise ValueError("prompt safety audit digest differs from the Wan manifest prompt")
        recorded = sample.safety.get("prompt_audit_digest")
        if recorded is None:
            raise ValueError(f"manifest sample {sample.sample_id!r} lacks safety.prompt_audit_digest")
        if str(recorded).lower() != audit.digest:
            raise ValueError(f"manifest safety.prompt_audit_digest differs for sample {sample.sample_id!r}")
    return audits


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
    codec_digest: str,
    conditioner_digest: str,
    tokenizer_digest: str,
    latent_geometry: VideoLatentGeometry,
) -> VideoCacheEntry:
    source = dataset.manifest_dataset[index]
    assignment = decoded.assignment
    contract_digest = wan_cache_contract_digest(model_recipe)
    normalization_digest = wan_latent_normalization_digest()
    source_payload = source.to_dict()
    conditioning_inputs_digest = canonical_sha256(
        {
            "schema": "worldfoundry-wan-conditioning-inputs",
            "task": source.task,
            "conditions": source_payload["conditions"],
        }
    )
    provenance = VideoCacheProvenance(
        media_sha256=source.media.sha256,
        prompt_sha256=audit.prompt_sha256,
        model_recipe_digest=contract_digest,
        codec_digest=codec_digest,
        conditioner_digest=conditioner_digest,
        tokenizer_digest=tokenizer_digest,
        conditioning_inputs_digest=conditioning_inputs_digest,
        safety_audit_digest=audit.digest,
        frame_sampling_digest=decoded.frame_sampling_digest,
        spatial_transform_digest=decoded.spatial_transform_digest,
        latent_normalization_digest=normalization_digest,
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
    codec_digest: str,
    conditioner_digest: str,
    tokenizer_digest: str,
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
                codec_digest=codec_digest,
                conditioner_digest=conditioner_digest,
                tokenizer_digest=tokenizer_digest,
                latent_geometry=geometry,
            )
        )
    unconditional = write_wan_unconditional_conditioning(
        store=store,
        context=unconditional_context,
        model_recipe=model_recipe,
        conditioner_digest=conditioner_digest,
        tokenizer_digest=tokenizer_digest,
    )
    cache_index = store.write_index(
        dataset_digest=dataset.dataset_digest,
        entries=entries,
    )
    return WanCachePreparationResult(
        cache_index,
        tuple(entries),
        audits,
        unconditional,
    )


def prepare_wan_training_cache(
    *,
    dataset: VideoDecodingDataset,
    store: VideoCacheStore,
    feature_encoder: WanFeatureEncoder,
    prompt_filter: ShieldGemmaPromptFilter,
    model_recipe: str,
    codec_digest: str,
    conditioner_digest: str,
    tokenizer_digest: str,
    safety_batch_size: int = 4,
) -> WanCachePreparationResult:
    """Safety-audit, decode, and encode one Wan cache in a shared lifecycle."""

    if not isinstance(prompt_filter, ShieldGemmaPromptFilter):
        raise TypeError("prompt_filter must be a ShieldGemmaPromptFilter")
    audits = _audit_prompts(
        dataset.manifest_dataset,
        prompt_filter,
        batch_size=safety_batch_size,
    )
    return prepare_wan_training_cache_from_audits(
        dataset=dataset,
        store=store,
        feature_encoder=feature_encoder,
        safety_audits=audits,
        model_recipe=model_recipe,
        codec_digest=codec_digest,
        conditioner_digest=conditioner_digest,
        tokenizer_digest=tokenizer_digest,
    )


def _strict_mapping(
    value: object,
    *,
    field_name: str,
    allowed: set[str],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    payload = {str(key): item for key, item in value.items()}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"{field_name} contains unknown fields: {unknown}")
    return payload


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
    options = dict(recipe.data.options)
    raw_buckets = options.get("video_buckets")
    if (
        not isinstance(raw_buckets, Sequence)
        or isinstance(
            raw_buckets,
            (str, bytes, bytearray),
        )
        or not raw_buckets
    ):
        raise ValueError("data.options.video_buckets must be a non-empty sequence")
    buckets: list[VideoResolutionBucket] = []
    for index, raw in enumerate(raw_buckets):
        payload = _strict_mapping(
            raw,
            field_name=f"data.options.video_buckets[{index}]",
            allowed={
                "num_frames",
                "height",
                "width",
                "conditioning_layout",
                "tasks",
                "aspect_bin",
            },
        )
        payload.setdefault("conditioning_layout", WAN_CONDITIONING_LAYOUT)
        buckets.append(VideoResolutionBucket(**payload))

    policy_payload = _strict_mapping(
        options.get("bucket_policy", {}),
        field_name="data.options.bucket_policy",
        allowed={
            "aspect_weight",
            "spatial_weight",
            "temporal_weight",
            "allow_spatial_upscale",
            "allow_temporal_padding",
        },
    )
    decode_payload = _strict_mapping(
        options.get("decode", {}),
        field_name="data.options.decode",
        allowed={
            "frame_sampling",
            "interpolation",
            "value_range",
            "decoder_thread_type",
            "verify_media_sha256",
            "verify_manifest_frame_count",
            "verify_manifest_geometry",
            "fps_tolerance",
        },
    )
    decode_payload.setdefault("value_range", "minus-one-one")
    geometry = VideoLatentGeometry(8, 8, 4, "first-frame")
    assignments = assign_video_buckets(
        tuple(manifest),
        buckets=buckets,
        geometry=geometry,
        conditioning_layout=WAN_CONDITIONING_LAYOUT,
        policy=VideoBucketSelectionPolicy(**policy_payload),
    )
    return VideoDecodingDataset(
        manifest,
        assignments,
        config=VideoDecodeConfig(**decode_payload),
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
    verify_media_hashes: bool = True,
    safety_batch_size: int = 4,
) -> WanCachePreparationResult:
    """Build UMT5 and VAE features in separate GPU-residency phases."""

    from safetensors.torch import load_file, save_file

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
        verify_files=True,
        verify_hashes=verify_media_hashes,
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
        gc.collect()
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()
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
    text_stage = Path(tempfile.mkdtemp(prefix=".wan-text-stage-", dir=store.root))
    staged_contexts: list[Path] = []
    try:
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
            stage_path = text_stage / f"{index:08d}.safetensors"
            save_file({"context": context}, stage_path)
            staged_contexts.append(stage_path)
        del text_encoder, text_components
        gc.collect()
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()

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
        for index, (decoded, audit, stage_path) in enumerate(zip(decoded_dataset, audits, staged_contexts)):
            context = load_file(stage_path, device="cpu")["context"]
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
                    codec_digest=wan_checkpoint_asset_digest(resolved_checkpoints["vae"]),
                    conditioner_digest=wan_checkpoint_asset_digest(resolved_checkpoints["text-encoder"]),
                    tokenizer_digest=wan_checkpoint_asset_digest(resolved_checkpoints["tokenizer"]),
                    latent_geometry=geometry,
                )
            )
        conditioner_digest = wan_checkpoint_asset_digest(resolved_checkpoints["text-encoder"])
        tokenizer_digest = wan_checkpoint_asset_digest(resolved_checkpoints["tokenizer"])
        unconditional = write_wan_unconditional_conditioning(
            store=store,
            context=unconditional_context,
            model_recipe=recipe.model.recipe,
            conditioner_digest=conditioner_digest,
            tokenizer_digest=tokenizer_digest,
        )
        cache_index = store.write_index(
            dataset_digest=decoded_dataset.dataset_digest,
            entries=entries,
        )
        return WanCachePreparationResult(
            cache_index,
            tuple(entries),
            audits,
            unconditional,
        )
    finally:
        shutil.rmtree(text_stage, ignore_errors=True)


__all__ = [
    "WanCachePreparationResult",
    "build_wan_video_decoding_dataset",
    "materialize_wan_training_cache",
    "prepare_wan_training_cache",
    "prepare_wan_training_cache_from_audits",
]

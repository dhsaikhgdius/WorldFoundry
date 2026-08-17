"""Native Cosmos cache preparation with separate text and VAE residency phases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

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
from .encoding import CosmosTextFeatureEncoder, CosmosVideoFeatureEncoder

COSMOS_PREDICT_RECIPES = frozenset(
    {
        "cosmos-predict2-2b-video2world",
        "cosmos-predict2-14b-video2world",
        "cosmos-predict2.5-2b",
        "cosmos-predict2.5-14b",
    }
)
COSMOS3_RECIPES = frozenset({"cosmos3-nano", "cosmos3-super"})
COSMOS_CACHE_RECIPES = COSMOS_PREDICT_RECIPES | COSMOS3_RECIPES

_PREDICT_GEOMETRY = VideoLatentGeometry(8, 8, 4, "first-frame")
_COSMOS3_GEOMETRY = VideoLatentGeometry(16, 16, 4, "first-frame")
_CONDITIONING_LAYOUTS = {
    "cosmos-predict2-2b-video2world": "t5-sequence",
    "cosmos-predict2-14b-video2world": "t5-sequence",
    "cosmos-predict2.5-2b": "reason1-sequence",
    "cosmos-predict2.5-14b": "reason1-sequence",
    "cosmos3-nano": "cosmos3-token-sequence",
    "cosmos3-super": "cosmos3-token-sequence",
}


def _geometry(model_recipe: str) -> VideoLatentGeometry:
    return _COSMOS3_GEOMETRY if model_recipe in COSMOS3_RECIPES else _PREDICT_GEOMETRY


def build_cosmos_video_decoding_dataset(
    recipe: TrainingRecipe,
    manifest: TrainingManifestDataset,
) -> VideoDecodingDataset:
    try:
        layout = _CONDITIONING_LAYOUTS[recipe.model.recipe]
    except KeyError as error:
        raise ValueError(f"unsupported Cosmos cache recipe: {recipe.model.recipe!r}") from error
    return build_video_decoding_dataset(
        recipe,
        manifest,
        geometry=_geometry(recipe.model.recipe),
        conditioning_layout=layout,
    )


def cosmos_latent_normalization(vae: object) -> dict[str, object]:
    """Record the deterministic affine already applied by the native Wan VAE."""

    mean = getattr(vae, "mean", None)
    std = getattr(vae, "std", None)
    if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
        raise TypeError("Cosmos native VAE must expose channel mean/std tensors")
    return {
        "posterior": "deterministic-mean",
        "operation": "(mean-latent-channel-mean)/channel-std",
        "channel_mean": mean.detach().cpu().float().tolist(),
        "channel_std": std.detach().cpu().float().tolist(),
    }


def _conditioning_layouts(conditioning: Mapping[str, torch.Tensor]) -> dict[str, str]:
    layouts = {
        "context": "sequence-features",
        "negative_context": "sequence-features",
        "input_ids": "sequence-tokens",
        "empty_input_ids": "sequence-tokens",
    }
    return {name: layouts[name] for name in conditioning}


def _write_cosmos_entry(
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
    video_encoder: CosmosVideoFeatureEncoder,
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
        latent_geometry=_geometry(model_recipe),
        latent_normalization=cosmos_latent_normalization(video_encoder.vae),
        codec=codec_identity,
        conditioner=conditioner_identity,
        tokenizer=tokenizer_identity,
        clean_latents=latents,
        conditioning=conditioning,
        conditioning_layouts=_conditioning_layouts(conditioning),
        latent_loss_mask=loss_mask,
        valid_latent_mask=valid_mask,
    )


def prepare_cosmos_training_cache_from_audits(
    *,
    dataset: VideoDecodingDataset,
    store: VideoCacheStore,
    text_encoder: CosmosTextFeatureEncoder,
    video_encoder: CosmosVideoFeatureEncoder,
    safety_audits: Sequence[PromptSafetyAudit],
    model_recipe: str,
    codec: Mapping[str, object],
    conditioner: Mapping[str, object],
    tokenizer: Mapping[str, object],
) -> VideoCachePreparationResult:
    """Prepare one round-trip cache from caller-owned tiny components."""

    audits = validate_video_prompt_audits(dataset.manifest_dataset, safety_audits)
    conditionings = [
        text_encoder.encode(
            sample_id=source.sample_id,
            prompt=source.prompt,
            frames=assignment.target_num_frames,
            height=assignment.target_height,
            width=assignment.target_width,
            fps=source.fps,
        )
        for source, assignment in zip(
            dataset.manifest_dataset,
            dataset.assignments,
            strict=True,
        )
    ]
    entries = [
        _write_cosmos_entry(
            store=store,
            dataset=dataset,
            index=index,
            audit=audit,
            model_recipe=model_recipe,
            codec_identity=codec,
            conditioner_identity=conditioner,
            tokenizer_identity=tokenizer,
            conditioning=conditionings[index],
            video_encoder=video_encoder,
        )
        for index, audit in enumerate(audits)
    ]
    index = store.write_index(entries=entries)
    return VideoCachePreparationResult(index, tuple(entries), audits)


def _encoder_identities(
    model_recipe: str,
    checkpoints: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    codec = checkpoint_spec_identity(checkpoints["vae"])
    if model_recipe.startswith("cosmos-predict2-"):
        conditioner = checkpoint_spec_identity(checkpoints["text-encoder"])
        tokenizer = checkpoint_spec_identity(checkpoints["text-tokenizer"])
    elif model_recipe.startswith("cosmos-predict2.5-"):
        conditioner = checkpoint_spec_identity(checkpoints["text-encoder"])
        tokenizer = checkpoint_spec_identity(checkpoints["tokenizer"])
    else:
        tokenizer = checkpoint_spec_identity(checkpoints["tokenizer"])
        conditioner = dict(tokenizer)
    return codec, conditioner, tokenizer


def _codec_options(recipe: TrainingRecipe) -> dict[str, object]:
    options: dict[str, object] = {}
    for source, destination in (
        ("vae_tiled", "tiled"),
        ("vae_tile_size", "tile_size"),
        ("vae_tile_stride", "tile_stride"),
    ):
        if source in recipe.model.options:
            options[destination] = recipe.model.options[source]
    return options


def _pair_option(value: object, default: tuple[int, int]) -> tuple[int, int]:
    raw = default if value is None else tuple(value)  # type: ignore[arg-type]
    return int(raw[0]), int(raw[1])


def materialize_cosmos_training_cache(
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
    """Build Cosmos conditioning, release it, then encode native VAE latents."""

    from worldfoundry.base_models.diffusion_model.assembly import NativeDiffusionAssembler
    from worldfoundry.base_models.diffusion_model.components import BuildPurpose, ComponentKey, ComponentKind
    from worldfoundry.base_models.diffusion_model.optimizations import AttentionBackend, RuntimePolicy
    from worldfoundry.base_models.diffusion_model.recipes.registry import default_native_diffusion_registry
    from worldfoundry.training.engine.video_flow import torch_dtype
    from worldfoundry.training.safety.shieldgemma import build_shieldgemma_prompt_filter

    if recipe.model.recipe not in COSMOS_CACHE_RECIPES:
        raise ValueError(f"Cosmos cache materialization does not support {recipe.model.recipe!r}")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    manifest = TrainingManifestDataset.from_file(
        manifest_path,
        split=recipe.data.split,
        verify_files=verify_media_files,
    )
    dataset = build_cosmos_video_decoding_dataset(recipe, manifest)
    store = VideoCacheStore(cache_dir)
    if (store.root / "index.json").exists():
        raise FileExistsError("Cosmos cache index already exists; materialization will not overwrite it")

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
        if "transformer" in overrides:
            raise ValueError("model.checkpoint and checkpoint_overrides['transformer'] cannot both be set")
        checkpoint = Path(recipe.model.checkpoint).expanduser()
        overrides["transformer"] = str(
            checkpoint if checkpoint.is_absolute() else root / checkpoint
        )
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
    codec_key = (
        ComponentKey(ComponentKind.LATENT_ENCODER, "codec")
        if recipe.model.recipe.startswith("cosmos-predict2-")
        else ComponentKey(ComponentKind.LATENT_INITIALIZER)
    )
    codec_options = _codec_options(recipe)

    with staged_video_conditioning(store.root, family="cosmos") as stage:
        component_options = (
            {conditioner_key: {"use_system_prompt": False}}
            if recipe.model.recipe in COSMOS3_RECIPES
            else None
        )
        text_components = assembler.build_components(
            native_recipe,
            purpose=BuildPurpose.TRAINING,
            policy=policy,
            checkpoint_overrides=overrides,
            component_options=component_options,
            component_keys=(conditioner_key,),
        )
        text_encoder = CosmosTextFeatureEncoder(
            text_components[conditioner_key],
            model_recipe=recipe.model.recipe,
            device=resolved_device,
            dtype=policy.dtype,
        )
        for index, (source, assignment) in enumerate(
            zip(manifest, dataset.assignments, strict=True)
        ):
            conditioning = text_encoder.encode(
                sample_id=source.sample_id,
                prompt=source.prompt,
                frames=assignment.target_num_frames,
                height=assignment.target_height,
                width=assignment.target_width,
                fps=source.fps,
            )
            stage.write(index, conditioning)
        del text_encoder, text_components
        release_accelerator_memory(resolved_device)

        video_components = assembler.build_components(
            native_recipe,
            purpose=BuildPurpose.TRAINING,
            policy=policy,
            checkpoint_overrides=overrides,
            component_options={codec_key: codec_options} if codec_options else None,
            component_keys=(codec_key,),
        )
        cosmos3 = recipe.model.recipe in COSMOS3_RECIPES
        video_encoder = CosmosVideoFeatureEncoder(
            video_components[codec_key],
            cosmos3=cosmos3,
            latent_channels=int(native_recipe.options["latent_channels"]),
            temporal_compression=int(native_recipe.options["temporal_compression"]),
            spatial_compression=int(native_recipe.options["spatial_compression"]),
            tiled=bool(recipe.model.options.get("vae_tiled", False)),
            tile_size=_pair_option(recipe.model.options.get("vae_tile_size"), (34, 34)),
            tile_stride=_pair_option(recipe.model.options.get("vae_tile_stride"), (18, 16)),
        )
        entries = [
            _write_cosmos_entry(
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
    "COSMOS3_RECIPES",
    "COSMOS_CACHE_RECIPES",
    "COSMOS_PREDICT_RECIPES",
    "build_cosmos_video_decoding_dataset",
    "cosmos_latent_normalization",
    "materialize_cosmos_training_cache",
    "prepare_cosmos_training_cache_from_audits",
]

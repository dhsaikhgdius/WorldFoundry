"""Deterministic image decoding and SANA cache precomputation."""

from __future__ import annotations

import gc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.contracts import (
    Conditioning,
    DiffusionRequest,
    SamplingConfig,
)
from worldfoundry.training.safety.shieldgemma import (
    PromptSafetyAudit,
    ShieldGemmaPromptFilter,
)

from .checkpoint_assets import checkpoint_asset_identity
from .dataset import TrainingManifestDataset
from .manifest import TrainingSample, resolve_local_media_path
from .sana_cache import (
    SanaCacheEntry,
    SanaCacheIndex,
    SanaCacheProvenance,
    SanaCacheStore,
)
from .shared_conditioning import SharedConditioningArtifact, SharedConditioningStore

SANA_PIXEL_TRANSFORM_SCHEMA = "worldfoundry-sana-exact-image-transform"


def _validate_sana_cache_recipe(recipe: object) -> object:
    from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
    from worldfoundry.training.recipes.spec import TrainingRecipe

    if not isinstance(recipe, (TrainingRecipe, PostTrainingRecipe)):
        raise TypeError("recipe must be TrainingRecipe or PostTrainingRecipe")
    if not recipe.model.recipe.startswith("sana-"):
        raise ValueError("SANA cache materialization requires a SANA model recipe")
    return recipe


def prompt_enhancement_config(
    *,
    enabled: bool,
    max_text_length: int,
    prefix: str,
) -> dict[str, object]:
    """Return the prompt enhancement behavior used by the conditioner."""

    if not isinstance(enabled, bool):
        raise TypeError("prompt enhancement enabled must be a bool")
    if isinstance(max_text_length, bool) or int(max_text_length) < 2:
        raise ValueError("max_text_length must be at least two")
    return {
        "enabled": enabled,
        "max_text_length": int(max_text_length),
        "prefix": str(prefix) if enabled else "",
    }


@dataclass(frozen=True, slots=True)
class ExactSanaImageTransform:
    """Decode one already-sized image into RGB ``[-1, 1]`` tensor space."""

    apply_exif_orientation: bool = True
    schema: str = SANA_PIXEL_TRANSFORM_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SANA_PIXEL_TRANSFORM_SCHEMA:
            raise ValueError(f"unsupported SANA pixel transform schema: {self.schema!r}")
        if not isinstance(self.apply_exif_orientation, bool):
            raise TypeError("apply_exif_orientation must be a bool")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "apply_exif_orientation": self.apply_exif_orientation,
            "color_mode": "RGB",
            "geometry": "require-exact-manifest-dimensions",
            "layout": "channels-single-frame-height-width",
            "value_transform": "uint8-div-127.5-minus-1",
        }

    def decode(self, sample: TrainingSample, *, manifest_path: str | Path) -> torch.Tensor:
        if sample.num_frames != 1:
            raise ValueError("exact SANA image transform rejects multi-frame samples")
        path = resolve_local_media_path(sample.media, manifest_path=manifest_path)
        if path is None:
            raise ValueError("SANA cache precomputation currently requires local media")
        try:
            import numpy as np
            from PIL import Image, ImageOps
        except ModuleNotFoundError as error:
            raise RuntimeError("SANA image cache precomputation requires Pillow and NumPy") from error
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source) if self.apply_exif_orientation else source.copy()
            image = image.convert("RGB")
            if image.size != (sample.width, sample.height):
                raise ValueError(
                    f"decoded image dimensions {image.size} differ from manifest "
                    f"{(sample.width, sample.height)} for {sample.sample_id!r}"
                )
            array = np.array(image, dtype=np.float32, copy=True)
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        tensor = tensor.div_(127.5).sub_(1.0)
        return tensor.unsqueeze(1)


def _component_module(component: object, *names: str) -> nn.Module | None:
    if isinstance(component, nn.Module):
        return component
    for name in names:
        value = getattr(component, name, None)
        if isinstance(value, nn.Module):
            return value
    return None


class SanaFeatureEncoder:
    """Frozen DCAE/Gemma feature path used only while building a cache."""

    def __init__(self, codec: object, conditioner: object) -> None:
        if not callable(getattr(codec, "encode", None)):
            raise TypeError("SANA cache codec must expose encode(images)")
        if not callable(getattr(conditioner, "encode", None)):
            raise TypeError("SANA cache conditioner must expose encode(request, device, dtype)")
        codec_module = _component_module(codec, "model")
        conditioner_module = _component_module(conditioner, "encoder", "model")
        if codec_module is None or conditioner_module is None:
            raise TypeError("SANA cache components must expose their torch modules")
        for module in (codec_module, conditioner_module):
            module.requires_grad_(False)
            module.eval()
        reference = next(codec_module.parameters(), None)
        if reference is None:
            reference = next(codec_module.buffers(), None)
        if reference is None:
            raise ValueError("SANA codec module has no parameter or buffer for placement")
        conditioner_reference = next(conditioner_module.parameters(), None)
        if conditioner_reference is not None and conditioner_reference.device != reference.device:
            raise ValueError("SANA codec and conditioner must be placed on the same device")
        self.codec = codec
        self.conditioner = conditioner
        self.codec_module = codec_module
        self.conditioner_module = conditioner_module
        self.device = reference.device
        self.dtype = reference.dtype if reference.is_floating_point() else torch.float32
        self.scaling_factor = float(getattr(codec, "scaling_factor", 1.0))
        if self.scaling_factor <= 0:
            raise ValueError("SANA codec scaling_factor must be positive")
        self.max_text_length = int(getattr(conditioner, "max_length", 300))

    def encode(
        self,
        *,
        sample_id: str,
        prompt: str,
        pixels: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pixels.shape != (3, 1, image_height, image_width):
            raise ValueError("SANA cache pixels must be unbatched [3,1,H,W]")
        images = pixels[:, 0].unsqueeze(0).to(device=self.device, dtype=self.dtype)
        request = DiffusionRequest(
            prompt=(prompt,),
            height=image_height,
            width=image_width,
            num_frames=1,
            sampling=SamplingConfig(guidance_scale=1.0),
            metadata={"sample_ids": (sample_id,)},
        )
        self.codec_module.eval()
        self.conditioner_module.eval()
        with torch.no_grad():
            latents = self.codec.encode(images)
            encoded = self.conditioner.encode(request, device=self.device, dtype=self.dtype)
        if not isinstance(latents, torch.Tensor) or latents.ndim != 4 or latents.shape[0] != 1:
            raise TypeError("SANA codec must return one BCHW tensor")
        if not isinstance(encoded, Conditioning):
            raise TypeError("SANA conditioner must return Conditioning")
        context = encoded.positive.get("context")
        context_mask = encoded.positive.get("context_mask")
        if not isinstance(context, torch.Tensor) or not isinstance(context_mask, torch.Tensor):
            raise TypeError("SANA conditioner must return context and context_mask tensors")
        if context.shape[0] != 1 or context_mask.shape[0] != 1:
            raise ValueError("SANA conditioner batch dimension must be one")
        return latents[0].detach(), context[0].detach(), context_mask[0].detach()

    def encode_unconditional(
        self,
        *,
        image_height: int,
        image_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the official empty-prompt CFG branch exactly once."""

        request = DiffusionRequest(
            prompt=("",),
            negative_prompt=("",),
            height=image_height,
            width=image_width,
            num_frames=1,
            sampling=SamplingConfig(guidance_scale=1.0),
        )
        self.conditioner_module.eval()
        with torch.no_grad():
            encoded = self.conditioner.encode(request, device=self.device, dtype=self.dtype)
        if not isinstance(encoded, Conditioning):
            raise TypeError("SANA conditioner must return Conditioning")
        context = encoded.negative.get("context")
        context_mask = encoded.negative.get("context_mask")
        if not isinstance(context, torch.Tensor) or not isinstance(context_mask, torch.Tensor):
            raise TypeError("SANA empty-prompt branch must return context and context_mask")
        if context.shape[0] != 1 or context_mask.shape[0] != 1:
            raise ValueError("SANA empty-prompt conditioning must contain one sample")
        return context[0].detach(), context_mask[0].detach()


@dataclass(frozen=True, slots=True)
class SanaCachePreparationResult:
    index: SanaCacheIndex
    entries: tuple[SanaCacheEntry, ...]
    safety_audits: tuple[PromptSafetyAudit, ...]
    unconditional_conditioning: SharedConditioningArtifact


def _audit_prompts(
    manifest: TrainingManifestDataset,
    prompt_filter: ShieldGemmaPromptFilter,
    *,
    batch_size: int,
) -> tuple[PromptSafetyAudit, ...]:
    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("safety_batch_size must be a positive integer")
    audits: list[PromptSafetyAudit] = []
    for offset in range(0, len(manifest), int(batch_size)):
        prompts = tuple(sample.prompt for sample in manifest[offset : offset + int(batch_size)])
        audits.extend(prompt_filter.require_safe(prompts))
    return tuple(audits)


def prepare_sana_training_cache(
    *,
    manifest: TrainingManifestDataset,
    store: SanaCacheStore,
    feature_encoder: SanaFeatureEncoder,
    prompt_filter: ShieldGemmaPromptFilter,
    model_recipe: str,
    codec: Mapping[str, object],
    conditioner: Mapping[str, object],
    tokenizer: Mapping[str, object],
    prompt_enhancement: Mapping[str, object],
    spatial_compression: int = 32,
    image_transform: ExactSanaImageTransform | None = None,
    safety_batch_size: int = 4,
) -> SanaCachePreparationResult:
    """Audit, encode, and atomically index one manifest-selected SANA cache."""

    if not isinstance(manifest, TrainingManifestDataset):
        raise TypeError("manifest must be TrainingManifestDataset")
    if not isinstance(store, SanaCacheStore):
        raise TypeError("store must be SanaCacheStore")
    if not isinstance(feature_encoder, SanaFeatureEncoder):
        raise TypeError("feature_encoder must be SanaFeatureEncoder")
    if not isinstance(prompt_filter, ShieldGemmaPromptFilter):
        raise TypeError("prompt_filter must be ShieldGemmaPromptFilter")
    if (store.root / "index.json").exists():
        raise FileExistsError("SANA cache index already exists; preparation will not overwrite it")
    transform = image_transform or ExactSanaImageTransform()
    audits = _audit_prompts(manifest, prompt_filter, batch_size=safety_batch_size)
    return prepare_sana_training_cache_from_audits(
        manifest=manifest,
        store=store,
        feature_encoder=feature_encoder,
        safety_audits=audits,
        model_recipe=model_recipe,
        codec=codec,
        conditioner=conditioner,
        tokenizer=tokenizer,
        prompt_enhancement=prompt_enhancement,
        spatial_compression=spatial_compression,
        image_transform=transform,
    )


def prepare_sana_training_cache_from_audits(
    *,
    manifest: TrainingManifestDataset,
    store: SanaCacheStore,
    feature_encoder: SanaFeatureEncoder,
    safety_audits: Sequence[PromptSafetyAudit],
    model_recipe: str,
    codec: Mapping[str, object],
    conditioner: Mapping[str, object],
    tokenizer: Mapping[str, object],
    prompt_enhancement: Mapping[str, object],
    spatial_compression: int = 32,
    image_transform: ExactSanaImageTransform | None = None,
) -> SanaCachePreparationResult:
    """Encode a manifest after a caller-owned ShieldGemma lifecycle."""

    if not isinstance(manifest, TrainingManifestDataset):
        raise TypeError("manifest must be TrainingManifestDataset")
    if not isinstance(store, SanaCacheStore):
        raise TypeError("store must be SanaCacheStore")
    if not isinstance(feature_encoder, SanaFeatureEncoder):
        raise TypeError("feature_encoder must be SanaFeatureEncoder")
    if (store.root / "index.json").exists():
        raise FileExistsError("SANA cache index already exists; preparation will not overwrite it")
    audits = tuple(safety_audits)
    if len(audits) != len(manifest) or not all(isinstance(audit, PromptSafetyAudit) for audit in audits):
        raise ValueError("one PromptSafetyAudit is required per manifest sample")
    if any(not audit.safe for audit in audits):
        raise ValueError("unsafe prompt audits cannot be used to prepare a SANA cache")
    transform = image_transform or ExactSanaImageTransform()
    if not len(manifest):
        raise ValueError("SANA cache preparation requires at least one sample")
    first_sample = manifest[0]
    unconditional_context, unconditional_mask = feature_encoder.encode_unconditional(
        image_height=first_sample.height,
        image_width=first_sample.width,
    )
    entries: list[SanaCacheEntry] = []
    for sample, audit in zip(manifest, audits):
        if audit.prompt != sample.prompt:
            raise ValueError("ShieldGemma audit prompt differs from the manifest prompt")
        if sample.safety.get("prompt_safe") is not True:
            raise ValueError(f"manifest prompt safety decision differs for sample {sample.sample_id!r}")
        if sample.safety.get("model_revision") != audit.model_revision:
            raise ValueError(f"manifest safety model revision differs for sample {sample.sample_id!r}")
        pixels = transform.decode(sample, manifest_path=manifest.manifest_path)
        latents, context, context_mask = feature_encoder.encode(
            sample_id=sample.sample_id,
            prompt=sample.prompt,
            pixels=pixels,
            image_height=sample.height,
            image_width=sample.width,
        )
        provenance = SanaCacheProvenance(
            media_uri=sample.media.uri,
            prompt=sample.prompt,
            model_recipe=model_recipe,
            codec=codec,
            conditioner=conditioner,
            tokenizer=tokenizer,
            safety_audit=audit.to_dict(),
            pixel_transform=transform.to_dict(),
            prompt_enhancement=prompt_enhancement,
            image_height=sample.height,
            image_width=sample.width,
            spatial_compression=spatial_compression,
            latent_scaling_factor=feature_encoder.scaling_factor,
            max_text_length=int(context.shape[1]),
        )
        entries.append(
            store.write_sample(
                sample_id=sample.sample_id,
                provenance=provenance,
                clean_latents=latents,
                context=context,
                context_mask=context_mask,
            )
        )
    reference = entries[0].provenance
    unconditional = SharedConditioningStore(store.root).write(
        branch="unconditional",
        prompt="",
        model_recipe=reference.model_recipe,
        conditioner=conditioner,
        tokenizer=tokenizer,
        tensors={
            "context": unconditional_context,
            "context_mask": unconditional_mask,
        },
        layouts={
            "context": "one-sequence-features",
            "context_mask": "sequence",
        },
    )
    index = store.write_index(entries=entries)
    return SanaCachePreparationResult(
        index=index,
        entries=tuple(entries),
        safety_audits=audits,
        unconditional_conditioning=unconditional,
    )


def _checkpoint_identity(spec: object) -> dict[str, object]:
    repository = getattr(spec, "repo_id", None) or "local-explicit"
    revision = getattr(spec, "revision", None) or "local-explicit"
    sources = tuple(str(source) for source in getattr(spec, "sources", ()))
    files = tuple(str(name) for name in getattr(spec, "files", ())) or tuple(
        str(name) for name in getattr(spec, "allow_patterns", ())
    )
    return checkpoint_asset_identity(
        repo_id=str(repository),
        revision=str(revision),
        files=files,
        file_size_bytes=dict(getattr(spec, "file_size_bytes", {})),
        sources=sources,
    )


def materialize_sana_training_cache(
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
) -> SanaCachePreparationResult:
    """Audit prompts, release ShieldGemma, then load only Gemma and DCAE."""

    recipe = _validate_sana_cache_recipe(recipe)

    from worldfoundry.base_models.diffusion_model.assembly import NativeDiffusionAssembler
    from worldfoundry.base_models.diffusion_model.components import (
        BuildPurpose,
        ComponentKey,
        ComponentKind,
    )
    from worldfoundry.base_models.diffusion_model.models.encoders.sana.component import (
        SANA_PROMPT_PREFIX,
    )
    from worldfoundry.base_models.diffusion_model.optimizations import AttentionBackend, RuntimePolicy
    from worldfoundry.base_models.diffusion_model.recipes.registry import (
        default_native_diffusion_registry,
    )
    from worldfoundry.training.safety.shieldgemma import build_shieldgemma_prompt_filter

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    manifest = TrainingManifestDataset.from_file(
        manifest_path,
        split=recipe.data.split,
        verify_files=verify_media_files,
    )
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

    native_recipe = default_native_diffusion_registry().resolve(recipe.model.recipe)
    conditioner_key = ComponentKey(ComponentKind.CONDITIONER)
    codec_key = ComponentKey(ComponentKind.LATENT_ENCODER, "codec")
    assembler = NativeDiffusionAssembler()
    overrides = dict(checkpoint_overrides or {})
    resolved_checkpoints = assembler.resolve_checkpoints(native_recipe, overrides)
    components = assembler.build_components(
        native_recipe,
        purpose=BuildPurpose.TRAINING,
        policy=RuntimePolicy(
            device=resolved_device,
            dtype={
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }[recipe.runtime.param_dtype],
            attention=AttentionBackend.TORCH,
        ),
        checkpoint_overrides=overrides,
        component_keys=(conditioner_key, codec_key),
    )
    feature_encoder = SanaFeatureEncoder(
        components[codec_key],
        components[conditioner_key],
    )
    conditioner = components[conditioner_key]
    enhancement = prompt_enhancement_config(
        enabled=bool(getattr(conditioner, "enhance_prompt", True)),
        max_text_length=feature_encoder.max_text_length,
        prefix=SANA_PROMPT_PREFIX,
    )
    return prepare_sana_training_cache_from_audits(
        manifest=manifest,
        store=SanaCacheStore(cache_dir),
        feature_encoder=feature_encoder,
        safety_audits=audits,
        model_recipe=recipe.model.recipe,
        codec=_checkpoint_identity(resolved_checkpoints["codec"]),
        conditioner=_checkpoint_identity(resolved_checkpoints["text-encoder"]),
        tokenizer=_checkpoint_identity(resolved_checkpoints["tokenizer"]),
        prompt_enhancement=enhancement,
        spatial_compression=int(native_recipe.options.get("spatial_compression", 32)),
    )


__all__ = [
    "ExactSanaImageTransform",
    "SANA_PIXEL_TRANSFORM_SCHEMA",
    "SanaCachePreparationResult",
    "SanaFeatureEncoder",
    "checkpoint_asset_identity",
    "materialize_sana_training_cache",
    "prepare_sana_training_cache",
    "prepare_sana_training_cache_from_audits",
    "prompt_enhancement_config",
]

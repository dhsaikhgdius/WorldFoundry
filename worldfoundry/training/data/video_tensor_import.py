"""Import official precomputed latent/context tensors into the native cache."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import quote

import torch

from worldfoundry.training.safety.shieldgemma import PromptSafetyAudit

from .dataset import TrainingManifestDataset
from .video_cache import VideoCacheProvenance, VideoCacheStore
from .video_precompute import (
    VideoCachePreparationResult,
    audit_video_prompts,
    validate_video_prompt_audits,
)
from .video_tensor_contracts import precomputed_video_tensor_contract


def materialize_precomputed_video_training_cache(
    recipe: object,
    *,
    manifest_path: str | Path,
    cache_dir: str | Path,
    source_dir: str | Path,
    device: str | torch.device = "cuda",
    safety_audits: Sequence[PromptSafetyAudit] | None = None,
    shieldgemma_checkpoint: object | None = None,
    verify_media_files: bool = True,
    safety_batch_size: int = 4,
    checkpoint_overrides: Mapping[str, object] | None = None,
) -> VideoCachePreparationResult:
    """Import one ``<sample-id>.safetensors`` file per manifest sample.

    Files contain ``clean_latents`` plus the model-family conditioning names;
    masks and a scalar sample weight are optional. This makes tensors emitted
    by the official preprocessing code usable without importing its trainer.
    """

    del checkpoint_overrides
    model_recipe = str(getattr(getattr(recipe, "model"), "recipe"))
    contract = precomputed_video_tensor_contract(model_recipe)
    geometry = contract.geometry
    conditioning_layout = contract.conditioning_layout
    normalization = contract.latent_normalization
    layouts = contract.tensor_layouts
    algorithm = getattr(recipe, "algorithm", None)
    manifest = TrainingManifestDataset.from_file(
        manifest_path,
        split=str(getattr(getattr(recipe, "data"), "split")),
        verify_files=verify_media_files,
    )
    resolved_device = torch.device(device)
    if safety_audits is None:
        from worldfoundry.training.safety.shieldgemma import build_shieldgemma_prompt_filter

        prompt_filter = build_shieldgemma_prompt_filter(
            shieldgemma_checkpoint,
            device=resolved_device,
            dtype=torch.bfloat16,
        )
        audits = audit_video_prompts(manifest, prompt_filter, batch_size=safety_batch_size)
    else:
        audits = tuple(safety_audits)
    audits = validate_video_prompt_audits(manifest, audits)

    source_root = Path(source_dir).expanduser().resolve()
    store = VideoCacheStore(cache_dir)
    if (store.root / "index.json").exists():
        raise FileExistsError("video cache index already exists; import will not overwrite it")
    from safetensors.torch import load_file

    entries = []
    for sample, audit in zip(manifest, audits, strict=True):
        tensor_path = source_root / f"{quote(sample.sample_id, safe='')}.safetensors"
        tensors = load_file(tensor_path, device="cpu")
        clean = tensors.pop("clean_latents")
        if clean.ndim != 4:
            raise ValueError(f"{tensor_path} clean_latents must be [C,T,H,W]")
        conditioning = {
            name.removeprefix("condition."): tensor
            for name, tensor in tensors.items()
            if name.removeprefix("condition.") in layouts
        }
        conditioning_layouts = {name: layouts[name] for name in conditioning}
        target_frames = int(clean.shape[1]) * geometry.temporal_compression
        target_height = int(clean.shape[2]) * geometry.spatial_compression_height
        target_width = int(clean.shape[3]) * geometry.spatial_compression_width
        provenance = VideoCacheProvenance(
            media_uri=sample.media.uri,
            prompt=sample.prompt,
            model_recipe=model_recipe,
            codec={"source_format": "official-precomputed-tensors"},
            conditioner={"source_format": "official-precomputed-tensors"},
            tokenizer={"source_format": "official-precomputed-tensors"},
            conditioning_inputs={"source_file": tensor_path.name},
            safety_audit=audit.to_dict(),
            frame_sampling={"mode": "official-precomputed", "frames": target_frames},
            spatial_transform={
                "mode": "official-precomputed",
                "height": target_height,
                "width": target_width,
            },
            latent_normalization=normalization,
            task=sample.task,
            conditioning_layout=conditioning_layout,
            aspect_bin=f"{target_width}:{target_height}",
            source_num_frames=sample.num_frames,
            source_height=sample.height,
            source_width=sample.width,
            source_fps=sample.fps,
            target_num_frames=target_frames,
            target_height=target_height,
            target_width=target_width,
            target_fps=float(getattr(algorithm, "default_fps", sample.fps)),
            latent_geometry=geometry,
        )
        latent_mask = tensors.get("latent_loss_mask")
        valid_mask = tensors.get("valid_latent_mask")
        if latent_mask is None:
            latent_mask = torch.ones((1, *clean.shape[-3:]), dtype=torch.float32)
        if valid_mask is None:
            valid_mask = torch.ones((1, *clean.shape[-3:]), dtype=torch.bool)
        entries.append(
            store.write_sample(
                sample_id=sample.sample_id,
                provenance=provenance,
                clean_latents=clean,
                conditioning=conditioning,
                conditioning_layouts=conditioning_layouts,
                latent_loss_mask=latent_mask,
                valid_latent_mask=valid_mask,
                sample_weight=tensors.get("sample_weight"),
            )
        )
    index = store.write_index(entries=entries)
    return VideoCachePreparationResult(index=index, entries=tuple(entries), safety_audits=audits)


__all__ = ["materialize_precomputed_video_training_cache"]

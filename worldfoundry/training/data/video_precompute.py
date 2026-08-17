"""Shared primitives for staged native video-cache materialization."""

from __future__ import annotations

import gc
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch

from worldfoundry.training.safety.shieldgemma import PromptSafetyAudit, ShieldGemmaPromptFilter

from .checkpoint_assets import checkpoint_asset_identity
from .dataset import TrainingManifestDataset
from .video_bucketing import (
    VideoBucketSelectionPolicy,
    VideoLatentGeometry,
    VideoResolutionBucket,
    assign_video_buckets,
)
from .video_cache import VideoCacheEntry, VideoCacheIndex, VideoCacheProvenance, VideoCacheStore
from .video_dataset import DecodedVideoSample, VideoDecodeConfig, VideoDecodingDataset


@dataclass(frozen=True, slots=True)
class VideoCachePreparationResult:
    index: VideoCacheIndex
    entries: tuple[VideoCacheEntry, ...]
    safety_audits: tuple[PromptSafetyAudit, ...]


class StagedVideoConditioning:
    """Temporary CPU tensor files bridging conditioner and VAE residency phases."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def write(self, index: int, tensors: Mapping[str, torch.Tensor]) -> Path:
        from safetensors.torch import save_file

        path = self.root / f"{index:08d}.safetensors"
        save_file(
            {str(name): tensor.detach().cpu().contiguous() for name, tensor in tensors.items()},
            path,
        )
        return path

    def read(self, index: int) -> dict[str, torch.Tensor]:
        from safetensors.torch import load_file

        return load_file(self.root / f"{index:08d}.safetensors", device="cpu")


@contextmanager
def staged_video_conditioning(
    cache_root: str | Path,
    *,
    family: str,
) -> Iterator[StagedVideoConditioning]:
    root = Path(tempfile.mkdtemp(prefix=f".{family}-conditioning-", dir=Path(cache_root)))
    try:
        yield StagedVideoConditioning(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def release_accelerator_memory(device: torch.device) -> None:
    """Collect released phase components before constructing the next phase."""

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def audit_video_prompts(
    manifest: TrainingManifestDataset,
    prompt_filter: ShieldGemmaPromptFilter,
    *,
    batch_size: int,
) -> tuple[PromptSafetyAudit, ...]:
    size = int(batch_size)
    if size <= 0:
        raise ValueError("safety_batch_size must be positive")
    audits: list[PromptSafetyAudit] = []
    for offset in range(0, len(manifest), size):
        audits.extend(prompt_filter.require_safe(tuple(sample.prompt for sample in manifest[offset : offset + size])))
    return tuple(audits)


def validate_video_prompt_audits(
    manifest: TrainingManifestDataset,
    safety_audits: Sequence[PromptSafetyAudit],
) -> tuple[PromptSafetyAudit, ...]:
    audits = tuple(safety_audits)
    if len(audits) != len(manifest):
        raise ValueError("one PromptSafetyAudit is required per video manifest sample")
    for sample, audit in zip(manifest, audits, strict=True):
        if not isinstance(audit, PromptSafetyAudit) or not audit.safe:
            raise ValueError("video cache preparation requires safe PromptSafetyAudit values")
        if audit.prompt != sample.prompt:
            raise ValueError("prompt safety audit differs from the video manifest prompt")
        if sample.safety.get("prompt_safe") is not True:
            raise ValueError(f"manifest prompt is not marked safe for sample {sample.sample_id!r}")
        if sample.safety.get("model_revision") != audit.model_revision:
            raise ValueError(f"manifest safety model revision differs for sample {sample.sample_id!r}")
    return audits


def _strict_options(value: object, *, field_name: str, allowed: set[str]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    payload = {str(key): item for key, item in value.items()}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"{field_name} contains unknown fields: {unknown}")
    return payload


def build_video_decoding_dataset(
    recipe: object,
    manifest: TrainingManifestDataset,
    *,
    geometry: VideoLatentGeometry,
    conditioning_layout: str,
) -> VideoDecodingDataset:
    """Apply the shared video bucket/decode recipe surface for one model family."""

    options = dict(getattr(getattr(recipe, "data"), "options"))
    raw_buckets = options.get("video_buckets")
    if not isinstance(raw_buckets, Sequence) or isinstance(raw_buckets, (str, bytes, bytearray)) or not raw_buckets:
        raise ValueError("data.options.video_buckets must be a non-empty sequence")
    buckets: list[VideoResolutionBucket] = []
    for index, raw in enumerate(raw_buckets):
        payload = _strict_options(
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
        payload.setdefault("conditioning_layout", conditioning_layout)
        buckets.append(VideoResolutionBucket(**payload))

    policy = _strict_options(
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
    decode = _strict_options(
        options.get("decode", {}),
        field_name="data.options.decode",
        allowed={
            "frame_sampling",
            "frame_sampling_seed",
            "interpolation",
            "resize_rounding",
            "spatial_transform",
            "value_range",
            "decoder_thread_type",
            "verify_manifest_frame_count",
            "verify_manifest_geometry",
            "fps_tolerance",
        },
    )
    decode.setdefault("value_range", "minus-one-one")
    assignments = assign_video_buckets(
        tuple(manifest),
        buckets=buckets,
        geometry=geometry,
        conditioning_layout=conditioning_layout,
        policy=VideoBucketSelectionPolicy(**policy),
    )
    return VideoDecodingDataset(
        manifest,
        assignments,
        config=VideoDecodeConfig(**decode),
    )


def checkpoint_spec_identity(spec: object) -> dict[str, object]:
    """Convert one resolved checkpoint declaration to cache-readable fields."""

    sources = tuple(str(source) for source in getattr(spec, "sources", ()))
    files = tuple(str(name) for name in getattr(spec, "files", ()))
    if not files:
        files = tuple(str(name) for name in getattr(spec, "allow_patterns", ()))
    if not files:
        files = sources
    return checkpoint_asset_identity(
        repo_id=getattr(spec, "repo_id", None) or "local-explicit",
        revision=getattr(spec, "revision", None) or "local-explicit",
        files=files,
        file_size_bytes=dict(getattr(spec, "file_size_bytes", {})),
        sources=sources,
    )


def write_video_cache_entry(
    *,
    store: VideoCacheStore,
    dataset: VideoDecodingDataset,
    index: int,
    decoded: DecodedVideoSample,
    audit: PromptSafetyAudit,
    model_recipe: str,
    latent_geometry: VideoLatentGeometry,
    latent_normalization: Mapping[str, object],
    codec: Mapping[str, object],
    conditioner: Mapping[str, object],
    tokenizer: Mapping[str, object],
    clean_latents: torch.Tensor,
    conditioning: Mapping[str, torch.Tensor],
    conditioning_layouts: Mapping[str, str],
    latent_loss_mask: torch.Tensor,
    valid_latent_mask: torch.Tensor,
) -> VideoCacheEntry:
    source = dataset.manifest_dataset[index]
    assignment = decoded.assignment
    provenance = VideoCacheProvenance(
        media_uri=source.media.uri,
        prompt=source.prompt,
        model_recipe=model_recipe,
        codec=codec,
        conditioner=conditioner,
        tokenizer=tokenizer,
        conditioning_inputs={"task": source.task, "conditions": source.to_dict()["conditions"]},
        safety_audit=audit.to_dict(),
        frame_sampling=decoded.frame_sampling,
        spatial_transform=decoded.spatial_transform,
        latent_normalization=latent_normalization,
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
        clean_latents=clean_latents,
        conditioning=conditioning,
        conditioning_layouts=conditioning_layouts,
        latent_loss_mask=latent_loss_mask,
        valid_latent_mask=valid_latent_mask,
    )


__all__ = [
    "StagedVideoConditioning",
    "VideoCachePreparationResult",
    "audit_video_prompts",
    "build_video_decoding_dataset",
    "checkpoint_spec_identity",
    "release_accelerator_memory",
    "staged_video_conditioning",
    "validate_video_prompt_audits",
    "write_video_cache_entry",
]

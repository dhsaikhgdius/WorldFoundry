"""Content-addressed video latent, conditioning, and mask cache.

The immutable object identity binds codec geometry, source/frame/spatial
transforms, latent normalization, conditioning inputs, encoder assets, and all
tensor contracts.  A run-local index maps sample ids to these objects and their
video bucket keys; the cache directory itself is never a training checkpoint.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, overload

import torch

from worldfoundry.core.io.file_utils import file_sha256
from worldfoundry.core.io.integrity import canonical_json as _core_canonical_json
from worldfoundry.core.io.integrity import canonical_sha256, replace_json_atomic, sync_directory
from worldfoundry.training.api.contracts import TrainingBatch

from .sana_cache import CacheTensorDescriptor
from .video_bucketing import VideoBucketKey, VideoLatentGeometry

VIDEO_CACHE_OBJECT_SCHEMA = "worldfoundry-video-training-cache-object"
VIDEO_CACHE_INDEX_SCHEMA = "worldfoundry-video-training-cache-index"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CONDITION_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_CONDITION_PREFIX = "condition."
_RESERVED_KEYS = frozenset(
    {
        "clean_latents",
        "latent_loss_mask",
        "valid_latent_mask",
        "sample_weight",
    }
)
_REQUIRED_KEYS = frozenset({"clean_latents"})
_RESERVED_LAYOUTS = {
    "clean_latents": "channels-frames-height-width",
    "latent_loss_mask": "channels-frames-height-width",
    "valid_latent_mask": "one-frames-height-width",
    "sample_weight": "scalar",
}


def _canonical_json(value: object) -> str:
    try:
        return _core_canonical_json(value)
    except (TypeError, ValueError) as error:
        raise TypeError("video cache metadata must be JSON serializable without NaN or infinity") from error


def _sha256(value: object, *, field_name: str) -> str:
    resolved = str(value).strip().lower()
    if _SHA256_PATTERN.fullmatch(resolved) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return resolved


def _nonempty(value: object, *, field_name: str) -> str:
    resolved = str(value).strip()
    if not resolved:
        raise ValueError(f"{field_name} cannot be empty")
    return resolved


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field_name} must be positive")
    return resolved


def _positive_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric, not bool")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return resolved


def _strict_mapping(
    value: object,
    *,
    field_name: str,
    allowed: set[str] | frozenset[str],
    required: set[str] | frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    payload = {str(key): item for key, item in value.items()}
    unknown = sorted(set(payload) - set(allowed))
    missing = sorted(set(required) - set(payload))
    if unknown or missing:
        raise ValueError(f"{field_name} fields mismatch; missing={missing}, unknown={unknown}")
    return payload


def _condition_name(value: object) -> str:
    resolved = _nonempty(value, field_name="conditioning tensor name")
    if _CONDITION_NAME_PATTERN.fullmatch(resolved) is None:
        raise ValueError(f"conditioning tensor name contains unsupported characters: {resolved!r}")
    if resolved in _RESERVED_KEYS or resolved.startswith(_CONDITION_PREFIX):
        raise ValueError(f"conditioning tensor name is reserved: {resolved!r}")
    return resolved


@dataclass(frozen=True, slots=True)
class VideoCacheProvenance:
    """Every non-tensor input that can change cached video features."""

    media_sha256: str
    prompt_sha256: str
    model_recipe_digest: str
    codec_digest: str
    conditioner_digest: str
    tokenizer_digest: str
    conditioning_inputs_digest: str
    safety_audit_digest: str
    frame_sampling_digest: str
    spatial_transform_digest: str
    latent_normalization_digest: str
    task: str
    conditioning_layout: str
    aspect_bin: str
    source_num_frames: int
    source_height: int
    source_width: int
    source_fps: float
    target_num_frames: int
    target_height: int
    target_width: int
    target_fps: float
    latent_geometry: VideoLatentGeometry

    def __post_init__(self) -> None:
        for name in (
            "media_sha256",
            "prompt_sha256",
            "model_recipe_digest",
            "codec_digest",
            "conditioner_digest",
            "tokenizer_digest",
            "conditioning_inputs_digest",
            "safety_audit_digest",
            "frame_sampling_digest",
            "spatial_transform_digest",
            "latent_normalization_digest",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), field_name=name))
        object.__setattr__(self, "task", _nonempty(self.task, field_name="task").lower().replace("-", "_"))
        object.__setattr__(
            self,
            "conditioning_layout",
            _nonempty(self.conditioning_layout, field_name="conditioning_layout").lower().replace("_", "-"),
        )
        object.__setattr__(self, "aspect_bin", _nonempty(self.aspect_bin, field_name="aspect_bin"))
        for name in (
            "source_num_frames",
            "source_height",
            "source_width",
            "target_num_frames",
            "target_height",
            "target_width",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), field_name=name))
        object.__setattr__(self, "source_fps", _positive_float(self.source_fps, field_name="source_fps"))
        object.__setattr__(self, "target_fps", _positive_float(self.target_fps, field_name="target_fps"))
        if not isinstance(self.latent_geometry, VideoLatentGeometry):
            raise TypeError("latent_geometry must be a VideoLatentGeometry")
        # Resolve the shape here so invalid pixel/codec alignment fails before
        # any expensive feature tensor is written.
        self.latent_geometry.latent_shape(
            num_frames=self.target_num_frames,
            height=self.target_height,
            width=self.target_width,
        )

    @property
    def bucket_key(self) -> VideoBucketKey:
        shape = self.latent_geometry.latent_shape(
            num_frames=self.target_num_frames,
            height=self.target_height,
            width=self.target_width,
        )
        return VideoBucketKey(
            task=self.task,
            latent_frames=shape.frames,
            latent_height=shape.height,
            latent_width=shape.width,
            aspect_bin=self.aspect_bin,
            conditioning_layout=self.conditioning_layout,
        )

    @property
    def batch_contract_digest(self) -> str:
        """Digest fields that must be identical inside one collated batch."""

        return canonical_sha256(
            {
                "model_recipe_digest": self.model_recipe_digest,
                "codec_digest": self.codec_digest,
                "conditioner_digest": self.conditioner_digest,
                "tokenizer_digest": self.tokenizer_digest,
                "latent_normalization_digest": self.latent_normalization_digest,
                "task": self.task,
                "conditioning_layout": self.conditioning_layout,
                "aspect_bin": self.aspect_bin,
                "target_num_frames": self.target_num_frames,
                "target_height": self.target_height,
                "target_width": self.target_width,
                "target_fps": self.target_fps,
                "latent_geometry": self.latent_geometry.to_dict(),
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "media_sha256": self.media_sha256,
            "prompt_sha256": self.prompt_sha256,
            "model_recipe_digest": self.model_recipe_digest,
            "codec_digest": self.codec_digest,
            "conditioner_digest": self.conditioner_digest,
            "tokenizer_digest": self.tokenizer_digest,
            "conditioning_inputs_digest": self.conditioning_inputs_digest,
            "safety_audit_digest": self.safety_audit_digest,
            "frame_sampling_digest": self.frame_sampling_digest,
            "spatial_transform_digest": self.spatial_transform_digest,
            "latent_normalization_digest": self.latent_normalization_digest,
            "task": self.task,
            "conditioning_layout": self.conditioning_layout,
            "aspect_bin": self.aspect_bin,
            "source_num_frames": self.source_num_frames,
            "source_height": self.source_height,
            "source_width": self.source_width,
            "source_fps": self.source_fps,
            "target_num_frames": self.target_num_frames,
            "target_height": self.target_height,
            "target_width": self.target_width,
            "target_fps": self.target_fps,
            "latent_geometry": self.latent_geometry.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: object) -> VideoCacheProvenance:
        fields = set(cls.__dataclass_fields__)
        payload = _strict_mapping(value, field_name="video cache provenance", allowed=fields, required=fields)
        payload["latent_geometry"] = VideoLatentGeometry.from_mapping(payload["latent_geometry"])
        return cls(**payload)


def _descriptor(tensor: torch.Tensor, *, layout: str) -> CacheTensorDescriptor:
    return CacheTensorDescriptor(
        dtype=str(tensor.dtype).removeprefix("torch."),
        shape=tuple(int(size) for size in tensor.shape),
        layout=_nonempty(layout, field_name="tensor layout"),
    )


def _validate_materialized_tensor(name: str, tensor: object) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"cached {name} must be a torch.Tensor")
    if tensor.device.type == "meta" or tensor.layout is not torch.strided:
        raise ValueError(f"cached {name} must be a dense materialized tensor")
    if tensor.is_complex() or tensor.is_quantized:
        raise ValueError(f"cached {name} cannot use complex or quantized storage")
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"cached {name} contains NaN or infinity")
    return tensor


def _validate_tensors(
    tensors: Mapping[str, torch.Tensor],
    layouts: Mapping[str, str],
    provenance: VideoCacheProvenance,
) -> Mapping[str, CacheTensorDescriptor]:
    keys = set(tensors)
    missing = sorted(_REQUIRED_KEYS - keys)
    unsupported = sorted(key for key in keys if key not in _RESERVED_KEYS and not key.startswith(_CONDITION_PREFIX))
    if missing or unsupported:
        raise ValueError(f"video cache tensor keys mismatch; missing={missing}, unsupported={unsupported}")
    if set(layouts) != keys:
        raise ValueError("one explicit layout is required for every cached video tensor")

    normalized = {name: _validate_materialized_tensor(name, tensor) for name, tensor in tensors.items()}
    latents = normalized["clean_latents"]
    shape = provenance.bucket_key.latent_shape
    if latents.ndim != 4 or tuple(latents.shape[-3:]) != (shape.frames, shape.height, shape.width):
        raise ValueError(
            "clean_latents must be unbatched [C,T,H,W] matching provenance; "
            f"got {tuple(latents.shape)}, expected spatial-temporal {(shape.frames, shape.height, shape.width)}"
        )
    if not latents.is_floating_point():
        raise TypeError("clean_latents must use a floating dtype")

    loss_mask = normalized.get("latent_loss_mask")
    if loss_mask is not None:
        if loss_mask.ndim != 4 or tuple(loss_mask.shape[-3:]) != tuple(latents.shape[-3:]):
            raise ValueError("latent_loss_mask must be [C,T,H,W] and match clean_latents")
        if int(loss_mask.shape[0]) not in {1, int(latents.shape[0])}:
            raise ValueError("latent_loss_mask channels must be one or match clean_latents")
        if not loss_mask.is_floating_point() or not bool((loss_mask >= 0).all()):
            raise ValueError("latent_loss_mask must contain non-negative floating weights")

    valid_mask = normalized.get("valid_latent_mask")
    if valid_mask is not None:
        if valid_mask.ndim != 4 or tuple(valid_mask.shape) != (1, *tuple(latents.shape[-3:])):
            raise ValueError("valid_latent_mask must be [1,T,H,W] matching clean_latents")
        if valid_mask.dtype not in {torch.bool, torch.int8, torch.int16, torch.int32, torch.int64}:
            raise TypeError("valid_latent_mask must use a bool or integer dtype")
        if not bool(((valid_mask == 0) | (valid_mask == 1)).all()):
            raise ValueError("valid_latent_mask must contain only zero and one")

    sample_weight = normalized.get("sample_weight")
    if sample_weight is not None:
        if sample_weight.ndim != 0 or not sample_weight.is_floating_point():
            raise ValueError("sample_weight must be one floating scalar")
        if not bool(torch.isfinite(sample_weight)) or not bool(sample_weight >= 0):
            raise ValueError("sample_weight must be finite and non-negative")

    for name, tensor in normalized.items():
        if not name.startswith(_CONDITION_PREFIX):
            continue
        condition_name = name.removeprefix(_CONDITION_PREFIX)
        _condition_name(condition_name)
        if tensor.ndim == 0:
            raise ValueError(f"conditioning tensor {condition_name!r} cannot be scalar")

    return MappingProxyType({name: _descriptor(tensor, layout=layouts[name]) for name, tensor in normalized.items()})


def _identity_payload(
    provenance: VideoCacheProvenance,
    descriptors: Mapping[str, CacheTensorDescriptor],
) -> dict[str, object]:
    return {
        "schema": VIDEO_CACHE_OBJECT_SCHEMA,
        "provenance": provenance.to_dict(),
        "tensors": {name: descriptors[name].to_dict() for name in sorted(descriptors)},
    }


def _object_metadata(identity: Mapping[str, object], identity_sha256: str) -> dict[str, str]:
    return {
        "worldfoundry": _canonical_json(
            {
                "schema": VIDEO_CACHE_OBJECT_SCHEMA,
                "identity_sha256": identity_sha256,
                "identity": identity,
            }
        )
    }


@dataclass(frozen=True, slots=True)
class VideoCacheEntry:
    sample_id: str
    identity_sha256: str
    object_sha256: str
    object_size_bytes: int
    object_path: str
    provenance: VideoCacheProvenance
    tensors: Mapping[str, CacheTensorDescriptor]

    def __post_init__(self) -> None:
        sample_id = _nonempty(self.sample_id, field_name="sample_id")
        identity = _sha256(self.identity_sha256, field_name="identity_sha256")
        object_digest = _sha256(self.object_sha256, field_name="object_sha256")
        size = _positive_int(self.object_size_bytes, field_name="object_size_bytes")
        expected_path = f"objects/{object_digest[:2]}/{object_digest}.safetensors"
        object_path = str(self.object_path).replace("\\", "/")
        if object_path != expected_path:
            raise ValueError(f"object_path must be content-addressed as {expected_path!r}")
        if not isinstance(self.provenance, VideoCacheProvenance):
            raise TypeError("provenance must be VideoCacheProvenance")
        descriptors = dict(self.tensors)
        if not _REQUIRED_KEYS <= set(descriptors):
            raise ValueError("video cache entry is missing clean_latents")
        if not all(isinstance(value, CacheTensorDescriptor) for value in descriptors.values()):
            raise TypeError("video cache tensor descriptors are invalid")
        expected_identity = canonical_sha256(_identity_payload(self.provenance, descriptors))
        if identity != expected_identity:
            raise ValueError("video cache logical identity digest does not match its metadata")
        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "identity_sha256", identity)
        object.__setattr__(self, "object_sha256", object_digest)
        object.__setattr__(self, "object_size_bytes", size)
        object.__setattr__(self, "object_path", object_path)
        object.__setattr__(self, "tensors", MappingProxyType(descriptors))

    @property
    def bucket_key(self) -> VideoBucketKey:
        return self.provenance.bucket_key

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "identity_sha256": self.identity_sha256,
            "object_sha256": self.object_sha256,
            "object_size_bytes": self.object_size_bytes,
            "object_path": self.object_path,
            "provenance": self.provenance.to_dict(),
            "tensors": {name: self.tensors[name].to_dict() for name in sorted(self.tensors)},
        }

    @classmethod
    def from_mapping(cls, value: object) -> VideoCacheEntry:
        fields = set(cls.__dataclass_fields__)
        payload = _strict_mapping(value, field_name="video cache entry", allowed=fields, required=fields)
        raw_provenance = payload.pop("provenance")
        raw_tensors = payload.pop("tensors")
        if not isinstance(raw_tensors, Mapping):
            raise TypeError("video cache entry tensors must be a mapping")
        return cls(
            **payload,
            provenance=VideoCacheProvenance.from_mapping(raw_provenance),
            tensors={
                str(name): CacheTensorDescriptor.from_mapping(descriptor) for name, descriptor in raw_tensors.items()
            },
        )


@dataclass(frozen=True, slots=True)
class VideoCacheIndex:
    dataset_digest: str
    entries: tuple[VideoCacheEntry, ...]
    index_sha256: str
    schema: str = VIDEO_CACHE_INDEX_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != VIDEO_CACHE_INDEX_SCHEMA:
            raise ValueError(f"unsupported video cache index schema: {self.schema!r}")
        dataset_digest = _sha256(self.dataset_digest, field_name="dataset_digest")
        entries = tuple(self.entries)
        if not entries or not all(isinstance(entry, VideoCacheEntry) for entry in entries):
            raise ValueError("video cache index entries must be non-empty VideoCacheEntry values")
        sample_ids = tuple(entry.sample_id for entry in entries)
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("video cache index sample_ids must be unique")
        identity_objects: dict[str, str] = {}
        for entry in entries:
            previous = identity_objects.setdefault(entry.identity_sha256, entry.object_sha256)
            if previous != entry.object_sha256:
                raise ValueError("one video cache logical identity cannot reference different tensor objects")
        payload = {
            "schema": self.schema,
            "dataset_digest": dataset_digest,
            "entries": [entry.to_dict() for entry in entries],
        }
        index_digest = _sha256(self.index_sha256, field_name="index_sha256")
        if index_digest != canonical_sha256(payload):
            raise ValueError("video cache index digest does not match its contents")
        object.__setattr__(self, "dataset_digest", dataset_digest)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "index_sha256", index_digest)

    @classmethod
    def build(cls, dataset_digest: str, entries: Sequence[VideoCacheEntry]) -> VideoCacheIndex:
        resolved_digest = _sha256(dataset_digest, field_name="dataset_digest")
        resolved_entries = tuple(entries)
        payload = {
            "schema": VIDEO_CACHE_INDEX_SCHEMA,
            "dataset_digest": resolved_digest,
            "entries": [entry.to_dict() for entry in resolved_entries],
        }
        return cls(
            dataset_digest=resolved_digest,
            entries=resolved_entries,
            index_sha256=canonical_sha256(payload),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "dataset_digest": self.dataset_digest,
            "entries": [entry.to_dict() for entry in self.entries],
            "index_sha256": self.index_sha256,
        }

    @classmethod
    def from_mapping(cls, value: object) -> VideoCacheIndex:
        payload = _strict_mapping(
            value,
            field_name="video cache index",
            allowed={"schema", "dataset_digest", "entries", "index_sha256"},
            required={"schema", "dataset_digest", "entries", "index_sha256"},
        )
        raw_entries = payload.pop("entries")
        if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes, bytearray)):
            raise TypeError("video cache index entries must be a sequence")
        return cls(entries=tuple(VideoCacheEntry.from_mapping(item) for item in raw_entries), **payload)


@dataclass(frozen=True, slots=True)
class VideoCachedSample:
    entry: VideoCacheEntry
    tensors: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        if not isinstance(self.entry, VideoCacheEntry):
            raise TypeError("entry must be a VideoCacheEntry")
        layouts = {name: descriptor.layout for name, descriptor in self.entry.tensors.items()}
        descriptors = _validate_tensors(self.tensors, layouts, self.entry.provenance)
        if dict(descriptors) != dict(self.entry.tensors):
            raise ValueError("loaded video tensor descriptors do not match the cache index")
        object.__setattr__(self, "tensors", MappingProxyType(dict(self.tensors)))


class VideoCacheStore:
    """Write and verify immutable content-addressed video feature objects."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.objects = self.root / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)

    def _resolved_object_path(self, entry: VideoCacheEntry) -> Path:
        path = self.root / entry.object_path
        resolved_parent = path.parent.resolve()
        if self.root != resolved_parent and self.root not in resolved_parent.parents:
            raise ValueError("video cache object path escapes the cache root")
        if path.is_symlink():
            raise ValueError(f"video cache objects cannot be symlinks: {path}")
        return path

    def write_sample(
        self,
        *,
        sample_id: str,
        provenance: VideoCacheProvenance,
        clean_latents: torch.Tensor,
        conditioning: Mapping[str, torch.Tensor] | None = None,
        conditioning_layouts: Mapping[str, str] | None = None,
        latent_loss_mask: torch.Tensor | None = None,
        valid_latent_mask: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
    ) -> VideoCacheEntry:
        if not isinstance(provenance, VideoCacheProvenance):
            raise TypeError("provenance must be a VideoCacheProvenance")
        raw_conditioning = {} if conditioning is None else dict(conditioning)
        raw_layouts = {} if conditioning_layouts is None else dict(conditioning_layouts)
        normalized_conditioning: dict[str, torch.Tensor] = {}
        normalized_layouts: dict[str, str] = {}
        for raw_name, tensor in raw_conditioning.items():
            name = _condition_name(raw_name)
            if name in normalized_conditioning:
                raise ValueError(f"duplicate conditioning tensor name: {name!r}")
            normalized_conditioning[name] = tensor
        if set(raw_layouts) != set(normalized_conditioning):
            raise ValueError("conditioning_layouts must exactly match conditioning tensor names")
        for raw_name, layout in raw_layouts.items():
            normalized_layouts[_condition_name(raw_name)] = _nonempty(layout, field_name="conditioning layout")

        tensors: dict[str, torch.Tensor] = {"clean_latents": clean_latents}
        layouts = {"clean_latents": _RESERVED_LAYOUTS["clean_latents"]}
        for name, tensor in normalized_conditioning.items():
            storage_name = f"{_CONDITION_PREFIX}{name}"
            tensors[storage_name] = tensor
            layouts[storage_name] = normalized_layouts[name]
        for name, tensor in (
            ("latent_loss_mask", latent_loss_mask),
            ("valid_latent_mask", valid_latent_mask),
            ("sample_weight", sample_weight),
        ):
            if tensor is not None:
                tensors[name] = tensor
                layouts[name] = _RESERVED_LAYOUTS[name]

        descriptors = _validate_tensors(tensors, layouts, provenance)
        identity = _identity_payload(provenance, descriptors)
        identity_sha256 = canonical_sha256(identity)
        storage = {name: tensor.detach().to(device="cpu").contiguous() for name, tensor in tensors.items()}
        metadata = _object_metadata(identity, identity_sha256)
        try:
            from safetensors.torch import save_file
        except ModuleNotFoundError as error:
            raise RuntimeError("video training cache requires safetensors") from error

        temporary_handle = tempfile.NamedTemporaryFile(
            prefix=".video-cache-",
            suffix=".safetensors",
            dir=self.objects,
            delete=False,
        )
        temporary = Path(temporary_handle.name)
        temporary_handle.close()
        try:
            save_file(storage, temporary, metadata=metadata)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            object_digest = file_sha256(temporary)
            object_size = temporary.stat().st_size
            destination_dir = self.objects / object_digest[:2]
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / f"{object_digest}.safetensors"
            if destination.exists():
                if destination.is_symlink():
                    raise ValueError(f"video cache objects cannot be symlinks: {destination}")
                if destination.stat().st_size != object_size or file_sha256(destination) != object_digest:
                    raise ValueError(f"existing content-addressed video cache object is corrupt: {destination}")
            else:
                try:
                    os.link(temporary, destination)
                except FileExistsError:
                    if destination.stat().st_size != object_size or file_sha256(destination) != object_digest:
                        raise ValueError(f"racing video cache writer produced a corrupt object: {destination}")
                sync_directory(destination_dir)
            return VideoCacheEntry(
                sample_id=sample_id,
                identity_sha256=identity_sha256,
                object_sha256=object_digest,
                object_size_bytes=object_size,
                object_path=f"objects/{object_digest[:2]}/{object_digest}.safetensors",
                provenance=provenance,
                tensors=descriptors,
            )
        finally:
            temporary.unlink(missing_ok=True)

    def audit_entry(self, entry: VideoCacheEntry, *, load_tensors: bool = True) -> VideoCachedSample | None:
        path = self._resolved_object_path(entry)
        if not path.is_file():
            raise FileNotFoundError(f"video cache object not found: {path}")
        if path.stat().st_size != entry.object_size_bytes:
            raise ValueError(f"video cache object size mismatch for {entry.sample_id!r}")
        if file_sha256(path) != entry.object_sha256:
            raise ValueError(f"video cache object SHA-256 mismatch for {entry.sample_id!r}")
        try:
            from safetensors import safe_open
        except ModuleNotFoundError as error:
            raise RuntimeError("video training cache requires safetensors") from error
        with safe_open(path, framework="pt", device="cpu") as handle:
            expected_identity = _identity_payload(entry.provenance, entry.tensors)
            if (handle.metadata() or {}) != _object_metadata(expected_identity, entry.identity_sha256):
                raise ValueError(f"video cache object metadata mismatch for {entry.sample_id!r}")
            if set(handle.keys()) != set(entry.tensors):
                raise ValueError(f"video cache object tensor keys mismatch for {entry.sample_id!r}")
            if not load_tensors:
                for name, descriptor in entry.tensors.items():
                    if tuple(handle.get_slice(name).get_shape()) != descriptor.shape:
                        raise ValueError(f"video cache object shape mismatch for {name!r}")
                return None
            tensors = {name: handle.get_tensor(name) for name in sorted(handle.keys())}
        return VideoCachedSample(entry=entry, tensors=tensors)

    def write_index(
        self,
        *,
        dataset_digest: str,
        entries: Sequence[VideoCacheEntry],
        filename: str = "index.json",
    ) -> VideoCacheIndex:
        if Path(filename).name != filename or filename in {".", ".."}:
            raise ValueError("video cache index filename must be one plain filename")
        index = VideoCacheIndex.build(dataset_digest, entries)
        destination = self.root / filename
        replace_json_atomic(destination, index.to_dict(), root=self.root)
        return index

    def read_index(self, filename: str = "index.json") -> VideoCacheIndex:
        if Path(filename).name != filename or filename in {".", ".."}:
            raise ValueError("video cache index filename must be one plain filename")
        path = self.root / filename
        if path.is_symlink():
            raise ValueError("video cache index cannot be a symlink")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return VideoCacheIndex.from_mapping(payload)


class VideoCachedDataset(Sequence[VideoCachedSample]):
    """Map-style dataset over an audited video cache index."""

    def __init__(
        self,
        root: str | Path,
        *,
        index_filename: str = "index.json",
        expected_dataset_digest: str | None = None,
        audit_on_open: bool = True,
        verify_on_read: bool = True,
    ) -> None:
        self.store = VideoCacheStore(root)
        self.index = self.store.read_index(index_filename)
        if expected_dataset_digest is not None:
            expected = _sha256(expected_dataset_digest, field_name="expected_dataset_digest")
            if self.index.dataset_digest != expected:
                raise ValueError(
                    f"video cache dataset digest mismatch: expected {expected}, got {self.index.dataset_digest}"
                )
        self._entries = self.index.entries
        self._verify_on_read = bool(verify_on_read)
        if audit_on_open:
            for entry in self._entries:
                self.store.audit_entry(entry, load_tensors=False)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(entry.sample_id for entry in self._entries)

    @property
    def bucket_keys(self) -> tuple[VideoBucketKey, ...]:
        return tuple(entry.bucket_key for entry in self._entries)

    @property
    def batch_contract_digests(self) -> tuple[str, ...]:
        """Return the exact compatibility key used by token-batch grouping."""

        return tuple(entry.provenance.batch_contract_digest for entry in self._entries)

    @property
    def dataset_digest(self) -> str:
        return self.index.dataset_digest

    @property
    def index_sha256(self) -> str:
        return self.index.index_sha256

    def __len__(self) -> int:
        return len(self._entries)

    @overload
    def __getitem__(self, index: int) -> VideoCachedSample: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[VideoCachedSample, ...]: ...

    def __getitem__(self, index: int | slice) -> VideoCachedSample | tuple[VideoCachedSample, ...]:
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(len(self))))
        entry = self._entries[index]
        if self._verify_on_read:
            sample = self.store.audit_entry(entry, load_tensors=True)
            assert sample is not None
            return sample
        try:
            from safetensors.torch import load_file
        except ModuleNotFoundError as error:
            raise RuntimeError("video training cache requires safetensors") from error
        return VideoCachedSample(
            entry=entry,
            tensors=load_file(self.store._resolved_object_path(entry), device="cpu"),
        )

    def __iter__(self) -> Iterator[VideoCachedSample]:
        for index in range(len(self)):
            yield self[index]


def collate_video_cached_samples(samples: Sequence[VideoCachedSample]) -> TrainingBatch:
    """Stack one strict video bucket and expose token accounting metadata."""

    values = tuple(samples)
    if not values:
        raise ValueError("cannot collate an empty video cache batch")
    if not all(isinstance(sample, VideoCachedSample) for sample in values):
        raise TypeError("all cached samples must be VideoCachedSample")
    reference = values[0]
    descriptors = dict(reference.entry.tensors)
    bucket_key = reference.entry.bucket_key
    contract_digest = reference.entry.provenance.batch_contract_digest
    for sample in values[1:]:
        if dict(sample.entry.tensors) != descriptors:
            raise ValueError("video cached samples must share tensor shapes, dtypes, and layouts")
        if sample.entry.bucket_key != bucket_key:
            raise ValueError("video cached samples cannot mix task/shape/aspect/conditioning buckets")
        if sample.entry.provenance.batch_contract_digest != contract_digest:
            raise ValueError("video cached samples belong to incompatible model/preprocessing contracts")

    stacked = {name: torch.stack([sample.tensors[name] for sample in values]) for name in sorted(descriptors)}
    conditions: dict[str, torch.Tensor] = {"clean_latents": stacked["clean_latents"]}
    for storage_name, tensor in stacked.items():
        if storage_name.startswith(_CONDITION_PREFIX):
            conditions[storage_name.removeprefix(_CONDITION_PREFIX)] = tensor
    for name in ("latent_loss_mask", "valid_latent_mask"):
        if name in stacked:
            conditions[name] = stacked[name]
    sample_weights = stacked.get("sample_weight")
    latent_tokens = len(values) * bucket_key.token_count
    provenance = reference.entry.provenance
    return TrainingBatch(
        sample_ids=tuple(sample.entry.sample_id for sample in values),
        prompts=tuple(f"sha256:{sample.entry.provenance.prompt_sha256}" for sample in values),
        conditions=conditions,
        sample_weights=sample_weights,
        metadata={
            "cache_schema": VIDEO_CACHE_OBJECT_SCHEMA,
            "cache_identity_sha256": tuple(sample.entry.identity_sha256 for sample in values),
            "cache_object_sha256": tuple(sample.entry.object_sha256 for sample in values),
            "bucket_key": bucket_key.to_dict(),
            "bucket_digest": bucket_key.digest,
            "samples_per_microbatch": len(values),
            "latent_tokens_per_sample": bucket_key.token_count,
            "latent_tokens_per_microbatch": latent_tokens,
            "target_num_frames": provenance.target_num_frames,
            "target_height": provenance.target_height,
            "target_width": provenance.target_width,
            "target_fps": provenance.target_fps,
            "batch_contract_digest": contract_digest,
        },
    )


__all__ = [
    "VIDEO_CACHE_INDEX_SCHEMA",
    "VIDEO_CACHE_OBJECT_SCHEMA",
    "VideoCacheEntry",
    "VideoCacheIndex",
    "VideoCacheProvenance",
    "VideoCacheStore",
    "VideoCachedDataset",
    "VideoCachedSample",
    "collate_video_cached_samples",
]

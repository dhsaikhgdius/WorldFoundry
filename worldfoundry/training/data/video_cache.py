"""Cached video latents, conditioning tensors, and masks."""

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
from urllib.parse import quote

import torch

from worldfoundry.core.io.integrity import replace_json_atomic, sync_directory
from worldfoundry.training.api.contracts import TrainingBatch

from .sana_cache import CacheTensorDescriptor
from .video_bucketing import VideoBucketKey, VideoLatentGeometry

VIDEO_CACHE_OBJECT_SCHEMA = "worldfoundry-video-training-cache-object"
VIDEO_CACHE_INDEX_SCHEMA = "worldfoundry-video-training-cache-index"
_CONDITION_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_CONDITION_PREFIX = "condition."
_RESERVED_KEYS = frozenset({"clean_latents", "latent_loss_mask", "valid_latent_mask", "sample_weight"})
_RESERVED_LAYOUTS = {
    "clean_latents": "channels-frames-height-width",
    "latent_loss_mask": "channels-frames-height-width",
    "valid_latent_mask": "one-frames-height-width",
    "sample_weight": "scalar",
}


def _nonempty(value: object, *, field_name: str) -> str:
    resolved = str(value).strip()
    if not resolved:
        raise ValueError(f"{field_name} cannot be empty")
    return resolved


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


def _positive_float(value: object, *, field_name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return resolved


def _json_mapping(value: Mapping[str, object], *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    try:
        payload = json.loads(json.dumps(dict(value), sort_keys=True))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field_name} must contain JSON values") from error
    return MappingProxyType(payload)


def _strict_mapping(
    value: object,
    *,
    field_name: str,
    allowed: set[str],
    required: set[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    payload = {str(key): item for key, item in value.items()}
    unknown = sorted(set(payload) - allowed)
    missing = sorted(required - set(payload))
    if unknown or missing:
        raise ValueError(f"{field_name} fields mismatch; missing={missing}, unknown={unknown}")
    return payload


def _condition_name(value: object) -> str:
    resolved = _nonempty(value, field_name="conditioning tensor name")
    if _CONDITION_NAME_PATTERN.fullmatch(resolved) is None or resolved in _RESERVED_KEYS:
        raise ValueError(f"invalid conditioning tensor name: {resolved!r}")
    return resolved


@dataclass(frozen=True, slots=True)
class VideoCacheProvenance:
    media_uri: str
    prompt: str
    model_recipe: str
    codec: Mapping[str, object]
    conditioner: Mapping[str, object]
    tokenizer: Mapping[str, object]
    conditioning_inputs: Mapping[str, object]
    safety_audit: Mapping[str, object]
    frame_sampling: Mapping[str, object]
    spatial_transform: Mapping[str, object]
    latent_normalization: Mapping[str, object]
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
        object.__setattr__(self, "media_uri", _nonempty(self.media_uri, field_name="media_uri"))
        object.__setattr__(self, "prompt", _nonempty(self.prompt, field_name="prompt"))
        object.__setattr__(self, "model_recipe", _nonempty(self.model_recipe, field_name="model_recipe"))
        for name in (
            "codec",
            "conditioner",
            "tokenizer",
            "conditioning_inputs",
            "safety_audit",
            "frame_sampling",
            "spatial_transform",
            "latent_normalization",
        ):
            object.__setattr__(self, name, _json_mapping(getattr(self, name), field_name=name))
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
    def batch_contract(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "model_recipe": self.model_recipe,
                "codec": dict(self.codec),
                "conditioner": dict(self.conditioner),
                "tokenizer": dict(self.tokenizer),
                "latent_normalization": dict(self.latent_normalization),
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
            "media_uri": self.media_uri,
            "prompt": self.prompt,
            "model_recipe": self.model_recipe,
            "codec": dict(self.codec),
            "conditioner": dict(self.conditioner),
            "tokenizer": dict(self.tokenizer),
            "conditioning_inputs": dict(self.conditioning_inputs),
            "safety_audit": dict(self.safety_audit),
            "frame_sampling": dict(self.frame_sampling),
            "spatial_transform": dict(self.spatial_transform),
            "latent_normalization": dict(self.latent_normalization),
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


def _validate_tensors(
    tensors: Mapping[str, torch.Tensor],
    layouts: Mapping[str, str],
    provenance: VideoCacheProvenance,
) -> Mapping[str, CacheTensorDescriptor]:
    if "clean_latents" not in tensors:
        raise ValueError("video cache requires clean_latents")
    if set(layouts) != set(tensors):
        raise ValueError("one layout is required for every cached video tensor")
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors.values()):
        raise TypeError("video cache values must be tensors")
    latents = tensors["clean_latents"]
    shape = provenance.bucket_key.latent_shape
    if latents.ndim != 4 or tuple(latents.shape[-3:]) != (shape.frames, shape.height, shape.width):
        raise ValueError("clean_latents must be [C,T,H,W] matching provenance")
    loss_mask = tensors.get("latent_loss_mask")
    if loss_mask is not None and tuple(loss_mask.shape[-3:]) != tuple(latents.shape[-3:]):
        raise ValueError("latent_loss_mask must match clean_latents")
    valid_mask = tensors.get("valid_latent_mask")
    if valid_mask is not None and tuple(valid_mask.shape) != (1, *tuple(latents.shape[-3:])):
        raise ValueError("valid_latent_mask must be [1,T,H,W] matching clean_latents")
    weight = tensors.get("sample_weight")
    if weight is not None and weight.ndim != 0:
        raise ValueError("sample_weight must be a scalar")
    for name, tensor in tensors.items():
        if name.startswith(_CONDITION_PREFIX):
            _condition_name(name.removeprefix(_CONDITION_PREFIX))
            if tensor.ndim == 0:
                raise ValueError("conditioning tensors cannot be scalar")
        elif name not in _RESERVED_KEYS:
            raise ValueError(f"unsupported video cache tensor: {name}")
    return MappingProxyType({name: _descriptor(tensor, layout=layouts[name]) for name, tensor in tensors.items()})


def _identity_payload(
    provenance: VideoCacheProvenance, descriptors: Mapping[str, CacheTensorDescriptor]
) -> dict[str, object]:
    return {
        "schema": VIDEO_CACHE_OBJECT_SCHEMA,
        "provenance": provenance.to_dict(),
        "tensors": {name: descriptors[name].to_dict() for name in sorted(descriptors)},
    }


def _metadata(provenance: VideoCacheProvenance, descriptors: Mapping[str, CacheTensorDescriptor]) -> dict[str, str]:
    return {
        "worldfoundry": json.dumps(
            _identity_payload(provenance, descriptors), sort_keys=True, separators=(",", ":")
        )
    }


@dataclass(frozen=True, slots=True)
class VideoCacheEntry:
    sample_id: str
    object_size_bytes: int
    object_path: str
    provenance: VideoCacheProvenance
    tensors: Mapping[str, CacheTensorDescriptor]

    def __post_init__(self) -> None:
        sample_id = _nonempty(self.sample_id, field_name="sample_id")
        expected_path = f"objects/{quote(sample_id, safe='')}.safetensors"
        if str(self.object_path).replace("\\", "/") != expected_path:
            raise ValueError(f"object_path must be {expected_path!r}")
        descriptors = dict(self.tensors)
        if "clean_latents" not in descriptors:
            raise ValueError("video cache entry requires clean_latents")
        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "object_size_bytes", _positive_int(self.object_size_bytes, field_name="object_size_bytes"))
        object.__setattr__(self, "object_path", expected_path)
        object.__setattr__(self, "tensors", MappingProxyType(descriptors))

    @property
    def bucket_key(self) -> VideoBucketKey:
        return self.provenance.bucket_key

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
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
        return cls(
            **payload,
            provenance=VideoCacheProvenance.from_mapping(raw_provenance),
            tensors={str(name): CacheTensorDescriptor.from_mapping(item) for name, item in raw_tensors.items()},
        )


@dataclass(frozen=True, slots=True)
class VideoCacheIndex:
    entries: tuple[VideoCacheEntry, ...]
    schema: str = VIDEO_CACHE_INDEX_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != VIDEO_CACHE_INDEX_SCHEMA:
            raise ValueError(f"unsupported video cache index schema: {self.schema!r}")
        entries = tuple(self.entries)
        if not entries or not all(isinstance(entry, VideoCacheEntry) for entry in entries):
            raise ValueError("video cache index requires entries")
        if len({entry.sample_id for entry in entries}) != len(entries):
            raise ValueError("video cache sample IDs must be unique")
        object.__setattr__(self, "entries", entries)

    @classmethod
    def build(cls, entries: Sequence[VideoCacheEntry]) -> VideoCacheIndex:
        return cls(entries=tuple(entries))

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.schema, "entries": [entry.to_dict() for entry in self.entries]}

    @classmethod
    def from_mapping(cls, value: object) -> VideoCacheIndex:
        payload = _strict_mapping(
            value,
            field_name="video cache index",
            allowed={"schema", "entries"},
            required={"schema", "entries"},
        )
        return cls(
            schema=payload["schema"],
            entries=tuple(VideoCacheEntry.from_mapping(item) for item in payload["entries"]),
        )


@dataclass(frozen=True, slots=True)
class VideoCachedSample:
    entry: VideoCacheEntry
    tensors: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        layouts = {name: descriptor.layout for name, descriptor in self.entry.tensors.items()}
        descriptors = _validate_tensors(self.tensors, layouts, self.entry.provenance)
        if dict(descriptors) != dict(self.entry.tensors):
            raise ValueError("loaded video tensor descriptors differ from the cache index")
        object.__setattr__(self, "tensors", MappingProxyType(dict(self.tensors)))


class VideoCacheStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.objects = self.root / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)

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
        conditioning = dict(conditioning or {})
        conditioning_layouts = dict(conditioning_layouts or {})
        if set(conditioning) != set(conditioning_layouts):
            raise ValueError("conditioning layouts must match conditioning tensor names")
        tensors: dict[str, torch.Tensor] = {"clean_latents": clean_latents}
        layouts = {"clean_latents": _RESERVED_LAYOUTS["clean_latents"]}
        for raw_name, tensor in conditioning.items():
            name = _condition_name(raw_name)
            tensors[f"{_CONDITION_PREFIX}{name}"] = tensor
            layouts[f"{_CONDITION_PREFIX}{name}"] = conditioning_layouts[raw_name]
        for name, tensor in (
            ("latent_loss_mask", latent_loss_mask),
            ("valid_latent_mask", valid_latent_mask),
            ("sample_weight", sample_weight),
        ):
            if tensor is not None:
                tensors[name] = tensor
                layouts[name] = _RESERVED_LAYOUTS[name]
        descriptors = _validate_tensors(tensors, layouts, provenance)
        try:
            from safetensors.torch import save_file
        except ModuleNotFoundError as error:
            raise RuntimeError("video training cache requires safetensors") from error
        destination = self.objects / f"{quote(sample_id, safe='')}.safetensors"
        handle = tempfile.NamedTemporaryFile(prefix=".video-cache-", suffix=".safetensors", dir=self.objects, delete=False)
        temporary = Path(handle.name)
        handle.close()
        try:
            save_file(
                {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()},
                temporary,
                metadata=_metadata(provenance, descriptors),
            )
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            sync_directory(self.objects)
        finally:
            temporary.unlink(missing_ok=True)
        return VideoCacheEntry(
            sample_id=sample_id,
            object_size_bytes=destination.stat().st_size,
            object_path=f"objects/{quote(sample_id, safe='')}.safetensors",
            provenance=provenance,
            tensors=descriptors,
        )

    def audit_entry(self, entry: VideoCacheEntry, *, load_tensors: bool = True) -> VideoCachedSample | None:
        path = self.root / entry.object_path
        if not path.is_file():
            raise FileNotFoundError(f"video cache object not found: {path}")
        if path.stat().st_size != entry.object_size_bytes:
            raise ValueError(f"video cache object size mismatch for {entry.sample_id!r}")
        try:
            from safetensors import safe_open
        except ModuleNotFoundError as error:
            raise RuntimeError("video training cache requires safetensors") from error
        with safe_open(path, framework="pt", device="cpu") as handle:
            if (handle.metadata() or {}) != _metadata(entry.provenance, entry.tensors):
                raise ValueError(f"video cache object metadata mismatch for {entry.sample_id!r}")
            if set(handle.keys()) != set(entry.tensors):
                raise ValueError(f"video cache tensor keys mismatch for {entry.sample_id!r}")
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
        entries: Sequence[VideoCacheEntry],
        filename: str = "index.json",
    ) -> VideoCacheIndex:
        index = VideoCacheIndex.build(entries)
        replace_json_atomic(self.root / filename, index.to_dict(), root=self.root)
        return index

    def read_index(self, filename: str = "index.json") -> VideoCacheIndex:
        with (self.root / filename).open("r", encoding="utf-8") as handle:
            return VideoCacheIndex.from_mapping(json.load(handle))


class VideoCachedDataset(Sequence[VideoCachedSample]):
    def __init__(
        self,
        root: str | Path,
        *,
        index_filename: str = "index.json",
        expected_sample_ids: Sequence[str] | None = None,
        audit_on_open: bool = True,
        verify_on_read: bool = True,
    ) -> None:
        self.store = VideoCacheStore(root)
        self.index = self.store.read_index(index_filename)
        self._entries = self.index.entries
        if expected_sample_ids is not None and tuple(expected_sample_ids) != self.sample_ids:
            raise ValueError("video cache sample IDs differ from the expected dataset")
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
    def batch_contracts(self) -> tuple[Mapping[str, object], ...]:
        return tuple(entry.provenance.batch_contract for entry in self._entries)

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
        from safetensors.torch import load_file

        return VideoCachedSample(entry=entry, tensors=load_file(self.store.root / entry.object_path, device="cpu"))

    def __iter__(self) -> Iterator[VideoCachedSample]:
        for index in range(len(self)):
            yield self[index]


def collate_video_cached_samples(samples: Sequence[VideoCachedSample]) -> TrainingBatch:
    values = tuple(samples)
    if not values:
        raise ValueError("cannot collate an empty video cache batch")
    reference = values[0]
    descriptors = dict(reference.entry.tensors)
    bucket_key = reference.entry.bucket_key
    contract = reference.entry.provenance.batch_contract
    for sample in values[1:]:
        if dict(sample.entry.tensors) != descriptors:
            raise ValueError("video cached samples must share tensor descriptors")
        if sample.entry.bucket_key != bucket_key:
            raise ValueError("video cached samples cannot mix bucket keys")
        if sample.entry.provenance.batch_contract != contract:
            raise ValueError("video cached samples belong to incompatible model or preprocessing contracts")
    stacked = {name: torch.stack([sample.tensors[name] for sample in values]) for name in sorted(descriptors)}
    conditions = {"clean_latents": stacked["clean_latents"]}
    for storage_name, tensor in stacked.items():
        if storage_name.startswith(_CONDITION_PREFIX):
            conditions[storage_name.removeprefix(_CONDITION_PREFIX)] = tensor
    for name in ("latent_loss_mask", "valid_latent_mask"):
        if name in stacked:
            conditions[name] = stacked[name]
    latent_tokens = len(values) * bucket_key.token_count
    provenance = reference.entry.provenance
    return TrainingBatch(
        sample_ids=tuple(sample.entry.sample_id for sample in values),
        prompts=tuple(sample.entry.provenance.prompt for sample in values),
        conditions=conditions,
        sample_weights=stacked.get("sample_weight"),
        metadata={
            "cache_schema": VIDEO_CACHE_OBJECT_SCHEMA,
            "cache_entries": tuple(sample.entry.to_dict() for sample in values),
            "bucket_key": bucket_key.to_dict(),
            "samples_per_microbatch": len(values),
            "latent_tokens_per_sample": bucket_key.token_count,
            "latent_tokens_per_microbatch": latent_tokens,
            "target_num_frames": provenance.target_num_frames,
            "target_height": provenance.target_height,
            "target_width": provenance.target_width,
            "target_fps": provenance.target_fps,
            "batch_contract": dict(contract),
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

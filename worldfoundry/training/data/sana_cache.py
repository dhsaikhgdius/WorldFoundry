"""Latent and text-conditioning cache for SANA training."""

from __future__ import annotations

import json
import math
import os
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

SANA_CACHE_OBJECT_SCHEMA = "worldfoundry-sana-training-cache-object"
SANA_CACHE_INDEX_SCHEMA = "worldfoundry-sana-training-cache-index"
_OBJECT_KEYS = frozenset({"clean_latents", "context", "context_mask", "latent_loss_mask", "sample_weight"})
_REQUIRED_OBJECT_KEYS = frozenset({"clean_latents", "context", "context_mask"})
_LAYOUTS = {
    "clean_latents": "channels-height-width",
    "context": "branch-sequence-features",
    "context_mask": "sequence",
    "latent_loss_mask": "channels-height-width",
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


def sana_cache_contract(
    model_recipe: str,
    *,
    latent_channels: int,
    spatial_compression: int,
    max_text_length: int,
    context_features: int,
) -> dict[str, object]:
    """Return the explicit denoiser-facing cache contract."""

    return {
        "model_recipe": _nonempty(model_recipe, field_name="model_recipe").lower().replace("_", "-"),
        "latent_channels": _positive_int(latent_channels, field_name="latent_channels"),
        "spatial_compression": _positive_int(spatial_compression, field_name="spatial_compression"),
        "max_text_length": _positive_int(max_text_length, field_name="max_text_length"),
        "context_features": _positive_int(context_features, field_name="context_features"),
    }


@dataclass(frozen=True, slots=True)
class SanaCacheProvenance:
    """Inputs that determine one cached SANA sample."""

    media_uri: str
    prompt: str
    model_recipe: str
    codec: Mapping[str, object]
    conditioner: Mapping[str, object]
    tokenizer: Mapping[str, object]
    safety_audit: Mapping[str, object]
    pixel_transform: Mapping[str, object]
    prompt_enhancement: Mapping[str, object]
    image_height: int
    image_width: int
    spatial_compression: int
    latent_scaling_factor: float
    max_text_length: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "media_uri", _nonempty(self.media_uri, field_name="media_uri"))
        object.__setattr__(self, "prompt", _nonempty(self.prompt, field_name="prompt"))
        object.__setattr__(self, "model_recipe", _nonempty(self.model_recipe, field_name="model_recipe"))
        for name in ("codec", "conditioner", "tokenizer", "safety_audit", "pixel_transform", "prompt_enhancement"):
            object.__setattr__(self, name, _json_mapping(getattr(self, name), field_name=name))
        object.__setattr__(self, "image_height", _positive_int(self.image_height, field_name="image_height"))
        object.__setattr__(self, "image_width", _positive_int(self.image_width, field_name="image_width"))
        object.__setattr__(
            self, "spatial_compression", _positive_int(self.spatial_compression, field_name="spatial_compression")
        )
        object.__setattr__(
            self,
            "latent_scaling_factor",
            _positive_float(self.latent_scaling_factor, field_name="latent_scaling_factor"),
        )
        object.__setattr__(self, "max_text_length", _positive_int(self.max_text_length, field_name="max_text_length"))

    def to_dict(self) -> dict[str, object]:
        return {
            "media_uri": self.media_uri,
            "prompt": self.prompt,
            "model_recipe": self.model_recipe,
            "codec": dict(self.codec),
            "conditioner": dict(self.conditioner),
            "tokenizer": dict(self.tokenizer),
            "safety_audit": dict(self.safety_audit),
            "pixel_transform": dict(self.pixel_transform),
            "prompt_enhancement": dict(self.prompt_enhancement),
            "image_height": self.image_height,
            "image_width": self.image_width,
            "spatial_compression": self.spatial_compression,
            "latent_scaling_factor": self.latent_scaling_factor,
            "max_text_length": self.max_text_length,
        }

    def batch_contract(self) -> dict[str, object]:
        return {
            "model_recipe": self.model_recipe,
            "codec": dict(self.codec),
            "conditioner": dict(self.conditioner),
            "tokenizer": dict(self.tokenizer),
            "pixel_transform": dict(self.pixel_transform),
            "prompt_enhancement": dict(self.prompt_enhancement),
            "image_height": self.image_height,
            "image_width": self.image_width,
            "spatial_compression": self.spatial_compression,
            "latent_scaling_factor": self.latent_scaling_factor,
            "max_text_length": self.max_text_length,
        }

    @classmethod
    def from_mapping(cls, value: object) -> SanaCacheProvenance:
        fields = set(cls.__dataclass_fields__)
        return cls(**_strict_mapping(value, field_name="cache provenance", allowed=fields, required=fields))


@dataclass(frozen=True, slots=True)
class CacheTensorDescriptor:
    dtype: str
    shape: tuple[int, ...]
    layout: str

    def __post_init__(self) -> None:
        dtype = _nonempty(self.dtype, field_name="tensor dtype").removeprefix("torch.")
        shape = tuple(int(size) for size in self.shape)
        if any(size <= 0 for size in shape):
            raise ValueError(f"cached tensor dimensions must be positive; got {shape}")
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "layout", _nonempty(self.layout, field_name="tensor layout"))

    def to_dict(self) -> dict[str, object]:
        return {"dtype": self.dtype, "shape": list(self.shape), "layout": self.layout}

    @classmethod
    def from_mapping(cls, value: object) -> CacheTensorDescriptor:
        payload = _strict_mapping(
            value,
            field_name="tensor descriptor",
            allowed={"dtype", "shape", "layout"},
            required={"dtype", "shape", "layout"},
        )
        return cls(dtype=payload["dtype"], shape=tuple(payload["shape"]), layout=payload["layout"])


def _descriptor(name: str, tensor: torch.Tensor) -> CacheTensorDescriptor:
    return CacheTensorDescriptor(
        dtype=str(tensor.dtype).removeprefix("torch."),
        shape=tuple(int(size) for size in tensor.shape),
        layout=_LAYOUTS[name],
    )


def _validate_tensors(
    tensors: Mapping[str, torch.Tensor], provenance: SanaCacheProvenance
) -> Mapping[str, CacheTensorDescriptor]:
    keys = set(tensors)
    if not _REQUIRED_OBJECT_KEYS <= keys or keys - _OBJECT_KEYS:
        raise ValueError("SANA cache tensor keys differ from the supported contract")
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors.values()):
        raise TypeError("SANA cache values must be tensors")
    latents = tensors["clean_latents"]
    context = tensors["context"]
    context_mask = tensors["context_mask"]
    if latents.ndim != 3 or not latents.is_floating_point():
        raise ValueError("clean_latents must be an unbatched floating CHW tensor")
    expected_spatial = (
        provenance.image_height // provenance.spatial_compression,
        provenance.image_width // provenance.spatial_compression,
    )
    if tuple(latents.shape[-2:]) != expected_spatial:
        raise ValueError("clean_latents spatial shape differs from the configured compression")
    if context.ndim != 3 or context.shape[0] != 1 or context.shape[1] != provenance.max_text_length:
        raise ValueError("context must have shape [1,max_text_length,features]")
    if context_mask.ndim != 1 or context_mask.shape[0] != provenance.max_text_length:
        raise ValueError("context_mask must have shape [max_text_length]")
    loss_mask = tensors.get("latent_loss_mask")
    if loss_mask is not None and tuple(loss_mask.shape[-2:]) != tuple(latents.shape[-2:]):
        raise ValueError("latent_loss_mask spatial shape must match clean_latents")
    weight = tensors.get("sample_weight")
    if weight is not None and weight.ndim != 0:
        raise ValueError("sample_weight must be a scalar")
    return MappingProxyType({name: _descriptor(name, tensor) for name, tensor in tensors.items()})


def _identity_payload(
    provenance: SanaCacheProvenance, descriptors: Mapping[str, CacheTensorDescriptor]
) -> dict[str, object]:
    return {
        "schema": SANA_CACHE_OBJECT_SCHEMA,
        "provenance": provenance.to_dict(),
        "tensors": {name: descriptors[name].to_dict() for name in sorted(descriptors)},
    }


def _metadata(provenance: SanaCacheProvenance, descriptors: Mapping[str, CacheTensorDescriptor]) -> dict[str, str]:
    return {
        "worldfoundry": json.dumps(
            _identity_payload(provenance, descriptors), sort_keys=True, separators=(",", ":")
        )
    }


@dataclass(frozen=True, slots=True)
class SanaCacheEntry:
    sample_id: str
    object_size_bytes: int
    object_path: str
    provenance: SanaCacheProvenance
    tensors: Mapping[str, CacheTensorDescriptor]

    def __post_init__(self) -> None:
        sample_id = _nonempty(self.sample_id, field_name="sample_id")
        expected_path = f"objects/{quote(sample_id, safe='')}.safetensors"
        if str(self.object_path).replace("\\", "/") != expected_path:
            raise ValueError(f"object_path must be {expected_path!r}")
        if not isinstance(self.provenance, SanaCacheProvenance):
            raise TypeError("provenance must be SanaCacheProvenance")
        descriptors = dict(self.tensors)
        if not _REQUIRED_OBJECT_KEYS <= set(descriptors) or set(descriptors) - _OBJECT_KEYS:
            raise ValueError("cache entry has invalid tensor descriptor keys")
        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "object_size_bytes", _positive_int(self.object_size_bytes, field_name="object_size_bytes"))
        object.__setattr__(self, "object_path", expected_path)
        object.__setattr__(self, "tensors", MappingProxyType(descriptors))

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "object_size_bytes": self.object_size_bytes,
            "object_path": self.object_path,
            "provenance": self.provenance.to_dict(),
            "tensors": {name: self.tensors[name].to_dict() for name in sorted(self.tensors)},
        }

    @classmethod
    def from_mapping(cls, value: object) -> SanaCacheEntry:
        fields = set(cls.__dataclass_fields__)
        payload = _strict_mapping(value, field_name="cache entry", allowed=fields, required=fields)
        raw_tensors = payload.pop("tensors")
        raw_provenance = payload.pop("provenance")
        return cls(
            **payload,
            provenance=SanaCacheProvenance.from_mapping(raw_provenance),
            tensors={str(name): CacheTensorDescriptor.from_mapping(item) for name, item in raw_tensors.items()},
        )


@dataclass(frozen=True, slots=True)
class SanaCacheIndex:
    entries: tuple[SanaCacheEntry, ...]
    schema: str = SANA_CACHE_INDEX_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SANA_CACHE_INDEX_SCHEMA:
            raise ValueError(f"unsupported SANA cache index schema: {self.schema!r}")
        entries = tuple(self.entries)
        if not entries or not all(isinstance(entry, SanaCacheEntry) for entry in entries):
            raise ValueError("cache index requires entries")
        if len({entry.sample_id for entry in entries}) != len(entries):
            raise ValueError("cache index sample IDs must be unique")
        object.__setattr__(self, "entries", entries)

    @classmethod
    def build(cls, entries: Sequence[SanaCacheEntry]) -> SanaCacheIndex:
        return cls(entries=tuple(entries))

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.schema, "entries": [entry.to_dict() for entry in self.entries]}

    @classmethod
    def from_mapping(cls, value: object) -> SanaCacheIndex:
        payload = _strict_mapping(
            value,
            field_name="cache index",
            allowed={"schema", "entries"},
            required={"schema", "entries"},
        )
        return cls(
            schema=payload["schema"],
            entries=tuple(SanaCacheEntry.from_mapping(item) for item in payload["entries"]),
        )


@dataclass(frozen=True, slots=True)
class SanaCachedSample:
    entry: SanaCacheEntry
    tensors: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        descriptors = _validate_tensors(self.tensors, self.entry.provenance)
        if dict(descriptors) != dict(self.entry.tensors):
            raise ValueError("loaded tensor descriptors do not match the cache index")
        object.__setattr__(self, "tensors", MappingProxyType(dict(self.tensors)))


class SanaCacheStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.objects = self.root / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)

    def _object_path(self, sample_id: str) -> Path:
        return self.objects / f"{quote(sample_id, safe='')}.safetensors"

    def write_sample(
        self,
        *,
        sample_id: str,
        provenance: SanaCacheProvenance,
        clean_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        latent_loss_mask: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
    ) -> SanaCacheEntry:
        tensors = {"clean_latents": clean_latents, "context": context, "context_mask": context_mask}
        if latent_loss_mask is not None:
            tensors["latent_loss_mask"] = latent_loss_mask
        if sample_weight is not None:
            tensors["sample_weight"] = sample_weight
        descriptors = _validate_tensors(tensors, provenance)
        try:
            from safetensors.torch import save_file
        except ModuleNotFoundError as error:
            raise RuntimeError("SANA training cache requires safetensors") from error
        destination = self._object_path(sample_id)
        handle = tempfile.NamedTemporaryFile(prefix=".sana-cache-", suffix=".safetensors", dir=self.objects, delete=False)
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
        return SanaCacheEntry(
            sample_id=sample_id,
            object_size_bytes=destination.stat().st_size,
            object_path=f"objects/{quote(sample_id, safe='')}.safetensors",
            provenance=provenance,
            tensors=descriptors,
        )

    def audit_entry(self, entry: SanaCacheEntry, *, load_tensors: bool = True) -> SanaCachedSample | None:
        path = self.root / entry.object_path
        if not path.is_file():
            raise FileNotFoundError(f"cache object not found: {path}")
        if path.stat().st_size != entry.object_size_bytes:
            raise ValueError(f"cache object size mismatch for {entry.sample_id!r}")
        try:
            from safetensors import safe_open
        except ModuleNotFoundError as error:
            raise RuntimeError("SANA training cache requires safetensors") from error
        with safe_open(path, framework="pt", device="cpu") as handle:
            if (handle.metadata() or {}) != _metadata(entry.provenance, entry.tensors):
                raise ValueError(f"cache object metadata mismatch for {entry.sample_id!r}")
            if set(handle.keys()) != set(entry.tensors):
                raise ValueError(f"cache object tensor keys mismatch for {entry.sample_id!r}")
            if not load_tensors:
                for name, descriptor in entry.tensors.items():
                    if tuple(handle.get_slice(name).get_shape()) != descriptor.shape:
                        raise ValueError(f"cache object shape mismatch for {name!r}")
                return None
            tensors = {name: handle.get_tensor(name) for name in sorted(handle.keys())}
        return SanaCachedSample(entry=entry, tensors=tensors)

    def write_index(
        self,
        *,
        entries: Sequence[SanaCacheEntry],
        filename: str = "index.json",
    ) -> SanaCacheIndex:
        index = SanaCacheIndex.build(entries)
        replace_json_atomic(self.root / filename, index.to_dict(), root=self.root)
        return index

    def read_index(self, filename: str = "index.json") -> SanaCacheIndex:
        with (self.root / filename).open("r", encoding="utf-8") as handle:
            return SanaCacheIndex.from_mapping(json.load(handle))


class SanaCachedDataset(Sequence[SanaCachedSample]):
    def __init__(
        self,
        root: str | Path,
        *,
        index_filename: str = "index.json",
        expected_sample_ids: Sequence[str] | None = None,
        audit_on_open: bool = True,
        verify_on_read: bool = True,
    ) -> None:
        self.store = SanaCacheStore(root)
        self.index = self.store.read_index(index_filename)
        self._entries = self.index.entries
        if expected_sample_ids is not None and tuple(expected_sample_ids) != self.sample_ids:
            raise ValueError("cache sample IDs differ from the expected dataset")
        self._verify_on_read = bool(verify_on_read)
        if audit_on_open:
            for entry in self._entries:
                self.store.audit_entry(entry, load_tensors=False)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(entry.sample_id for entry in self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    @overload
    def __getitem__(self, index: int) -> SanaCachedSample: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[SanaCachedSample, ...]: ...

    def __getitem__(self, index: int | slice) -> SanaCachedSample | tuple[SanaCachedSample, ...]:
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(len(self))))
        entry = self._entries[index]
        if self._verify_on_read:
            sample = self.store.audit_entry(entry, load_tensors=True)
            assert sample is not None
            return sample
        from safetensors.torch import load_file

        return SanaCachedSample(entry=entry, tensors=load_file(self.store.root / entry.object_path, device="cpu"))

    def __iter__(self) -> Iterator[SanaCachedSample]:
        for index in range(len(self)):
            yield self[index]


def collate_sana_cached_samples(samples: Sequence[SanaCachedSample]) -> TrainingBatch:
    values = tuple(samples)
    if not values:
        raise ValueError("cannot collate an empty SANA cache batch")
    reference = values[0]
    for sample in values[1:]:
        if dict(sample.entry.tensors) != dict(reference.entry.tensors):
            raise ValueError("cached samples in one batch must share tensor descriptors")
        if sample.entry.provenance.batch_contract() != reference.entry.provenance.batch_contract():
            raise ValueError("cached samples in one batch belong to incompatible preprocessing buckets")
    tensor_keys = sorted(reference.tensors)
    stacked = {name: torch.stack([sample.tensors[name] for sample in values]) for name in tensor_keys}
    conditions = {
        "clean_latents": stacked["clean_latents"],
        "context": stacked["context"],
        "context_mask": stacked["context_mask"],
    }
    if "latent_loss_mask" in stacked:
        conditions["latent_loss_mask"] = stacked["latent_loss_mask"]
    return TrainingBatch(
        sample_ids=tuple(sample.entry.sample_id for sample in values),
        prompts=tuple(sample.entry.provenance.prompt for sample in values),
        conditions=conditions,
        sample_weights=stacked.get("sample_weight"),
        metadata={
            "cache_schema": SANA_CACHE_OBJECT_SCHEMA,
            "cache_entries": tuple(sample.entry.to_dict() for sample in values),
            "image_height": reference.entry.provenance.image_height,
            "image_width": reference.entry.provenance.image_width,
            "latent_scaling_factor": reference.entry.provenance.latent_scaling_factor,
            "max_text_length": reference.entry.provenance.max_text_length,
        },
    )


__all__ = [
    "CacheTensorDescriptor",
    "SANA_CACHE_INDEX_SCHEMA",
    "SANA_CACHE_OBJECT_SCHEMA",
    "SanaCacheEntry",
    "SanaCacheIndex",
    "SanaCacheProvenance",
    "SanaCacheStore",
    "SanaCachedDataset",
    "SanaCachedSample",
    "collate_sana_cached_samples",
    "sana_cache_contract",
]

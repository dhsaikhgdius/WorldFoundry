"""Content-addressed latent and text-conditioning cache for SANA training.

The cache deliberately stores no raw prompts.  A logical identity binds source
content, encoder assets, safety review, transforms, and tensor contracts.  The
serialized safetensors bytes are then addressed independently by their own
SHA-256, so neither a stale index nor an overwritten object can silently alter
training inputs.
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
from worldfoundry.core.io.integrity import (
    canonical_sha256,
    replace_json_atomic,
    sync_directory,
    text_sha256,
)
from worldfoundry.training.api.contracts import PreparedBatch, TrainingBatch

SANA_CACHE_OBJECT_SCHEMA = "worldfoundry-sana-training-cache-object"
SANA_CACHE_INDEX_SCHEMA = "worldfoundry-sana-training-cache-index"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_OBJECT_KEYS = frozenset(
    {
        "clean_latents",
        "context",
        "context_mask",
        "latent_loss_mask",
        "sample_weight",
    }
)
_REQUIRED_OBJECT_KEYS = frozenset({"clean_latents", "context", "context_mask"})
_LAYOUTS = {
    "clean_latents": "channels-height-width",
    "context": "branch-sequence-features",
    "context_mask": "sequence",
    "latent_loss_mask": "channels-height-width",
    "sample_weight": "scalar",
}


def _canonical_json(value: object) -> str:
    try:
        return _core_canonical_json(value)
    except (TypeError, ValueError) as error:
        raise TypeError("cache metadata must be JSON serializable without NaN or infinity") from error


def sana_cache_contract_digest(
    model_recipe: str,
    *,
    latent_channels: int,
    spatial_compression: int,
    max_text_length: int,
    context_features: int,
) -> str:
    """Digest the denoiser-facing contract shared by cache creation and use."""

    return canonical_sha256(
        {
            "schema": "worldfoundry-sana-training-cache-contract",
            "model_recipe": _nonempty(model_recipe, field_name="model_recipe").lower().replace("_", "-"),
            "latent_channels": _positive_int(latent_channels, field_name="latent_channels"),
            "spatial_compression": _positive_int(
                spatial_compression,
                field_name="spatial_compression",
            ),
            "max_text_length": _positive_int(max_text_length, field_name="max_text_length"),
            "context_features": _positive_int(context_features, field_name="context_features"),
        }
    )


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
    if unknown:
        raise ValueError(f"{field_name} contains unknown fields: {unknown}")
    if missing:
        raise ValueError(f"{field_name} is missing required fields: {missing}")
    return payload


@dataclass(frozen=True, slots=True)
class SanaCacheProvenance:
    """Every non-tensor input that can change cached SANA features."""

    media_sha256: str
    prompt_sha256: str
    model_recipe_digest: str
    codec_digest: str
    conditioner_digest: str
    tokenizer_digest: str
    safety_audit_digest: str
    pixel_transform_digest: str
    prompt_enhancement_digest: str
    image_height: int
    image_width: int
    spatial_compression: int
    latent_scaling_factor: float
    max_text_length: int

    def __post_init__(self) -> None:
        for name in (
            "media_sha256",
            "prompt_sha256",
            "model_recipe_digest",
            "codec_digest",
            "conditioner_digest",
            "tokenizer_digest",
            "safety_audit_digest",
            "pixel_transform_digest",
            "prompt_enhancement_digest",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), field_name=name))
        object.__setattr__(
            self,
            "image_height",
            _positive_int(self.image_height, field_name="image_height"),
        )
        object.__setattr__(
            self,
            "image_width",
            _positive_int(self.image_width, field_name="image_width"),
        )
        object.__setattr__(
            self,
            "spatial_compression",
            _positive_int(self.spatial_compression, field_name="spatial_compression"),
        )
        object.__setattr__(
            self,
            "latent_scaling_factor",
            _positive_float(self.latent_scaling_factor, field_name="latent_scaling_factor"),
        )
        max_text_length = _positive_int(self.max_text_length, field_name="max_text_length")
        if max_text_length < 2:
            raise ValueError("max_text_length must be at least two")
        object.__setattr__(self, "max_text_length", max_text_length)

    def to_dict(self) -> dict[str, object]:
        return {
            "media_sha256": self.media_sha256,
            "prompt_sha256": self.prompt_sha256,
            "model_recipe_digest": self.model_recipe_digest,
            "codec_digest": self.codec_digest,
            "conditioner_digest": self.conditioner_digest,
            "tokenizer_digest": self.tokenizer_digest,
            "safety_audit_digest": self.safety_audit_digest,
            "pixel_transform_digest": self.pixel_transform_digest,
            "prompt_enhancement_digest": self.prompt_enhancement_digest,
            "image_height": self.image_height,
            "image_width": self.image_width,
            "spatial_compression": self.spatial_compression,
            "latent_scaling_factor": self.latent_scaling_factor,
            "max_text_length": self.max_text_length,
        }

    @classmethod
    def from_mapping(cls, value: object) -> SanaCacheProvenance:
        fields = set(cls.__dataclass_fields__)
        return cls(
            **_strict_mapping(
                value,
                field_name="cache provenance",
                allowed=fields,
                required=fields,
            )
        )


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
        shape = payload["shape"]
        if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes, bytearray)):
            raise TypeError("tensor descriptor shape must be a sequence")
        return cls(dtype=payload["dtype"], shape=tuple(shape), layout=payload["layout"])


def _descriptor(name: str, tensor: torch.Tensor) -> CacheTensorDescriptor:
    return CacheTensorDescriptor(
        dtype=str(tensor.dtype).removeprefix("torch."),
        shape=tuple(int(size) for size in tensor.shape),
        layout=_LAYOUTS[name],
    )


def _validate_tensors(
    tensors: Mapping[str, torch.Tensor],
    provenance: SanaCacheProvenance,
) -> Mapping[str, CacheTensorDescriptor]:
    keys = set(tensors)
    unknown = sorted(keys - _OBJECT_KEYS)
    missing = sorted(_REQUIRED_OBJECT_KEYS - keys)
    if unknown:
        raise ValueError(f"SANA cache contains unsupported tensor keys: {unknown}")
    if missing:
        raise ValueError(f"SANA cache is missing tensor keys: {missing}")

    normalized: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"cached {name} must be a torch.Tensor")
        if tensor.device.type == "meta" or tensor.layout is not torch.strided:
            raise ValueError(f"cached {name} must be a dense materialized tensor")
        if tensor.is_complex() or tensor.is_quantized:
            raise ValueError(f"cached {name} cannot use complex or quantized storage")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"cached {name} contains NaN or infinity")
        normalized[name] = tensor

    latents = normalized["clean_latents"]
    context = normalized["context"]
    context_mask = normalized["context_mask"]
    if latents.ndim != 3:
        raise ValueError(f"clean_latents must be unbatched CHW; got {tuple(latents.shape)}")
    if not latents.is_floating_point():
        raise TypeError("clean_latents must use a floating dtype")
    expected_height = provenance.image_height // provenance.spatial_compression
    expected_width = provenance.image_width // provenance.spatial_compression
    if provenance.image_height % provenance.spatial_compression:
        raise ValueError("image_height must be divisible by spatial_compression")
    if provenance.image_width % provenance.spatial_compression:
        raise ValueError("image_width must be divisible by spatial_compression")
    if tuple(latents.shape[-2:]) != (expected_height, expected_width):
        raise ValueError(
            "latent spatial shape does not match image dimensions and compression: "
            f"{tuple(latents.shape[-2:])} vs {(expected_height, expected_width)}"
        )
    if context.ndim != 3 or int(context.shape[0]) != 1:
        raise ValueError(f"context must be unbatched [1,L,C]; got {tuple(context.shape)}")
    if not context.is_floating_point():
        raise TypeError("context must use a floating dtype")
    if int(context.shape[1]) != provenance.max_text_length:
        raise ValueError("context sequence length does not match max_text_length")
    if context_mask.ndim != 1 or int(context_mask.shape[0]) != provenance.max_text_length:
        raise ValueError("context_mask must be [max_text_length]")
    if context_mask.dtype not in {torch.bool, torch.int8, torch.int16, torch.int32, torch.int64}:
        raise TypeError("context_mask must use a bool or integer dtype")
    if not bool(((context_mask == 0) | (context_mask == 1)).all()):
        raise ValueError("context_mask must contain only zero and one")

    loss_mask = normalized.get("latent_loss_mask")
    if loss_mask is not None:
        if loss_mask.ndim != 3 or tuple(loss_mask.shape[-2:]) != tuple(latents.shape[-2:]):
            raise ValueError("latent_loss_mask must be unbatched [C,H,W] and match latents")
        if int(loss_mask.shape[0]) not in {1, int(latents.shape[0])}:
            raise ValueError("latent_loss_mask channels must be one or match clean_latents")
        if not loss_mask.is_floating_point():
            raise TypeError("latent_loss_mask must use a floating dtype")
        if not bool((loss_mask >= 0).all()):
            raise ValueError("latent_loss_mask cannot contain negative weights")

    sample_weight = normalized.get("sample_weight")
    if sample_weight is not None:
        if sample_weight.ndim != 0 or not sample_weight.is_floating_point():
            raise ValueError("sample_weight must be one floating scalar")
        if not bool(sample_weight >= 0):
            raise ValueError("sample_weight cannot be negative")

    return MappingProxyType({name: _descriptor(name, tensor) for name, tensor in normalized.items()})


def _identity_payload(
    provenance: SanaCacheProvenance,
    descriptors: Mapping[str, CacheTensorDescriptor],
) -> dict[str, object]:
    return {
        "schema": SANA_CACHE_OBJECT_SCHEMA,
        "provenance": provenance.to_dict(),
        "tensors": {name: descriptors[name].to_dict() for name in sorted(descriptors)},
    }


def _object_metadata(identity_payload: Mapping[str, object], identity_sha256: str) -> dict[str, str]:
    # Safetensors metadata is internally represented as a hash map.  Keeping a
    # single envelope key makes byte serialization deterministic across writes.
    return {
        "worldfoundry": _canonical_json(
            {
                "schema": SANA_CACHE_OBJECT_SCHEMA,
                "identity_sha256": identity_sha256,
                "identity": identity_payload,
            }
        )
    }


@dataclass(frozen=True, slots=True)
class SanaCacheEntry:
    sample_id: str
    identity_sha256: str
    object_sha256: str
    object_size_bytes: int
    object_path: str
    provenance: SanaCacheProvenance
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
        if not isinstance(self.provenance, SanaCacheProvenance):
            raise TypeError("provenance must be SanaCacheProvenance")
        descriptors = dict(self.tensors)
        if not _REQUIRED_OBJECT_KEYS <= set(descriptors) or set(descriptors) - _OBJECT_KEYS:
            raise ValueError("cache entry has invalid tensor descriptor keys")
        if not all(isinstance(value, CacheTensorDescriptor) for value in descriptors.values()):
            raise TypeError("cache entry tensors must contain CacheTensorDescriptor values")
        expected_identity = canonical_sha256(_identity_payload(self.provenance, descriptors))
        if identity != expected_identity:
            raise ValueError("cache entry logical identity digest does not match its metadata")
        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "identity_sha256", identity)
        object.__setattr__(self, "object_sha256", object_digest)
        object.__setattr__(self, "object_size_bytes", size)
        object.__setattr__(self, "object_path", object_path)
        object.__setattr__(self, "tensors", MappingProxyType(descriptors))

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
    def from_mapping(cls, value: object) -> SanaCacheEntry:
        fields = set(cls.__dataclass_fields__)
        payload = _strict_mapping(
            value,
            field_name="cache entry",
            allowed=fields,
            required=fields,
        )
        raw_tensors = payload.pop("tensors")
        raw_provenance = payload.pop("provenance")
        if not isinstance(raw_tensors, Mapping):
            raise TypeError("cache entry tensors must be a mapping")
        return cls(
            **payload,
            provenance=SanaCacheProvenance.from_mapping(raw_provenance),
            tensors={
                str(name): CacheTensorDescriptor.from_mapping(descriptor) for name, descriptor in raw_tensors.items()
            },
        )


@dataclass(frozen=True, slots=True)
class SanaCacheIndex:
    dataset_digest: str
    entries: tuple[SanaCacheEntry, ...]
    index_sha256: str
    schema: str = SANA_CACHE_INDEX_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SANA_CACHE_INDEX_SCHEMA:
            raise ValueError(f"unsupported SANA cache index schema: {self.schema!r}")
        dataset_digest = _sha256(self.dataset_digest, field_name="dataset_digest")
        entries = tuple(self.entries)
        if not entries or not all(isinstance(entry, SanaCacheEntry) for entry in entries):
            raise ValueError("cache index entries must be a non-empty sequence of SanaCacheEntry")
        sample_ids = tuple(entry.sample_id for entry in entries)
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("cache index sample_ids must be unique")
        payload = {
            "schema": self.schema,
            "dataset_digest": dataset_digest,
            "entries": [entry.to_dict() for entry in entries],
        }
        expected = canonical_sha256(payload)
        index_digest = _sha256(self.index_sha256, field_name="index_sha256")
        if index_digest != expected:
            raise ValueError("cache index digest does not match its contents")
        object.__setattr__(self, "dataset_digest", dataset_digest)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "index_sha256", index_digest)

    @classmethod
    def build(cls, dataset_digest: str, entries: Sequence[SanaCacheEntry]) -> SanaCacheIndex:
        resolved_entries = tuple(entries)
        payload = {
            "schema": SANA_CACHE_INDEX_SCHEMA,
            "dataset_digest": _sha256(dataset_digest, field_name="dataset_digest"),
            "entries": [entry.to_dict() for entry in resolved_entries],
        }
        return cls(
            dataset_digest=payload["dataset_digest"],
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
    def from_mapping(cls, value: object) -> SanaCacheIndex:
        payload = _strict_mapping(
            value,
            field_name="cache index",
            allowed={"schema", "dataset_digest", "entries", "index_sha256"},
            required={"schema", "dataset_digest", "entries", "index_sha256"},
        )
        raw_entries = payload.pop("entries")
        if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes, bytearray)):
            raise TypeError("cache index entries must be a sequence")
        return cls(entries=tuple(SanaCacheEntry.from_mapping(item) for item in raw_entries), **payload)


@dataclass(frozen=True, slots=True)
class SanaCachedSample:
    entry: SanaCacheEntry
    tensors: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        if not isinstance(self.entry, SanaCacheEntry):
            raise TypeError("entry must be SanaCacheEntry")
        descriptors = _validate_tensors(self.tensors, self.entry.provenance)
        if dict(descriptors) != dict(self.entry.tensors):
            raise ValueError("loaded tensor descriptors do not match the cache index")
        object.__setattr__(self, "tensors", MappingProxyType(dict(self.tensors)))


class SanaCacheStore:
    """Write and verify immutable content-addressed cache objects."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.objects = self.root / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)

    def _resolved_object_path(self, entry: SanaCacheEntry) -> Path:
        path = self.root / entry.object_path
        resolved_parent = path.parent.resolve()
        if self.root != resolved_parent and self.root not in resolved_parent.parents:
            raise ValueError("cache object path escapes the cache root")
        if path.is_symlink():
            raise ValueError(f"cache objects cannot be symlinks: {path}")
        return path

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
        tensors = {
            "clean_latents": clean_latents,
            "context": context,
            "context_mask": context_mask,
        }
        if latent_loss_mask is not None:
            tensors["latent_loss_mask"] = latent_loss_mask
        if sample_weight is not None:
            tensors["sample_weight"] = sample_weight
        descriptors = _validate_tensors(tensors, provenance)
        identity_payload = _identity_payload(provenance, descriptors)
        identity_sha256 = canonical_sha256(identity_payload)
        storage = {name: tensor.detach().to(device="cpu").contiguous() for name, tensor in tensors.items()}
        metadata = _object_metadata(identity_payload, identity_sha256)
        try:
            from safetensors.torch import save_file
        except ModuleNotFoundError as error:
            raise RuntimeError("SANA training cache requires safetensors") from error

        temporary_handle = tempfile.NamedTemporaryFile(
            prefix=".sana-cache-",
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
                    raise ValueError(f"cache objects cannot be symlinks: {destination}")
                if destination.stat().st_size != object_size or file_sha256(destination) != object_digest:
                    raise ValueError(f"existing content-addressed cache object is corrupt: {destination}")
            else:
                try:
                    os.link(temporary, destination)
                except FileExistsError:
                    if destination.stat().st_size != object_size or file_sha256(destination) != object_digest:
                        raise ValueError(f"racing cache writer produced a corrupt object: {destination}")
                sync_directory(destination_dir)
            return SanaCacheEntry(
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

    def audit_entry(self, entry: SanaCacheEntry, *, load_tensors: bool = True) -> SanaCachedSample | None:
        path = self._resolved_object_path(entry)
        if not path.is_file():
            raise FileNotFoundError(f"cache object not found: {path}")
        actual_size = path.stat().st_size
        if actual_size != entry.object_size_bytes:
            raise ValueError(
                f"cache object size mismatch for {entry.sample_id!r}: "
                f"expected {entry.object_size_bytes}, got {actual_size}"
            )
        actual_digest = file_sha256(path)
        if actual_digest != entry.object_sha256:
            raise ValueError(
                f"cache object SHA-256 mismatch for {entry.sample_id!r}: "
                f"expected {entry.object_sha256}, got {actual_digest}"
            )
        try:
            from safetensors import safe_open
        except ModuleNotFoundError as error:
            raise RuntimeError("SANA training cache requires safetensors") from error
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            expected_identity = _identity_payload(entry.provenance, entry.tensors)
            if metadata != _object_metadata(expected_identity, entry.identity_sha256):
                raise ValueError(f"cache object metadata mismatch for {entry.sample_id!r}")
            keys = set(handle.keys())
            if keys != set(entry.tensors):
                raise ValueError(f"cache object tensor keys mismatch for {entry.sample_id!r}")
            if not load_tensors:
                for name, descriptor in entry.tensors.items():
                    tensor = handle.get_slice(name)
                    if tuple(tensor.get_shape()) != descriptor.shape:
                        raise ValueError(f"cache object shape mismatch for {name!r}")
                return None
            tensors = {name: handle.get_tensor(name) for name in sorted(keys)}
        return SanaCachedSample(entry=entry, tensors=tensors)

    def write_index(
        self,
        *,
        dataset_digest: str,
        entries: Sequence[SanaCacheEntry],
        filename: str = "index.json",
    ) -> SanaCacheIndex:
        if Path(filename).name != filename or filename in {".", ".."}:
            raise ValueError("cache index filename must be one plain filename")
        index = SanaCacheIndex.build(dataset_digest, entries)
        destination = self.root / filename
        replace_json_atomic(destination, index.to_dict(), root=self.root)
        return index

    def read_index(self, filename: str = "index.json") -> SanaCacheIndex:
        if Path(filename).name != filename or filename in {".", ".."}:
            raise ValueError("cache index filename must be one plain filename")
        path = self.root / filename
        if path.is_symlink():
            raise ValueError("cache index cannot be a symlink")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return SanaCacheIndex.from_mapping(payload)

    def write_prepared_batch(
        self,
        prepared: PreparedBatch,
        provenances: Sequence[SanaCacheProvenance],
    ) -> tuple[SanaCacheEntry, ...]:
        """Split a SANA PreparedBatch into immutable per-sample objects."""

        if not isinstance(prepared, PreparedBatch):
            raise TypeError("prepared must be PreparedBatch")
        if len(provenances) != prepared.batch_size:
            raise ValueError("one cache provenance is required per prepared sample")
        if not isinstance(prepared.clean_latents, torch.Tensor):
            raise TypeError("SANA prepared clean_latents must be one tensor")
        context = prepared.conditioning.get("context")
        context_mask = prepared.conditioning.get("context_mask")
        if not isinstance(context, torch.Tensor) or not isinstance(context_mask, torch.Tensor):
            raise TypeError("SANA prepared conditioning requires context and context_mask tensors")
        loss_mask = prepared.loss_mask
        if loss_mask is not None and not isinstance(loss_mask, torch.Tensor):
            raise TypeError("SANA prepared loss_mask must be one tensor")
        weights = prepared.sample_weights
        if weights is not None and not isinstance(weights, torch.Tensor):
            raise TypeError("SANA prepared sample_weights must be one tensor")

        entries: list[SanaCacheEntry] = []
        for index, (sample_id, provenance) in enumerate(zip(prepared.sample_ids, provenances)):
            entries.append(
                self.write_sample(
                    sample_id=sample_id,
                    provenance=provenance,
                    clean_latents=prepared.clean_latents[index],
                    context=context[index],
                    context_mask=context_mask[index],
                    latent_loss_mask=None if loss_mask is None else loss_mask[index],
                    sample_weight=None if weights is None else weights[index],
                )
            )
        return tuple(entries)


class SanaCachedDataset(Sequence[SanaCachedSample]):
    """Map-style dataset over a strictly audited SANA cache index."""

    def __init__(
        self,
        root: str | Path,
        *,
        index_filename: str = "index.json",
        expected_dataset_digest: str | None = None,
        audit_on_open: bool = True,
        verify_on_read: bool = True,
    ) -> None:
        self.store = SanaCacheStore(root)
        self.index = self.store.read_index(index_filename)
        if expected_dataset_digest is not None:
            expected = _sha256(expected_dataset_digest, field_name="expected_dataset_digest")
            if self.index.dataset_digest != expected:
                raise ValueError(f"cache dataset digest mismatch: expected {expected}, got {self.index.dataset_digest}")
        self._entries = self.index.entries
        self._verify_on_read = bool(verify_on_read)
        if audit_on_open:
            for entry in self._entries:
                self.store.audit_entry(entry, load_tensors=False)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(entry.sample_id for entry in self._entries)

    @property
    def dataset_digest(self) -> str:
        return self.index.dataset_digest

    @property
    def index_sha256(self) -> str:
        return self.index.index_sha256

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
        # Safetensors still validates its header and bounds.  This mode is only
        # appropriate after an audit on an immutable/read-only cache mount.
        try:
            from safetensors.torch import load_file
        except ModuleNotFoundError as error:
            raise RuntimeError("SANA training cache requires safetensors") from error
        path = self.store._resolved_object_path(entry)
        return SanaCachedSample(entry=entry, tensors=load_file(path, device="cpu"))

    def __iter__(self) -> Iterator[SanaCachedSample]:
        for index in range(len(self)):
            yield self[index]


def collate_sana_cached_samples(samples: Sequence[SanaCachedSample]) -> TrainingBatch:
    """Stack one shape-compatible SANA bucket into the public TrainingBatch."""

    values = tuple(samples)
    if not values:
        raise ValueError("cannot collate an empty SANA cache batch")
    if not all(isinstance(sample, SanaCachedSample) for sample in values):
        raise TypeError("all cached samples must be SanaCachedSample")

    reference = values[0]
    tensor_keys = set(reference.tensors)
    provenance = reference.entry.provenance
    for sample in values[1:]:
        if set(sample.tensors) != tensor_keys:
            raise ValueError("cached samples in one batch must expose identical tensor keys")
        if dict(sample.entry.tensors) != dict(reference.entry.tensors):
            raise ValueError("cached samples in one batch must share tensor shapes, dtypes, and layouts")
        current = sample.entry.provenance
        if (
            current.image_height,
            current.image_width,
            current.spatial_compression,
            current.latent_scaling_factor,
            current.max_text_length,
            current.model_recipe_digest,
            current.codec_digest,
            current.conditioner_digest,
            current.tokenizer_digest,
            current.pixel_transform_digest,
            current.prompt_enhancement_digest,
        ) != (
            provenance.image_height,
            provenance.image_width,
            provenance.spatial_compression,
            provenance.latent_scaling_factor,
            provenance.max_text_length,
            provenance.model_recipe_digest,
            provenance.codec_digest,
            provenance.conditioner_digest,
            provenance.tokenizer_digest,
            provenance.pixel_transform_digest,
            provenance.prompt_enhancement_digest,
        ):
            raise ValueError("cached samples in one batch belong to incompatible preprocessing buckets")

    stacked = {name: torch.stack([sample.tensors[name] for sample in values]) for name in sorted(tensor_keys)}
    conditions: dict[str, torch.Tensor] = {
        "clean_latents": stacked["clean_latents"],
        "context": stacked["context"],
        "context_mask": stacked["context_mask"],
    }
    if "latent_loss_mask" in stacked:
        conditions["latent_loss_mask"] = stacked["latent_loss_mask"]
    sample_weights = stacked.get("sample_weight")
    return TrainingBatch(
        sample_ids=tuple(sample.entry.sample_id for sample in values),
        prompts=tuple(f"sha256:{sample.entry.provenance.prompt_sha256}" for sample in values),
        conditions=conditions,
        sample_weights=sample_weights,
        metadata={
            "cache_schema": SANA_CACHE_OBJECT_SCHEMA,
            "cache_identity_sha256": tuple(sample.entry.identity_sha256 for sample in values),
            "cache_object_sha256": tuple(sample.entry.object_sha256 for sample in values),
            "image_height": provenance.image_height,
            "image_width": provenance.image_width,
            "latent_scaling_factor": provenance.latent_scaling_factor,
            "max_text_length": provenance.max_text_length,
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
    "file_sha256",
    "sana_cache_contract_digest",
    "text_sha256",
]

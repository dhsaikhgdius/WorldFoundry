"""Safetensors artifacts for fully tuned native models."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
from pathlib import Path
from types import MappingProxyType

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from worldfoundry.core.io.integrity import sync_directory, write_exclusive_json

FULL_MODEL_ARTIFACT_SCHEMA = "worldfoundry-full-model"
FULL_MODEL_MANIFEST_NAME = "worldfoundry_model.json"
FULL_MODEL_INDEX_NAME = "model.safetensors.index.json"
DEFAULT_MAX_SHARD_SIZE_BYTES = 2 * 1024**3

_SAFETENSORS_DTYPES = {
    "bool": "BOOL",
    "uint8": "U8",
    "int8": "I8",
    "int16": "I16",
    "int32": "I32",
    "int64": "I64",
    "float16": "F16",
    "bfloat16": "BF16",
    "float32": "F32",
    "float64": "F64",
    "complex64": "C64",
}
_DTYPE_SIZE_BYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
    "complex64": 8,
}


@dataclass(frozen=True, slots=True)
class FullModelArtifact:
    """A validated full-model artifact."""

    path: Path
    file_size_bytes: Mapping[str, int]
    tensor_count: int
    tensor_element_count: int
    parameter_count: int
    trainable_parameter_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        sizes = {str(name): int(size) for name, size in self.file_size_bytes.items()}
        if not sizes or any(not name or size < 0 for name, size in sizes.items()):
            raise ValueError("full-model file sizes are invalid")
        object.__setattr__(self, "file_size_bytes", MappingProxyType(sizes))
        for name in (
            "tensor_count",
            "tensor_element_count",
            "parameter_count",
            "trainable_parameter_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 0:
                raise ValueError(f"full-model {name} must be a non-negative integer")


def _positive_shard_size(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("max_shard_size_bytes must be an integer, not bool")
    size = int(value)
    if size <= 0:
        raise ValueError("max_shard_size_bytes must be positive")
    return size


def _dtype_name(tensor: torch.Tensor) -> str:
    name = str(tensor.dtype).removeprefix("torch.")
    if name not in _SAFETENSORS_DTYPES:
        raise TypeError(f"Safetensors full-model export does not support dtype {tensor.dtype}")
    return name


def _state_inventory(state: Mapping[str, object]) -> tuple[dict[str, dict[str, object]], int]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("full-model state_dict must be a non-empty mapping")
    inventory: dict[str, dict[str, object]] = {}
    total_size = 0
    for key in sorted(state):
        if not isinstance(key, str) or not key:
            raise ValueError("full-model state_dict keys must be non-empty strings")
        tensor = state[key]
        if isinstance(tensor, DTensor):
            raise RuntimeError(f"full-model export left a DTensor value for {key!r}")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"full-model state value {key!r} is not a torch.Tensor")
        if tensor.device.type == "meta":
            raise ValueError(f"full-model state value {key!r} is still on the meta device")
        if tensor.layout is not torch.strided:
            raise TypeError(f"full-model state value {key!r} must use strided layout")
        dtype = _dtype_name(tensor)
        numel = int(tensor.numel())
        size_bytes = numel * int(tensor.element_size())
        inventory[key] = {
            "shape": [int(value) for value in tensor.shape],
            "dtype": dtype,
            "numel": numel,
            "size_bytes": size_bytes,
        }
        total_size += size_bytes
    return inventory, total_size


def _partition_keys(
    inventory: Mapping[str, Mapping[str, object]],
    *,
    max_shard_size_bytes: int,
) -> tuple[tuple[str, ...], ...]:
    shards: list[tuple[str, ...]] = []
    active: list[str] = []
    active_size = 0
    for key, descriptor in inventory.items():
        size = int(descriptor["size_bytes"])
        if active and active_size + size > max_shard_size_bytes:
            shards.append(tuple(active))
            active = []
            active_size = 0
        active.append(key)
        active_size += size
        if size >= max_shard_size_bytes:
            shards.append(tuple(active))
            active = []
            active_size = 0
    if active:
        shards.append(tuple(active))
    return tuple(shards)


def _shard_names(count: int) -> tuple[str, ...]:
    if count == 1:
        return ("model.safetensors",)
    return tuple(f"model-{index:05d}-of-{count:05d}.safetensors" for index in range(1, count + 1))


def _sync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def save_full_model(
    model: nn.Module,
    output_dir: str | Path,
    *,
    model_state_dict: Mapping[str, object] | None = None,
    max_shard_size_bytes: int = DEFAULT_MAX_SHARD_SIZE_BYTES,
) -> FullModelArtifact:
    """Atomically export one complete native model."""

    if not isinstance(model, nn.Module):
        raise TypeError("full-model export requires an nn.Module")
    shard_limit = _positive_shard_size(max_shard_size_bytes)
    state = model.state_dict() if model_state_dict is None else model_state_dict
    inventory, total_size = _state_inventory(state)
    shards = _partition_keys(inventory, max_shard_size_bytes=shard_limit)
    shard_names = _shard_names(len(shards))
    weight_map = {key: shard_name for shard_name, keys in zip(shard_names, shards, strict=True) for key in keys}
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"full-model output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}-incomplete-",
            dir=destination.parent,
        )
    )
    try:
        try:
            from safetensors.torch import save_file
        except ModuleNotFoundError as error:
            raise RuntimeError("full-model export requires the train-core Safetensors dependency") from error
        for shard_name, keys in zip(shard_names, shards, strict=True):
            payload = {key: state[key].detach().to(device="cpu").contiguous().clone() for key in keys}
            shard_path = temporary / shard_name
            save_file(payload, str(shard_path), metadata={"format": "pt"})
            _sync_file(shard_path)
            del payload
        if len(shards) > 1:
            write_exclusive_json(
                temporary / FULL_MODEL_INDEX_NAME,
                {
                    "metadata": {"total_size": total_size},
                    "weight_map": weight_map,
                },
                root=temporary,
            )
        payload_files: dict[str, int] = {}
        for candidate in sorted(temporary.iterdir()):
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError(f"full-model staging entry is invalid: {candidate}")
            payload_files[candidate.name] = candidate.stat().st_size
        parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
        trainable_count = sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)
        manifest = {
            "schema": FULL_MODEL_ARTIFACT_SCHEMA,
            "format": "safetensors",
            "tensor_count": len(inventory),
            "tensor_element_count": sum(int(value["numel"]) for value in inventory.values()),
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_count,
            "total_size_bytes": total_size,
            "tensors": inventory,
            "weight_map": weight_map,
            "files": payload_files,
        }
        write_exclusive_json(
            temporary / FULL_MODEL_MANIFEST_NAME,
            manifest,
            root=temporary,
        )
        sync_directory(temporary)
        os.replace(temporary, destination)
        sync_directory(destination.parent)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return inspect_full_model(destination)


def _safe_relative_file(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty relative filename")
    path = Path(value)
    if path.is_absolute() or len(path.parts) != 1 or path.name != value:
        raise ValueError(f"unsafe {field_name}: {value!r}")
    return value


def inspect_full_model(input_dir: str | Path) -> FullModelArtifact:
    """Validate the manifest, payload sizes, and tensor headers."""

    source = Path(input_dir).expanduser().resolve()
    if not source.is_dir() or source.is_symlink():
        raise FileNotFoundError(f"full-model artifact directory does not exist: {source}")
    for candidate in source.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"full-model artifacts cannot contain symlinks: {candidate}")
    manifest_path = source / FULL_MODEL_MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid full-model manifest: {manifest_path}") from error
    expected_fields = {
        "schema",
        "format",
        "tensor_count",
        "tensor_element_count",
        "parameter_count",
        "trainable_parameter_count",
        "total_size_bytes",
        "tensors",
        "weight_map",
        "files",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_fields:
        raise ValueError("full-model manifest fields differ from the active schema")
    if manifest["schema"] != FULL_MODEL_ARTIFACT_SCHEMA or manifest["format"] != "safetensors":
        raise ValueError("unsupported full-model artifact schema or format")
    tensors = manifest["tensors"]
    weight_map = manifest["weight_map"]
    files = manifest["files"]
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("full-model tensor inventory must be a non-empty object")
    if not isinstance(weight_map, dict) or set(weight_map) != set(tensors):
        raise ValueError("full-model weight map differs from the tensor inventory")
    if not isinstance(files, dict) or not files:
        raise ValueError("full-model file inventory must be a non-empty object")

    tensor_count = int(manifest["tensor_count"])
    tensor_elements = int(manifest["tensor_element_count"])
    parameter_count = int(manifest["parameter_count"])
    trainable_count = int(manifest["trainable_parameter_count"])
    total_size = int(manifest["total_size_bytes"])
    if (
        tensor_count != len(tensors)
        or min(tensor_elements, parameter_count, trainable_count, total_size) < 0
        or trainable_count > parameter_count
    ):
        raise ValueError("full-model aggregate counts are invalid")

    normalized_tensors: dict[str, dict[str, object]] = {}
    calculated_elements = 0
    calculated_size = 0
    for key, descriptor in tensors.items():
        if not isinstance(key, str) or not key or not isinstance(descriptor, dict):
            raise ValueError("full-model tensor descriptor is invalid")
        if set(descriptor) != {"shape", "dtype", "numel", "size_bytes"}:
            raise ValueError(f"full-model tensor descriptor fields differ for {key!r}")
        shape = descriptor["shape"]
        dtype = descriptor["dtype"]
        if (
            not isinstance(shape, list)
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape)
            or dtype not in _SAFETENSORS_DTYPES
        ):
            raise ValueError(f"full-model tensor shape or dtype is invalid for {key!r}")
        numel = int(descriptor["numel"])
        size_bytes = int(descriptor["size_bytes"])
        expected_numel = prod(shape)
        if numel != expected_numel or size_bytes != numel * _DTYPE_SIZE_BYTES[str(dtype)]:
            raise ValueError(f"full-model tensor counts are invalid for {key!r}")
        normalized_tensors[key] = {
            "shape": shape,
            "dtype": dtype,
            "numel": numel,
            "size_bytes": size_bytes,
        }
        calculated_elements += numel
        calculated_size += size_bytes
    if calculated_elements != tensor_elements or calculated_size != total_size:
        raise ValueError("full-model aggregate tensor inventory is inconsistent")

    normalized_files: dict[str, int] = {}
    for raw_name, raw_size in files.items():
        name = _safe_relative_file(raw_name, field_name="full-model payload filename")
        if isinstance(raw_size, bool) or not isinstance(raw_size, int):
            raise ValueError(f"full-model file size is invalid for {name!r}")
        size = raw_size
        candidate = source / name
        if (
            size < 0
            or not candidate.is_file()
            or candidate.is_symlink()
            or candidate.stat().st_size != size
        ):
            raise ValueError(f"full-model payload verification failed for {name!r}")
        normalized_files[name] = size
    actual_files = {
        candidate.name
        for candidate in source.iterdir()
        if candidate.is_file() and candidate.name != FULL_MODEL_MANIFEST_NAME
    }
    if actual_files != set(normalized_files):
        raise ValueError("full-model payload file set differs from the manifest")

    shard_names = set()
    for key, raw_name in weight_map.items():
        name = _safe_relative_file(raw_name, field_name=f"weight-map filename for {key!r}")
        if name not in normalized_files or not name.endswith(".safetensors"):
            raise ValueError(f"full-model weight map points to an invalid shard for {key!r}")
        shard_names.add(name)
    expected_index = len(shard_names) > 1
    if (FULL_MODEL_INDEX_NAME in normalized_files) != expected_index:
        raise ValueError("full-model shard index presence is invalid")
    expected_payload = shard_names | ({FULL_MODEL_INDEX_NAME} if expected_index else set())
    if set(normalized_files) != expected_payload:
        raise ValueError("full-model payload inventory contains files outside the weight map")
    if expected_index:
        try:
            index = json.loads((source / FULL_MODEL_INDEX_NAME).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("full-model Safetensors index is invalid") from error
        if index != {"metadata": {"total_size": total_size}, "weight_map": weight_map}:
            raise ValueError("full-model Safetensors index differs from the manifest")

    try:
        from safetensors import safe_open
    except ModuleNotFoundError as error:
        raise RuntimeError("full-model inspection requires the train-core Safetensors dependency") from error
    discovered: set[str] = set()
    for shard_name in sorted(shard_names):
        expected_keys = {key for key, value in weight_map.items() if value == shard_name}
        with safe_open(str(source / shard_name), framework="pt", device="cpu") as handle:
            actual_keys = set(handle.keys())
            if actual_keys != expected_keys:
                raise ValueError(f"full-model shard key inventory differs for {shard_name!r}")
            for key in actual_keys:
                tensor_slice = handle.get_slice(key)
                descriptor = normalized_tensors[key]
                if (
                    list(tensor_slice.get_shape()) != descriptor["shape"]
                    or tensor_slice.get_dtype() != _SAFETENSORS_DTYPES[str(descriptor["dtype"])]
                ):
                    raise ValueError(f"full-model tensor header differs for {key!r}")
            discovered.update(actual_keys)
    if discovered != set(tensors):
        raise ValueError("full-model shards do not cover the complete tensor inventory")
    return FullModelArtifact(
        path=source,
        file_size_bytes=normalized_files,
        tensor_count=tensor_count,
        tensor_element_count=tensor_elements,
        parameter_count=parameter_count,
        trainable_parameter_count=trainable_count,
    )


def load_full_model(model: nn.Module, input_dir: str | Path) -> FullModelArtifact:
    """Strictly restore a full artifact into a native model."""

    if not isinstance(model, nn.Module):
        raise TypeError("full-model load requires an nn.Module")
    artifact = inspect_full_model(input_dir)
    manifest = json.loads((artifact.path / FULL_MODEL_MANIFEST_NAME).read_text(encoding="utf-8"))
    active = model.state_dict()
    tensors = manifest["tensors"]
    if set(active) != set(tensors):
        raise ValueError("full-model artifact tensor keys differ from the target model")
    for key, tensor in active.items():
        descriptor = tensors[key]
        if list(tensor.shape) != descriptor["shape"] or str(tensor.dtype).removeprefix("torch.") != descriptor["dtype"]:
            raise ValueError(f"full-model artifact tensor contract differs for {key!r}")
    try:
        from safetensors.torch import load_file
    except ModuleNotFoundError as error:
        raise RuntimeError("full-model load requires the train-core Safetensors dependency") from error
    loaded: dict[str, torch.Tensor] = {}
    for shard_name in sorted(set(manifest["weight_map"].values())):
        shard = load_file(str(artifact.path / shard_name), device="cpu")
        overlap = set(loaded) & set(shard)
        if overlap:
            raise ValueError(f"full-model shards contain duplicate tensors: {sorted(overlap)}")
        loaded.update(shard)
    result = model.load_state_dict(loaded, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("strict full-model load returned incompatible keys")
    return artifact


__all__ = [
    "DEFAULT_MAX_SHARD_SIZE_BYTES",
    "FULL_MODEL_ARTIFACT_SCHEMA",
    "FULL_MODEL_INDEX_NAME",
    "FULL_MODEL_MANIFEST_NAME",
    "FullModelArtifact",
    "inspect_full_model",
    "load_full_model",
    "save_full_model",
]

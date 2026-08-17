"""Shared conditioning tensors stored once per cache branch."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, overload

import torch

from worldfoundry.core.io.integrity import sync_directory, write_exclusive_json

from .sana_cache import CacheTensorDescriptor

SHARED_CONDITIONING_OBJECT_SCHEMA = "worldfoundry-shared-training-conditioning-object"
SHARED_CONDITIONING_MANIFEST_SCHEMA = "worldfoundry-shared-training-conditioning-manifest"
_IDENTIFIER_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]*")
_TENSOR_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _identifier(value: object, *, field_name: str) -> str:
    resolved = str(value).strip().lower().replace("_", "-")
    if _IDENTIFIER_PATTERN.fullmatch(resolved) is None:
        raise ValueError(f"{field_name} contains unsupported characters: {value!r}")
    return resolved


def _text(value: object, *, field_name: str) -> str:
    resolved = str(value).strip()
    if not resolved and field_name != "prompt":
        raise ValueError(f"{field_name} cannot be empty")
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


def _validate_tensor(name: str, tensor: object, *, layout: object) -> CacheTensorDescriptor:
    if _TENSOR_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(f"shared conditioning tensor name is invalid: {name!r}")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"shared conditioning {name!r} must be a torch.Tensor")
    if tensor.ndim == 0 or tensor.numel() == 0:
        raise ValueError(f"shared conditioning {name!r} must be non-empty")
    if tensor.device.type == "meta" or tensor.layout is not torch.strided:
        raise ValueError(f"shared conditioning {name!r} must be materialized and strided")
    resolved_layout = _text(layout, field_name=f"{name}.layout")
    return CacheTensorDescriptor(
        dtype=str(tensor.dtype).removeprefix("torch."),
        shape=tuple(int(size) for size in tensor.shape),
        layout=resolved_layout,
    )


@dataclass(frozen=True, slots=True)
class SharedConditioningIdentity:
    """Inputs that determine one shared conditioning branch."""

    branch: str
    prompt: str
    model_recipe: str
    conditioner: Mapping[str, object]
    tokenizer: Mapping[str, object]
    tensors: Mapping[str, CacheTensorDescriptor]
    schema: str = SHARED_CONDITIONING_OBJECT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SHARED_CONDITIONING_OBJECT_SCHEMA:
            raise ValueError(f"unsupported shared conditioning schema: {self.schema!r}")
        object.__setattr__(self, "branch", _identifier(self.branch, field_name="branch"))
        object.__setattr__(self, "prompt", _text(self.prompt, field_name="prompt"))
        object.__setattr__(self, "model_recipe", _text(self.model_recipe, field_name="model_recipe"))
        object.__setattr__(self, "conditioner", _json_mapping(self.conditioner, field_name="conditioner"))
        object.__setattr__(self, "tokenizer", _json_mapping(self.tokenizer, field_name="tokenizer"))
        tensors = {str(name): descriptor for name, descriptor in self.tensors.items()}
        if not tensors or not all(isinstance(item, CacheTensorDescriptor) for item in tensors.values()):
            raise ValueError("shared conditioning requires tensor descriptors")
        object.__setattr__(self, "tensors", MappingProxyType(tensors))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "branch": self.branch,
            "prompt": self.prompt,
            "model_recipe": self.model_recipe,
            "conditioner": dict(self.conditioner),
            "tokenizer": dict(self.tokenizer),
            "tensors": {name: self.tensors[name].to_dict() for name in sorted(self.tensors)},
        }

    @classmethod
    def from_mapping(cls, value: object) -> SharedConditioningIdentity:
        fields = set(cls.__dataclass_fields__)
        payload = _strict_mapping(value, field_name="shared conditioning identity", allowed=fields, required=fields)
        raw_tensors = payload.pop("tensors")
        if not isinstance(raw_tensors, Mapping):
            raise TypeError("shared conditioning tensor descriptors must be a mapping")
        return cls(
            **payload,
            tensors={
                str(name): CacheTensorDescriptor.from_mapping(descriptor) for name, descriptor in raw_tensors.items()
            },
        )


@dataclass(frozen=True, slots=True)
class SharedConditioningArtifact:
    identity: SharedConditioningIdentity
    object_size_bytes: int
    object_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, SharedConditioningIdentity):
            raise TypeError("identity must be SharedConditioningIdentity")
        if isinstance(self.object_size_bytes, bool) or int(self.object_size_bytes) <= 0:
            raise ValueError("shared conditioning object_size_bytes must be positive")
        expected_path = f"shared-objects/{self.identity.branch}.safetensors"
        object_path = str(self.object_path).replace("\\", "/")
        if object_path != expected_path:
            raise ValueError(f"shared conditioning object_path must be {expected_path!r}")
        object.__setattr__(self, "object_size_bytes", int(self.object_size_bytes))
        object.__setattr__(self, "object_path", object_path)

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_dict(),
            "object_size_bytes": self.object_size_bytes,
            "object_path": self.object_path,
        }

    @classmethod
    def from_mapping(cls, value: object) -> SharedConditioningArtifact:
        fields = set(cls.__dataclass_fields__)
        payload = _strict_mapping(value, field_name="shared conditioning artifact", allowed=fields, required=fields)
        return cls(identity=SharedConditioningIdentity.from_mapping(payload.pop("identity")), **payload)


@dataclass(frozen=True, slots=True)
class SharedConditioningSample:
    artifact: SharedConditioningArtifact
    tensors: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, SharedConditioningArtifact):
            raise TypeError("artifact must be SharedConditioningArtifact")
        tensors = {str(name): tensor for name, tensor in self.tensors.items()}
        if set(tensors) != set(self.artifact.identity.tensors):
            raise ValueError("shared conditioning tensor keys differ from the artifact")
        for name, descriptor in self.artifact.identity.tensors.items():
            if _validate_tensor(name, tensors[name], layout=descriptor.layout) != descriptor:
                raise ValueError(f"shared conditioning tensor descriptor differs for {name!r}")
        object.__setattr__(self, "tensors", MappingProxyType(tensors))


def _serialized_identity(identity: SharedConditioningIdentity) -> str:
    return json.dumps(identity.to_dict(), sort_keys=True, separators=(",", ":"))


def _object_metadata(identity: SharedConditioningIdentity) -> dict[str, str]:
    return {"worldfoundry": _serialized_identity(identity)}


class SharedConditioningStore:
    """Store one safetensors object and manifest for each named branch."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.objects = self.root / "shared-objects"
        self.objects.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _manifest_filename(branch: str) -> str:
        return f"{_identifier(branch, field_name='branch')}-conditioning.json"

    def write(
        self,
        *,
        branch: str,
        prompt: str,
        model_recipe: str,
        conditioner: Mapping[str, object],
        tokenizer: Mapping[str, object],
        tensors: Mapping[str, torch.Tensor],
        layouts: Mapping[str, str],
    ) -> SharedConditioningArtifact:
        values = {str(name): tensor for name, tensor in tensors.items()}
        resolved_layouts = {str(name): layout for name, layout in layouts.items()}
        if not values or set(values) != set(resolved_layouts):
            raise ValueError("shared conditioning layouts must match tensor names")
        identity = SharedConditioningIdentity(
            branch=branch,
            prompt=prompt,
            model_recipe=model_recipe,
            conditioner=conditioner,
            tokenizer=tokenizer,
            tensors={
                name: _validate_tensor(name, tensor, layout=resolved_layouts[name]) for name, tensor in values.items()
            },
        )
        manifest = self.root / self._manifest_filename(identity.branch)
        if manifest.exists():
            existing = self.read(identity.branch)
            if existing.artifact.identity != identity or any(
                not torch.equal(existing.tensors[name], values[name].detach().cpu()) for name in values
            ):
                raise FileExistsError(f"shared conditioning branch already stores different values: {identity.branch}")
            return existing.artifact

        try:
            from safetensors.torch import save_file
        except ModuleNotFoundError as error:
            raise RuntimeError("shared conditioning cache requires safetensors") from error

        destination = self.objects / f"{identity.branch}.safetensors"
        temporary_handle = tempfile.NamedTemporaryFile(
            prefix=f".{identity.branch}-", suffix=".safetensors", dir=self.objects, delete=False
        )
        temporary = Path(temporary_handle.name)
        temporary_handle.close()
        try:
            storage = {name: tensor.detach().cpu().contiguous() for name, tensor in values.items()}
            save_file(storage, temporary, metadata=_object_metadata(identity))
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            sync_directory(self.objects)
            artifact = SharedConditioningArtifact(
                identity=identity,
                object_size_bytes=destination.stat().st_size,
                object_path=f"shared-objects/{identity.branch}.safetensors",
            )
            write_exclusive_json(
                manifest,
                {"schema": SHARED_CONDITIONING_MANIFEST_SCHEMA, "artifact": artifact.to_dict()},
                root=self.root,
            )
            sync_directory(self.root)
            return artifact
        finally:
            temporary.unlink(missing_ok=True)

    @overload
    def read(self, branch: str, *, load_tensors: Literal[False]) -> SharedConditioningArtifact: ...

    @overload
    def read(self, branch: str, *, load_tensors: Literal[True] = True) -> SharedConditioningSample: ...

    def read(
        self,
        branch: str,
        *,
        load_tensors: bool = True,
    ) -> SharedConditioningArtifact | SharedConditioningSample:
        resolved_branch = _identifier(branch, field_name="branch")
        manifest = self.root / self._manifest_filename(resolved_branch)
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid shared conditioning manifest: {manifest}") from error
        envelope = _strict_mapping(
            payload,
            field_name="shared conditioning manifest",
            allowed={"schema", "artifact"},
            required={"schema", "artifact"},
        )
        if envelope["schema"] != SHARED_CONDITIONING_MANIFEST_SCHEMA:
            raise ValueError(f"unsupported shared conditioning manifest schema: {envelope['schema']!r}")
        artifact = SharedConditioningArtifact.from_mapping(envelope["artifact"])
        if artifact.identity.branch != resolved_branch:
            raise ValueError("shared conditioning manifest branch differs from its filename")
        path = self.root / artifact.object_path
        if not path.is_file():
            raise FileNotFoundError(f"shared conditioning object not found: {path}")
        if path.stat().st_size != artifact.object_size_bytes:
            raise ValueError("shared conditioning object size mismatch")
        try:
            from safetensors import safe_open
        except ModuleNotFoundError as error:
            raise RuntimeError("shared conditioning cache requires safetensors") from error
        with safe_open(path, framework="pt", device="cpu") as handle:
            if (handle.metadata() or {}) != _object_metadata(artifact.identity):
                raise ValueError("shared conditioning object metadata mismatch")
            if set(handle.keys()) != set(artifact.identity.tensors):
                raise ValueError("shared conditioning object tensor keys mismatch")
            if not load_tensors:
                for name, descriptor in artifact.identity.tensors.items():
                    if tuple(handle.get_slice(name).get_shape()) != descriptor.shape:
                        raise ValueError(f"shared conditioning object shape mismatch for {name!r}")
                return artifact
            tensors = {name: handle.get_tensor(name) for name in sorted(handle.keys())}
        return SharedConditioningSample(artifact=artifact, tensors=tensors)


__all__ = [
    "SHARED_CONDITIONING_MANIFEST_SCHEMA",
    "SHARED_CONDITIONING_OBJECT_SCHEMA",
    "SharedConditioningArtifact",
    "SharedConditioningIdentity",
    "SharedConditioningSample",
    "SharedConditioningStore",
]

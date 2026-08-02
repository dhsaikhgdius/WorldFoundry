"""Immutable shared conditioning artifacts for native training caches.

Some conditioning branches are identical for every sample in a run.  The
canonical example is the empty/negative prompt embedding used by classifier-
free guidance during DMD.  Storing that multi-megabyte tensor in every cache
object wastes space; recreating it at training time makes resume identity
implicit.  This module stores it once while binding the tensor bytes to the
exact encoder assets and model-facing contract.
"""

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

from worldfoundry.core.io.file_utils import file_sha256
from worldfoundry.core.io.integrity import (
    canonical_json,
    canonical_sha256,
    sync_directory,
    write_exclusive_json,
)

from .sana_cache import CacheTensorDescriptor

SHARED_CONDITIONING_OBJECT_SCHEMA = "worldfoundry-shared-training-conditioning-object"
SHARED_CONDITIONING_MANIFEST_SCHEMA = "worldfoundry-shared-training-conditioning-manifest"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]*")
_TENSOR_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _sha256(value: object, *, field_name: str) -> str:
    resolved = str(value).strip().lower()
    if _SHA256_PATTERN.fullmatch(resolved) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return resolved


def _identifier(value: object, *, field_name: str) -> str:
    resolved = str(value).strip().lower().replace("_", "-")
    if _IDENTIFIER_PATTERN.fullmatch(resolved) is None:
        raise ValueError(f"{field_name} contains unsupported characters: {value!r}")
    return resolved


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
    if tensor.ndim == 0 or any(int(size) <= 0 for size in tensor.shape):
        raise ValueError(f"shared conditioning {name!r} must be a non-empty tensor")
    if tensor.device.type == "meta" or tensor.layout is not torch.strided:
        raise ValueError(f"shared conditioning {name!r} must be dense and materialized")
    if tensor.is_complex() or tensor.is_quantized:
        raise ValueError(f"shared conditioning {name!r} cannot be complex or quantized")
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"shared conditioning {name!r} contains NaN or infinity")
    resolved_layout = str(layout).strip()
    if not resolved_layout:
        raise ValueError(f"shared conditioning {name!r} requires a non-empty layout")
    return CacheTensorDescriptor(
        dtype=str(tensor.dtype).removeprefix("torch."),
        shape=tuple(int(size) for size in tensor.shape),
        layout=resolved_layout,
    )


@dataclass(frozen=True, slots=True)
class SharedConditioningIdentity:
    """Logical identity of one branch shared by every sample in a cache."""

    branch: str
    prompt_sha256: str
    model_recipe_digest: str
    conditioner_digest: str
    tokenizer_digest: str
    tensors: Mapping[str, CacheTensorDescriptor]
    schema: str = SHARED_CONDITIONING_OBJECT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SHARED_CONDITIONING_OBJECT_SCHEMA:
            raise ValueError(f"unsupported shared conditioning schema: {self.schema!r}")
        object.__setattr__(self, "branch", _identifier(self.branch, field_name="branch"))
        for name in (
            "prompt_sha256",
            "model_recipe_digest",
            "conditioner_digest",
            "tokenizer_digest",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), field_name=name))
        tensors = {str(name): descriptor for name, descriptor in self.tensors.items()}
        if not tensors or any(
            _TENSOR_NAME_PATTERN.fullmatch(name) is None or not isinstance(descriptor, CacheTensorDescriptor)
            for name, descriptor in tensors.items()
        ):
            raise ValueError("shared conditioning requires named tensor descriptors")
        object.__setattr__(self, "tensors", MappingProxyType(tensors))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "branch": self.branch,
            "prompt_sha256": self.prompt_sha256,
            "model_recipe_digest": self.model_recipe_digest,
            "conditioner_digest": self.conditioner_digest,
            "tokenizer_digest": self.tokenizer_digest,
            "tensors": {name: self.tensors[name].to_dict() for name in sorted(self.tensors)},
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_mapping(cls, value: object) -> SharedConditioningIdentity:
        fields = set(cls.__dataclass_fields__)
        payload = _strict_mapping(
            value,
            field_name="shared conditioning identity",
            allowed=fields,
            required=fields,
        )
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
    """Content address and logical identity of one shared branch object."""

    identity: SharedConditioningIdentity
    identity_sha256: str
    object_sha256: str
    object_size_bytes: int
    object_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, SharedConditioningIdentity):
            raise TypeError("identity must be SharedConditioningIdentity")
        identity_sha256 = _sha256(self.identity_sha256, field_name="identity_sha256")
        if identity_sha256 != self.identity.digest:
            raise ValueError("shared conditioning identity digest does not match its contents")
        object_sha256 = _sha256(self.object_sha256, field_name="object_sha256")
        if isinstance(self.object_size_bytes, bool) or int(self.object_size_bytes) <= 0:
            raise ValueError("shared conditioning object_size_bytes must be positive")
        expected_path = f"shared-objects/{object_sha256[:2]}/{object_sha256}.safetensors"
        object_path = str(self.object_path).replace("\\", "/")
        if object_path != expected_path:
            raise ValueError(f"shared conditioning object_path must be {expected_path!r}")
        object.__setattr__(self, "identity_sha256", identity_sha256)
        object.__setattr__(self, "object_sha256", object_sha256)
        object.__setattr__(self, "object_size_bytes", int(self.object_size_bytes))
        object.__setattr__(self, "object_path", object_path)

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_dict(),
            "identity_sha256": self.identity_sha256,
            "object_sha256": self.object_sha256,
            "object_size_bytes": self.object_size_bytes,
            "object_path": self.object_path,
        }

    @classmethod
    def from_mapping(cls, value: object) -> SharedConditioningArtifact:
        fields = set(cls.__dataclass_fields__)
        payload = _strict_mapping(
            value,
            field_name="shared conditioning artifact",
            allowed=fields,
            required=fields,
        )
        raw_identity = payload.pop("identity")
        return cls(
            **payload,
            identity=SharedConditioningIdentity.from_mapping(raw_identity),
        )


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
            actual = _validate_tensor(name, tensors[name], layout=descriptor.layout)
            if actual != descriptor:
                raise ValueError(f"shared conditioning tensor descriptor differs for {name!r}")
        object.__setattr__(self, "tensors", MappingProxyType(tensors))


def _object_metadata(identity: SharedConditioningIdentity) -> dict[str, str]:
    return {
        "worldfoundry": canonical_json(
            {
                "schema": SHARED_CONDITIONING_OBJECT_SCHEMA,
                "identity_sha256": identity.digest,
                "identity": identity.to_dict(),
            }
        )
    }


def _manifest_payload(artifact: SharedConditioningArtifact) -> dict[str, object]:
    body = {
        "schema": SHARED_CONDITIONING_MANIFEST_SCHEMA,
        "artifact": artifact.to_dict(),
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


class SharedConditioningStore:
    """Write and audit one-copy conditioning objects below a cache root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.objects = self.root / "shared-objects"
        self.objects.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _manifest_filename(branch: str) -> str:
        return f"{_identifier(branch, field_name='branch')}-conditioning.json"

    def _object_file(self, artifact: SharedConditioningArtifact) -> Path:
        path = self.root / artifact.object_path
        resolved_parent = path.parent.resolve()
        if self.root != resolved_parent and self.root not in resolved_parent.parents:
            raise ValueError("shared conditioning object path escapes the cache root")
        if path.is_symlink():
            raise ValueError(f"shared conditioning objects cannot be symlinks: {path}")
        return path

    def write(
        self,
        *,
        branch: str,
        prompt_sha256: str,
        model_recipe_digest: str,
        conditioner_digest: str,
        tokenizer_digest: str,
        tensors: Mapping[str, torch.Tensor],
        layouts: Mapping[str, str],
    ) -> SharedConditioningArtifact:
        values = {str(name): tensor for name, tensor in tensors.items()}
        resolved_layouts = {str(name): layout for name, layout in layouts.items()}
        if not values or set(values) != set(resolved_layouts):
            raise ValueError("shared conditioning layouts must exactly match tensor names")
        descriptors = {
            name: _validate_tensor(name, tensor, layout=resolved_layouts[name]) for name, tensor in values.items()
        }
        identity = SharedConditioningIdentity(
            branch=branch,
            prompt_sha256=prompt_sha256,
            model_recipe_digest=model_recipe_digest,
            conditioner_digest=conditioner_digest,
            tokenizer_digest=tokenizer_digest,
            tensors=descriptors,
        )
        try:
            from safetensors.torch import save_file
        except ModuleNotFoundError as error:
            raise RuntimeError("shared conditioning cache requires safetensors") from error

        temporary_handle = tempfile.NamedTemporaryFile(
            prefix=".shared-conditioning-",
            suffix=".safetensors",
            dir=self.objects,
            delete=False,
        )
        temporary = Path(temporary_handle.name)
        temporary_handle.close()
        try:
            storage = {name: tensor.detach().to(device="cpu").contiguous() for name, tensor in values.items()}
            save_file(storage, temporary, metadata=_object_metadata(identity))
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            object_sha256 = file_sha256(temporary)
            object_size = temporary.stat().st_size
            destination_dir = self.objects / object_sha256[:2]
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / f"{object_sha256}.safetensors"
            if destination.exists():
                if destination.is_symlink() or destination.stat().st_size != object_size:
                    raise ValueError(f"existing shared conditioning object is corrupt: {destination}")
                if file_sha256(destination) != object_sha256:
                    raise ValueError(f"existing shared conditioning object is corrupt: {destination}")
            else:
                try:
                    os.link(temporary, destination)
                except FileExistsError:
                    if (
                        destination.is_symlink()
                        or destination.stat().st_size != object_size
                        or file_sha256(destination) != object_sha256
                    ):
                        raise ValueError("racing shared conditioning writer produced a corrupt object")
                sync_directory(destination_dir)
            artifact = SharedConditioningArtifact(
                identity=identity,
                identity_sha256=identity.digest,
                object_sha256=object_sha256,
                object_size_bytes=object_size,
                object_path=(f"shared-objects/{object_sha256[:2]}/{object_sha256}.safetensors"),
            )
            manifest = self.root / self._manifest_filename(identity.branch)
            if manifest.exists():
                existing = self.read(identity.branch, load_tensors=False)
                if existing != artifact:
                    raise FileExistsError(f"shared conditioning manifest already binds different content: {manifest}")
                return existing
            write_exclusive_json(
                manifest,
                _manifest_payload(artifact),
                root=self.root,
            )
            sync_directory(self.root)
            return artifact
        finally:
            temporary.unlink(missing_ok=True)

    @overload
    def read(
        self,
        branch: str,
        *,
        load_tensors: Literal[False],
    ) -> SharedConditioningArtifact: ...

    @overload
    def read(
        self,
        branch: str,
        *,
        load_tensors: Literal[True] = True,
    ) -> SharedConditioningSample: ...

    def read(
        self,
        branch: str,
        *,
        load_tensors: bool = True,
    ) -> SharedConditioningArtifact | SharedConditioningSample:
        manifest = self.root / self._manifest_filename(branch)
        if manifest.is_symlink():
            raise ValueError("shared conditioning manifest cannot be a symlink")
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid shared conditioning manifest: {manifest}") from error
        envelope = _strict_mapping(
            payload,
            field_name="shared conditioning manifest",
            allowed={"schema", "artifact", "manifest_sha256"},
            required={"schema", "artifact", "manifest_sha256"},
        )
        if envelope["schema"] != SHARED_CONDITIONING_MANIFEST_SCHEMA:
            raise ValueError(f"unsupported shared conditioning manifest schema: {envelope['schema']!r}")
        expected_manifest_sha256 = canonical_sha256({"schema": envelope["schema"], "artifact": envelope["artifact"]})
        if _sha256(envelope["manifest_sha256"], field_name="manifest_sha256") != expected_manifest_sha256:
            raise ValueError("shared conditioning manifest digest does not match its contents")
        artifact = SharedConditioningArtifact.from_mapping(envelope["artifact"])
        if artifact.identity.branch != _identifier(branch, field_name="branch"):
            raise ValueError("shared conditioning manifest branch differs from its filename")
        path = self._object_file(artifact)
        if not path.is_file():
            raise FileNotFoundError(f"shared conditioning object not found: {path}")
        if path.stat().st_size != artifact.object_size_bytes:
            raise ValueError("shared conditioning object size mismatch")
        if file_sha256(path) != artifact.object_sha256:
            raise ValueError("shared conditioning object SHA-256 mismatch")
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
                tensors: dict[str, torch.Tensor] = {}
            else:
                tensors = {name: handle.get_tensor(name) for name in sorted(handle.keys())}
        if not load_tensors:
            return artifact
        return SharedConditioningSample(artifact=artifact, tensors=tensors)


__all__ = [
    "SHARED_CONDITIONING_MANIFEST_SCHEMA",
    "SHARED_CONDITIONING_OBJECT_SCHEMA",
    "SharedConditioningArtifact",
    "SharedConditioningIdentity",
    "SharedConditioningSample",
    "SharedConditioningStore",
]

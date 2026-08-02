"""Sidecar-first loader for the optional WorldFoundry dispatcher library.

Importing this module does not import PyTorch or initialize an accelerator.
The target PyTorch runtime is imported only after the sidecar and library hash
have passed structural validation, and the DSO is loaded only after ABI checks.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = 1
OPERATOR_ABI_VERSION = 1
OPERATOR_SCHEMA_HASH = hashlib.sha256(
    f"worldfoundry_native::_build_info() -> str|abi={OPERATOR_ABI_VERSION}".encode()
).hexdigest()


class NativeKernelPackageError(RuntimeError):
    """Base error raised by the optional native-kernel package."""


class NativeKernelManifestError(NativeKernelPackageError):
    """The build sidecar or its referenced DSO is missing or corrupt."""


class NativeKernelCompatibilityError(NativeKernelPackageError):
    """The installed DSO was built for a different runtime ABI."""


_LOAD_LOCK = threading.RLock()
_LOADED_KEY: tuple[Any, ...] | None = None
_LOADED_MANIFEST: dict[str, Any] | None = None
_POISONED_REASON: str | None = None


def _package_root(root: str | Path | None = None) -> Path:
    return Path(root).resolve() if root is not None else Path(__file__).resolve().parent


def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / "build_manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise NativeKernelManifestError(f"native build sidecar is missing: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NativeKernelManifestError(f"native build sidecar is unreadable: {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise NativeKernelManifestError("native build sidecar must contain a JSON object")
    return payload


def _required_int(manifest: Mapping[str, Any], key: str) -> int:
    value = manifest.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise NativeKernelManifestError(f"native build sidecar field {key!r} must be an integer")
    return value


def _required_string(manifest: Mapping[str, Any], key: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or not value:
        raise NativeKernelManifestError(f"native build sidecar field {key!r} must be a non-empty string")
    return value


def _required_bool(manifest: Mapping[str, Any], key: str) -> bool:
    value = manifest.get(key)
    if not isinstance(value, bool):
        raise NativeKernelManifestError(f"native build sidecar field {key!r} must be boolean")
    return value


def _required_digest(manifest: Mapping[str, Any], key: str, *, length: int = 64) -> str:
    value = _required_string(manifest, key).lower()
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise NativeKernelManifestError(
            f"native build sidecar field {key!r} must contain {length} lowercase hexadecimal characters"
        )
    return value


def _required_string_list(manifest: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = manifest.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise NativeKernelManifestError(
            f"native build sidecar field {key!r} must be a list of non-empty strings"
        )
    return tuple(value)


def _validate_build_metadata(manifest: Mapping[str, Any]) -> None:
    _required_digest(manifest, "build_id", length=24)
    _required_string(manifest, "package_version")
    _required_string(manifest, "source_revision")
    _required_digest(manifest, "source_tree_hash")
    _required_string(manifest, "torch_version")

    torch_cuda = manifest.get("torch_cuda")
    if torch_cuda is not None and (not isinstance(torch_cuda, str) or not torch_cuda):
        raise NativeKernelManifestError("native build sidecar field 'torch_cuda' must be null or a string")

    cxx11_abi = _required_bool(manifest, "cxx11_abi")
    _required_bool(manifest, "links_libtorch_python")

    compiler = manifest.get("compiler")
    if not isinstance(compiler, Mapping):
        raise NativeKernelManifestError("native build sidecar field 'compiler' must be an object")
    _required_string(compiler, "id")
    _required_string(compiler, "version")
    cxx_standard = _required_int(compiler, "cxx_standard")
    if cxx_standard < 17:
        raise NativeKernelCompatibilityError(f"native dispatcher requires C++17 or newer, found C++{cxx_standard}")

    libstdcxx = manifest.get("libstdcxx")
    if not isinstance(libstdcxx, Mapping):
        raise NativeKernelManifestError("native build sidecar field 'libstdcxx' must be an object")
    _required_string(libstdcxx, "path")
    libstdcxx_hash = _required_string(libstdcxx, "sha256")
    if libstdcxx_hash != "unknown":
        _required_digest(libstdcxx, "sha256")

    compile_flags = _required_string_list(manifest, "compile_flags")
    expected_abi_flag = f"_GLIBCXX_USE_CXX11_ABI={int(cxx11_abi)}"
    if f"-std=c++{cxx_standard}" not in compile_flags or expected_abi_flag not in compile_flags:
        raise NativeKernelManifestError("native compile flags disagree with compiler/CXX11 ABI metadata")

    nvcc_version = manifest.get("nvcc_version")
    if nvcc_version is not None and (not isinstance(nvcc_version, str) or not nvcc_version):
        raise NativeKernelManifestError("native build sidecar field 'nvcc_version' must be null or a string")
    for key in ("sm_targets", "sass_targets", "ptx_targets", "capabilities"):
        _required_string_list(manifest, key)


def _library_path(root: Path, manifest: Mapping[str, Any]) -> Path:
    relative = Path(_required_string(manifest, "library"))
    if relative.is_absolute():
        raise NativeKernelManifestError("native DSO path must be relative to the package")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise NativeKernelManifestError("native DSO path escapes the package root") from exc
    if not resolved.is_file():
        raise NativeKernelManifestError(f"native dispatcher DSO is missing: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_files(root: Path) -> tuple[dict[str, Any], Path]:
    manifest = _read_manifest(root)
    schema_version = _required_int(manifest, "manifest_schema_version")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise NativeKernelCompatibilityError(
            f"unsupported native manifest schema {schema_version}; expected {MANIFEST_SCHEMA_VERSION}"
        )
    operator_abi = _required_int(manifest, "operator_abi_version")
    if operator_abi != OPERATOR_ABI_VERSION:
        raise NativeKernelCompatibilityError(
            f"unsupported native operator ABI {operator_abi}; expected {OPERATOR_ABI_VERSION}"
        )
    operator_schema_hash = _required_digest(manifest, "operator_schema_hash")
    if operator_schema_hash != OPERATOR_SCHEMA_HASH:
        raise NativeKernelCompatibilityError(
            f"native operator schema hash {operator_schema_hash!r} != {OPERATOR_SCHEMA_HASH!r}"
        )
    _validate_build_metadata(manifest)
    expected_hash = _required_digest(manifest, "library_sha256")
    library = _library_path(root, manifest)
    actual_hash = _sha256(library)
    if actual_hash != expected_hash:
        raise NativeKernelManifestError(
            f"native dispatcher DSO hash mismatch: expected {expected_hash}, found {actual_hash}"
        )
    if manifest["links_libtorch_python"] is not False:
        expected_python_abi = manifest.get("python_abi")
        if expected_python_abi != sys.implementation.cache_tag:
            raise NativeKernelCompatibilityError(
                f"native wheel requires Python ABI {expected_python_abi!r}, current ABI is "
                f"{sys.implementation.cache_tag!r}"
            )
    return manifest, library


def _import_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise NativeKernelCompatibilityError(f"PyTorch is unavailable: {type(exc).__name__}: {exc}") from exc


def _runtime_cxx11_abi(torch: Any) -> bool:
    helper = getattr(torch, "compiled_with_cxx11_abi", None)
    if callable(helper):
        return bool(helper())
    return bool(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI"))


def _runtime_errors(manifest: Mapping[str, Any], torch: Any) -> list[str]:
    errors: list[str] = []
    expected_torch = _required_string(manifest, "torch_version")
    actual_torch = str(getattr(torch, "__version__", "unknown"))
    if actual_torch != expected_torch:
        errors.append(f"PyTorch build {actual_torch!r} != {expected_torch!r}")

    expected_cuda = manifest.get("torch_cuda")
    actual_cuda = getattr(getattr(torch, "version", None), "cuda", None)
    if actual_cuda != expected_cuda:
        errors.append(f"Torch CUDA {actual_cuda!r} != {expected_cuda!r}")

    expected_abi = _required_bool(manifest, "cxx11_abi")
    actual_abi = _runtime_cxx11_abi(torch)
    if actual_abi != expected_abi:
        errors.append(f"CXX11 ABI {actual_abi!r} != {expected_abi!r}")
    return errors


def _load_identity(manifest: Mapping[str, Any], library: Path) -> tuple[Any, ...]:
    return (
        str(library),
        manifest["library_sha256"],
        manifest["build_id"],
        manifest["source_tree_hash"],
        manifest["manifest_schema_version"],
        manifest["operator_abi_version"],
        manifest["operator_schema_hash"],
        manifest["torch_version"],
        manifest["torch_cuda"],
        manifest["cxx11_abi"],
    )


def _poison(reason: str) -> None:
    global _POISONED_REASON
    _POISONED_REASON = reason


def _raise_poisoned() -> None:
    assert _POISONED_REASON is not None
    raise NativeKernelPackageError(
        "native dispatcher loading previously reached dlopen and failed; "
        f"this process cannot retry safely: {_POISONED_REASON}"
    )


def inspect(*, root: str | Path | None = None) -> dict[str, Any]:
    """Validate the sidecar and DSO hash without importing PyTorch."""

    manifest, _ = _validate_files(_package_root(root))
    return dict(manifest)


def load(*, root: str | Path | None = None) -> dict[str, Any]:
    """Load the ABI-compatible dispatcher DSO exactly once."""

    global _LOADED_KEY, _LOADED_MANIFEST

    package_root = _package_root(root)
    with _LOAD_LOCK:
        if _POISONED_REASON is not None:
            _raise_poisoned()
        manifest, library = _validate_files(package_root)
        key = _load_identity(manifest, library)
        if _LOADED_KEY is not None:
            if _LOADED_KEY != key:
                raise NativeKernelCompatibilityError(
                    f"native dispatcher {_LOADED_KEY!r} is already loaded; refusing incompatible {key!r}"
                )
            assert _LOADED_MANIFEST is not None
            return dict(_LOADED_MANIFEST)
        torch = _import_torch()
        errors = _runtime_errors(manifest, torch)
        if errors:
            raise NativeKernelCompatibilityError("native runtime mismatch: " + "; ".join(errors))

        try:
            torch.ops.load_library(str(library))
            encoded_build_info = torch.ops.worldfoundry_native._build_info()
        except Exception as exc:
            reason = f"failed to load native dispatcher {library}: {type(exc).__name__}: {exc}"
            _poison(reason)
            raise NativeKernelPackageError(reason) from exc

        try:
            build_info = json.loads(str(encoded_build_info))
        except json.JSONDecodeError as exc:
            reason = "native _build_info returned invalid JSON"
            _poison(reason)
            raise NativeKernelPackageError(reason) from exc
        if not isinstance(build_info, Mapping):
            reason = "native _build_info must return a JSON object"
            _poison(reason)
            raise NativeKernelPackageError(reason)
        if build_info.get("build_id") != manifest["build_id"]:
            reason = "native build ID differs from its sidecar"
            _poison(reason)
            raise NativeKernelPackageError(reason)
        if build_info.get("operator_abi_version") != manifest["operator_abi_version"]:
            reason = "native operator ABI differs from its sidecar"
            _poison(reason)
            raise NativeKernelPackageError(reason)

        _LOADED_KEY = key
        _LOADED_MANIFEST = dict(manifest)
        return dict(_LOADED_MANIFEST)


def is_loaded() -> bool:
    """Return whether this process loaded a dispatcher through this loader."""

    with _LOAD_LOCK:
        return _LOADED_KEY is not None


def _reset_for_tests() -> None:
    """Reset Python loader state; this cannot unload a real DSO."""

    global _LOADED_KEY, _LOADED_MANIFEST, _POISONED_REASON
    with _LOAD_LOCK:
        _LOADED_KEY = None
        _LOADED_MANIFEST = None
        _POISONED_REASON = None


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "OPERATOR_ABI_VERSION",
    "OPERATOR_SCHEMA_HASH",
    "NativeKernelCompatibilityError",
    "NativeKernelManifestError",
    "NativeKernelPackageError",
    "inspect",
    "is_loaded",
    "load",
]

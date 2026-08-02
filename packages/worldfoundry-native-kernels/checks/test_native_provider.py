from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from worldfoundry.core.kernels import native_provider

REPO_ROOT = Path(__file__).resolve().parents[3]
NATIVE_PYTHON_ROOT = REPO_ROOT / "packages" / "worldfoundry-native-kernels" / "python"

sys.path.insert(0, str(NATIVE_PYTHON_ROOT))
from worldfoundry_native_kernels import _loader as package_loader  # noqa: E402

BUILD_ID = "1" * 24
SOURCE_HASH = "2" * 64
LIBSTDCXX_HASH = "3" * 64


class _FakeOps:
    def __init__(self, build_id: str, operator_abi_version: int) -> None:
        self.load_count = 0
        self.worldfoundry_native = SimpleNamespace(
            _build_info=lambda: json.dumps(
                {
                    "build_id": build_id,
                    "operator_abi_version": operator_abi_version,
                }
            )
        )

    def load_library(self, _path: str) -> None:
        self.load_count += 1


def _fake_torch(*, build_id: str = BUILD_ID, version: str = "2.7.0+test") -> SimpleNamespace:
    return SimpleNamespace(
        __version__=version,
        version=SimpleNamespace(cuda="12.8"),
        compiled_with_cxx11_abi=lambda: True,
        ops=_FakeOps(build_id, package_loader.OPERATOR_ABI_VERSION),
    )


def _write_package_root(tmp_path: Path, *, torch_version: str = "2.7.0+test") -> Path:
    root = tmp_path / "worldfoundry_native_kernels"
    library = root / "lib" / "_worldfoundry_native_schema.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"dispatcher-test-double")
    manifest = {
        "manifest_schema_version": package_loader.MANIFEST_SCHEMA_VERSION,
        "operator_abi_version": package_loader.OPERATOR_ABI_VERSION,
        "operator_schema_hash": package_loader.OPERATOR_SCHEMA_HASH,
        "build_id": BUILD_ID,
        "package_version": "0.1.0",
        "source_revision": "test-revision",
        "source_tree_hash": SOURCE_HASH,
        "torch_version": torch_version,
        "torch_cuda": "12.8",
        "cxx11_abi": True,
        "links_libtorch_python": False,
        "library": "lib/_worldfoundry_native_schema.so",
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "compiler": {"id": "GNU", "version": "13.3.0", "cxx_standard": 17},
        "libstdcxx": {"path": "/usr/lib/libstdc++.so.6", "sha256": LIBSTDCXX_HASH},
        "compile_flags": ["-std=c++17", "_GLIBCXX_USE_CXX11_ABI=1"],
        "nvcc_version": None,
        "sm_targets": [],
        "sass_targets": [],
        "ptx_targets": [],
        "capabilities": [],
    }
    (root / "build_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


@pytest.fixture(autouse=True)
def _reset_package_loader() -> None:
    package_loader._reset_for_tests()
    yield
    package_loader._reset_for_tests()


def test_control_plane_and_optional_package_imports_are_side_effect_free() -> None:
    script = """
import sys
import worldfoundry
assert "torch" not in sys.modules
assert "worldfoundry_native_kernels" not in sys.modules
import worldfoundry_native_kernels
assert "torch" not in sys.modules
"""
    environment = dict(os.environ)
    python_path = [str(NATIVE_PYTHON_ROOT), str(REPO_ROOT)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_runtime_mismatch_is_rejected_before_dlopen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _write_package_root(tmp_path, torch_version="2.8.0+different")
    fake_torch = _fake_torch()
    monkeypatch.setattr(package_loader, "_import_torch", lambda: fake_torch)

    with pytest.raises(package_loader.NativeKernelCompatibilityError, match="PyTorch build"):
        package_loader.load(root=root)

    assert fake_torch.ops.load_count == 0


def test_corrupt_library_is_rejected_before_torch_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _write_package_root(tmp_path)
    (root / "lib" / "_worldfoundry_native_schema.so").write_bytes(b"tampered")
    imported_torch = False

    def fail_import():
        nonlocal imported_torch
        imported_torch = True
        raise AssertionError("torch import must occur after file validation")

    monkeypatch.setattr(package_loader, "_import_torch", fail_import)
    with pytest.raises(package_loader.NativeKernelManifestError, match="hash mismatch"):
        package_loader.load(root=root)
    assert imported_torch is False


def test_incomplete_sidecar_is_rejected_before_torch_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_package_root(tmp_path)
    manifest_path = root / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("compiler")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    imported_torch = False

    def fail_import():
        nonlocal imported_torch
        imported_torch = True
        raise AssertionError("torch import must occur after sidecar validation")

    monkeypatch.setattr(package_loader, "_import_torch", fail_import)
    with pytest.raises(package_loader.NativeKernelManifestError, match="compiler"):
        package_loader.load(root=root)
    assert imported_torch is False


def test_compatible_dispatcher_load_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _write_package_root(tmp_path)
    fake_torch = _fake_torch()
    monkeypatch.setattr(package_loader, "_import_torch", lambda: fake_torch)

    first = package_loader.load(root=root)
    second = package_loader.load(root=root)

    assert first["build_id"] == BUILD_ID
    assert second == first
    assert package_loader.is_loaded() is True
    assert fake_torch.ops.load_count == 1


def test_concurrent_dispatcher_load_runs_dlopen_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _write_package_root(tmp_path)
    fake_torch = _fake_torch()
    monkeypatch.setattr(package_loader, "_import_torch", lambda: fake_torch)

    with ThreadPoolExecutor(max_workers=8) as executor:
        manifests = list(executor.map(lambda _index: package_loader.load(root=root), range(32)))

    assert {manifest["build_id"] for manifest in manifests} == {BUILD_ID}
    assert fake_torch.ops.load_count == 1


def test_post_dlopen_failure_poison_latch_prevents_concurrent_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_package_root(tmp_path)
    fake_torch = _fake_torch(build_id="f" * 24)
    monkeypatch.setattr(package_loader, "_import_torch", lambda: fake_torch)

    def attempt_load(_index: int) -> str:
        try:
            package_loader.load(root=root)
        except package_loader.NativeKernelPackageError as exc:
            return str(exc)
        raise AssertionError("incompatible build info unexpectedly loaded")

    with ThreadPoolExecutor(max_workers=8) as executor:
        errors = list(executor.map(attempt_load, range(32)))

    assert fake_torch.ops.load_count == 1
    assert package_loader.is_loaded() is False
    assert any("differs from its sidecar" in error for error in errors)
    assert any("cannot retry safely" in error for error in errors)


def test_loaded_identity_rejects_same_path_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _write_package_root(tmp_path)
    fake_torch = _fake_torch()
    monkeypatch.setattr(package_loader, "_import_torch", lambda: fake_torch)
    original = package_loader.load(root=root)

    library = root / "lib" / "_worldfoundry_native_schema.so"
    library.write_bytes(b"replacement-dispatcher")
    manifest_path = root / "build_manifest.json"
    replacement = json.loads(manifest_path.read_text(encoding="utf-8"))
    replacement["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(replacement), encoding="utf-8")

    with pytest.raises(package_loader.NativeKernelCompatibilityError, match="already loaded"):
        package_loader.load(root=root)

    assert original["library_sha256"] != replacement["library_sha256"]
    assert fake_torch.ops.load_count == 1


def test_provider_reports_absent_package_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(native_provider, "_import_native_package", lambda: None)

    status = native_provider.native_provider_status()

    assert status.available is False
    assert status.inspectable is False
    assert status.state == "absent"
    assert status.installed is False
    assert "not installed" in (status.reason or "")


def test_provider_adapts_manifest_and_explicit_load(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePackage:
        def __init__(self) -> None:
            self.loaded = False

        def inspect(self):
            return {
                "build_id": "build-2",
                "operator_abi_version": 1,
                "capabilities": [],
            }

        def load(self):
            self.loaded = True
            return self.inspect()

        def is_loaded(self):
            return self.loaded

    package = FakePackage()
    monkeypatch.setattr(native_provider, "_import_native_package", lambda: package)

    inspected = native_provider.native_provider_status()
    loaded = native_provider.load_native_provider(strict=True)

    assert inspected.available is False
    assert inspected.inspectable is True
    assert inspected.state == "inspectable"
    assert inspected.runtime_compatible is None
    assert inspected.loaded is False
    assert loaded.runtime_compatible is True
    assert loaded.loaded is True
    assert loaded.available is True
    assert loaded.state == "loaded"
    assert loaded.build_id == "build-2"


def test_provider_preserves_manifest_valid_runtime_mismatch_state(monkeypatch: pytest.MonkeyPatch) -> None:
    class CompatibilityError(RuntimeError):
        pass

    class FakePackage:
        NativeKernelCompatibilityError = CompatibilityError

        @staticmethod
        def inspect():
            return {
                "build_id": "build-3",
                "operator_abi_version": 1,
                "capabilities": [],
            }

        @staticmethod
        def load():
            raise CompatibilityError("Torch build mismatch")

        @staticmethod
        def is_loaded():
            return False

    monkeypatch.setattr(native_provider, "_import_native_package", lambda: FakePackage())

    status = native_provider.load_native_provider()

    assert status.state == "runtime_incompatible"
    assert status.inspectable is True
    assert status.available is False
    assert status.manifest_valid is True
    assert status.runtime_compatible is False
    assert "Torch build mismatch" in (status.reason or "")


def test_strict_provider_resolution_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(native_provider, "_import_native_package", lambda: None)

    with pytest.raises(native_provider.NativeProviderUnavailable, match="not installed"):
        native_provider.load_native_provider(strict=True)

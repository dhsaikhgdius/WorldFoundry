"""NK: main package stays free of optional native-kernels coupling."""

from __future__ import annotations

import importlib
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
NATIVE_PKG = REPO_ROOT / "packages" / "worldfoundry-native-kernels"


def _optional_dependency_blobs() -> dict[str, list[str]]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return dict(data.get("project", {}).get("optional-dependencies", {}))


def test_main_pyproject_does_not_depend_on_native_kernels() -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    assert "worldfoundry-native-kernels" not in text
    assert "worldfoundry_native_kernels" not in text
    for name, deps in _optional_dependency_blobs().items():
        joined = "\n".join(deps)
        assert "worldfoundry-native-kernels" not in joined, name
        assert "worldfoundry_native_kernels" not in joined, name


def test_native_kernels_lives_in_separate_package_tree() -> None:
    assert (NATIVE_PKG / "pyproject.toml").is_file()
    native = tomllib.loads((NATIVE_PKG / "pyproject.toml").read_text(encoding="utf-8"))
    assert native["project"]["name"] == "worldfoundry-native-kernels"
    main = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    assert main["project"]["name"] == "worldfoundry"
    assert main["project"]["name"] != native["project"]["name"]


def test_native_provider_status_absent_without_optional_package() -> None:
    """Optional wheel missing → fail-closed absent status (no hard ImportError).

    Loading ``worldfoundry.core.kernels`` may pull torch via the package
    ``__init__`` (portable kernel fallbacks). The optional
    ``worldfoundry_native_kernels`` package must still stay unloaded.
    """
    script = """
import importlib
import sys
np = importlib.import_module("worldfoundry.core.kernels.native_provider")
assert "worldfoundry_native_kernels" not in sys.modules
status = np.native_provider_status(load=False, strict=False)
assert status.state == "absent"
assert status.installed is False
assert status.available is False
assert "not installed" in (status.reason or "")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_native_provider_module_exports_status_helper() -> None:
    module = importlib.import_module("worldfoundry.core.kernels.native_provider")
    assert callable(module.native_provider_status)
    assert issubclass(module.NativeProviderUnavailable, RuntimeError)

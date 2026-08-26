"""SY-07: VMem runtime_env must not purge global utils/models."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from worldfoundry.synthesis.visual_generation.vmem import runtime_env as vmem_env


def test_purge_keeps_global_utils_and_fail_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_root = tmp_path / "vmem_runtime"
    runtime_root.mkdir()
    global_utils = tmp_path / "site" / "utils.py"
    global_utils.parent.mkdir(parents=True)
    global_utils.write_text("VALUE = 1\n", encoding="utf-8")

    module = types.ModuleType("utils")
    module.__file__ = str(global_utils)
    monkeypatch.setitem(sys.modules, "utils", module)

    with pytest.raises(RuntimeError, match="already loaded from"):
        vmem_env._purge_conflicting_runtime_modules(runtime_root)

    assert "utils" in sys.modules
    assert sys.modules["utils"] is module


def test_purge_drops_prior_vmem_runtime_utils(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    active_root = tmp_path / "active" / "vmem_runtime"
    stale_root = tmp_path / "stale" / "vmem_runtime"
    active_root.mkdir(parents=True)
    stale_root.mkdir(parents=True)
    stale_utils = stale_root / "utils.py"
    stale_utils.write_text("VALUE = 0\n", encoding="utf-8")

    module = types.ModuleType("utils")
    module.__file__ = str(stale_utils)
    monkeypatch.setitem(sys.modules, "utils", module)

    vmem_env._purge_conflicting_runtime_modules(active_root)
    assert "utils" not in sys.modules


def test_purge_reloads_modules_under_active_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_root = tmp_path / "vmem_runtime"
    runtime_root.mkdir()
    local_utils = runtime_root / "utils.py"
    local_utils.write_text("VALUE = 2\n", encoding="utf-8")

    module = types.ModuleType("utils")
    module.__file__ = str(local_utils)
    monkeypatch.setitem(sys.modules, "utils", module)

    vmem_env._purge_conflicting_runtime_modules(runtime_root)
    assert "utils" not in sys.modules

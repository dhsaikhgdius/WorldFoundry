from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "setup" / "validate_unified_lock_tier.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_unified_lock_tier", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_torch_version() -> None:
    mod = _load_module()
    assert mod.parse_torch_version("# comment\ntorch==2.11.0\n") == "2.11.0"
    with pytest.raises(ValueError, match="no torch"):
        mod.parse_torch_version("# placeholder\n")


def test_validate_lock_rejects_torch_missing_from_index(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "index_torch_versions", lambda *_args, **_kwargs: {"2.5.1", "2.4.1"})
    with pytest.raises(SystemExit, match="refusing cu121 lock"):
        mod.validate_lock_against_index(
            "torch==2.11.0\n",
            tier="cu121",
            index_url="https://download.pytorch.org/whl/cu121",
        )


def test_validate_lock_accepts_torch_on_index(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "index_torch_versions", lambda *_args, **_kwargs: {"2.11.0", "2.10.0"})
    assert (
        mod.validate_lock_against_index(
            "torch==2.11.0\n",
            tier="cu128",
            index_url="https://download.pytorch.org/whl/cu128",
        )
        == "2.11.0"
    )

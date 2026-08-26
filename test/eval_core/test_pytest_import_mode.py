"""TE-02: pytest import-mode / strict-config contract."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_pytest_ini_uses_importlib_mode() -> None:
    text = (REPO_ROOT / "pytest.ini").read_text(encoding="utf-8")
    assert "--import-mode=importlib" in text
    assert "--strict-config" in text


@pytest.mark.unit
def test_top_level_conftest_inserts_repo_root_once() -> None:
    source = (REPO_ROOT / "test" / "conftest.py").read_text(encoding="utf-8")
    assert "SRC_ROOT" not in source
    assert source.count("sys.path.insert") == 1

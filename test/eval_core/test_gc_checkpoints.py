"""Tests for scripts/model_zoo/gc_checkpoints.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "model_zoo" / "gc_checkpoints.py"


def _load():
    spec = importlib.util.spec_from_file_location("gc_checkpoints", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_list_orphan_dirs(tmp_path: Path) -> None:
    mod = _load()
    root = tmp_path / "hfd"
    (root / "keep__model").mkdir(parents=True)
    (root / "orphan__model").mkdir()
    orphans = mod.list_orphan_dirs(root, {"org/keep-model", "keep__model"})
    # keep__model matches local name for keep-model mapping via replace — also in keep_names
    names = {path.name for path in orphans}
    assert "orphan__model" in names
    assert "keep__model" not in names


def test_local_name_for_repo() -> None:
    mod = _load()
    assert mod._local_name_for_repo("org/name") == "org__name"

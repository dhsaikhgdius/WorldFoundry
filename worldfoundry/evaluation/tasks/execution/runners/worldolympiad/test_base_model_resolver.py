"""Unit tests for the WorldOlympiad ``base_model_resolver``.

These exercise the capability-registry resolution path without the vendored
runtime, GPU, or real weights: fake model dirs are staged under a tmp path and
exposed to the resolver through the ``WORLDFOUNDRY_HFD_ROOT`` /
``WORLDFOUNDRY_CKPT_DIR`` env vars that ``expand_capability_path`` reads. The
DA3 shim test verifies the symlink makes ``depth_anything_3`` resolvable into
the vendored ``depth_anything_v3`` package without executing the module (so
numpy/torch need not be installed to run the suite).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from worldfoundry.evaluation.tasks.execution.runners.worldolympiad.base_model_resolver import (
    DA3_IMPORT_NAME,
    ResolvedBaseModels,
    build_da3_code_shim,
    da3_code_dir,
    resolve_all,
    resolve_clip_cache_dir,
    resolve_clip_code_dir,
    resolve_da3_weights_dir,
    resolve_qwenvl_dir,
    resolve_sam3_path,
)


def _point_roots_at(monkeypatch, hfd: Path, ckpt: Path) -> None:
    """Point the capability path tokens at tmp roots and clear per-model env overrides."""
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(hfd))
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(ckpt))
    for name in ("WORLDFOUNDRY_QWEN3_VL_MODEL_DIR", "QWENVL_MODEL_PATH", "WORLDFOUNDRY_SAM3_CKPT"):
        monkeypatch.delenv(name, raising=False)


def test_resolve_qwenvl_unstaged_returns_none(tmp_path, monkeypatch):
    _point_roots_at(monkeypatch, tmp_path / "hfd", tmp_path / "ckpt")
    assert resolve_qwenvl_dir() is None


def test_resolve_qwenvl_staged_returns_dir(tmp_path, monkeypatch):
    qwenvl = tmp_path / "hfd" / "Qwen--Qwen3-VL-8B-Instruct"
    qwenvl.mkdir(parents=True)
    (qwenvl / "config.json").write_text("{}", encoding="utf-8")
    (qwenvl / "model.safetensors").write_bytes(b"\x00")
    # The capability gates on min_file_count=3 (mirrors keye_vl) so a fully staged
    # model dir has more than a config + a single weight shard.
    (qwenvl / "tokenizer.json").write_text("{}", encoding="utf-8")
    _point_roots_at(monkeypatch, qwenvl.parent, tmp_path / "ckpt")
    assert resolve_qwenvl_dir() == qwenvl.resolve()


def test_resolve_da3_weights_staged_returns_dir(tmp_path, monkeypatch):
    da3 = tmp_path / "hfd" / "depth-anything--DA3-LARGE-1.1"
    da3.mkdir(parents=True)
    (da3 / "config.json").write_text("{}", encoding="utf-8")
    (da3 / "model.safetensors").write_bytes(b"\x00")
    _point_roots_at(monkeypatch, da3.parent, tmp_path / "ckpt")
    assert resolve_da3_weights_dir() == da3.resolve()


def test_resolve_sam3_rejects_undersized_checkpoint(tmp_path, monkeypatch):
    # The sam3 asset gates on min_size_bytes=1_000_000_000; a stub file is "not ready".
    sam3 = tmp_path / "hfd" / "facebook--sam3.1" / "sam3.1_multiplex.pt"
    sam3.parent.mkdir(parents=True)
    sam3.write_bytes(b"\x00" * 4)
    _point_roots_at(monkeypatch, sam3.parents[1], tmp_path / "ckpt")
    assert resolve_sam3_path() is None


def test_da3_code_shim_makes_depth_anything_3_resolvable(tmp_path):
    shim_dir = build_da3_code_shim(tmp_path)
    assert shim_dir is not None
    link = shim_dir / DA3_IMPORT_NAME
    assert link.is_symlink()
    # The symlink resolves to the base_models depth_anything_v3 package.
    assert link.resolve() == da3_code_dir()
    # find_spec locates the package without executing it (numpy/torch need not be installed).
    sys.path.insert(0, str(shim_dir))
    try:
        assert importlib.util.find_spec(DA3_IMPORT_NAME) is not None
    finally:
        sys.path.remove(str(shim_dir))


def test_clip_code_dir_resolves_to_openai_clip_runtime():
    clip_dir = resolve_clip_code_dir()
    assert clip_dir is not None
    assert clip_dir.is_dir()
    assert (clip_dir / "clip" / "__init__.py").is_file()


def test_clip_cache_dir_respects_env_override(tmp_path, monkeypatch):
    target = tmp_path / "clipcache"
    monkeypatch.setenv("CLIP_DOWNLOAD_ROOT", str(target))
    assert resolve_clip_cache_dir() == target


def test_resolve_all_builds_shim_and_provenance(tmp_path, monkeypatch):
    _point_roots_at(monkeypatch, tmp_path / "hfd", tmp_path / "ckpt")
    resolved = resolve_all(shim_parent=tmp_path / "out")
    assert isinstance(resolved, ResolvedBaseModels)
    # The DA3 code shim is built whenever the base_models code package exists, independent of weights.
    assert resolved.da3_code_shim_dir is not None
    assert resolved.clip_code_dir is not None
    entries = resolved.pythonpath_entries()
    assert resolved.da3_code_shim_dir in entries
    assert resolved.clip_code_dir in entries
    prov = resolved.as_provenance()
    assert set(prov) == {
        "qwenvl_dir",
        "da3_weights_dir",
        "da3_code_shim_dir",
        "sam3_path",
        "clip_code_dir",
        "clip_cache_dir",
    }
    # env_overrides only carries models that resolved to a staged asset.
    overrides = resolved.env_overrides()
    assert "QWENVL_MODEL_PATH" not in overrides  # qwenvl unstaged in this env
    assert "CLIP_DOWNLOAD_ROOT" in overrides

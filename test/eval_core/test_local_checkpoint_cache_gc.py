"""Tests for local checkpoint cache GC (LRU by mtime, flock-aware)."""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path

from worldfoundry.runtime.local_checkpoint_cache import (
    _is_ready,
    gc_local_checkpoint_cache,
    resolve_local_checkpoint_cache_root,
)


def _make_tree(root: Path, name: str, *, age_seconds: float) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "model.bin").write_bytes(b"x")
    stamp = time.time() - age_seconds
    os.utime(path, (stamp, stamp))
    return path


def test_gc_keeps_newest_and_removes_oldest(tmp_path: Path) -> None:
    older = _make_tree(tmp_path, "ckpt-old", age_seconds=100)
    newer = _make_tree(tmp_path, "ckpt-new", age_seconds=0)

    report = gc_local_checkpoint_cache(cache_root=tmp_path, keep_newest=1)
    assert report["kept"] == [str(newer)]
    assert report["removed"] == [str(older)]
    assert newer.exists()
    assert not older.exists()


def test_gc_dry_run_reports_without_deleting(tmp_path: Path) -> None:
    older = _make_tree(tmp_path, "ckpt-old", age_seconds=100)
    _make_tree(tmp_path, "ckpt-new", age_seconds=0)

    report = gc_local_checkpoint_cache(cache_root=tmp_path, keep_newest=1, dry_run=True)
    assert report["dry_run"] is True
    assert report["removed"] == [str(older)]
    assert older.exists()


def test_gc_skips_trees_with_held_publish_lock(tmp_path: Path) -> None:
    older = _make_tree(tmp_path, "ckpt-old", age_seconds=100)
    _make_tree(tmp_path, "ckpt-new", age_seconds=0)
    lock_path = tmp_path / f".{older.name}.lock"
    with open(lock_path, "w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            report = gc_local_checkpoint_cache(cache_root=tmp_path, keep_newest=1)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    assert report["locked"] == [str(older)]
    assert report["removed"] == []
    assert older.exists()


def test_gc_noop_when_staging_root_unset(tmp_path: Path) -> None:
    assert resolve_local_checkpoint_cache_root(environ={}) is None
    report = gc_local_checkpoint_cache(environ={})
    assert report["cache_root"] is None
    assert report["removed"] == []


def test_gc_keep_default_comes_from_env(tmp_path: Path) -> None:
    for idx in range(3):
        _make_tree(tmp_path, f"ckpt-{idx}", age_seconds=100 - idx)
    report = gc_local_checkpoint_cache(
        cache_root=tmp_path,
        environ={"WORLDFOUNDRY_LOCAL_CHECKPOINT_CACHE_KEEP": "1"},
    )
    assert report["keep_newest"] == 1
    assert len(report["kept"]) == 1
    assert len(report["removed"]) == 2


def test_is_ready_rejects_truncated_tree(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    (target / ".worldfoundry-local-cache.json").write_text(
        json.dumps({"source": str(source), "size_bytes": 1_000_000}),
        encoding="utf-8",
    )
    assert _is_ready(target, source, ()) is False
    (target / "model.bin").write_bytes(b"x" * 1_000_000)
    assert _is_ready(target, source, ()) is True

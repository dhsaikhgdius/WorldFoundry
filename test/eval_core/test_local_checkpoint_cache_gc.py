"""Tests for local checkpoint cache GC."""

from __future__ import annotations

import time
from pathlib import Path

from worldfoundry.runtime.local_checkpoint_cache import gc_local_checkpoint_cache


def test_gc_local_checkpoint_cache_keeps_newest(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    older = root / "ckpt-old"
    newer = root / "ckpt-new"
    older.mkdir()
    newer.mkdir()
    old_ts = time.time() - 100
    new_ts = time.time()
    Path(older).touch()
    Path(newer).touch()
    import os

    os.utime(older, (old_ts, old_ts))
    os.utime(newer, (new_ts, new_ts))

    report = gc_local_checkpoint_cache(cache_root=root, keep_newest=1, dry_run=False)
    assert str(newer) in report["kept"]
    assert str(older) in report["removed"]
    assert newer.exists()
    assert not older.exists()

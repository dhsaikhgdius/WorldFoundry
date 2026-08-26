"""CPU unit tests for compile_cache garbage collection."""

from __future__ import annotations

import time
from pathlib import Path

from worldfoundry.runtime.compile_cache import gc_compile_cache, resolve_compile_cache_base


def test_resolve_compile_cache_base_honors_env(tmp_path: Path) -> None:
    root = tmp_path / "compile-root"
    assert resolve_compile_cache_base(environ={"WORLDFOUNDRY_COMPILE_CACHE_DIR": str(root)}) == root


def test_gc_compile_cache_removes_stale_fingerprint(tmp_path: Path) -> None:
    base = tmp_path / "compile"
    stale = base / "stale-fingerprint"
    stale.mkdir(parents=True)
    (stale / "inductor").mkdir()
    # Make the stale tree look old.
    old = time.time() - 10 * 86400
    Path(stale).touch()
    import os

    os.utime(stale, (old, old))

    env = {"WORLDFOUNDRY_COMPILE_CACHE_DIR": str(base)}
    # keep_current=False avoids needing torch fingerprint / writable inductor layout.
    report = gc_compile_cache(
        keep_current=False,
        max_age_days=1.0,
        dry_run=False,
        environ=env,
    )
    assert str(stale) in report["removed"]
    assert not stale.exists()


def test_gc_compile_cache_dry_run_and_skips_cuda(tmp_path: Path) -> None:
    base = tmp_path / "compile"
    (base / "cuda").mkdir(parents=True)
    victim = base / "old-fp"
    victim.mkdir()
    env = {"WORLDFOUNDRY_COMPILE_CACHE_DIR": str(base)}
    report = gc_compile_cache(keep_current=False, dry_run=True, environ=env)
    assert str(victim) in report["removed"]
    assert victim.exists()
    assert any(path.endswith("/cuda") or path.endswith("\\cuda") for path in report["skipped"])

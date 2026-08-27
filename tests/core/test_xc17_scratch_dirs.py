"""XC-17: default scratch dirs live under the cache root, not bare /tmp.

Guards ``worldfoundry.core.io.scratch.make_scratch_dir``, the replacement for
the bare ``tempfile.mkdtemp()`` defaults that lyra / matrix_game pipelines
used when no explicit ``output_dir`` was provided. Scratch directories must
be created under ``${WORLDFOUNDRY_CACHE_DIR}/scratch/<date>/`` and registered
for best-effort removal at interpreter exit.
"""

from __future__ import annotations

import atexit
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from worldfoundry.core.io import scratch as scratch_module
from worldfoundry.core.io.scratch import make_scratch_dir, scratch_root_path


@pytest.fixture(autouse=True)
def _isolated_cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache_root = tmp_path / "wf_cache"
    monkeypatch.setenv("WORLDFOUNDRY_CACHE_DIR", str(cache_root))
    monkeypatch.delenv("WORLDFOUNDRY_HOME", raising=False)
    return cache_root


class TestScratchRootPath:
    def test_root_is_under_cache_dir(self, _isolated_cache_root: Path) -> None:
        assert scratch_root_path() == _isolated_cache_root / "scratch"

    def test_root_is_not_created_by_resolution(self, _isolated_cache_root: Path) -> None:
        assert not scratch_root_path().exists()

    def test_env_mapping_override(self, tmp_path: Path) -> None:
        override = tmp_path / "other_cache"
        assert scratch_root_path(env={"WORLDFOUNDRY_CACHE_DIR": str(override)}) == override / "scratch"


class TestMakeScratchDir:
    def test_created_under_dated_scratch_root(self, _isolated_cache_root: Path) -> None:
        scratch = make_scratch_dir(prefix="lyra1_static_generated_", cleanup_at_exit=False)
        assert scratch.is_dir()
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        assert scratch.parent == _isolated_cache_root / "scratch" / today
        assert scratch.name.startswith("lyra1_static_generated_")

    def test_directories_are_unique(self) -> None:
        first = make_scratch_dir(prefix="lyra2_runtime_", cleanup_at_exit=False)
        second = make_scratch_dir(prefix="lyra2_runtime_", cleanup_at_exit=False)
        assert first != second
        assert first.parent == second.parent

    def test_registers_atexit_rmtree(self, monkeypatch: pytest.MonkeyPatch) -> None:
        registered: list[tuple] = []
        monkeypatch.setattr(
            scratch_module.atexit, "register", lambda func, *args, **kwargs: registered.append((func, args, kwargs))
        )
        scratch = make_scratch_dir(prefix="wf_test_")
        assert registered == [(shutil.rmtree, (scratch,), {"ignore_errors": True})]

        func, args, kwargs = registered[0]
        (scratch / "artifact.bin").write_bytes(b"payload")
        func(*args, **kwargs)
        assert not scratch.exists()

    def test_cleanup_callback_ignores_missing_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        registered: list[tuple] = []
        monkeypatch.setattr(
            scratch_module.atexit, "register", lambda func, *args, **kwargs: registered.append((func, args, kwargs))
        )
        scratch = make_scratch_dir(prefix="wf_test_")
        shutil.rmtree(scratch)
        func, args, kwargs = registered[0]
        func(*args, **kwargs)  # must not raise on already-deleted dir

    def test_cleanup_at_exit_false_skips_registration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        registered: list[tuple] = []
        monkeypatch.setattr(
            scratch_module.atexit, "register", lambda func, *args, **kwargs: registered.append((func, args, kwargs))
        )
        make_scratch_dir(prefix="wf_test_", cleanup_at_exit=False)
        assert registered == []

    def test_uses_real_atexit_module(self) -> None:
        # The module must register against the real atexit hook (not a shim),
        # so cleanup actually runs at interpreter shutdown.
        assert scratch_module.atexit is atexit

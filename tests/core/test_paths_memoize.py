from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from worldfoundry.core.io import paths as paths_mod


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    paths_mod.clear_path_resolution_caches()
    yield
    paths_mod.clear_path_resolution_caches()


def test_worldfoundry_path_tokens_memoized_for_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_CACHE_DIR", "/tmp/wf-cache-a")
    paths_mod.clear_path_resolution_caches()
    with patch.object(paths_mod, "_build_worldfoundry_path_tokens", wraps=paths_mod._build_worldfoundry_path_tokens) as build:
        first = paths_mod.worldfoundry_path_tokens()
        second = paths_mod.worldfoundry_path_tokens()
    assert first == second
    assert first["WORLDFOUNDRY_CACHE_DIR"] == "/tmp/wf-cache-a"
    assert build.call_count == 1


def test_worldfoundry_path_tokens_rebuilds_when_env_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_CACHE_DIR", "/tmp/wf-cache-a")
    paths_mod.clear_path_resolution_caches()
    assert paths_mod.worldfoundry_path_tokens()["WORLDFOUNDRY_CACHE_DIR"] == "/tmp/wf-cache-a"
    monkeypatch.setenv("WORLDFOUNDRY_CACHE_DIR", "/tmp/wf-cache-b")
    assert paths_mod.worldfoundry_path_tokens()["WORLDFOUNDRY_CACHE_DIR"] == "/tmp/wf-cache-b"


def test_worldfoundry_path_tokens_explicit_env_bypasses_memo() -> None:
    with patch.object(paths_mod, "_build_worldfoundry_path_tokens", wraps=paths_mod._build_worldfoundry_path_tokens) as build:
        paths_mod.worldfoundry_path_tokens({"WORLDFOUNDRY_CACHE_DIR": "/tmp/a"})
        paths_mod.worldfoundry_path_tokens({"WORLDFOUNDRY_CACHE_DIR": "/tmp/a"})
    assert build.call_count == 2


def test_hfd_download_complete_caches_by_mtime(tmp_path: Path) -> None:
    model = tmp_path / "owner--repo"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    with patch.object(paths_mod, "_hfd_download_complete_uncached", wraps=paths_mod._hfd_download_complete_uncached) as scan:
        assert paths_mod.hfd_download_complete(model) is True
        assert paths_mod.hfd_download_complete(model) is True
    assert scan.call_count == 1


def test_hfd_download_complete_rejects_incomplete_and_busts_cache(tmp_path: Path) -> None:
    model = tmp_path / "owner--repo"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"ok")
    assert paths_mod.hfd_download_complete(model) is True
    incomplete = model / "weights.bin.incomplete"
    incomplete.write_bytes(b"partial")
    # Force a newer directory mtime so the memo key cannot collide on
    # coarse filesystem timestamp resolution.
    st = model.stat()
    os.utime(model, ns=(st.st_atime_ns + 1_000_000_000, st.st_mtime_ns + 1_000_000_000))
    assert paths_mod.hfd_download_complete(model) is False

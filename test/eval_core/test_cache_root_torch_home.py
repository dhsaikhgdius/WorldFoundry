"""CA-02: asset cache root prefers WorldFoundry paths over TORCH_HOME."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_cache_path_prefers_worldfoundry_cache_over_torch_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from worldfoundry.core.io.cache import _cache_path

    wf = tmp_path / "wf-cache"
    torch_home = tmp_path / "torch-home"
    monkeypatch.setenv("WORLDFOUNDRY_CACHE_DIR", str(wf))
    monkeypatch.setenv("TORCH_HOME", str(torch_home))

    resolved = _cache_path("s3://bucket/key.bin")
    assert resolved == wf / "s3" / "bucket" / "key.bin"


@pytest.mark.unit
def test_cache_path_torch_home_is_deprecated_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from worldfoundry.core.io.cache import _cache_path

    torch_home = tmp_path / "torch-home"
    monkeypatch.delenv("WORLDFOUNDRY_CACHE_DIR", raising=False)
    monkeypatch.delenv("WORLDFOUNDRY_HOME", raising=False)
    monkeypatch.setenv("TORCH_HOME", str(torch_home))

    with pytest.warns(DeprecationWarning, match="TORCH_HOME"):
        resolved = _cache_path("file.bin")
    assert resolved == torch_home / "file.bin"

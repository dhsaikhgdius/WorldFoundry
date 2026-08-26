from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from worldfoundry.core.io import cache as io_cache
from worldfoundry.core.io.disk import DiskSpaceError


def test_populate_runs_disk_preflight(tmp_path: Path) -> None:
    source = tmp_path / "src.bin"
    source.write_bytes(b"payload")
    target = tmp_path / "cache" / "dst.bin"
    seen: dict[str, object] = {}

    def _fake_ensure(path, *, required_bytes, label, env_vars=(), settings=None, mkdir=True):
        seen["path"] = Path(path)
        seen["required_bytes"] = required_bytes
        seen["label"] = label
        seen["mkdir"] = mkdir

    with (
        patch.object(io_cache, "ensure_free_disk", side_effect=_fake_ensure),
        patch.object(io_cache, "cache_min_free_bytes", return_value=1234),
    ):
        resolved = io_cache.download_from_cache_or_uri(str(source), cache_fp=str(target), rank_sync=False)

    assert Path(resolved) == target
    assert target.read_bytes() == b"payload"
    assert seen["path"] == target.parent
    assert seen["required_bytes"] == 1234
    assert seen["label"] == "WorldFoundry inference cache"
    assert seen["mkdir"] is False


def test_populate_maps_enospc_to_disk_space_error(tmp_path: Path) -> None:
    source = tmp_path / "src.bin"
    source.write_bytes(b"payload")
    target = tmp_path / "cache" / "dst.bin"

    with (
        patch.object(io_cache, "ensure_free_disk"),
        patch.object(io_cache, "cache_min_free_bytes", return_value=1),
        patch.object(io_cache, "copy_uri", side_effect=OSError(28, "No space left on device")),
        pytest.raises(DiskSpaceError, match="inference cache"),
    ):
        io_cache.download_from_cache_or_uri(str(source), cache_fp=str(target), rank_sync=False)

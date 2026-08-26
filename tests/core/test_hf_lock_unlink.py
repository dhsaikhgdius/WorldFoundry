from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from worldfoundry.core.io import hf as hf_mod


def test_download_snapshot_unlinks_lock_after_success(tmp_path: Path) -> None:
    lock_seen: dict[str, Path] = {}

    class _FakeLock:
        def __init__(self, path: str):
            lock_seen["path"] = Path(path)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("held", encoding="utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    with (
        patch.object(hf_mod, "FileLock", side_effect=_FakeLock),
        patch.object(hf_mod, "ensure_free_disk"),
        patch.object(hf_mod, "_snapshot_download", return_value="/cache"),
        patch.object(hf_mod, "_hub_cache_dir", return_value=tmp_path),
    ):
        hf_mod._download_snapshot(
            "owner/repo",
            revision=None,
            cache_dir=tmp_path,
            allow_patterns=None,
            ignore_patterns=None,
        )
    assert "path" in lock_seen
    assert not lock_seen["path"].exists()


def test_download_snapshot_keeps_lock_path_on_failure(tmp_path: Path) -> None:
    lock_seen: dict[str, Path] = {}

    class _FakeLock:
        def __init__(self, path: str):
            lock_seen["path"] = Path(path)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("held", encoding="utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    with (
        patch.object(hf_mod, "FileLock", side_effect=_FakeLock),
        patch.object(hf_mod, "ensure_free_disk"),
        patch.object(hf_mod, "_snapshot_download", side_effect=RuntimeError("boom")),
        patch.object(hf_mod, "_hub_cache_dir", return_value=tmp_path),
        patch.object(hf_mod, "disk_space_error_from_exception", return_value=None),
    ):
        try:
            hf_mod._download_snapshot(
                "owner/repo",
                revision=None,
                cache_dir=tmp_path,
                allow_patterns=None,
                ignore_patterns=None,
            )
        except RuntimeError:
            pass
    assert lock_seen["path"].exists()

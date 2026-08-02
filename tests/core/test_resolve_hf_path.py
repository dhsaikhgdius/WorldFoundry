from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from worldfoundry.core.io.hf import (
    HF_URI_SCHEME,
    _allow_patterns_for_subpath,
    _parse_hf_uri,
    hf_download_or_fpath,
    resolve_hf_path,
    resolve_hf_snapshot_path,
)
from worldfoundry.core.io.easy_io import resolve_checkpoint_path


def test_resolve_hf_path_passthrough_for_non_string() -> None:
    assert resolve_hf_path(None) is None
    assert resolve_hf_path(42) == 42


def test_resolve_hf_path_returns_existing_local_path(tmp_path: Path) -> None:
    local_file = tmp_path / "checkpoint.safetensors"
    local_file.write_text("ok", encoding="utf-8")

    assert resolve_hf_path(str(local_file)) == str(local_file)


def test_resolve_hf_path_returns_non_hf_path_unchanged() -> None:
    missing = "/tmp/does-not-exist/worldfoundry-hf-test"
    assert resolve_hf_path(missing) == missing


def test_parse_hf_uri_rejects_invalid_paths() -> None:
    with pytest.raises(ValueError, match="Invalid HF path"):
        _parse_hf_uri(f"{HF_URI_SCHEME}owner-only")


def test_parse_hf_uri_splits_repo_and_subpath() -> None:
    repo_id, subpath = _parse_hf_uri("hf://Efficient-Large-Model/SANA-WM/dit/model.safetensors")
    assert repo_id == "Efficient-Large-Model/SANA-WM"
    assert subpath == "dit/model.safetensors"


def test_allow_patterns_cover_directory_subtrees() -> None:
    assert _allow_patterns_for_subpath("refiner") == [
        "refiner",
        "refiner/*",
        "refiner/**",
    ]
    assert _allow_patterns_for_subpath("") is None


@patch("worldfoundry.core.io.hf._snapshot_download", return_value="/cache/repo-root")
def test_resolve_hf_path_downloads_hf_uri(mock_snapshot) -> None:
    resolved = resolve_hf_path("hf://owner/repo/checkpoints/model.pth")

    mock_snapshot.assert_called_once_with(
        repo_id="owner/repo",
        allow_patterns=["checkpoints/model.pth", "checkpoints/model.pth/*", "checkpoints/model.pth/**"],
    )
    assert resolved == "/cache/repo-root/checkpoints/model.pth"


@patch("worldfoundry.core.io.hf._snapshot_download", return_value="/cache/repo-root")
def test_hf_download_or_fpath_is_alias(mock_snapshot) -> None:
    resolved = hf_download_or_fpath("hf://owner/repo")

    mock_snapshot.assert_called_once_with(repo_id="owner/repo", allow_patterns=None)
    assert resolved == "/cache/repo-root"


def test_resolve_checkpoint_path_delegates_hf_uri() -> None:
    with patch("worldfoundry.core.io.hf.resolve_hf_path", return_value="/cache/model.pth") as mock_resolve:
        resolved = resolve_checkpoint_path("hf://owner/repo/model.pth")

    mock_resolve.assert_called_once_with("hf://owner/repo/model.pth")
    assert resolved == "/cache/model.pth"


def test_resolve_checkpoint_path_expands_local_path() -> None:
    assert resolve_checkpoint_path("~/checkpoint.pth").endswith("checkpoint.pth")


def test_resolve_hf_snapshot_path_still_handles_repo_ids(tmp_path: Path) -> None:
    local_dir = tmp_path / "local-repo"
    local_dir.mkdir()
    assert resolve_hf_snapshot_path(str(local_dir)) == local_dir

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.setup.materialize_workspace_demo_assets import (
    DEFAULT_PINS_PATH,
    _hash_path,
    _record,
    iter_pinned_repo_assets,
    load_demo_asset_pins,
)


def test_committed_pins_refuse_head_and_cover_repo_assets():
    pins = load_demo_asset_pins(DEFAULT_PINS_PATH)
    rows = iter_pinned_repo_assets(pins)
    assert len(rows) >= 36
    assert all(row["revision"] and row["revision"].upper() != "HEAD" for row in rows)
    assert all(row["remote"].startswith("https://") for row in rows)
    targets = {row["target"] for row in rows}
    assert "longcat_video/motorcycle.mp4" in targets
    assert "matrix-game-1/official_initial_image/forest_00.jpg" in targets


def test_load_demo_asset_pins_rejects_head(tmp_path):
    bad = {
        "repos": {"Demo": {"remote": "https://example.com/demo.git", "revision": "HEAD"}},
        "repo_assets": [{"target": "x.png", "repo": "Demo", "path": "x.png"}],
    }
    path = tmp_path / "pins.yaml"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="concrete revision"):
        load_demo_asset_pins(path)


def test_record_marks_sha256_mismatch(tmp_path):
    target = tmp_path / "asset.bin"
    target.write_bytes(b"abc")
    sha, _, _ = _hash_path(target)
    row = _record(target, "ready", {"kind": "official_git"}, expected_sha256="deadbeef")
    assert row["status"] == "sha256_mismatch"
    assert row["sha256"] == sha


def test_record_accepts_matching_sha256(tmp_path):
    target = tmp_path / "asset.bin"
    target.write_bytes(b"abc")
    sha, _, _ = _hash_path(target)
    row = _record(target, "ready", {"kind": "official_git"}, expected_sha256=sha)
    assert row["status"] == "ready"

"""Tests for WORLDFOUNDRY_PATH_MIRRORS parsing."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.studio.runtime_paths import _mirror_project_roots, _parse_path_mirrors


def test_parse_path_mirrors_default_legacy() -> None:
    pairs = _parse_path_mirrors({})
    assert ("/share/project", "/bench-workspace") in pairs
    assert ("/bench-workspace", "/share/project") in pairs


def test_parse_path_mirrors_custom_env() -> None:
    pairs = _parse_path_mirrors({"WORLDFOUNDRY_PATH_MIRRORS": "/data/a:/fast/a"})
    assert ("/data/a", "/fast/a") in pairs
    assert ("/fast/a", "/data/a") in pairs
    assert ("/share/project", "/bench-workspace") not in pairs


def test_mirror_project_roots_applies_custom() -> None:
    roots = _mirror_project_roots(
        (Path("/data/a/WorldFoundry"),),
        environ={"WORLDFOUNDRY_PATH_MIRRORS": "/data/a:/fast/a"},
    )
    assert Path("/data/a/WorldFoundry") in roots
    assert Path("/fast/a/WorldFoundry") in roots

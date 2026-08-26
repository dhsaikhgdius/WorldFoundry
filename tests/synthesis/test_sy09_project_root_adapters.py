"""SY-09: visual_generation adapters resolve repo root via paths.project_root."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.core.io.paths import project_root


def test_spatia_runtime_project_root_matches_paths_helper() -> None:
    from worldfoundry.synthesis.visual_generation.spatia.worldfoundry_runtime import SpatiaRuntime

    assert SpatiaRuntime.project_root() == project_root(
        Path("worldfoundry/synthesis/visual_generation/spatia/worldfoundry_runtime.py")
    )
    assert (SpatiaRuntime.project_root() / "pyproject.toml").is_file()


def test_longcat_and_lingbot_source_roots_use_project_root() -> None:
    from worldfoundry.synthesis.visual_generation import longcat_video, lingbot_world_v2

    # Import modules that define SRC_ROOT / SOURCE_ROOT.
    from worldfoundry.synthesis.visual_generation.longcat_video import worldfoundry_runtime as longcat
    from worldfoundry.synthesis.visual_generation.lingbot_world_v2 import runtime as lingbot

    del longcat_video, lingbot_world_v2
    expected = project_root(__file__)
    assert longcat.SRC_ROOT == expected
    assert lingbot.SOURCE_ROOT == expected

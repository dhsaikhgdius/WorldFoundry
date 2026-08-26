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


def test_more_adapters_use_paths_project_root() -> None:
    from worldfoundry.synthesis.visual_generation.dreamdojo import worldfoundry_runtime as dreamdojo
    from worldfoundry.synthesis.visual_generation.hydra import worldfoundry_runtime as hydra
    from worldfoundry.synthesis.visual_generation.magic_world import worldfoundry_runtime as magic
    from worldfoundry.synthesis.visual_generation.minwm import worldfoundry_runtime as minwm
    from worldfoundry.synthesis.visual_generation.scope import worldfoundry_runtime as scope
    from worldfoundry.synthesis.visual_generation.warp_as_history import variants as wah_variants
    from worldfoundry.synthesis.visual_generation.warp_as_history import worldfoundry_runtime as wah

    expected = project_root(__file__)
    assert minwm._PROJECT_ROOT == expected
    assert hydra._PROJECT_ROOT == expected
    assert magic._PROJECT_ROOT == expected
    assert magic.MagicWorldRuntime.video_x_fun_root().is_relative_to(expected)
    assert scope._project_root() == expected
    assert wah.WarpAsHistoryRuntime._project_root() == expected
    assert wah_variants.project_root() == expected
    assert dreamdojo.DreamDojoRuntime._repo_src_root() == expected

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


def test_batch3_runtime_env_project_roots_use_paths_helper() -> None:
    import importlib.util
    import sys

    expected = project_root(__file__)
    repo = Path(__file__).resolve().parents[2]

    def load_file(rel: str, name: str):
        path = repo / rel
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    neoverse = load_file(
        "worldfoundry/synthesis/visual_generation/neoverse/runtime_env.py",
        "wf_sy09_neoverse_runtime_env",
    )
    vmem = load_file(
        "worldfoundry/synthesis/visual_generation/vmem/runtime_env.py",
        "wf_sy09_vmem_runtime_env",
    )
    longvie = load_file(
        "worldfoundry/synthesis/visual_generation/longvie/runtime_env.py",
        "wf_sy09_longvie_runtime_env",
    )
    fantasy = load_file(
        "worldfoundry/synthesis/visual_generation/fantasy_world/runtime_env.py",
        "wf_sy09_fantasy_runtime_env",
    )
    multiworld = load_file(
        "worldfoundry/synthesis/visual_generation/multiworld/runtime_env.py",
        "wf_sy09_multiworld_runtime_env",
    )
    solaris = load_file(
        "worldfoundry/synthesis/visual_generation/solaris/runtime_env.py",
        "wf_sy09_solaris_runtime_env",
    )

    assert neoverse.project_root() == expected
    assert vmem.project_root() == expected
    assert longvie.project_root() == expected
    assert fantasy.project_root() == expected
    assert multiworld.project_root() == expected
    assert solaris.project_root() == expected

    for rel in (
        "worldfoundry/synthesis/visual_generation/kairos/runtime.py",
        "worldfoundry/synthesis/visual_generation/lingbot_video/runtime.py",
        "worldfoundry/synthesis/visual_generation/pusa_vidgen/adapter.py",
        "worldfoundry/synthesis/visual_generation/hunyuan_world/hy_world_2p0_worldgen_runtime.py",
        "worldfoundry/synthesis/visual_generation/lingbot_world/runtime.py",
        "worldfoundry/synthesis/visual_generation/magi/worldfoundry_runner.py",
        "worldfoundry/synthesis/visual_generation/official_video_runtime.py",
        "worldfoundry/synthesis/visual_generation/neoverse/runtime_env.py",
        "worldfoundry/synthesis/visual_generation/vmem/runtime_env.py",
        "worldfoundry/synthesis/visual_generation/longvie/runtime_env.py",
        "worldfoundry/synthesis/visual_generation/fantasy_world/runtime_env.py",
        "worldfoundry/synthesis/visual_generation/multiworld/runtime_env.py",
        "worldfoundry/synthesis/visual_generation/solaris/runtime_env.py",
    ):
        text = (repo / rel).read_text(encoding="utf-8")
        assert "project_root" in text, rel
        assert "from worldfoundry.core.io.paths import" in text, rel


def test_batch4_adapters_use_paths_project_root() -> None:
    """geometry_priors / moverse / pandora / fastwam / longvie depth root."""
    import importlib.util
    import sys

    expected = project_root(__file__)
    repo = Path(__file__).resolve().parents[2]

    def load_file(rel: str, name: str):
        path = repo / rel
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    # Source contracts avoid heavy imports (torch / imageio) via package __init__.
    for rel in (
        "worldfoundry/synthesis/visual_generation/geometry_priors.py",
        "worldfoundry/synthesis/visual_generation/moverse/moverse_synthesis.py",
        "worldfoundry/synthesis/visual_generation/pandora/pandora_synthesis.py",
        "worldfoundry/synthesis/action_generation/fastwam/action_dit.py",
        "worldfoundry/synthesis/visual_generation/longvie/runtime_env.py",
    ):
        text = (repo / rel).read_text(encoding="utf-8")
        assert "from worldfoundry.core.io.paths import" in text, rel
        assert "project_root" in text, rel
        assert "Path(__file__).resolve().parents[" not in text, rel

    longvie = load_file(
        "worldfoundry/synthesis/visual_generation/longvie/runtime_env.py",
        "wf_sy09_longvie_runtime_env_batch4",
    )
    depth_root = longvie.video_depth_anything_runtime_root()
    assert depth_root == expected / "worldfoundry/base_models/three_dimensions/depth/video_depth_anything_longvie"
    assert depth_root.is_dir()


def test_batch5_orchestration_runners_use_paths_project_root() -> None:
    """Official runners / orchestration: no Path(__file__).parents[N] climbs."""
    repo = Path(__file__).resolve().parents[2]
    for rel in (
        "worldfoundry/evaluation/tasks/execution/orchestration/benchmark_runner.py",
        "worldfoundry/evaluation/tasks/execution/orchestration/runtime_preflight.py",
        "worldfoundry/evaluation/tasks/execution/runners/mind/run_mind_official_runner.py",
        "worldfoundry/evaluation/tasks/execution/runners/sana_wm_bench/run_sana_wm_bench_official_runner.py",
        "worldfoundry/evaluation/tasks/execution/runners/stevo_bench/run_stevo_bench_official_runner.py",
        "worldfoundry/evaluation/tasks/execution/runners/wbench/run_wbench_official_runner.py",
        "worldfoundry/evaluation/tasks/execution/runners/worldarena/run_worldarena_official_runner.py",
        "worldfoundry/evaluation/tasks/execution/runners/devil_dynamics/run_devil_dynamics_official_runner.py",
    ):
        text = (repo / rel).read_text(encoding="utf-8")
        assert "from worldfoundry.core.io.paths import" in text, rel
        assert "project_root" in text, rel
        assert "Path(__file__).resolve().parents[" not in text, rel


def test_batch6_more_runners_use_paths_project_root() -> None:
    """docker_runner / fetv / videophy / worldolympiad / worldreasonbench."""
    repo = Path(__file__).resolve().parents[2]
    for rel in (
        "worldfoundry/evaluation/tasks/embodied/docker_runner.py",
        "worldfoundry/evaluation/tasks/execution/runners/fetv/fetv_official_runtime.py",
        "worldfoundry/evaluation/tasks/execution/runners/videophy/videophy_runtime.py",
        "worldfoundry/evaluation/tasks/execution/runners/worldolympiad/run_worldolympiad_official_runner.py",
        "worldfoundry/evaluation/tasks/execution/runners/worldreasonbench/run_worldreasonbench_official_runner.py",
    ):
        text = (repo / rel).read_text(encoding="utf-8")
        assert "from worldfoundry.core.io.paths import" in text, rel
        assert "project_root" in text, rel
        assert "Path(__file__).resolve().parents[" not in text, rel

def test_batch7_pipelines_studio_use_paths_project_root() -> None:
    """pipelines yume/vggt/lyra + studio explorer/capability_registry."""
    repo = Path(__file__).resolve().parents[2]
    for rel in (
        "worldfoundry/pipelines/yume/pipeline_yume.py",
        "worldfoundry/pipelines/yume/pipeline_yume_1p5.py",
        "worldfoundry/pipelines/vggt_omega/pipeline_vggt_omega.py",
        "worldfoundry/pipelines/lyra/lyra_utils.py",
        "worldfoundry/studio/visualization/capability_registry.py",
        "worldfoundry/studio/native/world_explorer/__main__.py",
    ):
        text = (repo / rel).read_text(encoding="utf-8")
        assert "from worldfoundry.core.io.paths import" in text, rel
        assert "project_root" in text, rel
        assert "Path(__file__).resolve().parents[" not in text, rel
        assert "current.parents[" not in text, rel


def test_batch8_training_tests_use_paths_project_root() -> None:
    """Training test helpers climb via project_root, not Path.parents[N>=3]."""
    import re

    repo = Path(__file__).resolve().parents[2]
    deep = re.compile(r"Path\(__file__\)\.resolve\(\)\.parents\[([3-9]|\d{2,})\]")
    paths = sorted(
        str(p.relative_to(repo))
        for p in (repo / "tests/training").rglob("*.py")
        if deep.search(p.read_text(encoding="utf-8"))
    )
    assert paths == [], paths
    # Spot-check known former climbers still import the helper.
    for rel in (
        "tests/training/post_training/agentic/test_qwen_ray_formal.py",
        "tests/training/recipes/test_post_training_recipe_architecture.py",
        "tests/training/post_training/rl/algorithms/bagel_flow_unigrpo/test_recipe_builder.py",
    ):
        text = (repo / rel).read_text(encoding="utf-8")
        assert "from worldfoundry.core.io.paths import" in text, rel
        assert "project_root" in text, rel


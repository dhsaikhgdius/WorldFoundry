from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import yaml

from worldfoundry.evaluation.utils import load_manifest_collection
from worldfoundry.pipelines.component_pipelines import OpenVLAPipeline
from worldfoundry.runtime.conda import (
    is_cuda_profile_supported,
    load_runtime_conda_env_specs,
)
from worldfoundry.synthesis.visual_generation.animatediff.worldfoundry_runtime import DEFAULT_ANIMATEDIFF_REPO_ROOT
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profiles


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_runtime_conda_env_specs_cover_runtime_profiles() -> None:
    specs = load_runtime_conda_env_specs(env_root="/tmp/worldfoundry-conda-test")
    profiles = load_runtime_profiles()
    expected = {
        "step-video-t2v",
        "open-magvit2",
        "show-o",
        "animatediff",
        "zeroscope",
        "cameractrl",
        "motionctrl",
        "dreamdojo",
        "irasim",
        "pandora",
        "splatt3r",
        "pixelsplat",
        "openvla",
        "openpi",
        "giga-brain-0",
        "being-h05",
        "dreamzero",
        "gr00t",
        "starvla",
        "lingbot-va",
        "lapa",
        "octo",
        "rt-1",
        "diffusion-policy",
        "act",
        "roboflamingo",
    }

    assert expected.issubset(profiles)
    assert expected.issubset(specs)
    assert specs["zeroscope"].cuda_profile == "cu128"
    assert specs["zeroscope"].driver_compatible is True
    assert specs["step-video-t2v"].driver_compatible is False
    assert specs["animatediff"].env_name == "worldfoundry-unified-cu128"
    assert "diffusers" in specs["animatediff"].pip_packages
    assert "transformers" in specs["animatediff"].pip_packages
    assert "huggingface_hub" in specs["animatediff"].pip_packages
    assert specs["openvla"].cuda_profile == "cu113"
    assert specs["openvla"].driver_compatible is True
    assert "torch" in specs["openvla"].validation_imports
    assert specs["scope"].pythonpath_dirs == (
        "worldfoundry/synthesis/visual_generation/scope/scope_runtime",
    )
    assert specs["scope"].editable_install_dirs == ()
    assert specs["scope"].source_requirement_files == ()
    assert profiles["openvla"].task_family == "vla_policy"
    assert profiles["openvla"].backend_stage == "in_tree_runtime"
    assert profiles["openvla"].runtime_status == "in_tree_openvla_predict_action_verified"
    assert specs["openpi"].cuda_profile == "cu118"
    assert specs["openpi"].driver_compatible is True
    assert specs["giga-brain-0"].cuda_profile == "cu128"
    assert profiles["giga-brain-0"].task_family == "vla_policy"
    assert profiles["giga-brain-0"].artifact_kind == "action_trace"
    assert specs["being-h05"].cuda_profile == "prepare_only"
    assert profiles["being-h05"].task_family == "vla_policy"
    assert profiles["being-h05"].artifact_kind == "action_trace"
    assert specs["dreamzero"].cuda_profile == "prepare_only"
    assert profiles["dreamzero"].task_family == "world_action_model"
    assert profiles["dreamzero"].artifact_kind == "action_trace"
    assert profiles["dreamzero"].backend_stage == "in_tree_official_server_client"
    assert (
        profiles["dreamzero"].runtime_status
        == "in_tree_official_server_client_checkpoint_gpu_probe_verified_cuda129_multigpu_required"
    )
    assert specs["lingbot-va"].cuda_profile == "prepare_only"
    assert profiles["lingbot-va"].task_family == "embodied_action"
    assert profiles["lingbot-va"].artifact_kind == "action_trace"
    assert specs["lapa"].cuda_profile == "cu118"
    assert specs["lapa"].driver_compatible is True
    assert profiles["lapa"].runtime_status == "in_tree_lapa_7b_openx_jax_gpu_action_tokens_verified"
    assert specs["dreamdojo"].cuda_profile == "prepare_only"
    assert profiles["dreamdojo"].task_family == "world_model"
    assert profiles["dreamdojo"].artifact_kind == "generated_world"
    assert profiles["dreamdojo"].runtime_status == "in_tree_dreamdojo_runtime_ported_dataset_and_gpu_parity_pending"
    assert specs["octo"].cuda_profile == "cu118"
    assert specs["rt-1"].cuda_profile == "cpu"
    assert specs["diffusion-policy"].cuda_profile == "cu113"
    assert profiles["diffusion-policy"].task_family == "visuomotor_policy"
    assert profiles["openpi"].runtime_status == "in_tree_openpi_pi05_libero_jax_gpu_infer_verified"
    assert (
        profiles["giga-brain-0"].runtime_status
        == "in_tree_giga_brain_0_runtime_converted_lerobot_stats_predict_gpu_verified_non_leaderboard"
    )
    assert profiles["octo"].runtime_status == "in_tree_octo_small_jax_gpu_sample_actions_verified"
    assert profiles["rt-1"].runtime_status == "in_tree_rt1_runtime_ported_savedmodel_checkpoint_missing"
    assert profiles["diffusion-policy"].runtime_status == "in_tree_lowdim_pusht_predict_action_ready"
    assert profiles["act"].task_family == "action_chunking_policy"
    assert "torch" in specs["zeroscope"].pip_packages


def test_runtime_conda_envs_do_not_install_from_cache_or_external_model_repos() -> None:
    payload = load_manifest_collection(
        REPO_ROOT / "worldfoundry" / "data" / "models" / "runtime" / "environments",
        item_key="envs",
    )
    forbidden = (
        "cache/",
        str(REPO_ROOT / "cache"),
        str(Path("/", "share", "project", "bench", "model")),
    )

    for env in payload["envs"]:
        for key in ("source_requirement_files", "editable_install_dirs"):
            for item in env.get(key, []):
                assert not any(marker in str(item) for marker in forbidden), (env["model_id"], key, item)


def test_animatediff_default_official_repo_root_uses_integrated_code() -> None:
    expected = REPO_ROOT / "worldfoundry" / "synthesis" / "visual_generation" / "animatediff"

    assert DEFAULT_ANIMATEDIFF_REPO_ROOT.resolve() == expected
    assert "cache" not in DEFAULT_ANIMATEDIFF_REPO_ROOT.resolve().parts


def test_cuda_profile_support_matches_driver_capability() -> None:
    assert is_cuda_profile_supported("cu113", "11.4") is True
    assert is_cuda_profile_supported("cu118", "11.4") is False
    assert is_cuda_profile_supported("prepare_only", "11.4") is True


def test_runtime_profile_plan_records_per_model_conda_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_CONDA_ENV_ROOT", "/tmp/worldfoundry-conda-test")
    pipe = OpenVLAPipeline.from_pretrained(
        {"model_id": "openvla", "plan_only": True},
        device="cpu",
    )

    result = pipe(
        prompt="a robot picks up a cube",
        images="memory://rgb.png",
        interactions=[{"delta": [0.0] * 7}],
        output_path=tmp_path / "openvla_action_trace.json",
        return_dict=True,
    )

    plan = json.loads(Path(result["plan_path"]).read_text(encoding="utf-8"))
    assert plan["profile"]["conda_env"]["env_name"] == "worldfoundry-unified-cu128"
    assert plan["context"]["conda_env_name"] == "worldfoundry-unified-cu128"
    assert plan["context"]["conda_env_prefix"] == "/tmp/worldfoundry-conda-test/worldfoundry-unified-cu128"
    assert plan["context"]["conda_env_driver_status"] == "compatible"
    assert plan["context"]["python"]


def test_model_env_install_script_has_current_open_source_contract() -> None:
    script = REPO_ROOT / "scripts" / "setup" / "model_env_install.sh"
    result = subprocess.run(["bash", "-n", str(script)], cwd=REPO_ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr

    help_result = subprocess.run(["bash", str(script), "--help"], cwd=REPO_ROOT, text=True, capture_output=True)
    assert help_result.returncode == 0
    assert "--model MODEL" in help_result.stdout
    assert "--verify-only" in help_result.stdout
    assert "unified env is the default" in help_result.stdout
    assert "dry" not in help_result.stdout.lower()


def test_model_env_install_list_uses_unified_default_for_runtime_profiles(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "PYTHON": str(Path(os.environ.get("PYTHON", "")) if os.environ.get("PYTHON") else ""),
            "PYTHONPATH": str(REPO_ROOT),
            "CONDA_EXE": "true",
            "WORLDFOUNDRY_CONDA_ENVS_ROOT": str(tmp_path / "envs"),
            "WORLDFOUNDRY_CUDA_PROFILE": "cu128",
            "WORLDFOUNDRY_ALLOW_NO_CUDA": "1",
        }
    )
    if not env["PYTHON"]:
        env.pop("PYTHON")

    result = subprocess.run(
        ["bash", "scripts/setup/model_env_install.sh", "--list", "--env-root", str(tmp_path / "envs")],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    rows = result.stdout.splitlines()
    assert any(row.startswith("animatediff\tworldfoundry-unified-cu128\tcu128") for row in rows)
    assert any(row.startswith("matrix-game-2\tworldfoundry-unified-cu128\tcu128") for row in rows)


def test_runtime_conda_resolver_routes_compatible_profiles_to_unified_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_USE_UNIFIED_ENV", "1")
    monkeypatch.setenv("WORLDFOUNDRY_CUDA_PROFILE", "cu128")
    specs = load_runtime_conda_env_specs(env_root=tmp_path / "envs")

    assert specs["animatediff"].env_name == "worldfoundry-unified-cu128"
    assert specs["zeroscope"].env_name == "worldfoundry-unified-cu128"
    assert specs["matrix-game-2"].env_name == "worldfoundry-unified-cu128"
    assert specs["openpi"].env_name == "worldfoundry-unified-cu128"
    assert "jax==0.4.25" in specs["openpi"].pip_packages
    assert specs["openpi"].pythonpath_dirs == ("worldfoundry/synthesis/action_generation/openpi/openpi_runtime",)

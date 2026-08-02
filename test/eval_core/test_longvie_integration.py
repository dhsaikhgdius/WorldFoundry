from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.evaluation.api import GenerationRequest
from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
from worldfoundry.evaluation.models import WorldFoundryPipelineRunner, resolve_model_zoo_runner
from worldfoundry.pipelines.longvie.pipeline_longvie import LongVie1Pipeline
from worldfoundry.pipelines.longvie.pipeline_longvie import LongVie2Pipeline
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile
from worldfoundry.runtime.conda import load_runtime_conda_env_spec


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CATALOG_DIR = REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog"


def test_longvie_catalog_entries_are_runnable_pipeline_targets() -> None:
    registry = load_model_zoo_registry(MODEL_CATALOG_DIR)

    expected = {
        "longvie-1": "worldfoundry.pipelines.longvie.pipeline_longvie:LongVie1Pipeline",
        "longvie-2": "worldfoundry.pipelines.longvie.pipeline_longvie:LongVie2Pipeline",
    }
    for model_id, pipeline_target in expected.items():
        entry = registry.get(model_id)
        profile = load_runtime_profile(model_id)

        assert entry.integration_status == "integrated"
        assert entry.runner_entry_kind == "runnable_runner"
        assert entry.output_artifacts == ("generated_video",)
        assert entry.pipeline_target == pipeline_target
        assert profile.integration_status == "integrated"
        assert profile.artifact_kind == "generated_video"


def test_longvie_default_call_rejects_request_plan(tmp_path: Path) -> None:
    pipeline = LongVie2Pipeline.from_pretrained(
        model_path={
            "control_weight_path": str(tmp_path / "control.safetensors"),
            "dit_weight_path": str(tmp_path / "dit.safetensors"),
        },
        device="cpu",
    )
    plan_path = tmp_path / "longvie_plan.json"

    with pytest.raises(RuntimeError, match="requires execute=True"):
        pipeline(
            prompt="ride a horse through a valley",
            images="first.png",
            interactions={"dense_video": "depth.mp4", "sparse_video": "track.mp4"},
            output_path=plan_path,
            return_dict=True,
        )

    assert not plan_path.exists()


def test_longvie_default_failure_defers_weight_dir_resolution(tmp_path: Path) -> None:
    missing_weight_dir = tmp_path / "missing-longvie-weights"
    pipeline = LongVie1Pipeline.from_pretrained(model_path=missing_weight_dir, device="cpu")

    with pytest.raises(RuntimeError, match="requires execute=True"):
        pipeline(
            prompt="turn toward the camera",
            images="first.png",
            video={"dense_video": "depth.mp4", "sparse_video": "track.mp4"},
            output_path=tmp_path / "plan.json",
            return_dict=True,
        )

    assert pipeline.synthesis_model.runtime.weight_dir == str(missing_weight_dir)
    assert not (tmp_path / "plan.json").exists()


def test_longvie_runtime_env_specs_track_official_dependencies() -> None:
    for model_id in ("longvie-1", "longvie-2"):
        spec = load_runtime_conda_env_spec(model_id)

        assert spec is not None
        assert spec.env_name == "worldfoundry-longvie-official-cu118"
        assert spec.source_requirement_files == ()
        assert {"torch", "transformers", "einops", "modelscope", "decord", "peft"}.issubset(
            set(spec.validation_imports)
        )


def test_longvie_model_zoo_runner_uses_pipeline_protocol(tmp_path: Path) -> None:
    resolved = resolve_model_zoo_runner(
        "longvie-1",
        manifest_dir=MODEL_CATALOG_DIR,
        runtime={"device": "cpu"},
    )

    assert isinstance(resolved.runner, WorldFoundryPipelineRunner)
    assert isinstance(resolved.runner.pipeline, LongVie1Pipeline)
    result = resolved.runner.generate(
        [
            GenerationRequest(
                sample_id="longvie-smoke",
                task_name="longvie:smoke",
                inputs={
                    "prompt": "walk through a forest",
                    "image": "first.png",
                    "dense_video": "depth.mp4",
                    "sparse_video": "track.mp4",
                },
                generation_kwargs={"output_dir": str(tmp_path)},
            )
        ]
    )[0]

    assert result.status == "failed"
    assert result.artifacts == {}
    assert result.error == "RuntimeError: LongVie requires execute=True; request-plan artifacts are no longer emitted."


def test_longvie_runtime_logic_lives_under_base_models() -> None:
    synthesis_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/longvie"
    base_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/longvie"
    config_root = REPO_ROOT / "worldfoundry/data/models/runtime/configs/longvie"
    runtime_root = base_root / "longvie_runtime"

    assert not (synthesis_root / "_runtime_env.py").exists()
    assert (base_root / "runtime_env.py").is_file()
    assert (base_root / "worldfoundry_runtime.py").is_file()
    assert not (runtime_root / "accelerate_config_14B.yaml").exists()
    assert (config_root / "accelerate_config_14B.yaml").is_file()
    assert not (runtime_root / "train.sh").exists()
    assert not (runtime_root / "train_longvie_control.py").exists()
    assert not (runtime_root / "train_longvie_history_control.py").exists()
    assert not (runtime_root / "setup.py").exists()
    assert not (runtime_root / "download_wan2.1.py").exists()
    assert not (runtime_root / "sample_longvideo.sh").exists()
    assert not (runtime_root / "utils").exists()
    assert (runtime_root / "inference.py").is_file()
    assert not (runtime_root / "requirements.txt").exists()
    assert not (REPO_ROOT / "requirements" / "worldfoundry-vendor-longvie.txt").exists()

    assert not (runtime_root / "README.md").exists()

    text = (synthesis_root / "longvie_synthesis.py").read_text(encoding="utf-8")
    assert "worldfoundry.synthesis.visual_generation.longvie.worldfoundry_runtime" in text
    assert "._runtime_env" not in text
    assert "resolve_wan21" not in text
    assert "resolve_longvie" not in text
    assert "from diffsynth" not in text

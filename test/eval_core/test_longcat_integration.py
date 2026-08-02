from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from worldfoundry.evaluation.api import GenerationRequest
from worldfoundry.evaluation.models import resolve_model_zoo_runner
from worldfoundry.pipelines.longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from worldfoundry.synthesis.visual_generation.longcat_video.worldfoundry_runtime import (
    LongCatVideoRuntime,
)
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CATALOG_DIR = REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog"


def test_longcat_pipeline_default_call_rejects_preflight_artifact(tmp_path: Path) -> None:
    pipeline = LongCatVideoPipeline.from_pretrained(
        model_path=tmp_path / "checkpoint",
        device="cpu",
    )
    plan_path = tmp_path / "longcat_video_plan.json"

    with pytest.raises(RuntimeError, match="requires execute=True"):
        pipeline(
            prompt="a cyclist driving through downtown at dawn",
            output_path=plan_path,
            return_dict=True,
        )

    assert not plan_path.exists()


def test_longcat_profile_declares_generated_video() -> None:
    profile = load_runtime_profile("longcat-video")

    assert profile.artifact_kind == "generated_video"
    assert profile.backend_stage == "in_tree_runtime"
    assert profile.integration_status == "integrated"
    assert profile.artifact_filename == "longcat_video.mp4"


def test_longcat_runner_fails_without_execute(tmp_path: Path) -> None:
    resolved = resolve_model_zoo_runner(
        "longcat-video",
        manifest_dir=MODEL_CATALOG_DIR,
        runtime={"device": "cpu"},
    )

    results = resolved.runner.generate(
        [
            GenerationRequest(
                sample_id="longcat-smoke",
                task_name="longcat:smoke",
                inputs={"prompt": "a robotic dancer in a studio"},
                output_schema={"generated_video": {"kind": "video"}},
            )
        ]
    )

    assert results[0].status == "failed"
    assert not results[0].artifacts
    assert (
        results[0].error
        == "RuntimeError: LongCat-Video requires execute=True; preflight artifacts are no longer emitted."
    )


def test_longcat_inference_inputs_live_in_data_test_cases() -> None:
    runtime_root = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/longcat_video/longcat_video_runtime"
    )
    wrapper_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/longcat_video"
    test_case_root = REPO_ROOT / "worldfoundry/data/test_cases/longcat_video"

    assert (test_case_root / "girl.png").is_file()
    assert (test_case_root / "motorcycle.mp4").is_file()
    assert wrapper_root.is_dir()
    assert not (wrapper_root / "LICENSE").exists()
    assert not (wrapper_root / "run_demo_text_to_video.py").exists()
    assert not (wrapper_root / "longcat_video").exists()
    assert not (runtime_root / "assets").exists()
    assert not (runtime_root / "girl.png").exists()
    assert not (runtime_root / "motorcycle.mp4").exists()

    image_inference = (runtime_root / "run_inference_image_to_video.py").read_text(encoding="utf-8")
    video_inference = (runtime_root / "run_inference_video_continuation.py").read_text(encoding="utf-8")

    assert '"data" / "test_cases" / "longcat_video"' in image_inference
    assert '"data" / "test_cases" / "longcat_video"' in video_inference


def test_longcat_runtime_keeps_only_catalog_supported_inference_scripts() -> None:
    runtime_root = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/longcat_video/longcat_video_runtime"
    )
    removed = [
        "run_streamlit.py",
        "run_demo_interactive_video.py",
        "run_demo_avatar_single_audio_to_video.py",
        "run_demo_avatar_multi_audio_to_video.py",
        "LongCat-Video-Avatar-Tech-Report.pdf",
        "LongCat-Video-Avatar-1.5-Tech-Report.pdf",
        "longcat_video/pipeline_longcat_video_avatar.py",
        "longcat_video/modules/avatar",
        "longcat_video/modules/quantization.py",
        "longcat_video/audio_process",
    ]
    inference_scripts = [
        "run_inference_text_to_video.py",
        "run_inference_image_to_video.py",
        "run_inference_video_continuation.py",
        "run_inference_long_video.py",
    ]

    assert [name for name in removed if (runtime_root / name).exists()] == []
    assert [name for name in inference_scripts if not (runtime_root / name).is_file()] == []
    assert sorted(path.name for path in runtime_root.glob("run_demo*.py")) == []

    runtime = LongCatVideoRuntime(checkpoint_dir=REPO_ROOT / "missing-longcat-checkpoint", device="cpu")
    preflight = runtime.preflight()
    assert preflight["runtime_ready"] is True
    assert sorted(Path(path).name for path in preflight["runtime_scripts"].values()) == sorted(inference_scripts)
    assert preflight["missing_runtime_files"] == []

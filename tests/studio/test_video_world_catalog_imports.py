from __future__ import annotations

import importlib

import pytest

from worldfoundry.studio.catalog import _discover_ast_pipelines, find_entry


def test_abstract_variant_base_pipelines_are_not_workspace_models() -> None:
    discovered_classes = {info.class_name for info in _discover_ast_pipelines()}

    assert "EchoMemoryPipeline" not in discovered_classes
    assert "MatrixGame35Pipeline" not in discovered_classes
    assert "EchoMemoryContextK1Pipeline" in discovered_classes
    assert "MatrixGame35FirstPersonPipeline" in discovered_classes


@pytest.mark.parametrize("model_id", ["framepack", "wan2.1_i2v", "wan2.1_t2v"])
def test_video_catalog_pipeline_target_is_importable(model_id: str) -> None:
    entry = find_entry(model_id)

    module = importlib.import_module(entry.module_path)

    assert getattr(module, entry.class_name) is not None


def test_matrix_game_1_uses_the_staged_in_domain_visual_qa_image() -> None:
    entry = find_entry("matrix-game-1")

    assert entry.default_input_path.endswith(
        "worldfoundry/data/test_cases/matrix-game-1/official_initial_image/forest_00.jpg"
    )


def test_cogvideox_i2v_demo_prompt_matches_its_person_input() -> None:
    entry = find_entry("cogvideox_5b_i2v")

    assert entry.default_input_path.endswith("worldfoundry/data/test_cases/studio_demo/00/image.jpg")
    assert "sparkler" in entry.default_prompt.lower()


def test_studio_demo_i2v_defaults_match_the_sparkler_fixture() -> None:
    for model_id in (
        "dynamicrafter_1024_i2v",
        "dynamicrafter_512_i2v",
        "framepack",
        "longvie-1",
        "ltx_video_i2v",
        "ltx2_i2v",
        "ltx2_3_i2v",
        "skyreels-v3",
        "wan2.1_i2v",
    ):
        entry = find_entry(model_id)
        assert "studio_demo/00/image.jpg" in str(entry.default_input_path)
        assert "sparkler" in entry.default_prompt.lower()


@pytest.mark.parametrize(
    ("model_id", "expected"),
    (
        (
            "wan2.1_i2v",
            {
                "height": 480,
                "width": 832,
                "num_frames": 81,
                "fps": 16,
                "num_inference_steps": 40,
                "shift": 3.0,
                "guidance_scale": 5.0,
                "seed": 42,
            },
        ),
        (
            "wan2.1_t2v",
            {
                "height": 480,
                "width": 832,
                "num_frames": 81,
                "fps": 16,
                "num_inference_steps": 50,
                "shift": 8.0,
                "guidance_scale": 6.0,
                "seed": 42,
            },
        ),
    ),
)
def test_wan21_catalog_uses_native_diffusion_argument_names(
    model_id: str,
    expected: dict[str, object],
) -> None:
    entry = find_entry(model_id)

    assert entry.default_call_kwargs == expected
    assert "sample_steps" not in entry.call_params
    assert "offload_model" not in entry.call_params


@pytest.mark.parametrize(
    ("model_id", "frames", "steps", "height", "width"),
    (
        ("zeroscope", 24, 40, 320, 576),
        ("animatediff", 16, 25, 256, 256),
        ("cogvideox_2b_t2v", 49, 50, 480, 720),
        ("cogvideox_5b_t2v", 49, 50, 480, 720),
    ),
)
def test_quality_validation_defaults_are_not_smoke_reductions(
    model_id: str,
    frames: int,
    steps: int,
    height: int,
    width: int,
) -> None:
    defaults = find_entry(model_id).default_call_kwargs

    assert defaults["num_frames"] == frames
    assert defaults["num_inference_steps"] == steps
    assert defaults["height"] == height
    assert defaults["width"] == width

from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "transformers" dependency at import time; skip when it is unavailable.
pytest.importorskip("transformers")

from worldfoundry.base_models.diffusion_model.optimizations import AttentionBackend
from worldfoundry.base_models.diffusion_model.recipes.hunyuan_video import (
    hunyuan_video15_i2v_recipe,
    hunyuan_video15_t2v_recipe,
)
from worldfoundry.core.inference import get_model_inference_spec
from worldfoundry.pipelines.hunyuan_video.pipeline_hunyuan_video import NativeHunyuanVideoPipeline
from worldfoundry.studio.catalog import find_entry
from worldfoundry.synthesis.visual_generation.official_video_runtime import OfficialVideoRuntime


def test_longsana_workspace_uses_full_long_video_defaults() -> None:
    entry = find_entry("longsana-video-2b-480p")

    assert entry.default_call_kwargs["num_frames"] == 161
    assert entry.default_call_kwargs["num_inference_steps"] == 50
    assert (entry.default_call_kwargs["height"], entry.default_call_kwargs["width"]) == (480, 832)
    assert entry.default_call_kwargs["guidance_scale"] == 1.0
    assert "negative_prompt" in entry.call_params


def test_wan_14b_workspace_uses_full_720p_defaults() -> None:
    entry = find_entry("wan2.1-t2v-14b")

    assert entry.default_model_ref.endswith("Wan2.1-T2V-14B")
    assert entry.default_call_kwargs == {
        "num_frames": 81,
        "height": 720,
        "width": 1280,
        "num_inference_steps": 50,
        "guidance_scale": 6.0,
        "shift": 5.0,
        "fps": 16,
        "seed": 42,
    }


def test_framepack_workspace_uses_official_full_quality_defaults() -> None:
    runtime = OfficialVideoRuntime.from_model_id("framepack")

    assert runtime.runtime["defaults"]["seconds"] == 5
    assert runtime.runtime["defaults"]["latent_window_size"] == 9
    assert runtime.runtime["defaults"]["num_steps"] == 25


def test_wan_fun_camera_workspace_uses_upstream_full_demo_defaults() -> None:
    entry = find_entry("wan21-fun-1p3b-cam")

    assert entry.default_call_kwargs["num_frames"] == 49
    assert entry.default_call_kwargs["num_inference_steps"] == 50
    assert (entry.default_call_kwargs["height"], entry.default_call_kwargs["width"]) == (480, 832)


def test_autoregressive_world_demos_do_not_use_one_block_smoke_defaults() -> None:
    abot = find_entry("abot-world-0-5b-lf")
    longvie = find_entry("longvie-1")

    assert (abot.default_call_kwargs["num_frames"], abot.default_call_kwargs["num_blocks"]) == (57, 5)
    assert longvie.default_call_kwargs["num_frames"] == 81
    assert longvie.default_call_kwargs["num_inference_steps"] == 50


def test_pusa_workspace_uses_official_full_length_720p_defaults() -> None:
    entry = find_entry("pusa-vidgen")

    assert (entry.default_call_kwargs["height"], entry.default_call_kwargs["width"]) == (720, 1280)
    assert entry.default_call_kwargs["num_frames"] == 81
    # Four steps is the official LightX2V distilled schedule, not a smoke reduction.
    assert entry.default_call_kwargs["num_inference_steps"] == 4


def test_hunyuanvideo15_workspace_uses_full_nondistilled_quality_defaults() -> None:
    t2v = find_entry("hunyuanvideo-1.5-t2v")
    i2v = find_entry("hunyuanvideo-1.5-i2v")

    assert (t2v.default_call_kwargs["height"], t2v.default_call_kwargs["width"]) == (720, 1280)
    assert (i2v.default_call_kwargs["height"], i2v.default_call_kwargs["width"]) == (720, 544)
    for entry in (t2v, i2v):
        assert entry.default_call_kwargs["num_frames"] == 121
        assert entry.default_call_kwargs["num_inference_steps"] == 50
        assert entry.default_call_kwargs["guidance_scale"] == 6.0
        assert entry.default_load_kwargs["attention_backend"] == "flash"

    assert hunyuan_video15_t2v_recipe().checkpoints["transformer"].files == (
        "transformer/720p_t2v/diffusion_pytorch_model.safetensors",
    )
    assert hunyuan_video15_i2v_recipe().checkpoints["transformer"].files == (
        "transformer/720p_i2v/diffusion_pytorch_model.safetensors",
    )

    assert NativeHunyuanVideoPipeline._attention_policy(
        "auto", model_id="hunyuanvideo-1.5-t2v"
    ) is AttentionBackend.FLASH
    assert NativeHunyuanVideoPipeline._attention_policy(
        "torch", model_id="hunyuanvideo-1.5-t2v"
    ) is AttentionBackend.TORCH


def test_easyanimate_i2v_default_prompt_matches_its_reference_image() -> None:
    entry = find_entry("easyanimate_i2v")

    assert "sparkler" in entry.default_prompt.lower()
    assert entry.default_input_path.endswith("studio_demo/00/image.jpg")


def test_stable_video_infinity_uses_its_official_480p_demo_fixture() -> None:
    entry = find_entry("stable-video-infinity")
    spec = get_model_inference_spec("stable-video-infinity")

    assert entry.default_input_path.endswith("stable-video-infinity/svi-2.0/frame.jpg")
    assert "water shimmers" in entry.default_prompt.lower()
    assert spec is not None
    defaults = spec.tasks[0].default_call_kwargs
    assert defaults["num_frames"] == 81
    assert defaults["num_inference_steps"] == 50
    assert defaults["num_motion_frames"] == 5
    assert defaults["prompt_repeat_times"] == 2


def test_dreamx_world_defaults_are_directly_runnable() -> None:
    for model_id in ("dreamx-world-5b", "dreamx-world-5b-cam"):
        entry = find_entry(model_id)

        assert "sparkler" in entry.default_prompt.lower()
        assert entry.default_input_path.endswith("dreamx_world/007.jpg")


def test_neoverse_full_demo_offloads_inactive_components_without_reducing_quality() -> None:
    entry = find_entry("neoverse")

    assert entry.default_load_kwargs["enable_vram_management"] is True
    assert entry.default_call_kwargs["num_frames"] == 81
    assert entry.default_call_kwargs["num_inference_steps"] == 4
    assert (entry.default_load_kwargs["height"], entry.default_load_kwargs["width"]) == (336, 560)

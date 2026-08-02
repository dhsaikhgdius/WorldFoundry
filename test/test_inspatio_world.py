import os

import pytest

from worldfoundry.evaluation.utils import worldfoundry_data_path


def test_inspatio_world_gpu_integration(tmp_path):
    if os.environ.get("INSPATIO_WORLD_RUN_INTEGRATION") != "1":
        pytest.skip("Set INSPATIO_WORLD_RUN_INTEGRATION=1 with local checkpoints to run this GPU integration test.")

    device = os.environ.get("INSPATIO_WORLD_DEVICE", "cuda")
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("InSpatio-World GPU integration test requested but CUDA is unavailable.")

    from worldfoundry.pipelines.inspatio_world.pipeline_inspatio_world import InspatioWorldPipeline

    video_path = os.environ.get(
        "INSPATIO_WORLD_INPUT_VIDEO",
        str(worldfoundry_data_path("test_cases", "longcat_video", "motorcycle.mp4")),
    )
    pipeline = InspatioWorldPipeline.from_pretrained(
        model_path=os.environ.get("INSPATIO_WORLD_MODEL_PATH", "inspatio/world"),
        required_components={
            "wan_model_path": os.environ.get("INSPATIO_WORLD_WAN_MODEL_PATH", "Wan-AI/Wan2.1-T2V-1.3B"),
            "da3_model_path": os.environ.get("INSPATIO_WORLD_DA3_MODEL_PATH", "depth-anything/DA3NESTED-GIANT-LARGE"),
            "florence_model_path": os.environ.get("INSPATIO_WORLD_FLORENCE_MODEL_PATH", "microsoft/Florence-2-large"),
        },
        device=device,
    )
    result = pipeline(
        videos=video_path,
        traj_txt_path=os.environ.get("INSPATIO_WORLD_TRAJ", "x_y_circle_cycle.txt"),
        prompt="A motorcycle moves through a natural outdoor scene.",
        output_dir=str(tmp_path / "inspatio_world_output"),
        return_dict=True,
    )
    assert result["generated_video_paths"]

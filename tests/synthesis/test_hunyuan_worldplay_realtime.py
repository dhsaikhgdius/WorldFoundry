
import pytest

# This test module imports worldfoundry code that requires the optional
# "loguru" dependency at import time; skip when it is unavailable.
pytest.importorskip("loguru")
from types import SimpleNamespace

import torch

from worldfoundry.pipelines.hunyuan_world.pipeline_hunyuan_worldplay import (
    HunyuanWorldPlayPipeline,
)
from worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_worldplay import realtime
from worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_worldplay.models.autoencoders import (
    hunyuanvideo_15_vae_w_cache as vae_module,
)
from worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_worldplay.utils import (
    infer_utils,
)


def test_torch_compile_wrapper_reuses_compiled_callable(monkeypatch) -> None:
    state = SimpleNamespace(enable_torch_compile=True)
    compile_calls = []

    def fake_compile(function):
        compile_calls.append(function)
        return function

    monkeypatch.setattr(infer_utils, "get_infer_state", lambda: state)
    monkeypatch.setattr(infer_utils.torch, "compile", fake_compile)

    class Module:
        @infer_utils.torch_compile_wrapper()
        def forward(self, value):
            return value + 1

    module = Module()
    assert module.forward(1) == 2
    assert module.forward(2) == 3
    assert len(compile_calls) == 1


def test_causal_vae_stream_keeps_cache_and_native_cadence() -> None:
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cache_ids = []

        def forward(self, z, *, feat_cache, feat_idx, first_chunk=False):
            self.cache_ids.append(id(feat_cache))
            feat_cache[0] = z[:, :, -1:].clone()
            feat_idx[0] += 1
            temporal = 1 if first_chunk else 4
            return torch.zeros(z.shape[0], 3, temporal, 2, 2)

    vae = vae_module.AutoencoderKLConv3D.__new__(vae_module.AutoencoderKLConv3D)
    torch.nn.Module.__init__(vae)
    vae.decoder = Decoder()
    vae._cached_conv_counts = {"decoder": 1, "encoder": 0}
    vae.use_slicing = False
    vae.use_temporal_tiling = False
    vae.use_spatial_tiling = False
    vae._tile_parallelism_enabled = False

    first = vae.decode_stream(torch.zeros(1, 1, 4, 1, 1), return_dict=False)[0]
    second = vae.decode_stream(torch.zeros(1, 1, 4, 1, 1), return_dict=False)[0]
    assert first.shape[2] == 13
    assert second.shape[2] == 16
    assert len(set(vae.decoder.cache_ids)) == 1
    vae.reset_stream_decode()
    assert vae._stream_decode_active is False


def test_realtime_camera_history_grows_by_native_latent_blocks() -> None:
    session = realtime.HunyuanWorldPlayRealtimeSession.__new__(
        realtime.HunyuanWorldPlayRealtimeSession
    )
    session.operator = SimpleNamespace(
        forward_speed=0.08,
        yaw_speed_deg=3.0,
        pitch_speed_deg=3.0,
    )
    session._motions = []
    session._latents = None

    viewmats, intrinsics, actions = session._camera_conditions(["forward"], None)
    assert viewmats.shape == (1, 4, 4, 4)
    assert intrinsics.shape == (1, 4, 3, 3)
    assert actions.shape == (1, 4)

    session._latents = torch.empty(1, 1, 4, 1, 1)
    viewmats, intrinsics, actions = session._camera_conditions(["camera_r"], None)
    assert viewmats.shape == (1, 8, 4, 4)
    assert intrinsics.shape == (1, 8, 3, 3)
    assert actions.shape == (1, 8)


def test_pipeline_reports_hy_worldplay_native_realtime_spec() -> None:
    runtime = SimpleNamespace(
        model=SimpleNamespace(execution_device="cpu", target_dtype=torch.float32)
    )
    synthesis = SimpleNamespace(runtime=runtime)
    operator = SimpleNamespace()
    pipeline = HunyuanWorldPlayPipeline(
        synthesis_model=synthesis,
        operators=operator,
        device="cpu",
    )
    assert pipeline.prepare_realtime() == {
        "realtime_spec": {
            "fps": 24,
            "first_chunk_frames": 13,
            "steady_chunk_frames": 16,
            "controls": [
                "forward",
                "backward",
                "left",
                "right",
                "camera_up",
                "camera_down",
                "camera_l",
                "camera_r",
            ],
            "transport": "in-memory-rgb",
            "stateful": True,
        }
    }

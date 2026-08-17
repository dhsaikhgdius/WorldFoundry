from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "ftfy" dependency at import time; skip when it is unavailable.
pytest.importorskip("ftfy")

import torch

from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.component import (
    LTXTensorVideoCodec,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.video_vae import (
    _module_execution_target,
)
from worldfoundry.base_models.diffusion_model.models.initializers.sana import (
    SanaNoiseInitializer,
)


def test_sana_720p_geometry_pads_latent_height_without_reducing_resolution() -> None:
    initializer = SanaNoiseInitializer(
        channels=128,
        spatial_compression=32,
        temporal_compression=8,
        allow_spatial_padding=True,
    )

    shape = initializer.latent_shape(
        DiffusionRequest(prompt="test", height=720, width=1280, num_frames=81)
    )

    assert shape == (1, 128, 11, 23, 40)


def test_ltx_tensor_codec_crops_padded_decode_to_requested_pixels() -> None:
    class FakeEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

    class FakeDecoderBody:
        def tiled_decode(self, latents: torch.Tensor, tiling):
            del latents, tiling
            yield torch.zeros(1, 3, 9, 64, 64)

    class FakeDecoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.decoder = FakeDecoderBody()

    codec = LTXTensorVideoCodec(FakeEncoder(), FakeDecoder(), tiling=None)

    video = codec.decode(
        torch.zeros(1, 128, 2, 2, 2),
        DiffusionRequest(prompt="test", height=33, width=63, num_frames=9),
    )

    assert video.shape == (1, 3, 9, 33, 63)


def test_ltx_tensor_codec_uses_explicit_compute_target_for_wrapped_modules() -> None:
    class FakeEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

    class FakeDecoderBody:
        def __init__(self) -> None:
            self.received: torch.Tensor | None = None

        def tiled_decode(self, latents: torch.Tensor, tiling):
            del tiling
            self.received = latents
            yield torch.zeros(1, 3, 9, 32, 32, dtype=latents.dtype, device=latents.device)

    class FakeDecoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
            self.decoder = FakeDecoderBody()

    decoder = FakeDecoder()
    codec = LTXTensorVideoCodec(
        FakeEncoder(),
        decoder,
        tiling=None,
        compute_device="cpu",
        compute_dtype=torch.bfloat16,
    )

    codec.decode(torch.zeros(1, 128, 2, 1, 1, dtype=torch.float32))

    assert decoder.decoder.received is not None
    assert decoder.decoder.received.dtype is torch.bfloat16


def test_ltx_tiled_encoder_prefers_vram_wrapper_execution_target() -> None:
    class FakeManagedLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
            self.computation_device = torch.device("cuda:3")
            self.computation_dtype = torch.bfloat16

    module = torch.nn.Sequential(FakeManagedLayer())

    device, dtype = _module_execution_target(module)

    assert device == torch.device("cuda:3")
    assert dtype is torch.bfloat16

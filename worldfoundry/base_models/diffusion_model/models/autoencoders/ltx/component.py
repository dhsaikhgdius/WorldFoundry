"""Native LTX video/audio autoencoder components."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as functional

from ....components import ComponentBuildContext
from ....contracts import DiffusionRequest, ModalityState
from ....loaders import CheckpointSpec, ModuleLoadSpec, NativeModuleLoader, safetensors_json_metadata
from ....optimizations import RuntimePolicy
from ....runners import MultiModalLatentDecoder
from ...representations.ltx.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ...representations.ltx.types import AudioLatentShape, VideoLatentShape, VideoPixelShape
from .audio import AudioDecoderConfigurator, VocoderConfigurator, decode_audio
from .audio.vocoder import Snake, SnakeBeta
from .video import (
    SpatialTilingConfig,
    TemporalTilingConfig,
    TilingConfig,
    VideoDecoderConfigurator,
    VideoEncoderConfigurator,
)


class LTXVideoEncoderModule(torch.nn.Module):
    def __init__(self, checkpoint_config: Mapping[str, object]) -> None:
        super().__init__()
        self.encoder = VideoEncoderConfigurator.from_config(dict(checkpoint_config))

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.encoder(sample)


class LTXVideoDecoderModule(torch.nn.Module):
    def __init__(self, checkpoint_config: Mapping[str, object]) -> None:
        super().__init__()
        self.decoder = VideoDecoderConfigurator.from_config(dict(checkpoint_config))


class LTXAudioDecoderModule(torch.nn.Module):
    def __init__(self, checkpoint_config: Mapping[str, object]) -> None:
        super().__init__()
        self.decoder = AudioDecoderConfigurator.from_config(dict(checkpoint_config))


class LTXVocoderModule(torch.nn.Module):
    def __init__(self, checkpoint_config: Mapping[str, object]) -> None:
        super().__init__()
        self.vocoder = VocoderConfigurator.from_config(dict(checkpoint_config))


def _prefixed_state_dict(
    state_dict: Mapping[str, object],
    *,
    prefixes: Mapping[str, str],
) -> Mapping[str, object]:
    converted: dict[str, object] = {}
    for key, value in state_dict.items():
        for source, destination in prefixes.items():
            if key.startswith(source):
                suffix = key.removeprefix(source)
                if "per_channel_statistics." in destination and suffix not in {
                    "mean-of-means",
                    "std-of-means",
                }:
                    break
                converted[f"{destination}{suffix}"] = value
                break
    return converted


def convert_ltx_video_encoder_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    return _prefixed_state_dict(
        state_dict,
        prefixes={
            "vae.encoder.": "encoder.",
            "vae.per_channel_statistics.": "encoder.per_channel_statistics.",
        },
    )


def convert_ltx_video_decoder_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    return _prefixed_state_dict(
        state_dict,
        prefixes={
            "vae.decoder.": "decoder.",
            "vae.per_channel_statistics.": "decoder.per_channel_statistics.",
        },
    )


def convert_ltx_audio_decoder_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    return _prefixed_state_dict(
        state_dict,
        prefixes={
            "audio_vae.decoder.": "decoder.",
            "audio_vae.per_channel_statistics.": "decoder.per_channel_statistics.",
        },
    )


def convert_ltx_vocoder_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    return _prefixed_state_dict(state_dict, prefixes={"vocoder.": "vocoder."})


def _load_configured_module(
    checkpoint: CheckpointSpec,
    policy: RuntimePolicy,
    *,
    module_class: type[torch.nn.Module],
    converter,
) -> torch.nn.Module:
    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    return NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=module_class,
            config_resolver=lambda checkpoint: {"checkpoint_config": safetensors_json_metadata(checkpoint)},
            state_dict_converter=converter,
            vram_module_map={
                Snake: AutoWrappedModule,
                SnakeBeta: AutoWrappedModule,
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv1d: AutoWrappedModule,
                torch.nn.Conv2d: AutoWrappedModule,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.ConvTranspose1d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                torch.nn.GroupNorm: AutoWrappedModule,
            },
        ),
        checkpoint,
        policy,
    )


def load_ltx_video_encoder(context: ComponentBuildContext) -> LTXVideoEncoderModule:
    return load_ltx_video_encoder_checkpoint(
        context.require_checkpoint("weights"),
        context.policy,
    )


def load_ltx_video_encoder_checkpoint(
    checkpoint: CheckpointSpec,
    policy: RuntimePolicy,
) -> LTXVideoEncoderModule:
    module = _load_configured_module(
        checkpoint,
        policy,
        module_class=LTXVideoEncoderModule,
        converter=convert_ltx_video_encoder_state_dict,
    )
    if not isinstance(module, LTXVideoEncoderModule):
        raise TypeError(f"expected LTXVideoEncoderModule, got {type(module).__name__}")
    return module


def load_ltx_video_decoder_checkpoint(
    checkpoint: CheckpointSpec,
    policy: RuntimePolicy,
) -> LTXVideoDecoderModule:
    module = _load_configured_module(
        checkpoint,
        policy,
        module_class=LTXVideoDecoderModule,
        converter=convert_ltx_video_decoder_state_dict,
    )
    if not isinstance(module, LTXVideoDecoderModule):
        raise TypeError(f"expected LTXVideoDecoderModule, got {type(module).__name__}")
    return module


class LTXMediaDecoder(MultiModalLatentDecoder):
    """Decode final LTX modality states into video and synchronized audio."""

    def __init__(
        self,
        video: LTXVideoDecoderModule,
        audio: LTXAudioDecoderModule,
        vocoder: LTXVocoderModule,
        *,
        compute_dtype: torch.dtype,
        tiling: TilingConfig | None,
    ) -> None:
        self.video = video
        self.audio = audio
        self.vocoder = vocoder
        self.compute_dtype = compute_dtype
        self.tiling = tiling

    @staticmethod
    def _shapes(request: DiffusionRequest) -> tuple[VideoLatentShape, AudioLatentShape]:
        fps = float(request.inputs.get("frame_rate", 24.0))
        pixels = VideoPixelShape(
            batch=request.batch_size,
            frames=request.num_frames,
            height=request.height,
            width=request.width,
            fps=fps,
        )
        return VideoLatentShape.from_pixel_shape(pixels), AudioLatentShape.from_video_pixel_shape(pixels)

    def decode_modalities(
        self,
        states: Mapping[str, ModalityState],
        request: DiffusionRequest,
    ) -> Mapping[str, object]:
        video_shape, audio_shape = self._shapes(request)
        video_latent = VideoLatentPatchifier(1).unpatchify(states["video"].latent, video_shape)
        audio_latent = AudioPatchifier(1).unpatchify(states["audio"].latent, audio_shape)
        with torch.autocast(
            device_type=video_latent.device.type,
            dtype=self.compute_dtype,
            enabled=self.compute_dtype in {torch.float16, torch.bfloat16},
        ):
            chunks = list(
                self.video.decoder.decode_video(
                    video_latent.to(dtype=self.compute_dtype),
                    self.tiling,
                )
            )
            if not chunks:
                raise RuntimeError("LTX video decoder returned no chunks")
            decoded_audio = decode_audio(
                audio_latent.to(dtype=self.compute_dtype),
                self.audio.decoder,
                self.vocoder.vocoder,
            )
        return {
            "video": torch.cat(chunks, dim=0),
            "audio": decoded_audio.waveform,
            "audio_sampling_rate": decoded_audio.sampling_rate,
        }


class LTXVideoMediaDecoder(MultiModalLatentDecoder):
    """Decode the video-only LTX-Video latent state."""

    def __init__(
        self,
        video: LTXVideoDecoderModule,
        *,
        tiling: TilingConfig | None,
        first_stage_scale: float | None = None,
        latent_upsample_factor: int = 1,
        tone_map_compression: float = 0.0,
    ) -> None:
        self.video = video
        self.tiling = tiling
        if first_stage_scale is not None and not 0.0 < float(first_stage_scale) <= 1.0:
            raise ValueError("LTX first-stage scale must be in (0, 1]")
        if int(latent_upsample_factor) < 1:
            raise ValueError("LTX latent upsample factor must be positive")
        if not 0.0 <= float(tone_map_compression) <= 1.0:
            raise ValueError("LTX tone-map compression must be in [0, 1]")
        self.first_stage_scale = None if first_stage_scale is None else float(first_stage_scale)
        self.latent_upsample_factor = int(latent_upsample_factor)
        self.tone_map_compression = float(tone_map_compression)

    @staticmethod
    def _round_down_to_vae(value: int) -> int:
        result = int(value) - int(value) % 32
        if result < 32:
            raise ValueError("LTX stage dimensions must be at least 32 pixels")
        return result

    def _latent_pixel_size(self, request: DiffusionRequest) -> tuple[int, int]:
        if self.first_stage_scale is None:
            return request.height, request.width
        return (
            self._round_down_to_vae(int(request.height * self.first_stage_scale))
            * self.latent_upsample_factor,
            self._round_down_to_vae(int(request.width * self.first_stage_scale))
            * self.latent_upsample_factor,
        )

    def _tone_map(self, latent: torch.Tensor) -> torch.Tensor:
        if self.tone_map_compression == 0.0:
            return latent
        scale = self.tone_map_compression * 0.75
        sigmoid = torch.sigmoid(4.0 * scale * (latent.abs() - 1.0))
        return latent * (1.0 - 0.8 * scale * sigmoid)

    def decode_modalities(
        self,
        states: Mapping[str, ModalityState],
        request: DiffusionRequest,
    ) -> Mapping[str, object]:
        latent_height, latent_width = self._latent_pixel_size(request)
        pixels = VideoPixelShape(
            batch=request.batch_size,
            frames=request.num_frames,
            height=latent_height,
            width=latent_width,
            fps=float(request.inputs.get("frame_rate", 24.0)),
        )
        video_shape = VideoLatentShape.from_pixel_shape(pixels)
        video_latent = VideoLatentPatchifier(1).unpatchify(states["video"].latent, video_shape)
        video_latent = self._tone_map(video_latent)
        chunks = list(self.video.decoder.decode_video(video_latent, self.tiling))
        if not chunks:
            raise RuntimeError("LTX-Video decoder returned no chunks")
        video = torch.cat(chunks, dim=0)
        if (latent_height, latent_width) != (request.height, request.width):
            dtype = video.dtype
            video = functional.interpolate(
                video.permute(0, 3, 1, 2).float(),
                size=(request.height, request.width),
                mode="bilinear",
                align_corners=False,
            ).to(dtype=dtype).permute(0, 2, 3, 1)
        return {"video": video}


class LTXTensorVideoCodec:
    """BCTHW encoder/decoder role for models that diffuse directly in LTX latents."""

    spatial_compression_factor = 32
    temporal_compression_factor = 8
    latent_ch = 128

    def __init__(
        self,
        encoder: LTXVideoEncoderModule,
        decoder: LTXVideoDecoderModule,
        *,
        tiling: TilingConfig | None,
        compute_device: str | torch.device | None = None,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.tiling = tiling
        self.compute_device = None if compute_device is None else torch.device(compute_device)
        self.compute_dtype = compute_dtype

    def _input_target(self, module: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
        parameter = next(module.parameters())
        return (
            self.compute_device if self.compute_device is not None else parameter.device,
            self.compute_dtype if self.compute_dtype is not None else parameter.dtype,
        )

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or images.shape[1] != 3:
            raise ValueError(f"LTX tensor codec expects BCTHW RGB video, got {tuple(images.shape)}")
        device, dtype = self._input_target(self.encoder)
        images = images.to(device=device, dtype=dtype)
        return self.encoder.encoder.tiled_encode(images, self.tiling)

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        request: DiffusionRequest | None = None,
    ) -> torch.Tensor:
        if request is not None and bool(request.inputs.get("return_latent", False)):
            return latents
        device, dtype = self._input_target(self.decoder)
        latents = latents.to(device=device, dtype=dtype)
        chunks = list(self.decoder.decoder.tiled_decode(latents, self.tiling))
        if not chunks:
            raise RuntimeError("LTX tensor decoder returned no chunks")
        video = torch.cat(chunks, dim=2)
        if request is not None:
            decoded_height, decoded_width = video.shape[-2:]
            if decoded_height < request.height or decoded_width < request.width:
                raise ValueError(
                    "LTX tensor decoder output is smaller than the requested crop: "
                    f"{decoded_height}x{decoded_width} < {request.height}x{request.width}"
                )
            if (decoded_height, decoded_width) != (request.height, request.width):
                top = (decoded_height - request.height) // 2
                left = (decoded_width - request.width) // 2
                video = video[
                    ...,
                    top : top + request.height,
                    left : left + request.width,
                ]
        return video.clamp_(-1.0, 1.0)


def _tiling_from_options(options: Mapping[str, object]) -> TilingConfig | None:
    if options.get("tiled", True) is False:
        return None
    return TilingConfig(
        spatial_config=SpatialTilingConfig(
            tile_size_in_pixels=int(options.get("spatial_tile_size", 768)),
            tile_overlap_in_pixels=int(options.get("spatial_overlap", 64)),
        ),
        temporal_config=TemporalTilingConfig(
            tile_size_in_frames=int(options.get("temporal_tile_size", 80)),
            tile_overlap_in_frames=int(options.get("temporal_overlap", 24)),
        ),
    )


def build_ltx_media_decoder(context: ComponentBuildContext) -> LTXMediaDecoder:
    video = _load_configured_module(
        context.require_checkpoint("weights"),
        context.policy,
        module_class=LTXVideoDecoderModule,
        converter=convert_ltx_video_decoder_state_dict,
    )
    audio = _load_configured_module(
        context.require_checkpoint("weights"),
        context.policy,
        module_class=LTXAudioDecoderModule,
        converter=convert_ltx_audio_decoder_state_dict,
    )
    vocoder = _load_configured_module(
        context.require_checkpoint("weights"),
        context.policy,
        module_class=LTXVocoderModule,
        converter=convert_ltx_vocoder_state_dict,
    )
    if not isinstance(video, LTXVideoDecoderModule):
        raise TypeError("LTX video decoder construction returned an unexpected module")
    if not isinstance(audio, LTXAudioDecoderModule):
        raise TypeError("LTX audio decoder construction returned an unexpected module")
    if not isinstance(vocoder, LTXVocoderModule):
        raise TypeError("LTX vocoder construction returned an unexpected module")
    return LTXMediaDecoder(
        video,
        audio,
        vocoder,
        compute_dtype=context.policy.dtype,
        tiling=_tiling_from_options(context.component_options),
    )


def build_ltx_video_media_decoder(context: ComponentBuildContext) -> LTXVideoMediaDecoder:
    video = load_ltx_video_decoder_checkpoint(
        context.require_checkpoint("weights"),
        context.policy,
    )
    raw_scale = context.component_options.get("first_stage_scale")
    return LTXVideoMediaDecoder(
        video,
        tiling=_tiling_from_options(context.component_options),
        first_stage_scale=None if raw_scale is None else float(raw_scale),
        latent_upsample_factor=int(context.component_options.get("latent_upsample_factor", 1)),
        tone_map_compression=float(context.component_options.get("tone_map_compression", 0.0)),
    )


def build_ltx_tensor_video_codec(context: ComponentBuildContext) -> LTXTensorVideoCodec:
    """Build one shared LTX encoder/decoder pair for standard diffusion roles."""

    checkpoint = context.require_checkpoint("weights")
    return LTXTensorVideoCodec(
        load_ltx_video_encoder_checkpoint(checkpoint, context.policy),
        load_ltx_video_decoder_checkpoint(checkpoint, context.policy),
        tiling=_tiling_from_options(context.component_options),
        compute_device=context.policy.device,
        compute_dtype=context.policy.dtype,
    )


__all__ = [
    "LTXMediaDecoder",
    "LTXVideoMediaDecoder",
    "LTXTensorVideoCodec",
    "LTXVideoEncoderModule",
    "build_ltx_media_decoder",
    "build_ltx_video_media_decoder",
    "build_ltx_tensor_video_codec",
    "convert_ltx_audio_decoder_state_dict",
    "convert_ltx_video_decoder_state_dict",
    "convert_ltx_video_encoder_state_dict",
    "convert_ltx_vocoder_state_dict",
    "load_ltx_video_encoder",
    "load_ltx_video_decoder_checkpoint",
    "load_ltx_video_encoder_checkpoint",
]

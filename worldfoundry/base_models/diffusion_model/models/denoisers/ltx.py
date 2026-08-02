"""Native LTX audio-video denoiser component."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from ...components import ComponentBuildContext
from ...contracts import MultiModalDenoiserInput, MultiModalDenoiserOutput
from ...loaders import ModuleLoadSpec, NativeModuleLoader, safetensors_json_metadata
from ..networks.ltx.modality import Modality
from ..networks.ltx.model import LTXModel
from ..networks.ltx.perturbations import BatchedPerturbationConfig
from ..networks.ltx.transformer import BasicAVTransformerBlock
from .ltx_configurator import LTXModelConfigurator, LTXVideoOnlyModelConfigurator


class LTXAVTransformerModule(torch.nn.Module):
    """Checkpoint-configured LTX transformer with a stable loader surface."""

    def __init__(self, checkpoint_config: Mapping[str, object]) -> None:
        super().__init__()
        self.velocity_model = LTXModelConfigurator.from_config(dict(checkpoint_config))

    def forward(self, *args, **kwargs):
        return self.velocity_model(*args, **kwargs)


class LTXVideoTransformerModule(torch.nn.Module):
    """Checkpoint-configured LTX-Video transformer."""

    def __init__(self, checkpoint_config: Mapping[str, object]) -> None:
        super().__init__()
        self.velocity_model = LTXVideoOnlyModelConfigurator.from_config(dict(checkpoint_config))

    def forward(self, *args, **kwargs):
        return self.velocity_model(*args, **kwargs)


def convert_ltx_transformer_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    """Select transformer weights and strip the official container prefix."""

    prefix = "model.diffusion_model."
    excluded = (
        f"{prefix}video_embeddings_connector.",
        f"{prefix}audio_embeddings_connector.",
    )
    return {
        f"velocity_model.{key.removeprefix(prefix)}": value
        for key, value in state_dict.items()
        if key.startswith(prefix) and not key.startswith(excluded)
    }


def convert_ltx_video_transformer_state_dict(
    state_dict: Mapping[str, object],
) -> Mapping[str, object]:
    """Map an official LTX-Video 0.9.x transformer into the native module."""

    prefix = "model.diffusion_model."
    return {
        f"velocity_model.{key.removeprefix(prefix)}": value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


class LTXJointDenoiser:
    """Adapt the LTX joint velocity model to the canonical multi-modal contract."""

    def __init__(
        self,
        model: LTXAVTransformerModule,
        *,
        compute_dtype: torch.dtype,
    ) -> None:
        self.model = model
        self.compute_dtype = compute_dtype

    @staticmethod
    def _modality(
        state,
        *,
        sigma: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
    ) -> Modality:
        batch = state.latent.shape[0]
        sigma = sigma.reshape(-1)
        if sigma.numel() == 1:
            sigma = sigma.expand(batch)
        if sigma.numel() != batch:
            raise ValueError("LTX sigma must be scalar or contain one value per sample")
        mask_sigma = sigma.view(batch, *([1] * (state.denoise_mask.ndim - 1)))
        return Modality(
            latent=state.latent,
            sigma=sigma,
            timesteps=state.denoise_mask * mask_sigma,
            positions=state.positions,
            context=context,
            context_mask=context_mask,
            attention_mask=state.attention_mask,
        )

    def __call__(self, model_input: MultiModalDenoiserInput) -> MultiModalDenoiserOutput:
        try:
            video_state = model_input.modalities["video"]
            audio_state = model_input.modalities["audio"]
            video_context = model_input.conditioning["video_context"]
            audio_context = model_input.conditioning["audio_context"]
        except KeyError as error:
            raise KeyError(f"LTX joint denoising is missing {error.args[0]!r}") from error
        if not isinstance(video_context, torch.Tensor) or not isinstance(audio_context, torch.Tensor):
            raise TypeError("LTX video_context and audio_context must be tensors")
        context_mask = model_input.conditioning.get("context_mask")
        if context_mask is not None and not isinstance(context_mask, torch.Tensor):
            raise TypeError("LTX context_mask must be a tensor")

        video = self._modality(
            video_state,
            sigma=model_input.timestep,
            context=video_context,
            context_mask=context_mask,
        )
        audio = self._modality(
            audio_state,
            sigma=model_input.timestep,
            context=audio_context,
            context_mask=context_mask,
        )
        perturbations = BatchedPerturbationConfig.empty(video_state.latent.shape[0])
        with torch.autocast(
            device_type=video_state.latent.device.type,
            dtype=self.compute_dtype,
            enabled=self.compute_dtype in {torch.float16, torch.bfloat16},
        ):
            video_velocity, audio_velocity = self.model(video, audio, perturbations)
        if video_velocity is None or audio_velocity is None:
            raise RuntimeError("LTX audio-video transformer returned an absent modality")
        return MultiModalDenoiserOutput(
            samples={
                "video": video.latent - video.timesteps * video_velocity,
                "audio": audio.latent - audio.timesteps * audio_velocity,
            }
        )


class LTXVideoDenoiser:
    """Adapt the video-only velocity model to the joint-state runner contract."""

    def __init__(
        self,
        model: LTXVideoTransformerModule,
        *,
        compute_dtype: torch.dtype,
    ) -> None:
        self.model = model
        self.compute_dtype = compute_dtype

    def __call__(self, model_input: MultiModalDenoiserInput) -> MultiModalDenoiserOutput:
        try:
            state = model_input.modalities["video"]
            context = model_input.conditioning["video_context"]
        except KeyError as error:
            raise KeyError(f"LTX-Video denoising is missing {error.args[0]!r}") from error
        if not isinstance(context, torch.Tensor):
            raise TypeError("LTX-Video context must be a tensor")
        context_mask = model_input.conditioning.get("context_mask")
        if context_mask is not None and not isinstance(context_mask, torch.Tensor):
            raise TypeError("LTX-Video context_mask must be a tensor")
        video = LTXJointDenoiser._modality(
            state,
            sigma=model_input.timestep,
            context=context,
            context_mask=context_mask,
        )
        perturbations = BatchedPerturbationConfig.empty(state.latent.shape[0])
        with torch.autocast(
            device_type=state.latent.device.type,
            dtype=self.compute_dtype,
            enabled=self.compute_dtype in {torch.float16, torch.bfloat16},
        ):
            velocity, audio_velocity = self.model(video, None, perturbations)
        if velocity is None or audio_velocity is not None:
            raise RuntimeError("LTX-Video transformer returned an unexpected modality set")
        return MultiModalDenoiserOutput(samples={"video": video.latent - video.timesteps * velocity})


def build_ltx_joint_denoiser(context: ComponentBuildContext) -> LTXJointDenoiser:
    """Load one LTX-2.x AV transformer through the shared native loader."""

    from worldfoundry.core.vram import (
        AutoWrappedLinear,
        AutoWrappedModule,
        AutoWrappedNonRecurseModule,
    )

    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=LTXAVTransformerModule,
            config_resolver=lambda checkpoint: {"checkpoint_config": safetensors_json_metadata(checkpoint)},
            state_dict_converter=convert_ltx_transformer_state_dict,
            vram_module_map={
                LTXModel: AutoWrappedNonRecurseModule,
                BasicAVTransformerBlock: AutoWrappedNonRecurseModule,
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.LayerNorm: AutoWrappedModule,
                torch.nn.RMSNorm: AutoWrappedModule,
            },
            layer_container="velocity_model.transformer_blocks",
        ),
        context.require_checkpoint("weights"),
        context.policy,
    )
    if not isinstance(model, LTXAVTransformerModule):
        raise TypeError(f"expected LTXAVTransformerModule, got {type(model).__name__}")
    model.velocity_model.refresh_preprocessor_bindings()
    return LTXJointDenoiser(model, compute_dtype=context.policy.dtype)


def build_ltx_video_denoiser(context: ComponentBuildContext) -> LTXVideoDenoiser:
    """Load LTX-Video 0.9.x through the shared native loader."""

    from worldfoundry.core.vram import (
        AutoWrappedLinear,
        AutoWrappedModule,
        AutoWrappedNonRecurseModule,
    )

    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=LTXVideoTransformerModule,
            config_resolver=lambda checkpoint: {"checkpoint_config": safetensors_json_metadata(checkpoint)},
            state_dict_converter=convert_ltx_video_transformer_state_dict,
            vram_module_map={
                LTXModel: AutoWrappedNonRecurseModule,
                BasicAVTransformerBlock: AutoWrappedNonRecurseModule,
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.LayerNorm: AutoWrappedModule,
                torch.nn.RMSNorm: AutoWrappedModule,
            },
            layer_container="velocity_model.transformer_blocks",
        ),
        context.require_checkpoint("weights"),
        context.policy,
    )
    if not isinstance(model, LTXVideoTransformerModule):
        raise TypeError(f"expected LTXVideoTransformerModule, got {type(model).__name__}")
    model.velocity_model.refresh_preprocessor_bindings()
    return LTXVideoDenoiser(model, compute_dtype=context.policy.dtype)


__all__ = [
    "LTXAVTransformerModule",
    "LTXJointDenoiser",
    "LTXVideoDenoiser",
    "LTXVideoTransformerModule",
    "build_ltx_joint_denoiser",
    "build_ltx_video_denoiser",
    "convert_ltx_transformer_state_dict",
    "convert_ltx_video_transformer_state_dict",
]

"""Native SANA flow adapter for Score Identity Distillation."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from worldfoundry.base_models.diffusion_model.contracts import (
    Conditioning,
    DenoiserInput,
    DenoiserOutput,
    DiffusionRequest,
)
from worldfoundry.training.models.sana import SanaTrainAdapter
from worldfoundry.training.objectives.flow_matching import flow_interpolate
from worldfoundry.training.post_training.shared.prediction import NativeFlowPredictionAdapter


class SanaSIDPredictionAdapter:
    """Bind one independently loaded SANA role to the SiD functional seam."""

    noise_process_kind = "flow-matching"

    def __init__(
        self,
        adapter: SanaTrainAdapter,
        *,
        checkpoint_identity: str,
        autocast_dtype: torch.dtype | None = None,
    ) -> None:
        if not isinstance(adapter, SanaTrainAdapter):
            raise TypeError("adapter must be SanaTrainAdapter")
        identity = str(checkpoint_identity).strip()
        if not identity:
            raise ValueError("checkpoint_identity must be non-empty")
        self.adapter = adapter
        self.prediction = NativeFlowPredictionAdapter(
            adapter,
            autocast_dtype=autocast_dtype,
        )
        self.module = adapter.trainable_module
        self.trainable_module = self.module
        self.checkpoint_identity = identity
        self.fsdp_block_classes = adapter.fsdp_block_classes

    def add_noise(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        return flow_interpolate(clean_latents, noise, sigmas)

    def predict_velocity(
        self,
        noisy_latents: torch.Tensor,
        sigmas: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> torch.Tensor:
        return self.prediction.predict_velocity(
            noisy_latents,
            sigmas,
            sample_ids=sample_ids,
            conditioning=conditioning,
            training=training,
            branch=branch,
        )

    def predict_clean(
        self,
        noisy_latents: torch.Tensor,
        sigmas: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> torch.Tensor:
        return self.prediction.predict_clean(
            noisy_latents,
            sigmas,
            sample_ids=sample_ids,
            conditioning=conditioning,
            training=training,
            branch=branch,
        )


class DiffusersSanaDenoiser:
    """Translate a locally loaded Diffusers SANA transformer to the core denoiser contract."""

    def __init__(self, model: torch.nn.Module) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Diffusers SANA transformer must be an nn.Module")
        self.model = model

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        if not isinstance(model_input, DenoiserInput):
            raise TypeError("model_input must be DenoiserInput")
        context = model_input.conditioning.get("context")
        mask = model_input.conditioning.get("context_mask")
        if not isinstance(context, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise TypeError("Diffusers SANA requires context and context_mask tensors")
        if context.ndim == 4 and int(context.shape[1]) == 1:
            context = context[:, 0]
        if context.ndim != 3:
            raise ValueError("Diffusers SANA context must be [B,L,C] or [B,1,L,C]")
        latents = model_input.latents
        timesteps = model_input.timestep.to(device=latents.device, dtype=torch.float32).reshape(-1)
        if timesteps.numel() == 1:
            timesteps = timesteps.expand(latents.shape[0])
        if timesteps.numel() != latents.shape[0]:
            raise ValueError("Diffusers SANA requires one timestep per sample")
        guidance = None
        if bool(getattr(getattr(self.model, "config", None), "guidance_embeds", False)):
            guidance = model_input.conditioning.get("cfg_scale")
            if not isinstance(guidance, torch.Tensor):
                raise TypeError("guidance-embedded Diffusers SANA requires cfg_scale")
            guidance = guidance.to(device=latents.device, dtype=torch.float32).reshape(-1)
        output = self.model(
            hidden_states=latents,
            encoder_hidden_states=context.to(device=latents.device, dtype=latents.dtype),
            timestep=timesteps,
            guidance=guidance,
            encoder_attention_mask=mask.to(device=latents.device),
            return_dict=False,
        )[0]
        if not isinstance(output, torch.Tensor) or output.shape != latents.shape:
            raise ValueError("Diffusers SANA prediction must match the latent input")
        return DenoiserOutput(sample=output)


class DiffusersSanaConditioner:
    """Use a local Diffusers pipeline only as a frozen prompt encoder."""

    def __init__(self, pipeline: object) -> None:
        encoder = getattr(pipeline, "text_encoder", None)
        if not isinstance(encoder, torch.nn.Module):
            raise TypeError("Diffusers SANA pipeline must expose a text_encoder")
        if not callable(getattr(pipeline, "encode_prompt", None)):
            raise TypeError("Diffusers SANA pipeline must expose encode_prompt")
        self.pipeline = pipeline
        self.encoder = encoder

    def encode(
        self,
        request: DiffusionRequest,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Conditioning:
        if not isinstance(request, DiffusionRequest):
            raise TypeError("request must be DiffusionRequest")
        embeddings, mask, _, _ = self.pipeline.encode_prompt(
            list(request.prompts),
            do_classifier_free_guidance=False,
            device=device,
            max_sequence_length=300,
        )
        if not isinstance(embeddings, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise TypeError("Diffusers SANA prompt encoder returned invalid tensors")
        return Conditioning(
            positive={
                "context": embeddings.to(device=device, dtype=dtype).unsqueeze(1),
                "context_mask": mask.to(device=device),
            }
        )


def build_local_diffusers_sana_sid_adapter(
    path: str,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    parameter_dtype: torch.dtype | None = None,
    checkpoint_identity: str,
    load_conditioner: bool,
) -> tuple[SanaTrainAdapter, SanaSIDPredictionAdapter]:
    """Load local Diffusers assets into WorldFoundry's native training contracts."""

    from pathlib import Path

    source = Path(path).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"local Diffusers SANA directory does not exist: {source}")
    try:
        from diffusers import SanaPipeline, SanaTransformer2DModel
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError("local Diffusers SANA materialization requires diffusers") from error
    resolved_device = torch.device(device)
    resolved_parameter_dtype = dtype if parameter_dtype is None else parameter_dtype
    supported_dtypes = {torch.float32, torch.bfloat16, torch.float16}
    if dtype not in supported_dtypes or resolved_parameter_dtype not in supported_dtypes:
        raise ValueError("local Diffusers SANA requires float32, bfloat16, or float16 dtypes")
    transformer = SanaTransformer2DModel.from_pretrained(
        source,
        subfolder="transformer",
        torch_dtype=resolved_parameter_dtype,
        local_files_only=True,
        use_safetensors=True,
    ).to(resolved_device)
    conditioner = None
    if load_conditioner:
        pipeline = SanaPipeline.from_pretrained(
            source,
            transformer=None,
            vae=None,
            torch_dtype=dtype,
            local_files_only=True,
            use_safetensors=True,
        )
        pipeline.to(resolved_device)
        conditioner = DiffusersSanaConditioner(pipeline)
    train_adapter = SanaTrainAdapter(
        DiffusersSanaDenoiser(transformer),
        codec=None,
        conditioner=conditioner,
        expected_latent_channels=int(transformer.config.in_channels),
        spatial_compression=32,
    )
    prediction = SanaSIDPredictionAdapter(
        train_adapter,
        checkpoint_identity=checkpoint_identity,
        autocast_dtype=None if dtype is torch.float32 else dtype,
    )
    return train_adapter, prediction


__all__ = [
    "DiffusersSanaConditioner",
    "DiffusersSanaDenoiser",
    "SanaSIDPredictionAdapter",
    "build_local_diffusers_sana_sid_adapter",
]

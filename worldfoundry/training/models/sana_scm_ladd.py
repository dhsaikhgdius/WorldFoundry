"""Native SANA-Sprint execution adapters for SCM-LADD."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from math import isclose

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.contracts import DenoiserInput
from worldfoundry.base_models.diffusion_model.models.denoisers.sana import SanaDenoiser
from worldfoundry.base_models.diffusion_model.models.networks.sana.ladd import (
    SANAFeatureDiscriminatorHead,
    SANAFeatureDiscriminatorHeads,
)
from worldfoundry.training.post_training.distillation.scm_ladd.contracts import (
    SCMVelocityPrediction,
)


def _validate_inputs(
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    sample_ids: tuple[str, ...],
    conditioning: Mapping[str, object],
) -> torch.Tensor:
    if not isinstance(latents, torch.Tensor) or latents.ndim != 4:
        raise TypeError("SANA SCM latents must be a BCHW tensor")
    if not isinstance(timesteps, torch.Tensor):
        raise TypeError("SANA SCM timesteps must be a tensor")
    resolved = timesteps.to(device=latents.device, dtype=torch.float32).reshape(-1)
    if resolved.numel() == 1:
        resolved = resolved.expand(latents.shape[0])
    if resolved.numel() != latents.shape[0]:
        raise ValueError("SANA SCM requires one TrigFlow timestep per sample")
    if len(sample_ids) != latents.shape[0]:
        raise ValueError("SANA SCM sample ids do not match the latent batch")
    if not isinstance(conditioning, Mapping):
        raise TypeError("SANA SCM conditioning must be a mapping")
    return resolved


class SanaSCMVelocityAdapter:
    """Expose raw SanaMSCM velocity/log-variance to the native objective."""

    def __init__(
        self,
        denoiser: SanaDenoiser,
        *,
        role: str,
        checkpoint_identity: str,
        fp32_attention: bool,
        expected_latent_channels: int = 32,
        autocast_dtype: torch.dtype | None = None,
    ) -> None:
        if not isinstance(denoiser, SanaDenoiser):
            raise TypeError("denoiser must be SanaDenoiser")
        if role not in {"student", "teacher"}:
            raise ValueError("SANA SCM role must be student or teacher")
        identity = str(checkpoint_identity).strip()
        if not identity:
            raise ValueError("SANA SCM checkpoint_identity must be non-empty")
        if isinstance(expected_latent_channels, bool) or int(expected_latent_channels) <= 0:
            raise ValueError("expected_latent_channels must be positive")
        self.denoiser = denoiser
        self.module = denoiser.model
        self.trainable_module = self.module
        self.role = role
        self.checkpoint_identity = identity
        if not isinstance(fp32_attention, bool):
            raise TypeError("fp32_attention must be bool")
        set_fp32_attention = getattr(self.module, "set_fp32_attention", None)
        if not callable(set_fp32_attention):
            raise TypeError("SANA SCM graph must expose set_fp32_attention(enabled)")
        set_fp32_attention(fp32_attention)
        self.fp32_attention = fp32_attention
        self.expected_latent_channels = int(expected_latent_channels)
        self.autocast_dtype = autocast_dtype

        log_variance = getattr(self.module, "logvar_linear", None)
        guidance_embedder = getattr(self.module, "cfg_embedder", None)
        if role == "student":
            if not isinstance(log_variance, nn.Module):
                raise ValueError("SANA SCM student graph must contain the learned log-variance head")
            if not isinstance(guidance_embedder, nn.Module):
                raise ValueError("SANA SCM student graph must contain the CFG-scale embedder")
        else:
            if log_variance is not None or guidance_embedder is not None:
                raise ValueError("SANA SCM teacher graph must omit log-variance and CFG-scale embedders")
            self.module.requires_grad_(False)
            self.module.eval()

        blocks = getattr(self.module, "blocks", None)
        if not isinstance(blocks, nn.ModuleList) or not blocks:
            raise TypeError("SANA SCM graph must expose a non-empty ModuleList named 'blocks'")
        self.fsdp_block_classes = tuple(dict.fromkeys(type(block) for block in blocks))

    def predict_velocity(
        self,
        scaled_noisy_latents: torch.Tensor,
        trig_timesteps: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        guidance_embedding_scale: float,
        return_log_variance: bool = False,
        branch: str = "positive",
    ) -> SCMVelocityPrediction:
        timesteps = _validate_inputs(
            scaled_noisy_latents,
            trig_timesteps,
            sample_ids,
            conditioning,
        )
        if scaled_noisy_latents.shape[1] != self.expected_latent_channels:
            raise ValueError(
                f"SANA SCM expects {self.expected_latent_channels} latent channels, "
                f"got {scaled_noisy_latents.shape[1]}"
            )
        if not isinstance(training, bool) or not isinstance(return_log_variance, bool):
            raise TypeError("training and return_log_variance must be bool")
        if branch not in {"positive", "negative"}:
            raise ValueError("SANA SCM branch must be positive or negative")
        if return_log_variance and self.role != "student":
            raise ValueError("only the SANA SCM student owns learned log variance")
        if self.role == "teacher" and training:
            raise ValueError("the frozen SANA SCM teacher cannot enter training mode")
        if self.role == "student":
            configured = float(getattr(self.module, "cfg_embed_scale"))
            if not isclose(float(guidance_embedding_scale), configured, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    "SCM recipe guidance_embedding_scale differs from the loaded SANA student graph"
                )
            cfg_scale = conditioning.get("cfg_scale")
            if not isinstance(cfg_scale, torch.Tensor) or cfg_scale.reshape(-1).numel() != scaled_noisy_latents.shape[0]:
                raise ValueError("SANA SCM student conditioning requires one tensor cfg_scale per sample")

        self.module.train(training)
        autocast = (
            torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
            if scaled_noisy_latents.device.type == "cuda" and self.autocast_dtype is not None
            else nullcontext()
        )
        with autocast:
            output = self.denoiser.forward_with_options(
                DenoiserInput(
                    latents=scaled_noisy_latents,
                    timestep=timesteps,
                    next_timestep=torch.zeros_like(timesteps),
                    conditioning=conditioning,
                    step_index=0,
                    total_steps=1,
                    branch=branch,
                ),
                return_log_variance=return_log_variance,
                apply_output_scale=False,
            )
        if output.sample.shape != scaled_noisy_latents.shape:
            raise ValueError("SANA SCM velocity shape differs from the latent input")
        log_variance = output.extras.get("log_variance")
        if return_log_variance and not isinstance(log_variance, torch.Tensor):
            raise ValueError("SANA SCM student failed to return learned log variance")
        return SCMVelocityPrediction(
            velocity=output.sample,
            log_variance=log_variance if isinstance(log_variance, torch.Tensor) else None,
        )


class SanaLADDDiscriminatorAdapter:
    """Run frozen teacher features through independently optimized LADD heads."""

    def __init__(
        self,
        teacher: SanaSCMVelocityAdapter,
        heads: SANAFeatureDiscriminatorHeads,
        *,
        autocast_dtype: torch.dtype | None = None,
    ) -> None:
        if not isinstance(teacher, SanaSCMVelocityAdapter) or teacher.role != "teacher":
            raise TypeError("LADD feature role must be a SANA SCM teacher adapter")
        if not isinstance(heads, SANAFeatureDiscriminatorHeads):
            raise TypeError("heads must be SANAFeatureDiscriminatorHeads")
        blocks = getattr(teacher.module, "blocks", None)
        if not isinstance(blocks, nn.ModuleList):
            raise TypeError("SANA teacher must expose transformer blocks")
        if any(index >= len(blocks) for index in heads.block_ids):
            raise ValueError("LADD head block id exceeds the SANA teacher depth")
        self.teacher = teacher
        self.module = heads
        self.trainable_module = heads
        self.feature_module = teacher.module
        self.head_block_ids = heads.block_ids
        self.fsdp_block_classes = (SANAFeatureDiscriminatorHead,)
        self.autocast_dtype = autocast_dtype

    def predict_logits(
        self,
        scaled_noisy_latents: torch.Tensor,
        trig_timesteps: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        head_block_ids: tuple[int, ...],
    ) -> torch.Tensor:
        _validate_inputs(scaled_noisy_latents, trig_timesteps, sample_ids, conditioning)
        selected = tuple(int(value) for value in head_block_ids)
        if selected != self.head_block_ids:
            raise ValueError("requested LADD head blocks differ from the materialized discriminator")
        if not isinstance(training, bool):
            raise TypeError("training must be bool")
        self.module.train(training)
        self.feature_module.eval()

        features: list[torch.Tensor] = []
        handles: list[torch.utils.hooks.RemovableHandle] = []

        def capture(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            if not isinstance(output, torch.Tensor):
                raise TypeError("SANA teacher block must return a tensor feature")
            features.append(output)

        blocks: Sequence[nn.Module] = getattr(self.feature_module, "blocks")
        try:
            for index in selected:
                handles.append(blocks[index].register_forward_hook(capture))
            self.teacher.predict_velocity(
                scaled_noisy_latents,
                trig_timesteps,
                sample_ids=sample_ids,
                conditioning=conditioning,
                training=False,
                guidance_embedding_scale=float(getattr(self.feature_module, "cfg_embed_scale", 0.1)),
                return_log_variance=False,
            )
        finally:
            for handle in handles:
                handle.remove()
        if len(features) != len(selected):
            raise RuntimeError("SANA teacher did not execute every selected LADD feature block")
        autocast = (
            torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
            if scaled_noisy_latents.device.type == "cuda" and self.autocast_dtype is not None
            else nullcontext()
        )
        with autocast:
            return self.module(features)


__all__ = ["SanaLADDDiscriminatorAdapter", "SanaSCMVelocityAdapter"]

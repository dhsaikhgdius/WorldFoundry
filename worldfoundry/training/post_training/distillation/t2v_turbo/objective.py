"""Reward-feedback latent consistency distillation for native T2V-Turbo.

Unlike the generic latent-consistency engine, the released T2V-Turbo trainer
uses the current student itself for the stop-gradient target.  The frozen
teacher only advances the guided DDIM trajectory.  Keeping that distinction
here avoids inventing an EMA role that the released optimization never used.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from math import isfinite, prod
from typing import Literal, Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from worldfoundry.training.api.contracts import ObjectiveBatch, PreparedBatch, TrainingBatch, TrainStepResult
from worldfoundry.training.models.lvdm import (
    FramewiseLVDMCodec,
    freeze_module,
    latent_sample,
    module_device_dtype,
)
from worldfoundry.training.post_training.distillation.latent_consistency.math import (
    add_forward_diffusion_noise,
    append_dims,
    boundary_condition_scalings,
    classifier_free_guidance,
    consistency_prediction,
    deterministic_ddim_step,
    gather_schedule_coefficients,
    guidance_scale_embedding,
    latent_consistency_elementwise_loss,
    prediction_to_origin_and_epsilon,
)

TurboLossType = Literal["l2", "pseudo_huber"]


def t2v_turbo_scaled_linear_beta_schedule(
    num_train_timesteps: int,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Build the float32 scaled-linear schedule used by the released trainer."""

    return torch.linspace(
        0.00085**0.5,
        0.012**0.5,
        int(num_train_timesteps),
        dtype=torch.float32,
        device=device,
    ).square()


def _cuda_autocast(reference: Tensor, dtype: torch.dtype | None):
    if reference.device.type != "cuda" or dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


@runtime_checkable
class DifferentiableImageReward(Protocol):
    """Score decoded frames while preserving gradients to their pixels."""

    def __call__(self, images: Tensor, prompts: list[str]) -> Tensor: ...


@runtime_checkable
class DifferentiableVideoReward(Protocol):
    """Score decoded ``[B,T,C,H,W]`` videos with pixel gradients intact."""

    def __call__(self, videos: Tensor, prompts: list[str]) -> Tensor: ...


class LVDMEpsilonPredictor:
    """Thin call binding over the native VideoCrafter three-dimensional UNet."""

    def __init__(self, module: nn.Module) -> None:
        if not isinstance(module, nn.Module):
            raise TypeError("T2V-Turbo predictor module must be an nn.Module")
        self.module = module

    @classmethod
    def from_denoiser_component(cls, component: object) -> "LVDMEpsilonPredictor":
        module = getattr(component, "model", None)
        if not isinstance(module, nn.Module):
            raise TypeError("native T2V-Turbo denoiser must expose model")
        return cls(module)

    def predict(
        self,
        noisy: Tensor,
        timesteps: Tensor,
        *,
        context: Tensor,
        fps: Tensor | None,
        guidance_embedding: Tensor | None,
    ) -> Tensor:
        kwargs: dict[str, object] = {"context": context}
        if fps is not None:
            kwargs["fps"] = fps
        if guidance_embedding is not None:
            kwargs["timestep_cond"] = guidance_embedding
        output = self.module(noisy, timesteps, **kwargs)
        if isinstance(output, tuple):
            output = output[0]
        sample = getattr(output, "sample", output)
        if not isinstance(sample, Tensor) or sample.shape != noisy.shape:
            raise ValueError("T2V-Turbo predictor output must match noisy latents")
        return sample


class T2VTurboTrainAdapter:
    """Encode cached/raw video data and bind student/teacher native UNets."""

    prediction_type = "epsilon"
    lora_target_preset = "t2v-turbo-unet"

    def __init__(
        self,
        *,
        student: LVDMEpsilonPredictor,
        teacher: LVDMEpsilonPredictor,
        codec: FramewiseLVDMCodec | None = None,
        text_encoder: object | None = None,
        default_fps: int = 16,
        expected_latent_channels: int = 4,
        student_autocast_dtype: torch.dtype | None = None,
    ) -> None:
        if not isinstance(student, LVDMEpsilonPredictor) or not isinstance(teacher, LVDMEpsilonPredictor):
            raise TypeError("student and teacher must be LVDMEpsilonPredictor values")
        if student.module is teacher.module:
            raise ValueError("T2V-Turbo student and teacher modules must be distinct")
        self.student = student
        self.teacher = teacher
        self.codec = codec
        self.text_encoder = text_encoder
        self.trainable_module = student.module
        self.default_fps = int(default_fps)
        self.expected_latent_channels = int(expected_latent_channels)
        if student_autocast_dtype not in {None, torch.float16, torch.bfloat16}:
            raise ValueError("student_autocast_dtype must be float16, bfloat16, or None")
        self.student_autocast_dtype = student_autocast_dtype
        freeze_module(teacher.module)
        frozen: list[nn.Module] = [teacher.module]
        if isinstance(codec, nn.Module):
            frozen.append(codec)
        if isinstance(text_encoder, nn.Module):
            frozen.append(text_encoder)
        self.frozen_modules = tuple(frozen)
        for module in self.frozen_modules:
            freeze_module(module)
        blocks: list[nn.Module] = []
        for name in ("input_blocks", "output_blocks", "transformer_blocks"):
            values = getattr(self.trainable_module, name, ())
            if isinstance(values, (nn.ModuleList, nn.Sequential, list, tuple)):
                blocks.extend(value for value in values if isinstance(value, nn.Module))
        self.fsdp_block_classes = tuple(dict.fromkeys(type(value) for value in blocks))

    def replace_student_module(self, module: nn.Module) -> None:
        """Reconnect the predictor after PEFT wraps the student module."""

        if not isinstance(module, nn.Module):
            raise TypeError("replacement student must be an nn.Module")
        self.student.module = module
        self.trainable_module = module

    def _encode_text(self, prompts: list[str]) -> Tensor:
        if self.text_encoder is None:
            raise RuntimeError("T2V-Turbo requires cached contexts or a text encoder")
        encode = getattr(self.text_encoder, "encode", None)
        value = encode(prompts) if callable(encode) else self.text_encoder(prompts)  # type: ignore[operator]
        if isinstance(value, Tensor):
            return value
        mode = getattr(value, "mode", None)
        if callable(mode):
            resolved = mode()
            if isinstance(resolved, Tensor):
                return resolved
        return latent_sample(value)

    def prepare_batch(self, batch: TrainingBatch) -> PreparedBatch:
        device, dtype = module_device_dtype(self.trainable_module)
        cached = batch.conditions.get("clean_latents")
        if cached is None:
            if not isinstance(batch.pixel_values, Tensor) or self.codec is None:
                raise TypeError("T2V-Turbo requires pixels with a codec or cached clean_latents")
            with torch.no_grad():
                clean = self.codec.encode_video(batch.pixel_values.to(device=device, dtype=dtype))
        else:
            if batch.pixel_values is not None:
                raise ValueError("T2V-Turbo batch cannot contain both pixels and cached latents")
            if not isinstance(cached, Tensor):
                raise TypeError("clean_latents must be a tensor")
            clean = cached
        if clean.ndim != 5 or int(clean.shape[0]) != batch.batch_size:
            raise ValueError("T2V-Turbo clean latents must be [B,C,T,H,W]")
        if int(clean.shape[1]) != self.expected_latent_channels:
            raise ValueError(f"T2V-Turbo expected {self.expected_latent_channels} latent channels")
        clean = clean.detach().to(device=device, dtype=dtype)

        context = batch.conditions.get("context")
        unconditional = batch.conditions.get("unconditional_context")
        if context is None or unconditional is None:
            with torch.no_grad():
                context = self._encode_text(list(batch.prompts))
                unconditional = self._encode_text([""])
        if not isinstance(context, Tensor) or not isinstance(unconditional, Tensor):
            raise TypeError("T2V-Turbo contexts must be tensors")
        context = context.detach().to(device=device, dtype=dtype)
        unconditional = unconditional.detach().to(device=device, dtype=dtype)
        if int(unconditional.shape[0]) == 1:
            unconditional = unconditional.expand(batch.batch_size, *unconditional.shape[1:])
        if int(context.shape[0]) != batch.batch_size or int(unconditional.shape[0]) != batch.batch_size:
            raise ValueError("T2V-Turbo contexts must match batch size")
        fps = batch.conditions.get("fps")
        if fps is None:
            fps = batch.metadata.get("target_fps", self.default_fps)
        if not isinstance(fps, Tensor):
            if float(fps) != float(self.default_fps):
                raise ValueError(f"T2V-Turbo requires {self.default_fps} FPS conditioning")
            fps = torch.full((batch.batch_size,), int(fps), device=device, dtype=torch.long)
        else:
            fps = fps.to(device=device).reshape(batch.batch_size)
            if not bool(torch.all(fps == float(self.default_fps))):
                raise ValueError(f"T2V-Turbo requires {self.default_fps} FPS conditioning")
            fps = fps.to(dtype=torch.long)
        loss_mask = batch.conditions.get("latent_loss_mask")
        if isinstance(loss_mask, Tensor):
            loss_mask = loss_mask.to(device=device, dtype=torch.float32)
        sample_weights = batch.sample_weights
        if isinstance(sample_weights, Tensor):
            sample_weights = sample_weights.to(device=device, dtype=torch.float32)
        return PreparedBatch(
            sample_ids=batch.sample_ids,
            clean_latents=clean,
            conditioning={
                "context": context,
                "unconditional_context": unconditional,
                "fps": fps,
            },
            loss_mask=loss_mask,
            sample_weights=sample_weights,
            metadata={
                **dict(batch.metadata),
                "model_family": "t2v-turbo",
                "prompts": batch.prompts,
            },
        )

    def forward_train(self, batch: ObjectiveBatch) -> Tensor:
        noisy = batch.model_input
        if isinstance(noisy, Mapping) or not isinstance(noisy, Tensor):
            raise TypeError("T2V-Turbo model_input must be one tensor")
        context = batch.conditioning.get("context")
        fps = batch.conditioning.get("fps")
        guidance = batch.conditioning.get("guidance_embedding")
        if not all(isinstance(value, Tensor) for value in (context, fps, guidance)):
            raise TypeError("T2V-Turbo student conditioning is incomplete")
        self.trainable_module.train()
        return self.student.predict(
            noisy,
            batch.timesteps,
            context=context,
            fps=fps,
            guidance_embedding=guidance,
        )

    def teacher_prediction(
        self,
        noisy: Tensor,
        timesteps: Tensor,
        *,
        context: Tensor,
        fps: Tensor | None,
    ) -> Tensor:
        self.teacher.module.eval()
        with _cuda_autocast(noisy, torch.float16):
            return self.teacher.predict(
                noisy,
                timesteps,
                context=context,
                fps=fps,
                guidance_embedding=None,
            )

    def student_target_prediction(
        self,
        noisy: Tensor,
        timesteps: Tensor,
        *,
        context: Tensor,
        fps: Tensor,
        guidance_embedding: Tensor,
    ) -> Tensor:
        # The released trainer keeps the current student in train mode here;
        # only autograd is disabled.  In particular, this is not an EMA model.
        with _cuda_autocast(noisy, self.student_autocast_dtype):
            return self.student.predict(
                noisy,
                timesteps,
                context=context,
                fps=fps,
                guidance_embedding=guidance_embedding,
            )

    def decode_video(self, latents: Tensor) -> Tensor:
        if self.codec is None:
            raise RuntimeError("reward feedback requires a differentiable LVDM codec")
        return self.codec.decode_video(latents)


@dataclass(frozen=True, slots=True)
class T2VTurboConfig:
    """Released distillation schedule plus optional differentiable rewards."""

    num_train_timesteps: int = 1000
    num_ddim_timesteps: int = 50
    topk: int = 20
    guidance_min: float = 5.0
    guidance_max: float = 15.0
    guidance_embedding_dim: int = 256
    sigma_data: float = 0.5
    timestep_scaling: float = 10.0
    loss_type: TurboLossType = "pseudo_huber"
    pseudo_huber_c: float = 0.001
    distillation_weight: float = 1.0
    image_reward_weight: float = 0.0
    video_reward_weight: float = 0.0
    image_reward_frames: int = 5
    image_reward_batch_size: int = 1
    video_reward_frames: int = 8

    def __post_init__(self) -> None:
        integer_fields = (
            "num_train_timesteps",
            "num_ddim_timesteps",
            "topk",
            "guidance_embedding_dim",
            "image_reward_frames",
            "image_reward_batch_size",
            "video_reward_frames",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, int(value))
        if self.num_train_timesteps % self.num_ddim_timesteps:
            raise ValueError("num_train_timesteps must be divisible by num_ddim_timesteps")
        minimum = float(self.guidance_min)
        maximum = float(self.guidance_max)
        if not isfinite(minimum) or not isfinite(maximum) or minimum < 0.0 or maximum < minimum:
            raise ValueError("guidance range is invalid")
        if self.loss_type not in {"l2", "pseudo_huber"}:
            raise ValueError("loss_type must be l2 or pseudo_huber")
        positive = ("sigma_data", "timestep_scaling", "pseudo_huber_c")
        for name in positive:
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        for name in ("distillation_weight", "image_reward_weight", "video_reward_weight"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.distillation_weight + self.image_reward_weight + self.video_reward_weight == 0.0:
            raise ValueError("at least one T2V-Turbo objective weight must be positive")
        object.__setattr__(self, "guidance_min", minimum)
        object.__setattr__(self, "guidance_max", maximum)


def _per_sample_loss(
    elementwise: Tensor,
    *,
    mask: object | None,
    sample_weights: object | None,
) -> tuple[Tensor, Tensor, Tensor]:
    batch = int(elementwise.shape[0])
    if mask is None:
        expanded = torch.ones_like(elementwise, dtype=torch.float32)
    else:
        if not isinstance(mask, Tensor):
            raise TypeError("T2V-Turbo loss_mask must be a tensor")
        if mask.ndim + 1 == elementwise.ndim and int(mask.shape[0]) == batch:
            mask = mask.unsqueeze(1)
        expanded = torch.broadcast_to(mask, elementwise.shape).to(device=elementwise.device, dtype=torch.float32)
    per_denominator = expanded.reshape(batch, -1).sum(dim=1)
    per_sample = (elementwise.float() * expanded).reshape(batch, -1).sum(dim=1) / per_denominator.clamp_min(1.0)
    weights = torch.ones(batch, device=elementwise.device, dtype=torch.float32)
    if sample_weights is not None:
        if not isinstance(sample_weights, Tensor):
            raise TypeError("sample_weights must be a tensor")
        weights = sample_weights.to(device=elementwise.device, dtype=torch.float32)
    weights = weights * (per_denominator > 0)
    denominator = weights.sum()
    if not bool(denominator.detach() > 0):
        raise ValueError("T2V-Turbo loss has no active samples")
    numerator = (per_sample * weights).sum()
    return numerator / denominator, numerator, denominator


class T2VTurboObjective:
    """Same-student consistency target with optional image/video rewards."""

    prediction_type = "epsilon"

    def __init__(
        self,
        *,
        adapter: T2VTurboTrainAdapter,
        config: T2VTurboConfig | None = None,
        image_reward: DifferentiableImageReward | None = None,
        video_reward: DifferentiableVideoReward | None = None,
    ) -> None:
        if not isinstance(adapter, T2VTurboTrainAdapter):
            raise TypeError("adapter must be T2VTurboTrainAdapter")
        self.adapter = adapter
        self.config = config or T2VTurboConfig()
        if self.config.image_reward_weight and image_reward is None:
            raise ValueError("image_reward_weight requires a differentiable image reward")
        if self.config.video_reward_weight and video_reward is None:
            raise ValueError("video_reward_weight requires a differentiable video reward")
        self.image_reward = image_reward
        self.video_reward = video_reward
        for reward in (image_reward, video_reward):
            if isinstance(reward, nn.Module):
                freeze_module(reward)

        betas = t2v_turbo_scaled_linear_beta_schedule(self.config.num_train_timesteps)
        alpha_cumprods = torch.cumprod(1.0 - betas, dim=0)
        self.betas = betas
        self.alpha_cumprods = alpha_cumprods
        self.alphas = alpha_cumprods.sqrt()
        self.sigmas = (1.0 - alpha_cumprods).sqrt()
        ratio = self.config.num_train_timesteps // self.config.num_ddim_timesteps
        self.start_timesteps = torch.arange(1, self.config.num_ddim_timesteps + 1, dtype=torch.int64) * ratio - 1
        self.previous_alpha_cumprods = torch.cat((alpha_cumprods[:1], alpha_cumprods[self.start_timesteps[:-1]]))

    def corrupt(self, batch: PreparedBatch, *, generator: object | None = None) -> ObjectiveBatch:
        clean = batch.clean_latents
        if isinstance(clean, Mapping) or not isinstance(clean, Tensor):
            raise TypeError("T2V-Turbo requires one clean latent tensor")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator or None")
        pair_indices = torch.randint(
            self.config.num_ddim_timesteps,
            (batch.batch_size,),
            device=clean.device,
            generator=generator,
        )
        starts = self.start_timesteps.to(clean.device).gather(0, pair_indices)
        ends = (starts - self.config.topk).clamp_min(0)
        start_alpha = gather_schedule_coefficients(self.alphas, starts, clean)
        start_sigma = gather_schedule_coefficients(self.sigmas, starts, clean)
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        noisy = add_forward_diffusion_noise(clean, noise, start_alpha, start_sigma)
        guidance = self.config.guidance_min + (self.config.guidance_max - self.config.guidance_min) * torch.rand(
            batch.batch_size, device=clean.device, generator=generator
        )
        embedding = guidance_scale_embedding(
            guidance,
            embedding_dim=self.config.guidance_embedding_dim,
            dtype=clean.dtype,
        )
        conditioning = dict(batch.conditioning)
        conditioning.update(
            {
                "pair_indices": pair_indices,
                "end_timesteps": ends,
                "guidance_coefficients": guidance,
                "guidance_embedding": embedding,
            }
        )
        frame_count = int(clean.shape[2])
        if self.config.image_reward_weight:
            conditioning["image_reward_frame_indices"] = torch.randperm(
                frame_count,
                device=clean.device,
                generator=generator,
            )[: min(frame_count, self.config.image_reward_frames)]
            conditioning["image_reward_batch_indices"] = torch.randperm(
                batch.batch_size,
                device=clean.device,
                generator=generator,
            )[: min(batch.batch_size, self.config.image_reward_batch_size)]
        if self.config.video_reward_weight:
            selected_frames = min(frame_count, self.config.video_reward_frames)
            stride = max(frame_count // selected_frames, 1)
            start = torch.randint(stride, (), device=clean.device, generator=generator)
            conditioning["video_reward_frame_indices"] = (
                start + torch.arange(selected_frames, device=clean.device) * stride
            ).clamp_max(frame_count - 1)
        metadata = dict(batch.metadata)
        metadata.update({"objective": "t2v_turbo", "prediction_type": self.prediction_type})
        return ObjectiveBatch(
            sample_ids=batch.sample_ids,
            model_input=noisy,
            target=clean,
            sigmas=self.sigmas.to(clean.device).gather(0, starts),
            timesteps=starts,
            conditioning=conditioning,
            noise=noise,
            loss_mask=batch.loss_mask,
            sample_weights=batch.sample_weights,
            metadata=metadata,
        )

    def _consistency_prediction(
        self,
        model_output: Tensor,
        noisy: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        alpha = gather_schedule_coefficients(self.alphas, timesteps, noisy)
        sigma = gather_schedule_coefficients(self.sigmas, timesteps, noisy)
        origin, _ = prediction_to_origin_and_epsilon(
            model_output,
            noisy,
            alpha,
            sigma,
            prediction_type="epsilon",
        )
        skip, out = boundary_condition_scalings(
            timesteps,
            sigma_data=self.config.sigma_data,
            timestep_scaling=self.config.timestep_scaling,
        )
        return consistency_prediction(noisy, origin, append_dims(skip, noisy.ndim), append_dims(out, noisy.ndim))

    def _distillation_target(self, batch: ObjectiveBatch) -> Tensor:
        noisy = batch.model_input
        if isinstance(noisy, Mapping) or not isinstance(noisy, Tensor):
            raise TypeError("T2V-Turbo model_input must be a tensor")
        context = batch.conditioning["context"]
        unconditional = batch.conditioning["unconditional_context"]
        fps = batch.conditioning["fps"]
        guidance = batch.conditioning["guidance_coefficients"]
        guidance_embedding = batch.conditioning["guidance_embedding"]
        pair_indices = batch.conditioning["pair_indices"]
        ends = batch.conditioning["end_timesteps"]
        if not all(
            isinstance(value, Tensor)
            for value in (context, unconditional, fps, guidance, guidance_embedding, pair_indices, ends)
        ):
            raise TypeError("T2V-Turbo target conditioning is incomplete")
        start_alpha = gather_schedule_coefficients(self.alphas, batch.timesteps, noisy)
        start_sigma = gather_schedule_coefficients(self.sigmas, batch.timesteps, noisy)
        with torch.no_grad():
            conditional_output = self.adapter.teacher_prediction(
                noisy,
                batch.timesteps,
                context=context,
                fps=fps,
            )
            unconditional_output = self.adapter.teacher_prediction(
                noisy,
                batch.timesteps,
                context=unconditional,
                # The released trainer omits FPS on the unconditional teacher
                # branch, so the native UNet uses its checkpoint default.
                fps=None,
            )
            conditional_origin, conditional_epsilon = prediction_to_origin_and_epsilon(
                conditional_output,
                noisy,
                start_alpha,
                start_sigma,
                prediction_type="epsilon",
            )
            unconditional_origin, unconditional_epsilon = prediction_to_origin_and_epsilon(
                unconditional_output,
                noisy,
                start_alpha,
                start_sigma,
                prediction_type="epsilon",
            )
            guided_origin = classifier_free_guidance(conditional_origin, unconditional_origin, guidance)
            guided_epsilon = classifier_free_guidance(conditional_epsilon, unconditional_epsilon, guidance)
            previous_alpha = append_dims(
                self.previous_alpha_cumprods.to(noisy.device).gather(0, pair_indices),
                noisy.ndim,
            )
            previous_latents = deterministic_ddim_step(guided_origin, guided_epsilon, previous_alpha)
            target_output = self.adapter.student_target_prediction(
                previous_latents,
                ends,
                context=context,
                fps=fps,
                guidance_embedding=guidance_embedding,
            )
            return self._consistency_prediction(target_output, previous_latents, ends)

    def _image_reward_loss(self, prediction: Tensor, batch: ObjectiveBatch, prompts: tuple[str, ...]) -> Tensor:
        if self.image_reward is None or not self.config.image_reward_weight:
            return prediction.new_zeros((), dtype=torch.float32)
        sample_indices = batch.conditioning["image_reward_batch_indices"]
        frame_indices = batch.conditioning["image_reward_frame_indices"]
        if not isinstance(sample_indices, Tensor) or not isinstance(frame_indices, Tensor):
            raise TypeError("image reward indices must be tensors")
        selected = prediction.index_select(0, sample_indices).index_select(2, frame_indices)
        decoded = self.adapter.decode_video(selected)
        images = decoded.permute(0, 2, 1, 3, 4).reshape(-1, *decoded.shape[1:2], *decoded.shape[3:])
        selected_prompts = [prompts[index] for index in sample_indices.tolist()]
        scores = self.image_reward((images / 2.0 + 0.5).clamp(0.0, 1.0), selected_prompts)
        if not isinstance(scores, Tensor):
            raise TypeError("differentiable image reward must return a tensor")
        return -scores.float().mean() * self.config.image_reward_weight

    def _video_reward_loss(self, prediction: Tensor, batch: ObjectiveBatch, prompts: tuple[str, ...]) -> Tensor:
        if self.video_reward is None or not self.config.video_reward_weight:
            return prediction.new_zeros((), dtype=torch.float32)
        frame_indices = batch.conditioning["video_reward_frame_indices"]
        if not isinstance(frame_indices, Tensor):
            raise TypeError("video reward frame indices must be a tensor")
        selected = prediction.index_select(2, frame_indices)
        decoded = self.adapter.decode_video(selected).permute(0, 2, 1, 3, 4)
        scores = self.video_reward((decoded / 2.0 + 0.5).clamp(0.0, 1.0), list(prompts))
        if not isinstance(scores, Tensor):
            raise TypeError("differentiable video reward must return a tensor")
        return -scores.float().mean() * self.config.video_reward_weight

    def compute_loss(self, prediction: object, batch: ObjectiveBatch) -> TrainStepResult:
        if isinstance(prediction, Mapping) or not isinstance(prediction, Tensor):
            raise TypeError("T2V-Turbo student prediction must be a tensor")
        noisy = batch.model_input
        if isinstance(noisy, Mapping) or not isinstance(noisy, Tensor):
            raise TypeError("T2V-Turbo model_input must be a tensor")
        online = self._consistency_prediction(prediction, noisy, batch.timesteps)
        if self.config.distillation_weight:
            target = self._distillation_target(batch)
            elementwise = latent_consistency_elementwise_loss(
                online,
                target,
                loss_type=self.config.loss_type,
                pseudo_huber_c=self.config.pseudo_huber_c if self.config.loss_type == "pseudo_huber" else None,
            )
        else:
            # The released trainer assigns reward-only workers outside the
            # consistency role.  Preserve that useful property without tying
            # it to a fixed process count: a reward-only objective does not run
            # either frozen teacher branch or the stop-gradient student target.
            elementwise = torch.zeros_like(online, dtype=torch.float32)
        distillation, _, denominator = _per_sample_loss(
            elementwise,
            mask=batch.loss_mask,
            sample_weights=batch.sample_weights,
        )
        prompts = tuple(str(value) for value in batch.metadata.get("prompts", ()))
        if len(prompts) != batch.batch_size:
            raise ValueError("T2V-Turbo metadata must retain one prompt per sample")
        image_reward_loss = self._image_reward_loss(online, batch, prompts)
        video_reward_loss = self._video_reward_loss(online, batch, prompts)
        loss = self.config.distillation_weight * distillation + image_reward_loss + video_reward_loss
        numerator = loss * denominator
        target_values = batch.target
        if isinstance(target_values, Mapping) or not isinstance(target_values, Tensor):
            raise TypeError("T2V-Turbo target must be a tensor")
        latent_tokens = int(target_values.shape[0]) * prod(int(value) for value in target_values.shape[2:])
        return TrainStepResult(
            loss=loss,
            losses={
                "t2v_turbo": loss,
                "t2v_turbo/distillation": distillation,
                "t2v_turbo/image_reward": image_reward_loss,
                "t2v_turbo/video_reward": video_reward_loss,
            },
            metrics={
                "loss_numerator": numerator.detach(),
                "loss_denominator": denominator.detach(),
                "sigma_mean": batch.sigmas.float().mean().detach(),
                "sigma_min": batch.sigmas.float().min().detach(),
                "sigma_max": batch.sigmas.float().max().detach(),
            },
            sample_count=batch.batch_size,
            latent_token_count=latent_tokens,
            diagnostics={
                "target_role": "current_student_stop_gradient",
                "teacher_role": "guided_ddim_trajectory",
                "loss_type": self.config.loss_type,
            },
        )

    def prepared_loss_denominator(self, batch: PreparedBatch) -> Tensor:
        clean = batch.clean_latents
        if isinstance(clean, Mapping) or not isinstance(clean, Tensor):
            raise TypeError("T2V-Turbo clean_latents must be a tensor")
        weights = torch.ones(batch.batch_size, device=clean.device, dtype=torch.float32)
        if isinstance(batch.sample_weights, Tensor):
            weights = batch.sample_weights.to(device=clean.device, dtype=torch.float32)
        if isinstance(batch.loss_mask, Tensor):
            mask = batch.loss_mask
            if mask.ndim + 1 == clean.ndim and int(mask.shape[0]) == batch.batch_size:
                mask = mask.unsqueeze(1)
            active = torch.broadcast_to(mask, clean.shape).reshape(batch.batch_size, -1).sum(dim=1) > 0
            weights = weights * active
        return weights.sum()


__all__ = [
    "DifferentiableImageReward",
    "DifferentiableVideoReward",
    "LVDMEpsilonPredictor",
    "T2VTurboConfig",
    "T2VTurboObjective",
    "T2VTurboTrainAdapter",
    "t2v_turbo_scaled_linear_beta_schedule",
]

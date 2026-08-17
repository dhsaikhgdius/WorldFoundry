"""Two-teacher-step to one-student-step progressive objective."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import ProgressiveDistillationConfig
from .contracts import (
    ProgressiveDistillationBatch,
    ProgressivePredictionAdapter,
    ProgressiveRandomInputs,
)
from .math import (
    add_forward_noise,
    alpha_sigma,
    cosine_logsnr,
    deterministic_ddim_step,
    implied_clean_target,
    prediction_to_clean_epsilon_velocity,
    progressive_loss_per_sample,
)


@dataclass(frozen=True, slots=True)
class ProgressiveDistillationLossResult:
    loss: Tensor
    metrics: Mapping[str, object]


def _adapter_module(
    adapter: ProgressivePredictionAdapter,
    *,
    role: str,
) -> nn.Module:
    if not isinstance(adapter, ProgressivePredictionAdapter):
        raise TypeError(f"{role} must implement ProgressivePredictionAdapter")
    module = adapter.module
    if not isinstance(module, nn.Module):
        raise TypeError(f"{role}.module must be an nn.Module")
    return module


def _prediction(
    adapter: ProgressivePredictionAdapter,
    noisy: Tensor,
    logsnr: Tensor,
    batch: ProgressiveDistillationBatch,
    *,
    training: bool,
    role: str,
) -> Tensor:
    value = adapter.predict_model_output(
        noisy,
        logsnr,
        sample_ids=batch.sample_ids,
        conditioning=batch.conditioning,
        training=training,
    )
    if not isinstance(value, Tensor) or value.shape != noisy.shape:
        raise ValueError(f"{role} model output must match noisy latents")
    if value.device != noisy.device or not value.is_floating_point():
        raise ValueError(f"{role} model output must remain floating on the latent device")
    if not bool(torch.isfinite(value.detach()).all()):
        raise FloatingPointError(f"{role} model output is non-finite")
    return value


class ProgressiveDistillationObjective:
    """Execute the released two-half-step DDIM target construction."""

    def __init__(
        self,
        *,
        student: ProgressivePredictionAdapter,
        teacher: ProgressivePredictionAdapter,
        config: ProgressiveDistillationConfig,
    ) -> None:
        student_module = _adapter_module(student, role="progressive student")
        teacher_module = _adapter_module(teacher, role="progressive teacher")
        if student_module is teacher_module:
            raise ValueError("progressive student and teacher must be distinct modules")
        if {id(value) for value in student_module.parameters()} & {
            id(value) for value in teacher_module.parameters()
        }:
            raise ValueError("progressive student and teacher cannot share parameters")
        if not isinstance(config, ProgressiveDistillationConfig):
            raise TypeError("config must be ProgressiveDistillationConfig")
        teacher_module.requires_grad_(False)
        teacher_module.eval()
        self.student = student
        self.teacher = teacher
        self.student_module = student_module
        self.teacher_module = teacher_module
        self.config = config

    @staticmethod
    def loss_denominator(batch: ProgressiveDistillationBatch) -> int:
        if not isinstance(batch, ProgressiveDistillationBatch):
            raise TypeError("batch must be ProgressiveDistillationBatch")
        return batch.batch_size

    @staticmethod
    def _validate_random_inputs(
        clean: Tensor,
        random_inputs: ProgressiveRandomInputs,
        *,
        student_num_steps: int,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(random_inputs, ProgressiveRandomInputs):
            raise TypeError("random_inputs must be ProgressiveRandomInputs")
        noise = random_inputs.noise
        indices = random_inputs.timestep_indices
        if not isinstance(noise, Tensor) or noise.shape != clean.shape:
            raise ValueError("progressive noise must match clean_latents")
        if noise.device != clean.device or noise.dtype != clean.dtype:
            raise ValueError("progressive noise must share latent device and dtype")
        if not isinstance(indices, Tensor) or indices.shape != (clean.shape[0],):
            raise ValueError("progressive timestep_indices must have shape [B]")
        if indices.device != clean.device or indices.dtype != torch.int64:
            raise ValueError("progressive timestep_indices must be int64 on the latent device")
        if not bool(((indices >= 0) & (indices < student_num_steps)).all()):
            raise ValueError("progressive timestep index lies outside the active stage")
        return noise, indices

    def loss(
        self,
        batch: ProgressiveDistillationBatch,
        *,
        random_inputs: ProgressiveRandomInputs,
        student_num_steps: int,
    ) -> ProgressiveDistillationLossResult:
        if not isinstance(batch, ProgressiveDistillationBatch):
            raise TypeError("batch must be ProgressiveDistillationBatch")
        if (
            isinstance(student_num_steps, bool)
            or not isinstance(student_num_steps, int)
            or student_num_steps <= 0
        ):
            raise ValueError("student_num_steps must be a positive integer")
        clean = batch.clean_latents
        if not isinstance(clean, Tensor) or not clean.is_floating_point():
            raise TypeError("clean_latents must be a floating torch.Tensor")
        noise, indices = self._validate_random_inputs(
            clean,
            random_inputs,
            student_num_steps=student_num_steps,
        )
        dtype = torch.float32
        start_times = (indices.to(dtype=dtype) + 1.0) / float(student_num_steps)
        middle_times = start_times - 0.5 / float(student_num_steps)
        end_times = start_times - 1.0 / float(student_num_steps)
        schedule_kwargs = {
            "logsnr_min": self.config.logsnr_min,
            "logsnr_max": self.config.logsnr_max,
        }
        start_logsnr = cosine_logsnr(start_times, **schedule_kwargs)
        middle_logsnr = cosine_logsnr(middle_times, **schedule_kwargs)
        end_logsnr = cosine_logsnr(end_times, **schedule_kwargs)
        start_alpha, start_sigma = alpha_sigma(start_logsnr, clean)
        noisy = add_forward_noise(clean, noise, start_alpha, start_sigma)

        self.teacher_module.eval()
        with torch.no_grad():
            first_output = _prediction(
                self.teacher,
                noisy,
                start_logsnr,
                batch,
                training=False,
                role="progressive teacher first half-step",
            )
            first_clean, first_epsilon, _ = prediction_to_clean_epsilon_velocity(
                first_output,
                noisy,
                start_alpha,
                start_sigma,
                prediction_type=self.config.prediction_type,
            )
            middle_latents = deterministic_ddim_step(
                first_clean,
                first_epsilon,
                middle_logsnr,
            )
            middle_alpha, middle_sigma = alpha_sigma(
                middle_logsnr,
                middle_latents,
            )
            second_output = _prediction(
                self.teacher,
                middle_latents,
                middle_logsnr,
                batch,
                training=False,
                role="progressive teacher second half-step",
            )
            second_clean, second_epsilon, _ = prediction_to_clean_epsilon_velocity(
                second_output,
                middle_latents,
                middle_alpha,
                middle_sigma,
                prediction_type=self.config.prediction_type,
            )
            teacher_end = deterministic_ddim_step(
                second_clean,
                second_epsilon,
                end_logsnr,
            )
            target_clean = implied_clean_target(
                noisy,
                teacher_end,
                second_clean,
                start_logsnr,
                end_logsnr,
                indices,
            )
            target_alpha, target_sigma = alpha_sigma(start_logsnr, target_clean)
            target_epsilon = (noisy.float() - target_alpha * target_clean) / target_sigma
            target_velocity = target_alpha * target_epsilon - target_sigma * target_clean

        student_output = _prediction(
            self.student,
            noisy,
            start_logsnr,
            batch,
            training=True,
            role="progressive student",
        )
        predicted_clean, predicted_epsilon, predicted_velocity = (
            prediction_to_clean_epsilon_velocity(
                student_output,
                noisy,
                start_alpha,
                start_sigma,
                prediction_type=self.config.prediction_type,
            )
        )
        per_sample = progressive_loss_per_sample(
            predicted_clean,
            predicted_epsilon,
            predicted_velocity,
            target_clean,
            target_epsilon,
            target_velocity,
            loss_weight=self.config.loss_weight,
        )
        loss = per_sample.mean()
        if not bool(torch.isfinite(loss.detach())):
            raise FloatingPointError("progressive distillation loss is non-finite")
        return ProgressiveDistillationLossResult(
            loss=loss,
            metrics={
                "loss_denominator": torch.tensor(
                    batch.batch_size,
                    device=loss.device,
                    dtype=torch.float32,
                ),
                "student_num_steps": student_num_steps,
                "teacher_num_steps": student_num_steps * 2,
                "timestep_index_mean": indices.float().mean().detach(),
                "start_time_mean": start_times.mean().detach(),
            },
        )


__all__ = [
    "ProgressiveDistillationLossResult",
    "ProgressiveDistillationObjective",
]

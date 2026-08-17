"""Official-semantics causal consistency teacher/student objective."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ..causal.contracts import CausalCleanPredictionAdapter, CausalVelocityPredictionAdapter
from .config import (
    CausalConsistencyConfig,
    build_causal_consistency_schedule,
)
from .contracts import CausalConsistencyTrainingBatch
from .math import (
    adjacent_flow_euler_step,
    classifier_free_velocity,
    flow_corrupt,
    full_frame_timesteps,
)


@dataclass(frozen=True, slots=True)
class CausalConsistencyLossResult:
    loss: Tensor
    metrics: Mapping[str, object]


class CausalConsistencyObjective:
    """One adjacent causal-teacher Euler step and an EMA consistency target."""

    def __init__(
        self,
        *,
        student: CausalCleanPredictionAdapter,
        teacher: CausalVelocityPredictionAdapter,
        ema_student: CausalCleanPredictionAdapter,
        config: CausalConsistencyConfig,
    ) -> None:
        if not isinstance(student, CausalCleanPredictionAdapter):
            raise TypeError("student must implement CausalCleanPredictionAdapter")
        if not isinstance(teacher, CausalVelocityPredictionAdapter):
            raise TypeError("teacher must implement CausalVelocityPredictionAdapter")
        if not isinstance(ema_student, CausalCleanPredictionAdapter):
            raise TypeError("ema_student must implement CausalCleanPredictionAdapter")
        modules = (student.module, teacher.module, ema_student.module)
        if not all(isinstance(module, torch.nn.Module) for module in modules):
            raise TypeError("causal consistency adapters must expose nn.Module roles")
        if len({id(module) for module in modules}) != 3:
            raise ValueError("student, causal teacher, and EMA target must be distinct modules")
        if not isinstance(config, CausalConsistencyConfig):
            raise TypeError("config must be CausalConsistencyConfig")
        teacher.module.requires_grad_(False)
        ema_student.module.requires_grad_(False)
        teacher.module.eval()
        ema_student.module.eval()
        self.student = student
        self.teacher = teacher
        self.ema_student = ema_student
        self.config = config
        self.schedule = build_causal_consistency_schedule(config)

    @property
    def pair_count(self) -> int:
        return self.schedule.pair_count

    def loss_denominator(self, batch: CausalConsistencyTrainingBatch) -> int:
        clean = batch.clean_latents
        if not isinstance(clean, Tensor):
            raise TypeError("native Causal Consistency requires torch.Tensor latents")
        return clean.numel()

    def loss(
        self,
        batch: CausalConsistencyTrainingBatch,
        *,
        pair_index: int,
        noise: Tensor,
    ) -> CausalConsistencyLossResult:
        if not isinstance(batch, CausalConsistencyTrainingBatch):
            raise TypeError("batch must be CausalConsistencyTrainingBatch")
        clean = batch.clean_latents
        if not isinstance(clean, Tensor) or not clean.is_floating_point():
            raise TypeError("clean_latents must be a floating torch.Tensor")
        if isinstance(pair_index, bool) or not isinstance(pair_index, int):
            raise TypeError("pair_index must be an integer")
        if not 0 <= pair_index < self.pair_count:
            raise ValueError("pair_index is outside the adjacent consistency schedule")
        if not isinstance(noise, Tensor) or noise.shape != clean.shape or noise.dtype != clean.dtype:
            raise ValueError("noise must match clean_latents shape and dtype")
        if noise.device != clean.device:
            raise ValueError("noise and clean_latents must share a device")

        timestep = self.schedule.timesteps[pair_index]
        next_timestep = self.schedule.timesteps[pair_index + 1]
        sigma = self.schedule.sigmas[pair_index]
        latent_t = flow_corrupt(clean, noise, sigma)
        timesteps = full_frame_timesteps(
            latent_t,
            timestep,
            frame_dim=self.config.frame_dim,
        )
        next_timesteps = full_frame_timesteps(
            latent_t,
            next_timestep,
            frame_dim=self.config.frame_dim,
        )

        self.teacher.module.eval()
        self.ema_student.module.eval()
        with torch.no_grad():
            conditional_velocity = self.teacher.predict_velocity(
                latent_t,
                timesteps,
                clean_context=clean,
                sample_ids=batch.sample_ids,
                conditioning=batch.conditioning,
                training=False,
            )
            unconditional_velocity = self.teacher.predict_velocity(
                latent_t,
                timesteps,
                clean_context=clean,
                sample_ids=batch.sample_ids,
                conditioning=batch.unconditional_conditioning,
                training=False,
            )
            if (
                not isinstance(conditional_velocity, Tensor)
                or conditional_velocity.shape != clean.shape
                or not isinstance(unconditional_velocity, Tensor)
                or unconditional_velocity.shape != clean.shape
            ):
                raise ValueError("causal teacher velocities must match clean_latents")
            guided_velocity = classifier_free_velocity(
                conditional_velocity,
                unconditional_velocity,
                self.config.guidance_scale,
            )
            latent_next = adjacent_flow_euler_step(
                latent_t,
                guided_velocity,
                timestep=timestep,
                next_timestep=next_timestep,
                num_train_timesteps=self.schedule.num_train_timesteps,
            )

        prediction = self.student.predict_clean(
            latent_t,
            timesteps,
            clean_context=clean,
            sample_ids=batch.sample_ids,
            conditioning=batch.conditioning,
            training=True,
        )
        if not isinstance(prediction, Tensor) or prediction.shape != clean.shape:
            raise ValueError("causal consistency student prediction must match clean_latents")
        with torch.no_grad():
            target = self.ema_student.predict_clean(
                latent_next,
                next_timesteps,
                clean_context=clean,
                sample_ids=batch.sample_ids,
                conditioning=batch.conditioning,
                training=False,
            )
            if not isinstance(target, Tensor) or target.shape != clean.shape:
                raise ValueError("EMA consistency target must match clean_latents")
            target = target.detach()
        loss = F.mse_loss(prediction, target, reduction="mean")
        if not bool(torch.isfinite(loss.detach())):
            raise FloatingPointError("non-finite causal consistency loss")
        per_sample = (prediction - target).float().square().flatten(1).mean(dim=1)
        return CausalConsistencyLossResult(
            loss=loss,
            metrics={
                "loss_denominator": clean.numel(),
                "pair_index": pair_index,
                "timestep": timestep,
                "next_timestep": next_timestep,
                "per_sample_mse": per_sample.detach(),
            },
        )


__all__ = ["CausalConsistencyLossResult", "CausalConsistencyObjective"]

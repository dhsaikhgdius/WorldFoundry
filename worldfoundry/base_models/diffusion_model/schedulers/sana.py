"""Stateless schedulers for Sana recipes."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..components import ComponentBuildContext
from ..contracts import SamplingConfig, SchedulerStep
from .flow_dpm import FlowDPMSolverMultistepScheduler
from .wan import WanFlowMatchEulerScheduler


def build_sana_flow_match_scheduler(context: ComponentBuildContext) -> WanFlowMatchEulerScheduler:
    """Reuse the canonical shifted flow Euler trajectory for Sana."""

    return WanFlowMatchEulerScheduler(
        num_train_timesteps=int(context.component_options.get("num_train_timesteps", 1000)),
        shift=float(context.component_options.get("shift", 3.0)),
    )


class SanaFlowDPMScheduler:
    """DPM-Solver++ order-2 adapter for released bidirectional Sana graphs."""

    def __init__(self, *, shift: float = 8.0, num_train_timesteps: int = 1000) -> None:
        self.shift = float(shift)
        self.solver = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=int(num_train_timesteps),
            solver_order=2,
            prediction_type="flow_prediction",
            shift=1.0,
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            lower_order_final=True,
            final_sigmas_type="zero",
        )

    def schedule(
        self,
        sampling: SamplingConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[SchedulerStep, ...]:
        del dtype
        unsupported = set(sampling.scheduler_options) - {"shift"}
        if unsupported:
            raise ValueError(f"unsupported SanaFlowDPMScheduler options: {sorted(unsupported)}")
        self.solver.set_timesteps(
            sampling.num_inference_steps,
            device=device,
            shift=float(sampling.scheduler_options.get("shift", self.shift)),
        )
        timesteps = self.solver.timesteps
        terminal = torch.zeros((), device=device, dtype=timesteps.dtype)
        return tuple(
            SchedulerStep(
                index=index,
                timestep=timestep,
                next_timestep=(timesteps[index + 1] if index + 1 < len(timesteps) else terminal),
            )
            for index, timestep in enumerate(timesteps)
        )

    @staticmethod
    def scale_model_input(latents: Tensor, step: SchedulerStep) -> Tensor:
        del step
        return latents

    def step(
        self,
        model_output: Tensor,
        step: SchedulerStep,
        latents: Tensor,
        *,
        generator: torch.Generator,
    ) -> Tensor:
        return self.solver.step(
            model_output,
            step.timestep,
            latents,
            generator=generator,
            return_dict=False,
        )[0]


def build_sana_flow_dpm_scheduler(context: ComponentBuildContext) -> SanaFlowDPMScheduler:
    return SanaFlowDPMScheduler(
        shift=float(context.component_options.get("shift", 8.0)),
        num_train_timesteps=int(context.component_options.get("num_train_timesteps", 1000)),
    )


class SanaStreamingEulerScheduler(WanFlowMatchEulerScheduler):
    """Euler flow schedule including released 2/4-step streaming timesteps."""

    _FIXED_TIMESTEPS = {
        2: (1000.0, 743.0),
        4: (1000.0, 961.0, 893.0, 743.0),
    }

    def schedule(
        self,
        sampling: SamplingConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[SchedulerStep, ...]:
        fixed = self._FIXED_TIMESTEPS.get(sampling.num_inference_steps)
        if fixed is None:
            return super().schedule(sampling, device=device, dtype=dtype)
        unsupported = set(sampling.scheduler_options) - {"shift"}
        if unsupported:
            raise ValueError(
                f"unsupported SanaStreamingEulerScheduler options: {sorted(unsupported)}"
            )
        timesteps = torch.tensor(
            (*fixed, 0.0),
            device=device,
            dtype=self.timestep_dtype,
        )
        return tuple(
            SchedulerStep(
                index=index,
                timestep=timesteps[index],
                next_timestep=timesteps[index + 1],
            )
            for index in range(sampling.num_inference_steps)
        )


def build_sana_streaming_euler_scheduler(
    context: ComponentBuildContext,
) -> SanaStreamingEulerScheduler:
    return SanaStreamingEulerScheduler(
        num_train_timesteps=int(context.component_options.get("num_train_timesteps", 1000)),
        shift=float(context.component_options.get("shift", 8.0)),
    )


class SanaSCMScheduler:
    """Native TrigFlow/sCM sampler used by Sana Sprint checkpoints."""

    def __init__(self, *, sigma_data: float = 0.5) -> None:
        self.sigma_data = float(sigma_data)
        if self.sigma_data <= 0:
            raise ValueError("Sana Sprint sigma_data must be positive")

    def schedule(
        self,
        sampling: SamplingConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[SchedulerStep, ...]:
        del dtype
        unsupported = set(sampling.scheduler_options) - {"timesteps"}
        if unsupported:
            raise ValueError(f"unsupported SanaSCMScheduler options: {sorted(unsupported)}")
        explicit = sampling.scheduler_options.get("timesteps")
        if explicit is not None:
            values = tuple(float(value) for value in explicit)  # type: ignore[arg-type]
            if len(values) != sampling.num_inference_steps + 1:
                raise ValueError("Sana Sprint explicit timesteps must contain num_inference_steps + 1 values")
            timesteps = torch.tensor(values, device=device, dtype=torch.float32)
        elif sampling.num_inference_steps == 2:
            timesteps = torch.tensor((math.pi / 2, 1.3, 0.0), device=device, dtype=torch.float32)
        else:
            timesteps = torch.linspace(
                math.pi / 2,
                0.0,
                sampling.num_inference_steps + 1,
                device=device,
                dtype=torch.float32,
            )
        return tuple(
            SchedulerStep(index=index, timestep=timesteps[index], next_timestep=timesteps[index + 1])
            for index in range(sampling.num_inference_steps)
        )

    def scale_model_input(self, latents: Tensor, step: SchedulerStep) -> Tensor:
        del step
        return latents / self.sigma_data

    def step(
        self,
        model_output: Tensor,
        step: SchedulerStep,
        latents: Tensor,
        *,
        generator: torch.Generator,
    ) -> Tensor:
        current = step.timestep.to(device=latents.device, dtype=latents.dtype)
        following = step.next_timestep.to(device=latents.device, dtype=latents.dtype)
        pred_x0 = torch.cos(current) * latents - torch.sin(current) * model_output.to(latents.dtype)
        noise = torch.randn(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        ) * self.sigma_data
        return torch.cos(following) * pred_x0 + torch.sin(following) * noise


def build_sana_scm_scheduler(context: ComponentBuildContext) -> SanaSCMScheduler:
    return SanaSCMScheduler(sigma_data=float(context.component_options.get("sigma_data", 0.5)))


__all__ = [
    "SanaSCMScheduler",
    "SanaFlowDPMScheduler",
    "SanaStreamingEulerScheduler",
    "build_sana_flow_dpm_scheduler",
    "build_sana_flow_match_scheduler",
    "build_sana_scm_scheduler",
    "build_sana_streaming_euler_scheduler",
]

"""Canonical flow-matching schedules used by Wan inference families."""

from __future__ import annotations

import torch
from torch import Tensor

from ..components import ComponentBuildContext
from ..contracts import SamplingConfig, SchedulerStep


def _validate_schedule_values(
    *,
    num_inference_steps: int,
    shift: float,
    sigma_max: float,
    sigma_min: float,
    denoising_strength: float,
) -> None:
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    if shift <= 0:
        raise ValueError("shift must be positive")
    if sigma_max <= sigma_min:
        raise ValueError("sigma_max must be greater than sigma_min")
    if not 0 <= denoising_strength <= 1:
        raise ValueError("denoising_strength must be in [0, 1]")


def shift_flow_sigmas(sigmas: Tensor, shift: float) -> Tensor:
    """Apply Wan's rational flow-shift transform without changing dtype/device."""

    shift = float(shift)
    if shift <= 0:
        raise ValueError("shift must be positive")
    return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)


def build_wan_sigmas(
    num_inference_steps: int,
    *,
    shift: float = 3.0,
    sigma_max: float = 1.0,
    sigma_min: float = 0.0,
    denoising_strength: float = 1.0,
    include_terminal: bool = False,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build Wan's descending shifted sigma trajectory.

    By default the returned tensor contains one sigma per model evaluation.
    ``include_terminal=True`` appends the final integration point, producing
    ``num_inference_steps + 1`` values for a stateless native runner.
    """

    num_inference_steps = int(num_inference_steps)
    shift = float(shift)
    sigma_max = float(sigma_max)
    sigma_min = float(sigma_min)
    denoising_strength = float(denoising_strength)
    _validate_schedule_values(
        num_inference_steps=num_inference_steps,
        shift=shift,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        denoising_strength=denoising_strength,
    )
    if not dtype.is_floating_point:
        raise ValueError("Wan sigma dtype must be floating point")

    sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
    path = torch.linspace(
        sigma_start,
        sigma_min,
        num_inference_steps + 1,
        device=device,
        dtype=dtype,
    )
    shifted = shift_flow_sigmas(path, shift)
    return shifted if include_terminal else shifted[:-1]


def _broadcast_sigma(sigma: Tensor, reference: Tensor) -> Tensor:
    values = sigma.to(device=reference.device, dtype=reference.dtype).reshape(-1)
    if values.numel() == 1:
        return values.reshape((1,) * reference.ndim)
    if not reference.ndim or values.numel() != reference.shape[0]:
        raise ValueError(
            "sigma must be scalar or have one value per sample: "
            f"got {values.numel()} values for shape {tuple(reference.shape)}"
        )
    return values.reshape(values.shape[0], *((1,) * (reference.ndim - 1)))


def add_flow_noise(original: Tensor, noise: Tensor, sigma: Tensor) -> Tensor:
    """Interpolate clean samples and noise at a flow-matching sigma."""

    if original.shape != noise.shape:
        raise ValueError(f"original and noise shapes must match: {original.shape} != {noise.shape}")
    weight = _broadcast_sigma(sigma, noise)
    return ((1.0 - weight) * original + weight * noise).type_as(noise)


def flow_prediction_to_x0(flow: Tensor, noisy: Tensor, sigma: Tensor) -> Tensor:
    """Convert a Wan flow prediction into its clean-sample prediction."""

    if flow.shape != noisy.shape:
        raise ValueError(f"flow and noisy shapes must match: {flow.shape} != {noisy.shape}")
    original_dtype = flow.dtype
    flow64 = flow.double()
    noisy64 = noisy.double()
    weight = _broadcast_sigma(sigma, flow64)
    return (noisy64 - weight * flow64).to(original_dtype)


class WanFlowMatchEulerScheduler:
    """Stateless native Euler scheduler matching Wan's shifted flow trajectory."""

    def __init__(
        self,
        *,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.0,
        timestep_dtype: torch.dtype = torch.float32,
    ) -> None:
        if int(num_train_timesteps) <= 0:
            raise ValueError("num_train_timesteps must be positive")
        _validate_schedule_values(
            num_inference_steps=1,
            shift=float(shift),
            sigma_max=float(sigma_max),
            sigma_min=float(sigma_min),
            denoising_strength=1.0,
        )
        if not timestep_dtype.is_floating_point:
            raise ValueError("timestep_dtype must be floating point")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.timestep_dtype = timestep_dtype

    def schedule(
        self,
        sampling: SamplingConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[SchedulerStep, ...]:
        del dtype
        supported_options = {
            "denoising_strength",
            "shift",
            "sigma_max",
            "sigma_min",
        }
        unsupported = set(sampling.scheduler_options) - supported_options
        if unsupported:
            raise ValueError(f"unsupported WanFlowMatchEulerScheduler options: {sorted(unsupported)}")
        options = sampling.scheduler_options
        sigmas = build_wan_sigmas(
            sampling.num_inference_steps,
            shift=float(options.get("shift", self.shift)),
            sigma_max=float(options.get("sigma_max", self.sigma_max)),
            sigma_min=float(options.get("sigma_min", self.sigma_min)),
            denoising_strength=float(options.get("denoising_strength", 1.0)),
            include_terminal=True,
            device=device,
            dtype=self.timestep_dtype,
        )
        timesteps = sigmas * self.num_train_timesteps
        return tuple(
            SchedulerStep(
                index=index,
                timestep=timesteps[index],
                next_timestep=timesteps[index + 1],
            )
            for index in range(sampling.num_inference_steps)
        )

    def scale_model_input(self, latents: Tensor, step: SchedulerStep) -> Tensor:
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
        del generator
        timestep_scale = float(self.num_train_timesteps)
        sigma = step.timestep / timestep_scale
        next_sigma = step.next_timestep / timestep_scale
        delta = (next_sigma - sigma).to(device=latents.device, dtype=latents.dtype)
        return latents + delta * model_output.to(latents.dtype)


def build_wan_flow_match_euler_scheduler(
    context: ComponentBuildContext,
) -> WanFlowMatchEulerScheduler:
    """Build the shared Wan Euler scheduler from component options."""

    options = context.component_options
    return WanFlowMatchEulerScheduler(
        num_train_timesteps=int(options.get("num_train_timesteps", 1000)),
        shift=float(options.get("shift", 3.0)),
        sigma_max=float(options.get("sigma_max", 1.0)),
        sigma_min=float(options.get("sigma_min", 0.0)),
    )


class WanFlowUniPCScheduler:
    """Native scheduler-contract adapter for Wan's canonical UniPC solver."""

    def __init__(
        self,
        *,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        use_karras_sigma: bool = False,
    ) -> None:
        from .flow_unipc import FlowUniPCMultistepScheduler

        self.shift = float(shift)
        self.use_karras_sigma = bool(use_karras_sigma)
        self.solver = FlowUniPCMultistepScheduler(
            num_train_timesteps=int(num_train_timesteps),
            shift=1,
            use_dynamic_shifting=False,
        )

    def schedule(
        self,
        sampling: SamplingConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[SchedulerStep, ...]:
        del dtype
        unsupported = set(sampling.scheduler_options) - {"shift", "use_karras_sigma"}
        if unsupported:
            raise ValueError(f"unsupported WanFlowUniPCScheduler options: {sorted(unsupported)}")
        use_karras_sigma = bool(sampling.scheduler_options.get("use_karras_sigma", self.use_karras_sigma))
        # The NVIDIA solver's historical Karras branch interprets ``N`` as
        # intervals and returns N+1 model-evaluation points. The canonical
        # scheduler contract defines N as evaluations, so translate here.
        solver_steps = sampling.num_inference_steps - 1 if use_karras_sigma and sampling.num_inference_steps > 1 else sampling.num_inference_steps
        self.solver.set_timesteps(
            solver_steps,
            device=device,
            shift=float(sampling.scheduler_options.get("shift", self.shift)),
            use_kerras_sigma=use_karras_sigma,
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

    def scale_model_input(self, latents: Tensor, step: SchedulerStep) -> Tensor:
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
            return_dict=False,
            generator=generator,
        )[0]


def build_wan_flow_unipc_scheduler(context: ComponentBuildContext) -> WanFlowUniPCScheduler:
    """Build the shared Wan UniPC solver from component options."""

    return WanFlowUniPCScheduler(
        num_train_timesteps=int(context.component_options.get("num_train_timesteps", 1000)),
        shift=float(context.component_options.get("shift", 5.0)),
        use_karras_sigma=bool(context.component_options.get("use_karras_sigma", False)),
    )


class InferenceFlowMatchScheduler:
    """Transitional stateful API for causal Wan runtimes.

    New native recipes should use :class:`WanFlowMatchEulerScheduler`. This
    compatibility surface preserves the exact API used by existing causal
    model wrappers while sharing the canonical schedule and conversion code.
    """

    def __init__(
        self,
        num_inference_steps: int = 100,
        *,
        num_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        extra_one_step: bool = False,
    ) -> None:
        if int(num_timesteps) <= 0:
            raise ValueError("num_timesteps must be positive")
        self.num_timesteps = int(num_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.extra_one_step = bool(extra_one_step)
        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self,
        num_inference_steps: int,
        denoising_strength: float = 1.0,
    ) -> None:
        num_inference_steps = int(num_inference_steps)
        if self.extra_one_step:
            self.sigmas = build_wan_sigmas(
                num_inference_steps,
                shift=self.shift,
                sigma_max=self.sigma_max,
                sigma_min=self.sigma_min,
                denoising_strength=denoising_strength,
            )
        else:
            _validate_schedule_values(
                num_inference_steps=num_inference_steps,
                shift=self.shift,
                sigma_max=self.sigma_max,
                sigma_min=self.sigma_min,
                denoising_strength=float(denoising_strength),
            )
            sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * float(denoising_strength)
            unshifted = torch.linspace(
                sigma_start,
                self.sigma_min,
                num_inference_steps,
                dtype=torch.float32,
            )
            self.sigmas = shift_flow_sigmas(unshifted, self.shift)
        self.timesteps = self.sigmas * self.num_timesteps

    def _indices(self, timestep: Tensor, device: torch.device) -> Tensor:
        timesteps = self.timesteps.to(device=device, dtype=torch.float64)
        values = timestep.reshape(-1).to(device=device, dtype=torch.float64)
        return torch.argmin(
            (timesteps.unsqueeze(0) - values.unsqueeze(1)).abs(),
            dim=1,
        )

    def sigma_at(self, timestep: Tensor, reference: Tensor) -> Tensor:
        indices = self._indices(timestep, reference.device)
        values = self.sigmas.to(device=reference.device, dtype=reference.dtype)[indices]
        return _broadcast_sigma(values, reference)

    def add_noise(self, original_samples: Tensor, noise: Tensor, timestep: Tensor) -> Tensor:
        return add_flow_noise(
            original_samples,
            noise,
            self.sigma_at(timestep, noise),
        )

    def flow_to_x0(self, flow: Tensor, noisy: Tensor, timestep: Tensor) -> Tensor:
        sigma = self.sigma_at(timestep, flow.double())
        return flow_prediction_to_x0(flow, noisy, sigma)

    def flow_step(
        self,
        flow: Tensor,
        noisy: Tensor,
        timestep: Tensor,
        next_timestep: Tensor,
    ) -> Tensor:
        sigma = self.sigma_at(timestep, noisy)
        next_sigma = self.sigma_at(next_timestep, noisy)
        return noisy + flow.to(noisy.dtype) * (next_sigma - sigma)


__all__ = [
    "InferenceFlowMatchScheduler",
    "WanFlowMatchEulerScheduler",
    "WanFlowUniPCScheduler",
    "add_flow_noise",
    "build_wan_flow_match_euler_scheduler",
    "build_wan_flow_unipc_scheduler",
    "build_wan_sigmas",
    "flow_prediction_to_x0",
    "shift_flow_sigmas",
]

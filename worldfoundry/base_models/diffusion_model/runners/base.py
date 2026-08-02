"""Canonical diffusion runner with an explicit framework-owned sampling loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch import Tensor

from ..contracts import (
    ConditionEncoder,
    Conditioning,
    Denoiser,
    DenoiserInput,
    DenoiserOutput,
    DiffusionOutput,
    DiffusionRequest,
    DiffusionScheduler,
    EncodedLatentInitializer,
    FinalDenoiseScheduler,
    LatentDecoder,
    LatentInitialization,
    LatentInitializer,
    LatentEncoder,
)
from ..extensions import DiffusionExtension, DiffusionRunContext


@dataclass(frozen=True, slots=True)
class RunnerComponents:
    """The five canonical components required by the standard runner."""

    denoiser: Denoiser
    conditioner: ConditionEncoder
    latent_initializer: LatentInitializer
    scheduler: DiffusionScheduler
    decoder: LatentDecoder
    latent_encoder: LatentEncoder | None = None


class NativeDiffusionRunner:
    """Framework-owned denoising loop composed from native component contracts."""

    def __init__(
        self,
        *,
        model_id: str,
        components: RunnerComponents,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        extensions: Iterable[DiffusionExtension] = (),
        guidance_mode: str = "standard",
    ) -> None:
        self.model_id = str(model_id)
        if not self.model_id.strip():
            raise ValueError("model_id cannot be empty")
        self.components = components
        self.device = torch.device(device)
        self.dtype = dtype
        self.extensions = tuple(extensions)
        self.guidance_mode = str(guidance_mode).strip().lower().replace("_", "-")
        if self.guidance_mode not in {"standard", "positive"}:
            raise ValueError(f"unsupported classifier-free guidance mode: {guidance_mode!r}")
        extension_ids = [extension.extension_id for extension in self.extensions]
        if len(extension_ids) != len(set(extension_ids)):
            raise ValueError("extension_id values must be unique within one runner")

    def _generator(self, seed: int) -> torch.Generator:
        try:
            generator = torch.Generator(device=self.device)
        except (RuntimeError, TypeError):
            generator = torch.Generator(device=self.device.type)
        return generator.manual_seed(seed)

    def _is_runtime_device(self, device: torch.device) -> bool:
        """Accept the active device when the configured index is implicit."""

        if device.type != self.device.type:
            return False
        return self.device.index is None or device.index == self.device.index

    @staticmethod
    def _branch_conditioning(
        conditioning: Conditioning,
        branch: str,
    ) -> Mapping[str, object]:
        values: dict[str, object] = dict(conditioning.shared)
        values.update(conditioning.positive if branch == "positive" else conditioning.negative)
        return values

    def _call_denoiser(
        self,
        context: DiffusionRunContext,
        *,
        latents: Tensor,
        branch: str,
    ) -> DenoiserOutput:
        return self._call_denoiser_with_conditioning(
            context,
            latents=latents,
            branch=branch,
            conditioning=self._branch_conditioning(context.conditioning, branch),
        )

    def _call_denoiser_with_conditioning(
        self,
        context: DiffusionRunContext,
        *,
        latents: Tensor,
        branch: str,
        conditioning: Mapping[str, object],
    ) -> DenoiserOutput:
        """Evaluate one branch with an explicitly composed condition mapping."""

        step = context.step
        if step is None:
            raise RuntimeError("denoiser called before a scheduler step was selected")
        model_input = DenoiserInput(
            latents=latents,
            timestep=step.timestep,
            next_timestep=step.next_timestep,
            conditioning=conditioning,
            step_index=step.index,
            total_steps=context.request.sampling.num_inference_steps,
            branch=branch,
        )
        for extension in self.extensions:
            model_input = extension.before_denoiser(context, model_input)
            if not isinstance(model_input, DenoiserInput):
                raise TypeError(
                    f"{extension.extension_id}.before_denoiser must return "
                    f"DenoiserInput, got {type(model_input).__name__}"
                )
        model_output = self.components.denoiser(model_input)
        if not isinstance(model_output, DenoiserOutput):
            raise TypeError(f"denoiser must return DenoiserOutput, got {type(model_output).__name__}")
        for extension in reversed(self.extensions):
            model_output = extension.after_denoiser(context, model_output)
            if not isinstance(model_output, DenoiserOutput):
                raise TypeError(
                    f"{extension.extension_id}.after_denoiser must return "
                    f"DenoiserOutput, got {type(model_output).__name__}"
                )
        if model_output.sample.shape != latents.shape:
            raise ValueError(
                "denoiser output shape must match latent shape: "
                f"{tuple(model_output.sample.shape)} != {tuple(latents.shape)}"
            )
        return model_output

    def predict(
        self,
        context: DiffusionRunContext,
        model_latents: Tensor,
    ) -> DenoiserOutput:
        """Predict one guided update.

        Families with embedded guidance or multi-stage denoisers can override
        this method while keeping the same lifecycle and extension contracts.
        """

        positive = self._call_denoiser(
            context,
            latents=model_latents,
            branch="positive",
        )
        scale = context.request.sampling.guidance_scale
        if not context.conditioning.negative:
            return positive
        if self.guidance_mode == "standard" and scale == 1.0:
            return positive
        if self.guidance_mode == "positive" and scale == 0.0:
            return positive
        negative = self._call_denoiser(
            context,
            latents=model_latents,
            branch="negative",
        )
        if self.guidance_mode == "positive":
            guided = positive.sample + scale * (positive.sample - negative.sample)
        else:
            guided = negative.sample + scale * (positive.sample - negative.sample)
        return DenoiserOutput(
            sample=guided,
            extras={"positive": positive.extras, "negative": negative.extras},
        )

    @torch.no_grad()
    def run(self, request: DiffusionRequest) -> DiffusionOutput:
        """Execute one native diffusion request."""

        generator = self._generator(request.sampling.seed)
        conditioning = self.components.conditioner.encode(
            request,
            device=self.device,
            dtype=self.dtype,
        )
        if not isinstance(conditioning, Conditioning):
            raise TypeError(f"conditioner.encode must return Conditioning, got {type(conditioning).__name__}")
        context = DiffusionRunContext(
            request=request,
            components=self.components,
            conditioning=conditioning,
            generator=generator,
        )

        try:
            for extension in self.extensions:
                extension.on_run_start(context)
            for extension in self.extensions:
                context.conditioning = extension.prepare_conditioning(
                    context,
                    context.conditioning,
                )
                if not isinstance(context.conditioning, Conditioning):
                    raise TypeError(
                        f"{extension.extension_id}.prepare_conditioning must return "
                        f"Conditioning, got {type(context.conditioning).__name__}"
                    )

            if self.components.latent_encoder is not None and isinstance(
                self.components.latent_initializer, EncodedLatentInitializer
            ):
                initialization = self.components.latent_initializer.initialize_with_encoder(
                    request,
                    latent_encoder=self.components.latent_encoder,
                    generator=generator,
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                initialization = self.components.latent_initializer.initialize(
                    request,
                    generator=generator,
                    device=self.device,
                    dtype=self.dtype,
                )
            initialization_artifacts: Mapping[str, object] = {}
            if isinstance(initialization, LatentInitialization):
                overlap = sorted(set(context.conditioning.shared) & set(initialization.conditioning))
                if overlap:
                    raise ValueError(f"latent initialization conditions overlap existing shared values: {overlap}")
                shared = dict(context.conditioning.shared)
                shared.update(initialization.conditioning)
                context.conditioning = Conditioning(
                    positive=context.conditioning.positive,
                    negative=context.conditioning.negative,
                    shared=shared,
                )
                latents = initialization.latents
                initialization_artifacts = initialization.artifacts
            else:
                latents = initialization
            if not isinstance(latents, Tensor):
                raise TypeError("latent_initializer.initialize must return a tensor or LatentInitialization")
            if not self._is_runtime_device(latents.device):
                raise ValueError(f"latent initializer returned {latents.device}, expected {self.device}")

            schedule = tuple(
                self.components.scheduler.schedule(
                    request.sampling,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            if len(schedule) != request.sampling.num_inference_steps:
                raise ValueError(
                    "scheduler returned an unexpected number of steps: "
                    f"{len(schedule)} != {request.sampling.num_inference_steps}"
                )
            for expected_index, step in enumerate(schedule):
                if step.index != expected_index:
                    raise ValueError(
                        "scheduler step indices must be contiguous and zero-based: "
                        f"got {step.index} at position {expected_index}"
                    )

            for step in schedule:
                context.step = step
                model_latents = self.components.scheduler.scale_model_input(
                    latents,
                    step,
                )
                prediction = self.predict(context, model_latents)
                latents = self.components.scheduler.step(
                    prediction.sample,
                    step,
                    latents,
                    generator=generator,
                )
                if not isinstance(latents, Tensor):
                    raise TypeError("scheduler.step must return a tensor")
                for extension in self.extensions:
                    latents = extension.after_step(context, latents)
                    if not isinstance(latents, Tensor):
                        raise TypeError(
                            f"{extension.extension_id}.after_step must return a tensor, got {type(latents).__name__}"
                        )

            final_denoise = False
            if isinstance(self.components.scheduler, FinalDenoiseScheduler):
                final_step = self.components.scheduler.final_denoise_step()
                if final_step is not None:
                    final_denoise = True
                    context.step = final_step
                    model_latents = self.components.scheduler.scale_model_input(
                        latents,
                        final_step,
                    )
                    prediction = self.predict(context, model_latents)
                    latents = prediction.sample
                    for extension in self.extensions:
                        latents = extension.after_step(context, latents)
                        if not isinstance(latents, Tensor):
                            raise TypeError(
                                f"{extension.extension_id}.after_step must return a tensor, "
                                f"got {type(latents).__name__}"
                            )

            sample = self.components.decoder.decode(latents, request)
            if not isinstance(sample, Tensor):
                raise TypeError("decoder.decode must return a tensor")
            for extension in self.extensions:
                sample = extension.after_decode(context, sample)
                if not isinstance(sample, Tensor):
                    raise TypeError(
                        f"{extension.extension_id}.after_decode must return a tensor, got {type(sample).__name__}"
                    )
            output = DiffusionOutput(
                sample=sample,
                latents=latents,
                artifacts=initialization_artifacts,
                metadata={
                    "model_id": self.model_id,
                    "seed": request.sampling.seed,
                    "num_inference_steps": request.sampling.num_inference_steps,
                    "guidance_scale": request.sampling.guidance_scale,
                    "guidance_mode": self.guidance_mode,
                    "final_denoise": final_denoise,
                    "extensions": [extension.extension_id for extension in self.extensions],
                },
            )
            for extension in reversed(self.extensions):
                extension.on_run_end(context)
            return output
        except BaseException as error:
            for extension in reversed(self.extensions):
                extension.on_run_error(context, error)
            raise


class DualConditionGuidanceRunner(NativeDiffusionRunner):
    """Framework-owned three-branch CFG for text plus a secondary condition."""

    def __init__(
        self,
        *,
        secondary_guidance_scale: float = 1.0,
        secondary_guidance_input: str = "secondary_guidance_scale",
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.secondary_guidance_scale = float(secondary_guidance_scale)
        self.secondary_guidance_input = str(secondary_guidance_input)

    def predict(
        self,
        context: DiffusionRunContext,
        model_latents: Tensor,
    ) -> DenoiserOutput:
        positive = self._call_denoiser(context, latents=model_latents, branch="positive")
        text_scale = context.request.sampling.guidance_scale
        if text_scale == 1.0 or not context.conditioning.negative:
            return positive

        negative = self._call_denoiser(context, latents=model_latents, branch="negative")
        unconditional_values = dict(self._branch_conditioning(context.conditioning, "negative"))
        unconditional_values["drop_secondary_condition"] = True
        unconditional = self._call_denoiser_with_conditioning(
            context,
            latents=model_latents,
            branch="unconditional",
            conditioning=unconditional_values,
        )
        secondary_scale = float(
            context.request.inputs.get(
                self.secondary_guidance_input,
                self.secondary_guidance_scale,
            )
        )
        guided = (
            unconditional.sample
            + secondary_scale * (negative.sample - unconditional.sample)
            + text_scale * (positive.sample - negative.sample)
        )
        return DenoiserOutput(
            sample=guided,
            extras={
                "positive": positive.extras,
                "negative": negative.extras,
                "unconditional": unconditional.extras,
            },
        )


__all__ = ["DualConditionGuidanceRunner", "NativeDiffusionRunner", "RunnerComponents"]

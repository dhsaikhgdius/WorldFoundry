"""Composable research extensions for the canonical diffusion loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ..contracts import (
    Conditioning,
    DenoiserInput,
    DenoiserOutput,
    DiffusionRequest,
    SchedulerStep,
)

if TYPE_CHECKING:
    from ..runners import RunnerComponents


@dataclass(slots=True)
class DiffusionRunContext:
    """Per-run state visible to explicitly installed extensions."""

    request: DiffusionRequest
    components: "RunnerComponents"
    conditioning: Conditioning
    generator: torch.Generator
    state: dict[str, object] = field(default_factory=dict)
    step: SchedulerStep | None = None


class DiffusionExtension:
    """No-op base class for memory, control, and research interventions.

    Extensions are installed on one runner instance.  They do not mutate class
    definitions or process-wide registries, so multiple research variants can
    coexist safely in the same Python process.
    """

    extension_id = "extension"

    def on_run_start(self, context: DiffusionRunContext) -> None:
        """Initialize extension-owned state."""

    def prepare_conditioning(
        self,
        context: DiffusionRunContext,
        conditioning: Conditioning,
    ) -> Conditioning:
        """Add or transform conditions once before sampling."""

        return conditioning

    def before_denoiser(
        self,
        context: DiffusionRunContext,
        model_input: DenoiserInput,
    ) -> DenoiserInput:
        """Inject step-local conditions immediately before the network call."""

        return model_input

    def after_denoiser(
        self,
        context: DiffusionRunContext,
        model_output: DenoiserOutput,
    ) -> DenoiserOutput:
        """Observe or transform one branch prediction."""

        return model_output

    def after_step(
        self,
        context: DiffusionRunContext,
        latents: Tensor,
    ) -> Tensor:
        """Observe or transform latents after a scheduler update."""

        return latents

    def after_decode(
        self,
        context: DiffusionRunContext,
        sample: Tensor,
    ) -> Tensor:
        """Observe or transform the decoded result."""

        return sample

    def on_run_end(self, context: DiffusionRunContext) -> None:
        """Release extension-owned resources after a successful run."""

    def on_run_error(
        self,
        context: DiffusionRunContext,
        error: BaseException,
    ) -> None:
        """Release extension-owned resources after a failed run."""


__all__ = ["DiffusionExtension", "DiffusionRunContext"]

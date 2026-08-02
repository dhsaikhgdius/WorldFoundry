"""Typed contracts shared by canonical diffusion models and runners."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence, runtime_checkable

import torch
from torch import Tensor


def _frozen_mapping(value: Mapping | None) -> Mapping:
    return MappingProxyType(dict(value or {}))


def _string_tuple(value: str | Sequence[str], *, field_name: str) -> tuple[str, ...]:
    items = (value,) if isinstance(value, str) else tuple(str(item) for item in value)
    if not items:
        raise ValueError(f"{field_name} cannot be empty")
    return items


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Model-independent settings for one denoising run."""

    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    seed: int = 0
    scheduler_options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if self.guidance_scale < 0:
            raise ValueError("guidance_scale must be non-negative")
        object.__setattr__(
            self,
            "scheduler_options",
            _frozen_mapping(self.scheduler_options),
        )


@dataclass(frozen=True, slots=True)
class DiffusionRequest:
    """Normalized request consumed by the native runner.

    Model-specific conditions such as images, camera trajectories, actions, or
    memory references live in ``inputs``.  They remain explicit values rather
    than process-wide mutable configuration.
    """

    prompt: str | Sequence[str]
    negative_prompt: str | Sequence[str] | None = None
    height: int = 512
    width: int = 512
    num_frames: int = 1
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    inputs: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        prompts = _string_tuple(self.prompt, field_name="prompt")
        if self.height <= 0 or self.width <= 0:
            raise ValueError("height and width must be positive")
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if self.negative_prompt is not None:
            negative = _string_tuple(
                self.negative_prompt,
                field_name="negative_prompt",
            )
            if len(negative) not in (1, len(prompts)):
                raise ValueError("negative_prompt must contain one item or match the prompt batch")
        object.__setattr__(self, "inputs", _frozen_mapping(self.inputs))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    @property
    def prompts(self) -> tuple[str, ...]:
        return _string_tuple(self.prompt, field_name="prompt")

    @property
    def negative_prompts(self) -> tuple[str, ...] | None:
        if self.negative_prompt is None:
            return None
        values = _string_tuple(self.negative_prompt, field_name="negative_prompt")
        if len(values) == 1 and len(self.prompts) > 1:
            return values * len(self.prompts)
        return values

    @property
    def batch_size(self) -> int:
        return len(self.prompts)


@dataclass(frozen=True, slots=True)
class Conditioning:
    """Positive, negative, and branch-independent denoiser conditions."""

    positive: Mapping[str, object]
    negative: Mapping[str, object] = field(default_factory=dict)
    shared: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "positive", _frozen_mapping(self.positive))
        object.__setattr__(self, "negative", _frozen_mapping(self.negative))
        object.__setattr__(self, "shared", _frozen_mapping(self.shared))


@dataclass(frozen=True, slots=True)
class LatentInitialization:
    """Initial noise plus run-local conditions produced during initialization.

    Image/video encoders often belong to the latent codec rather than the text
    conditioner. Returning their results here keeps that state explicit and
    lets the framework merge it into the immutable conditioning object before
    the first denoiser call.
    """

    latents: Tensor
    conditioning: Mapping[str, object] = field(default_factory=dict)
    artifacts: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.latents, Tensor):
            raise TypeError("LatentInitialization.latents must be a tensor")
        object.__setattr__(self, "conditioning", _frozen_mapping(self.conditioning))
        object.__setattr__(self, "artifacts", _frozen_mapping(self.artifacts))


@dataclass(frozen=True, slots=True)
class SchedulerStep:
    """One immutable point in a scheduler-owned denoising trajectory."""

    index: int
    timestep: Tensor
    next_timestep: Tensor


@dataclass(frozen=True, slots=True)
class DenoiserInput:
    """Complete input for one conditional denoiser evaluation."""

    latents: Tensor
    timestep: Tensor
    next_timestep: Tensor
    conditioning: Mapping[str, object]
    step_index: int
    total_steps: int
    branch: str = "positive"

    def with_updates(self, **changes: object) -> "DenoiserInput":
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class DenoiserOutput:
    """Noise, velocity, or flow prediction returned by a denoiser."""

    sample: Tensor
    extras: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extras", _frozen_mapping(self.extras))

    def with_updates(self, **changes: object) -> "DenoiserOutput":
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class DiffusionOutput:
    """Decoded output plus final latent and reproducibility metadata."""

    sample: Tensor
    latents: Tensor
    artifacts: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", _frozen_mapping(self.artifacts))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    def with_metadata(self, **values: object) -> "DiffusionOutput":
        metadata = dict(self.metadata)
        metadata.update(values)
        return replace(self, metadata=metadata)


@runtime_checkable
class Denoiser(Protocol):
    """A trainable network that predicts one denoising update."""

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        """Evaluate one positive or negative conditioning branch."""


@runtime_checkable
class ConditionEncoder(Protocol):
    """Encode prompts and model-specific inputs into denoiser conditions."""

    def encode(
        self,
        request: DiffusionRequest,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Conditioning:
        """Return conditions for one request."""


@runtime_checkable
class LatentInitializer(Protocol):
    """Create initial noise or image/video-conditioned latents."""

    def initialize(
        self,
        request: DiffusionRequest,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | LatentInitialization:
        """Return the initial latent tensor."""


@runtime_checkable
class EncodedLatentInitializer(Protocol):
    """Initializer that consumes a recipe-bound shared latent encoder."""

    def initialize_with_encoder(
        self,
        request: DiffusionRequest,
        *,
        latent_encoder: "LatentEncoder",
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | LatentInitialization:
        """Encode request media and create the initial latent trajectory."""


@runtime_checkable
class LatentEncoder(Protocol):
    """Encode pixels into a model's latent representation."""

    def encode(self, images: Tensor) -> Tensor:
        """Encode a normalized image batch."""


@runtime_checkable
class LatentDecoder(Protocol):
    """Decode final latents into pixels or another generated representation."""

    def decode(self, latents: Tensor, request: DiffusionRequest) -> Tensor:
        """Decode a completed latent trajectory."""


@runtime_checkable
class DiffusionScheduler(Protocol):
    """Own the immutable denoising schedule and numerical update rule."""

    def schedule(
        self,
        sampling: SamplingConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Sequence[SchedulerStep]:
        """Build a schedule for one run without mutating global state."""

    def scale_model_input(self, latents: Tensor, step: SchedulerStep) -> Tensor:
        """Scale latents before a denoiser evaluation."""

    def step(
        self,
        model_output: Tensor,
        step: SchedulerStep,
        latents: Tensor,
        *,
        generator: torch.Generator,
    ) -> Tensor:
        """Advance the latent trajectory by one step."""


@runtime_checkable
class FinalDenoiseScheduler(Protocol):
    """Optional scheduler capability for a terminal clean prediction.

    Some native samplers advance through every interval in their numerical
    schedule and then evaluate the denoiser once more at the last non-zero
    noise level.  Keeping that request on the scheduler lets the shared runner
    implement the lifecycle without a model-name branch.
    """

    def final_denoise_step(self) -> SchedulerStep | None:
        """Return the terminal prediction point, or ``None`` when unused."""


@dataclass(frozen=True, slots=True)
class ModalityState:
    """One modality's latent, immutable conditioning mask, and positions."""

    latent: Tensor
    denoise_mask: Tensor
    positions: Tensor
    clean_latent: Tensor
    attention_mask: Tensor | None = None

    def with_updates(self, **changes: object) -> "ModalityState":
        return replace(self, **changes)

    def clone(self) -> "ModalityState":
        return ModalityState(
            latent=self.latent.clone(),
            denoise_mask=self.denoise_mask.clone(),
            positions=self.positions.clone(),
            clean_latent=self.clean_latent.clone(),
            attention_mask=self.attention_mask.clone() if self.attention_mask is not None else None,
        )


@dataclass(frozen=True, slots=True)
class MultiModalDenoiserInput:
    """Joint state passed to a denoiser that couples several modalities."""

    modalities: Mapping[str, ModalityState]
    timestep: Tensor
    conditioning: Mapping[str, object]
    step_index: int
    total_steps: int
    branch: str = "positive"

    def __post_init__(self) -> None:
        object.__setattr__(self, "modalities", _frozen_mapping(self.modalities))
        object.__setattr__(self, "conditioning", _frozen_mapping(self.conditioning))


@dataclass(frozen=True, slots=True)
class MultiModalDenoiserOutput:
    """Denoised predictions keyed by modality name."""

    samples: Mapping[str, Tensor]
    extras: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", _frozen_mapping(self.samples))
        object.__setattr__(self, "extras", _frozen_mapping(self.extras))


@runtime_checkable
class LatentProcessor(Protocol):
    """Transform completed modality states between framework-owned stages."""

    def process(
        self,
        states: Mapping[str, ModalityState],
        request: DiffusionRequest,
    ) -> Mapping[str, Tensor]:
        """Return dense modality latents used to initialize the next stage."""


__all__ = [
    "ConditionEncoder",
    "Conditioning",
    "Denoiser",
    "DenoiserInput",
    "DenoiserOutput",
    "DiffusionOutput",
    "DiffusionRequest",
    "DiffusionScheduler",
    "EncodedLatentInitializer",
    "FinalDenoiseScheduler",
    "LatentDecoder",
    "LatentEncoder",
    "LatentInitialization",
    "LatentInitializer",
    "LatentProcessor",
    "ModalityState",
    "MultiModalDenoiserInput",
    "MultiModalDenoiserOutput",
    "SamplingConfig",
    "SchedulerStep",
]

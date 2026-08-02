"""Shared forward-data/backward-simulation generator rollout for SenseFlow."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import SenseFlowConfig
from .contracts import SenseFlowPredictionAdapter, SenseFlowTrainingBatch


def _levels(reference: Tensor, value: float) -> Tensor:
    return torch.full(
        (int(reference.shape[0]),),
        float(value),
        device=reference.device,
        dtype=torch.float32,
    )


def _randn_like(reference: Tensor, *, generator: torch.Generator) -> Tensor:
    return torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


def _clean_prediction(
    student: SenseFlowPredictionAdapter,
    sample: Tensor,
    sigmas: Tensor,
    batch: SenseFlowTrainingBatch,
    *,
    training: bool,
) -> Tensor:
    prediction = student.predict_clean(
        sample,
        sigmas,
        sample_ids=batch.sample_ids,
        conditioning=batch.conditioning,
        training=training,
    )
    if not isinstance(prediction, Tensor) or prediction.shape != sample.shape:
        raise ValueError("SenseFlow student predict_clean must preserve the latent shape")
    return prediction


@dataclass(frozen=True, slots=True)
class SenseFlowAnchorRollout:
    anchor_sample: Tensor
    generated_clean: Tensor
    anchor_sigmas: Tensor
    anchor_index: int
    anchor_timestep: int
    backward_simulation: bool


def simulate_senseflow_anchor(
    student: SenseFlowPredictionAdapter,
    batch: SenseFlowTrainingBatch,
    config: SenseFlowConfig,
    *,
    generator: torch.Generator,
    training: bool,
    anchor_index: int | None = None,
    backward_simulation: bool | None = None,
) -> SenseFlowAnchorRollout:
    """Sample one batch-shared anchor, matching Algorithm 1 for any local batch size."""

    if not isinstance(student, SenseFlowPredictionAdapter):
        raise TypeError("student must implement SenseFlowPredictionAdapter")
    if not isinstance(batch, SenseFlowTrainingBatch):
        raise TypeError("batch must be SenseFlowTrainingBatch")
    if not isinstance(config, SenseFlowConfig):
        raise TypeError("config must be SenseFlowConfig")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be torch.Generator")
    reference = batch.real_latents
    if not isinstance(reference, Tensor):
        raise TypeError("SenseFlow real_latents must be a torch.Tensor")
    schedule = config.schedule
    if anchor_index is None:
        selected = int(
            torch.randint(
                0,
                len(schedule.sigmas),
                (),
                device=reference.device,
                generator=generator,
            ).item()
        )
    else:
        if isinstance(anchor_index, bool) or not isinstance(anchor_index, int):
            raise TypeError("anchor_index must be an integer or None")
        selected = anchor_index
    if not 0 <= selected < len(schedule.sigmas):
        raise ValueError("anchor_index falls outside the SenseFlow schedule")
    if backward_simulation is None:
        use_backward = bool(
            torch.rand((), device=reference.device, generator=generator).item()
            < config.backward_simulation_probability
        )
    elif isinstance(backward_simulation, bool):
        use_backward = backward_simulation
    else:
        raise TypeError("backward_simulation must be bool or None")

    if use_backward:
        clean_source = _randn_like(reference, generator=generator)
        for index in range(selected):
            with torch.no_grad():
                noisy = student.add_noise(
                    clean_source,
                    _randn_like(clean_source, generator=generator),
                    _levels(clean_source, schedule.sigmas[index]),
                )
                if not isinstance(noisy, Tensor) or noisy.shape != reference.shape:
                    raise ValueError("SenseFlow student add_noise must preserve the latent shape")
                clean = _clean_prediction(
                    student,
                    noisy,
                    _levels(noisy, schedule.sigmas[index]),
                    batch,
                    training=False,
                )
                clean_source = clean
    else:
        clean_source = reference

    current = student.add_noise(
        clean_source,
        _randn_like(clean_source, generator=generator),
        _levels(clean_source, schedule.sigmas[selected]),
    )
    if not isinstance(current, Tensor) or current.shape != reference.shape:
        raise ValueError("SenseFlow student add_noise must preserve the latent shape")

    anchor_sigmas = _levels(current, schedule.sigmas[selected])
    if training:
        generated = _clean_prediction(
            student,
            current,
            anchor_sigmas,
            batch,
            training=True,
        )
    else:
        with torch.no_grad():
            generated = _clean_prediction(
                student,
                current,
                anchor_sigmas,
                batch,
                training=False,
            )
    return SenseFlowAnchorRollout(
        anchor_sample=current,
        generated_clean=generated,
        anchor_sigmas=anchor_sigmas,
        anchor_index=selected,
        anchor_timestep=schedule.timesteps[selected],
        backward_simulation=use_backward,
    )


__all__ = ["SenseFlowAnchorRollout", "simulate_senseflow_anchor"]

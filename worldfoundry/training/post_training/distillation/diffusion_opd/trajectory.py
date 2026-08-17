"""Student on-policy rollout and exact teacher/student replay."""

from __future__ import annotations

import torch

from ...rl.rollout_strategies.transition import FlowTransitionStrategy
from ...rl.transitions.flow_sde import flow_ode_step
from ...shared.contracts import FlowPredictionAdapter
from .contracts import (
    DiffusionOPDReplayResult,
    DiffusionOPDRolloutBatch,
    DiffusionOPDTrajectory,
)


def _sigma_at(sigmas: object, index: int, batch_size: int) -> object:
    return sigmas[index].expand(batch_size)


class DiffusionOPDTrajectorySampler:
    """Roll out the current student with SDE only at supervised transitions."""

    def __init__(
        self,
        student: FlowPredictionAdapter,
        *,
        transition_strategy: FlowTransitionStrategy,
        sigmas: tuple[float, ...],
        step_indices: tuple[int, ...],
        trajectory_dtype: object,
    ) -> None:
        if not isinstance(student, FlowPredictionAdapter):
            raise TypeError("DiffusionOPD student must implement FlowPredictionAdapter")
        if not isinstance(transition_strategy, FlowTransitionStrategy):
            raise TypeError("transition_strategy must implement FlowTransitionStrategy")
        self.student = student
        self.module = student.module
        self.transition_strategy = transition_strategy
        self.sigmas = tuple(sigmas)
        self.step_indices = tuple(step_indices)
        self.trajectory_dtype = trajectory_dtype

    def sample(
        self,
        batch: DiffusionOPDRolloutBatch,
        *,
        generator: object | None = None,
    ) -> DiffusionOPDTrajectory:
        if not isinstance(batch, DiffusionOPDRolloutBatch):
            raise TypeError("DiffusionOPD sampler requires DiffusionOPDRolloutBatch")
        schedule = batch.sigmas.to(
            device=batch.initial_latents.device,
            dtype=torch.float32,
        )
        expected = torch.tensor(self.sigmas, device=schedule.device, dtype=torch.float32)
        if schedule.shape != expected.shape or not torch.equal(schedule, expected):
            raise ValueError("DiffusionOPD batch schedule differs from the recipe")
        current = batch.initial_latents.to(dtype=self.trajectory_dtype)
        states = [current]
        scales: list[object] = []
        selected = set(self.step_indices)
        for index in range(len(self.sigmas) - 1):
            sigma = _sigma_at(schedule, index, batch.batch_size)
            sigma_next = _sigma_at(schedule, index + 1, batch.batch_size)
            with torch.no_grad():
                velocity = self.student.predict_velocity(
                    current,
                    sigma,
                    sample_ids=batch.sample_ids,
                    conditioning=batch.conditioning,
                    training=False,
                )
                if index in selected:
                    transition = self.transition_strategy.step(
                        velocity,
                        current,
                        sigma,
                        sigma_next,
                        generator=generator,
                        trajectory_dtype=self.trajectory_dtype,
                    )
                    current = transition.next_sample
                    scales.append(transition.scale.detach())
                else:
                    current = flow_ode_step(
                        velocity,
                        current,
                        sigma,
                        sigma_next,
                    ).to(dtype=self.trajectory_dtype)
            states.append(current.detach())
        return DiffusionOPDTrajectory(
            sample_ids=batch.sample_ids,
            domain=batch.domain,
            latents=torch.stack(states, dim=1),
            sigmas=schedule,
            step_indices=self.step_indices,
            transition_scales=torch.stack(scales, dim=1),
            conditioning=batch.conditioning,
        )


class NativeDiffusionOPDTrajectoryReplay:
    """Recompute transition means on the student's fixed rollout states."""

    def __init__(
        self,
        prediction: FlowPredictionAdapter,
        *,
        transition_strategy: FlowTransitionStrategy,
    ) -> None:
        if not isinstance(prediction, FlowPredictionAdapter):
            raise TypeError("DiffusionOPD replay requires FlowPredictionAdapter")
        if not isinstance(transition_strategy, FlowTransitionStrategy):
            raise TypeError("transition_strategy must implement FlowTransitionStrategy")
        self.prediction = prediction
        self.module = prediction.module
        self.transition_strategy = transition_strategy

    def replay(
        self,
        trajectory: DiffusionOPDTrajectory,
        *,
        training: bool,
    ) -> DiffusionOPDReplayResult:
        if not isinstance(trajectory, DiffusionOPDTrajectory):
            raise TypeError("DiffusionOPD replay requires DiffusionOPDTrajectory")
        means: list[object] = []
        scales: list[object] = []
        for index in trajectory.step_indices:
            current = trajectory.latents[:, index]
            observed_next = trajectory.latents[:, index + 1]
            sigma = _sigma_at(trajectory.sigmas, index, trajectory.batch_size)
            sigma_next = _sigma_at(
                trajectory.sigmas,
                index + 1,
                trajectory.batch_size,
            )
            velocity = self.prediction.predict_velocity(
                current,
                sigma,
                sample_ids=trajectory.sample_ids,
                conditioning=trajectory.conditioning,
                training=training,
            )
            transition = self.transition_strategy.step(
                velocity,
                current,
                sigma,
                sigma_next,
                next_sample=observed_next,
                trajectory_dtype=trajectory.latents.dtype,
            )
            means.append(transition.mean)
            scales.append(transition.scale)
        return DiffusionOPDReplayResult(
            transition_means=torch.stack(means, dim=1),
            transition_scales=torch.stack(scales, dim=1),
        )


__all__ = [
    "DiffusionOPDTrajectorySampler",
    "NativeDiffusionOPDTrajectoryReplay",
]

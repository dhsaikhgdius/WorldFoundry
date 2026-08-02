"""Official AnyFlow on-policy rollout schedules."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import AnyFlowBidirectionalOnPolicyConfig, AnyFlowOnPolicyConfig
from .contracts import (
    AnyFlowBidirectionalAdapter,
    AnyFlowFARAdapter,
    AnyFlowTrainingBatch,
)
from .math import flowmap_inference_schedule, flowmap_step
from .synchronization import AnyFlowDecisionRNG


@dataclass(frozen=True, slots=True)
class AnyFlowRolloutChoice:
    step_count: int
    gradient_interval: int

    def __post_init__(self) -> None:
        if isinstance(self.step_count, bool) or int(self.step_count) <= 0:
            raise ValueError("step_count must be positive")
        if isinstance(self.gradient_interval, bool) or not 0 <= int(self.gradient_interval) < int(self.step_count):
            raise ValueError("gradient_interval must select one rollout interval")


def sample_rollout_choice(
    config: AnyFlowOnPolicyConfig | AnyFlowBidirectionalOnPolicyConfig,
    decisions: AnyFlowDecisionRNG,
    *,
    reference: Tensor,
) -> AnyFlowRolloutChoice:
    """Sample step budget and gradient interval identically on every rank."""

    if not isinstance(
        config,
        (AnyFlowOnPolicyConfig, AnyFlowBidirectionalOnPolicyConfig),
    ):
        raise TypeError("config must be an AnyFlow on-policy config")
    if not isinstance(decisions, AnyFlowDecisionRNG):
        raise TypeError("decisions must be AnyFlowDecisionRNG")
    step_count = decisions.choice(config.inference_steps, reference=reference)
    gradient_interval = decisions.randrange(step_count, reference=reference)
    return AnyFlowRolloutChoice(
        step_count=step_count,
        gradient_interval=gradient_interval,
    )


def _frame_times(value: Tensor, reference: Tensor, scale: int) -> Tensor:
    batch, frames = int(reference.shape[0]), int(reference.shape[2])
    return value.to(device=reference.device, dtype=torch.float32).reshape(1, 1).expand(
        batch,
        frames,
    ) * float(scale)


def _selected_rollout_transitions(
    schedule: Tensor,
    choice: AnyFlowRolloutChoice,
) -> tuple[tuple[Tensor, Tensor], ...]:
    """Compress a schedule around the sampled interval like AnyFlow upstream.

    The released pipelines do not execute every inference interval during
    on-policy training.  They evaluate at most three arbitrary-span FlowMap
    transitions: the schedule start to the selected interval, the selected
    interval itself, and its destination to zero.  Empty endpoint transitions
    are omitted.
    """

    selected = int(choice.gradient_interval)
    transitions: list[tuple[Tensor, Tensor]] = []
    if selected > 0:
        transitions.append((schedule[0], schedule[selected]))
    transitions.append((schedule[selected], schedule[selected + 1]))
    if selected + 1 < int(choice.step_count):
        transitions.append((schedule[selected + 1], schedule[-1]))
    return tuple(transitions)


def anyflow_rollout(
    student: AnyFlowFARAdapter,
    batch: AnyFlowTrainingBatch,
    initial_noise: Tensor,
    choice: AnyFlowRolloutChoice,
    config: AnyFlowOnPolicyConfig,
    *,
    differentiable: bool,
) -> Tensor:
    """Run the released chunk-autoregressive, compressed on-policy rollout."""

    if not isinstance(student, AnyFlowFARAdapter):
        raise TypeError("student must implement AnyFlowFARAdapter")
    if not isinstance(batch, AnyFlowTrainingBatch):
        raise TypeError("batch must be AnyFlowTrainingBatch")
    if not isinstance(choice, AnyFlowRolloutChoice):
        raise TypeError("choice must be AnyFlowRolloutChoice")
    if not isinstance(config, AnyFlowOnPolicyConfig):
        raise TypeError("config must be AnyFlowOnPolicyConfig")
    if not isinstance(initial_noise, Tensor) or initial_noise.shape != batch.clean_latents.shape:
        raise ValueError("initial_noise must match AnyFlow clean_latents")
    if int(initial_noise.shape[2]) != config.far.partition.frame_count:
        raise ValueError("rollout latent frames must equal the configured FAR partition")

    schedule = flowmap_inference_schedule(
        choice.step_count,
        shift=config.flow_map.timestep_shift,
        device=initial_noise.device,
    )
    transitions = _selected_rollout_transitions(schedule, choice)
    partition = config.far.partition
    rollout_state = student.create_rollout_state(
        partition=partition,
        reference=initial_noise,
    )
    completed: list[Tensor] = []
    for chunk_index, (start, stop) in enumerate(partition.spans()):
        current = initial_noise[:, :, start:stop]
        for current_time, destination_time in transitions:
            model_t = _frame_times(
                current_time,
                current,
                config.flow_map.num_train_timesteps,
            )
            model_r = _frame_times(
                destination_time,
                current,
                config.flow_map.num_train_timesteps,
            )
            if differentiable:
                velocity = student.rollout_velocity(
                    current,
                    model_t,
                    model_r,
                    partition=partition,
                    chunk_index=chunk_index,
                    rollout_state=rollout_state,
                    sample_ids=batch.sample_ids,
                    conditioning=batch.conditioning,
                    training=True,
                )
                if not isinstance(velocity, Tensor) or velocity.shape != current.shape:
                    raise ValueError("AnyFlow rollout velocity must preserve the latent shape")
                current = flowmap_step(
                    current,
                    velocity,
                    current_time.expand(int(current.shape[0])),
                    destination_time.expand(int(current.shape[0])),
                )
            else:
                with torch.no_grad():
                    velocity = student.rollout_velocity(
                        current,
                        model_t,
                        model_r,
                        partition=partition,
                        chunk_index=chunk_index,
                        rollout_state=rollout_state,
                        sample_ids=batch.sample_ids,
                        conditioning=batch.conditioning,
                        training=False,
                    )
                    if not isinstance(velocity, Tensor) or velocity.shape != current.shape:
                        raise ValueError("AnyFlow rollout velocity must preserve the latent shape")
                    current = flowmap_step(
                        current,
                        velocity,
                        current_time.expand(int(current.shape[0])),
                        destination_time.expand(int(current.shape[0])),
                    )
        completed.append(current)
        if chunk_index + 1 < partition.chunk_count:
            with torch.no_grad():
                student.commit_rollout_chunk(
                    torch.cat(completed, dim=2),
                    partition=partition,
                    chunk_index=chunk_index,
                    rollout_state=rollout_state,
                    sample_ids=batch.sample_ids,
                    conditioning=batch.conditioning,
                )
    output = torch.cat(completed, dim=2)
    if differentiable and not output.requires_grad:
        raise RuntimeError("AnyFlow differentiable rollout lost its model gradient")
    return output


def anyflow_bidirectional_rollout(
    student: AnyFlowBidirectionalAdapter,
    batch: AnyFlowTrainingBatch,
    initial_noise: Tensor,
    choice: AnyFlowRolloutChoice,
    config: AnyFlowBidirectionalOnPolicyConfig,
    *,
    differentiable: bool,
) -> Tensor:
    """Run the released full-video compressed on-policy rollout."""

    if not isinstance(student, AnyFlowBidirectionalAdapter):
        raise TypeError("student must implement AnyFlowBidirectionalAdapter")
    if not isinstance(batch, AnyFlowTrainingBatch):
        raise TypeError("batch must be AnyFlowTrainingBatch")
    if not isinstance(choice, AnyFlowRolloutChoice):
        raise TypeError("choice must be AnyFlowRolloutChoice")
    if not isinstance(config, AnyFlowBidirectionalOnPolicyConfig):
        raise TypeError("config must be AnyFlowBidirectionalOnPolicyConfig")
    if not isinstance(initial_noise, Tensor) or initial_noise.shape != (batch.clean_latents.shape):
        raise ValueError("initial_noise must match AnyFlow clean_latents")
    schedule = flowmap_inference_schedule(
        choice.step_count,
        shift=config.flow_map.timestep_shift,
        device=initial_noise.device,
    )
    transitions = _selected_rollout_transitions(schedule, choice)
    current = initial_noise
    for current_time, destination_time in transitions:
        model_t = _frame_times(
            current_time,
            current,
            config.flow_map.num_train_timesteps,
        )
        model_r = _frame_times(
            destination_time,
            current,
            config.flow_map.num_train_timesteps,
        )
        if differentiable:
            velocity = student.rollout_velocity(
                current,
                model_t,
                model_r,
                sample_ids=batch.sample_ids,
                conditioning=batch.conditioning,
                training=True,
            )
            if not isinstance(velocity, Tensor) or velocity.shape != current.shape:
                raise ValueError("AnyFlow rollout velocity must preserve the latent shape")
            current = flowmap_step(
                current,
                velocity,
                current_time.expand(int(current.shape[0])),
                destination_time.expand(int(current.shape[0])),
            )
        else:
            with torch.no_grad():
                velocity = student.rollout_velocity(
                    current,
                    model_t,
                    model_r,
                    sample_ids=batch.sample_ids,
                    conditioning=batch.conditioning,
                    training=False,
                )
                if not isinstance(velocity, Tensor) or velocity.shape != current.shape:
                    raise ValueError("AnyFlow rollout velocity must preserve the latent shape")
                current = flowmap_step(
                    current,
                    velocity,
                    current_time.expand(int(current.shape[0])),
                    destination_time.expand(int(current.shape[0])),
                )
    if differentiable and not current.requires_grad:
        raise RuntimeError("AnyFlow differentiable rollout lost its model gradient")
    return current


__all__ = [
    "AnyFlowRolloutChoice",
    "anyflow_bidirectional_rollout",
    "anyflow_rollout",
    "sample_rollout_choice",
]

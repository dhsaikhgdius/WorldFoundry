"""Native trajectory rollout and exact replay for rectified-flow policies."""

from __future__ import annotations

from collections.abc import Mapping

from ..shared.contracts import FlowPredictionAdapter
from .contracts import FlowReplayResult, FlowTrajectory, FlowTrajectoryReplayBatch
from .rollout_strategies.transition import (
    FlowTransitionStrategy,
    VariancePreservingFlowTransition,
    flow_transition_strategy_from_identity,
)
from .transitions.flow_sde import flow_ode_step


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("native trajectory rollout requires the 'train-core' extra") from error
    return torch


def _normalized_sigmas(sigmas: object, *, batch_size: int, device: object) -> object:
    torch = _require_torch()
    if not torch.is_tensor(sigmas):
        sigmas = torch.as_tensor(sigmas, dtype=torch.float32, device=device)
    else:
        sigmas = sigmas.to(device=device, dtype=torch.float32)
    if sigmas.ndim == 1:
        if sigmas.numel() < 2:
            raise ValueError("trajectory schedule needs at least two sigmas")
        differences = sigmas[1:] - sigmas[:-1]
    elif sigmas.ndim == 2 and int(sigmas.shape[0]) == batch_size:
        if int(sigmas.shape[1]) < 2:
            raise ValueError("trajectory schedule needs at least two sigmas")
        differences = sigmas[:, 1:] - sigmas[:, :-1]
    else:
        raise ValueError("sigmas must have shape [S+1] or [B,S+1]")
    if not bool(torch.isfinite(sigmas).all()):
        raise ValueError("trajectory sigmas must be finite")
    if not bool((sigmas >= 0).all() and (sigmas <= 1).all() and (differences < 0).all()):
        raise ValueError("trajectory sigmas must be in [0,1] and strictly descending")
    if not bool((sigmas[..., :-1] > 0).all()):
        raise ValueError("every transition source sigma must be positive")
    return sigmas


def _sigma_at(sigmas: object, index: int, batch_size: int) -> object:
    if sigmas.ndim == 1:
        return sigmas[index].expand(batch_size)
    return sigmas[:, index]


def _slice_batched_mapping(
    values: Mapping[str, object],
    *,
    start: int,
    end: int,
    batch_size: int,
) -> dict[str, object]:
    torch = _require_torch()
    result: dict[str, object] = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.shape[0]) == batch_size:
            result[str(key)] = value[start:end]
        else:
            result[str(key)] = value
    return result


class FlowTrajectorySampler:
    """Generate an auditable mixed ODE/SDE trajectory with a native policy."""

    def __init__(
        self,
        policy: FlowPredictionAdapter,
        *,
        eta: float | None = None,
        sigma_max: float | None = None,
        transition_strategy: FlowTransitionStrategy | None = None,
        trajectory_dtype: object | None = None,
        forward_batch_size: int | None = None,
    ) -> None:
        torch = _require_torch()
        if not isinstance(policy, FlowPredictionAdapter):
            raise TypeError("policy must implement FlowPredictionAdapter")
        if transition_strategy is None:
            transition_strategy = VariancePreservingFlowTransition(
                eta=1.0 if eta is None else eta,
                sigma_max=0.99 if sigma_max is None else sigma_max,
            )
        elif eta is not None or sigma_max is not None:
            raise ValueError("transition_strategy cannot be combined with eta or sigma_max")
        if not isinstance(transition_strategy, FlowTransitionStrategy):
            raise TypeError("transition_strategy must implement FlowTransitionStrategy")
        if trajectory_dtype is not None and trajectory_dtype not in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        }:
            raise ValueError("trajectory_dtype must be a floating torch dtype")
        if forward_batch_size is not None and (isinstance(forward_batch_size, bool) or int(forward_batch_size) <= 0):
            raise ValueError("forward_batch_size must be a positive integer")
        self.policy = policy
        self.module = policy.module
        self.transition_strategy = transition_strategy
        self.eta = transition_strategy.eta
        self.sigma_max = getattr(transition_strategy, "sigma_max", None)
        self.trajectory_dtype = trajectory_dtype
        self.forward_batch_size = None if forward_batch_size is None else int(forward_batch_size)

    def _predict_velocity(
        self,
        current: object,
        sigma: object,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
    ) -> object:
        torch = _require_torch()
        batch_size = int(current.shape[0])
        chunk_size = self.forward_batch_size or batch_size
        if chunk_size >= batch_size:
            return self.policy.predict_velocity(
                current,
                sigma,
                sample_ids=sample_ids,
                conditioning=conditioning,
                training=training,
            )
        predictions: list[object] = []
        for start in range(0, batch_size, chunk_size):
            end = min(batch_size, start + chunk_size)
            predictions.append(
                self.policy.predict_velocity(
                    current[start:end],
                    sigma[start:end],
                    sample_ids=sample_ids[start:end],
                    conditioning=_slice_batched_mapping(
                        conditioning,
                        start=start,
                        end=end,
                        batch_size=batch_size,
                    ),
                    training=training,
                )
            )
        return torch.cat(predictions, dim=0)

    def sample(
        self,
        initial_latents: object,
        sigmas: object,
        *,
        sample_ids: tuple[str, ...],
        group_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        policy_revision: str,
        sde_step_indices: tuple[int, ...] | None = None,
        generator: object | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> FlowTrajectory:
        torch = _require_torch()
        if not torch.is_tensor(initial_latents) or initial_latents.ndim < 2:
            raise TypeError("initial_latents must be a [B,...] torch.Tensor")
        batch_size = int(initial_latents.shape[0])
        if len(sample_ids) != batch_size or len(group_ids) != batch_size:
            raise ValueError("sample_ids/group_ids must match initial_latents")
        schedule = _normalized_sigmas(sigmas, batch_size=batch_size, device=initial_latents.device)
        transition_count = int(schedule.shape[-1]) - 1
        if sde_step_indices is None:
            selected = tuple(range(transition_count))
        else:
            selected = tuple(int(index) for index in sde_step_indices)
        if not selected or selected != tuple(sorted(set(selected))):
            raise ValueError("sde_step_indices must be non-empty, sorted, and unique")
        if selected[0] < 0 or selected[-1] >= transition_count:
            raise ValueError("sde_step_indices fall outside the schedule")
        selected_set = set(selected)
        storage_dtype = self.trajectory_dtype or initial_latents.dtype
        current = initial_latents.to(dtype=storage_dtype)
        states = [current]
        log_probs: list[object] = []
        means: list[object] = []
        scales: list[object] = []
        for index in range(transition_count):
            sigma = _sigma_at(schedule, index, batch_size)
            sigma_next = _sigma_at(schedule, index + 1, batch_size)
            with torch.no_grad():
                velocity = self._predict_velocity(
                    current,
                    sigma,
                    sample_ids=sample_ids,
                    conditioning=conditioning,
                    training=False,
                )
            if index in selected_set:
                transition = self.transition_strategy.step(
                    velocity,
                    current,
                    sigma,
                    sigma_next,
                    generator=generator,
                    trajectory_dtype=storage_dtype,
                )
                if transition.log_prob is None:
                    raise RuntimeError("positive-eta SDE transition did not produce log-prob")
                current = transition.next_sample
                log_probs.append(transition.log_prob.detach())
                means.append(transition.mean.detach())
                scales.append(transition.scale.detach())
            else:
                current = flow_ode_step(velocity, current, sigma, sigma_next).to(dtype=storage_dtype)
            states.append(current.detach())

        return FlowTrajectory(
            sample_ids=sample_ids,
            group_ids=group_ids,
            policy_revision=policy_revision,
            latents=torch.stack(states, dim=1),
            sigmas=schedule,
            step_indices=selected,
            old_log_probs=torch.stack(log_probs, dim=1),
            transition_means=torch.stack(means, dim=1),
            transition_scales=torch.stack(scales, dim=1),
            conditioning=conditioning,
            transition_identity=self.transition_strategy.identity,
            metadata={} if metadata is None else metadata,
        )


class NativeFlowTrajectoryReplay:
    """Recompute selected transition likelihoods under the active policy."""

    def __init__(self, policy: FlowPredictionAdapter) -> None:
        if not isinstance(policy, FlowPredictionAdapter):
            raise TypeError("policy must implement FlowPredictionAdapter")
        self.policy = policy
        self.module = policy.module

    def replay(
        self,
        trajectory: FlowTrajectory | FlowTrajectoryReplayBatch,
        *,
        training: bool,
    ) -> FlowReplayResult:
        torch = _require_torch()
        if not isinstance(trajectory, (FlowTrajectory, FlowTrajectoryReplayBatch)):
            raise TypeError("trajectory must be FlowTrajectory or FlowTrajectoryReplayBatch")
        transition_strategy = flow_transition_strategy_from_identity(trajectory.transition_identity)
        log_probs: list[object] = []
        means: list[object] = []
        scales: list[object] = []
        velocities: list[object] = []
        for index in trajectory.step_indices:
            current = trajectory.latents[:, index]
            observed_next = trajectory.latents[:, index + 1]
            sigma = _sigma_at(trajectory.sigmas, index, trajectory.batch_size)
            sigma_next = _sigma_at(trajectory.sigmas, index + 1, trajectory.batch_size)
            velocity = self.policy.predict_velocity(
                current,
                sigma,
                sample_ids=trajectory.sample_ids,
                conditioning=trajectory.conditioning,
                training=training,
            )
            transition = transition_strategy.step(
                velocity,
                current,
                sigma,
                sigma_next,
                next_sample=observed_next,
                trajectory_dtype=trajectory.latents.dtype,
            )
            if transition.log_prob is None:
                raise RuntimeError("stored stochastic trajectory replay produced no log-prob")
            log_probs.append(transition.log_prob)
            means.append(transition.mean)
            scales.append(transition.scale)
            velocities.append(velocity)

        replay_scales = torch.stack(scales, dim=1)
        saved_scales = trajectory.transition_scales.to(device=replay_scales.device, dtype=replay_scales.dtype)
        if replay_scales.shape != saved_scales.shape or not torch.equal(replay_scales.detach(), saved_scales):
            raise ValueError("replayed transition scales differ from rollout")
        sqrt_dt = torch.stack(
            [
                torch.sqrt(
                    _sigma_at(trajectory.sigmas, index, trajectory.batch_size)
                    - _sigma_at(
                        trajectory.sigmas,
                        index + 1,
                        trajectory.batch_size,
                    )
                )
                for index in trajectory.step_indices
            ],
            dim=1,
        ).to(device=replay_scales.device, dtype=replay_scales.dtype)
        sqrt_dt_broadcast = sqrt_dt.reshape((*sqrt_dt.shape, *((1,) * (replay_scales.ndim - 2))))
        std_dev_t = replay_scales / sqrt_dt_broadcast
        return FlowReplayResult(
            log_probs=torch.stack(log_probs, dim=1),
            transition_means=torch.stack(means, dim=1),
            transition_scales=replay_scales,
            velocities=torch.stack(velocities, dim=1),
            std_dev_t=std_dev_t,
            sqrt_dt=sqrt_dt,
        )


def slice_flow_trajectory(
    trajectory: FlowTrajectory,
    start: int,
    end: int,
) -> FlowTrajectoryReplayBatch:
    """Select a contiguous policy microbatch without changing its semantics."""

    if not isinstance(trajectory, FlowTrajectory):
        raise TypeError("trajectory must be FlowTrajectory")
    if isinstance(start, bool) or isinstance(end, bool) or not 0 <= int(start) < int(end) <= trajectory.batch_size:
        raise ValueError("trajectory slice must be a non-empty in-range interval")
    begin = int(start)
    stop = int(end)
    conditioning = _slice_batched_mapping(
        trajectory.conditioning,
        start=begin,
        end=stop,
        batch_size=trajectory.batch_size,
    )
    metadata = dict(trajectory.metadata)
    metadata["policy_microbatch"] = {"start": begin, "end": stop}
    return FlowTrajectoryReplayBatch(
        source=trajectory,
        start=begin,
        end=stop,
        conditioning=conditioning,
        metadata=metadata,
    )


__all__ = [
    "FlowTrajectorySampler",
    "NativeFlowTrajectoryReplay",
    "slice_flow_trajectory",
]

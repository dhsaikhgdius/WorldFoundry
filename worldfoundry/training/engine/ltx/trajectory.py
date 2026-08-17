"""Audio-conditioned video-policy rollout and replay for LTX-2.x."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from worldfoundry.training.post_training.rl.contracts import (
    FlowReplayResult,
    FlowTrajectory,
    FlowTrajectoryReplayBatch,
)
from worldfoundry.training.post_training.rl.rollout_strategies.transition import (
    FlowTransitionStrategy,
    flow_transition_strategy_from_identity,
)
from worldfoundry.training.post_training.rl.transitions.flow_sde import flow_ode_step

LTX_AUDIO_TRAJECTORY = "ltx_audio_trajectory"
LTX_AUDIO_TRANSITION_MEANS = "ltx_audio_transition_means"
LTX_AUDIO_TRANSITION_SCALES = "ltx_audio_transition_scales"


@dataclass(frozen=True, slots=True)
class LTXFlowReplayResult(FlowReplayResult):
    """Video replay fields plus the LTX-2.3 audio transition distribution."""

    audio_transition_means: torch.Tensor | None = None
    audio_transition_scales: torch.Tensor | None = None

    def __post_init__(self) -> None:
        FlowReplayResult.__post_init__(self)
        if (self.audio_transition_means is None) != (self.audio_transition_scales is None):
            raise ValueError("LTX audio transition means and scales must be provided together")
        if self.audio_transition_means is not None:
            if self.audio_transition_means.shape[:2] != self.log_probs.shape:
                raise ValueError("LTX audio transition means must start with [B,K]")
            torch.broadcast_shapes(
                self.audio_transition_scales.shape,
                self.audio_transition_means.shape,
            )


def _sigma_at(sigmas: torch.Tensor, index: int, batch_size: int) -> torch.Tensor:
    if sigmas.ndim == 1:
        return sigmas[index].expand(batch_size)
    return sigmas[:, index]


def _slice_conditioning(
    values: Mapping[str, object],
    *,
    start: int,
    end: int,
    batch_size: int,
) -> dict[str, object]:
    return {
        key: value[start:end]
        if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.shape[0]) == batch_size
        else value
        for key, value in values.items()
    }


def _combine_modality_log_prob(
    video_log_prob: torch.Tensor,
    audio_log_prob: torch.Tensor,
    *,
    video: torch.Tensor,
    audio: torch.Tensor,
) -> torch.Tensor:
    """Match one concatenated AV policy by weighting each modality by its elements."""

    video_elements = int(video[0].numel())
    audio_elements = int(audio[0].numel())
    return (video_log_prob * video_elements + audio_log_prob * audio_elements) / (video_elements + audio_elements)


class LTXAudioConditionedTrajectorySampler:
    """Advance an LTX video policy with either ODE or policy-owned audio state."""

    def __init__(
        self,
        policy: object,
        *,
        transition_strategy: FlowTransitionStrategy,
        trajectory_dtype: torch.dtype,
        audio_joint_sde: bool,
        init_same_noise: bool,
        forward_batch_size: int | None = None,
    ) -> None:
        self.policy = policy
        self.module = policy.module
        self.transition_strategy = transition_strategy
        self.trajectory_dtype = trajectory_dtype
        self.audio_joint_sde = bool(audio_joint_sde)
        self.init_same_noise = bool(init_same_noise)
        self.forward_batch_size = forward_batch_size

    def _predict(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        sigma: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(video.shape[0])
        chunk_size = self.forward_batch_size or batch_size
        if chunk_size >= batch_size:
            return self.policy.predict_joint_velocity(
                video,
                audio,
                sigma,
                sample_ids=sample_ids,
                conditioning=conditioning,
                training=training,
            )
        video_predictions: list[torch.Tensor] = []
        audio_predictions: list[torch.Tensor] = []
        for start in range(0, batch_size, chunk_size):
            end = min(batch_size, start + chunk_size)
            video_prediction, audio_prediction = self.policy.predict_joint_velocity(
                video[start:end],
                audio[start:end],
                sigma[start:end],
                sample_ids=sample_ids[start:end],
                conditioning=_slice_conditioning(
                    conditioning,
                    start=start,
                    end=end,
                    batch_size=batch_size,
                ),
                training=training,
            )
            video_predictions.append(video_prediction)
            audio_predictions.append(audio_prediction)
        return torch.cat(video_predictions), torch.cat(audio_predictions)

    def sample(
        self,
        initial_latents: torch.Tensor,
        sigmas: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        group_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        policy_revision: str,
        sde_step_indices: tuple[int, ...] | None = None,
        generator: torch.Generator | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> FlowTrajectory:
        batch_size = int(initial_latents.shape[0])
        schedule = torch.as_tensor(sigmas, device=initial_latents.device, dtype=torch.float32)
        transition_count = int(schedule.shape[-1]) - 1
        selected = tuple(range(transition_count)) if sde_step_indices is None else tuple(sde_step_indices)
        selected_set = set(selected)

        video = initial_latents.to(dtype=self.trajectory_dtype)
        if self.init_same_noise:
            group_order = tuple(dict.fromkeys(group_ids))
            first_indices = tuple(group_ids.index(group_id) for group_id in group_order)
            group_audio = self.policy.initial_audio_latents(
                video[list(first_indices)],
                generator=generator,
                dtype=self.trajectory_dtype,
            )
            group_index = {group_id: index for index, group_id in enumerate(group_order)}
            audio = torch.stack([group_audio[group_index[group_id]] for group_id in group_ids])
        else:
            audio = self.policy.initial_audio_latents(
                video,
                generator=generator,
                dtype=self.trajectory_dtype,
            )
        video_states = [video]
        audio_states = [audio]
        old_log_probs: list[torch.Tensor] = []
        video_means: list[torch.Tensor] = []
        video_scales: list[torch.Tensor] = []
        audio_means: list[torch.Tensor] = []
        audio_scales: list[torch.Tensor] = []

        for index in range(transition_count):
            sigma = _sigma_at(schedule, index, batch_size)
            sigma_next = _sigma_at(schedule, index + 1, batch_size)
            with torch.no_grad():
                video_velocity, audio_velocity = self._predict(
                    video,
                    audio,
                    sigma,
                    sample_ids=sample_ids,
                    conditioning=conditioning,
                    training=False,
                )

            if index in selected_set:
                video_transition = self.transition_strategy.step(
                    video_velocity,
                    video,
                    sigma,
                    sigma_next,
                    generator=generator,
                    trajectory_dtype=self.trajectory_dtype,
                )
                video = video_transition.next_sample
                if video_transition.log_prob is None:
                    raise RuntimeError("LTX video SDE transition produced no log-probability")
                log_prob = video_transition.log_prob
                video_means.append(video_transition.mean.detach())
                video_scales.append(video_transition.scale.detach())
                if self.audio_joint_sde:
                    audio_transition = self.transition_strategy.step(
                        audio_velocity,
                        audio,
                        sigma,
                        sigma_next,
                        generator=generator,
                        trajectory_dtype=self.trajectory_dtype,
                    )
                    if audio_transition.log_prob is None:
                        raise RuntimeError("LTX joint audio SDE transition produced no log-probability")
                    audio = audio_transition.next_sample
                    audio_means.append(audio_transition.mean.detach())
                    audio_scales.append(audio_transition.scale.detach())
                    log_prob = _combine_modality_log_prob(
                        log_prob,
                        audio_transition.log_prob,
                        video=video,
                        audio=audio,
                    )
                else:
                    audio = flow_ode_step(
                        audio_velocity,
                        audio,
                        sigma,
                        sigma_next,
                    ).to(self.trajectory_dtype)
            else:
                video = flow_ode_step(video_velocity, video, sigma, sigma_next).to(self.trajectory_dtype)
                log_prob = None
                audio = flow_ode_step(audio_velocity, audio, sigma, sigma_next).to(self.trajectory_dtype)

            video_states.append(video.detach())
            audio_states.append(audio.detach())
            if log_prob is not None:
                old_log_probs.append(log_prob.detach())

        trajectory_conditioning = dict(conditioning)
        trajectory_conditioning[LTX_AUDIO_TRAJECTORY] = torch.stack(audio_states, dim=1)
        if self.audio_joint_sde:
            trajectory_conditioning[LTX_AUDIO_TRANSITION_MEANS] = torch.stack(
                audio_means,
                dim=1,
            )
            trajectory_conditioning[LTX_AUDIO_TRANSITION_SCALES] = torch.stack(
                audio_scales,
                dim=1,
            )
        return FlowTrajectory(
            sample_ids=sample_ids,
            group_ids=group_ids,
            policy_revision=policy_revision,
            latents=torch.stack(video_states, dim=1),
            sigmas=schedule,
            step_indices=selected,
            old_log_probs=torch.stack(old_log_probs, dim=1),
            transition_means=torch.stack(video_means, dim=1),
            transition_scales=torch.stack(video_scales, dim=1),
            conditioning=trajectory_conditioning,
            transition_identity=self.transition_strategy.identity,
            metadata={} if metadata is None else metadata,
        )


def build_ltx_ray_trajectory_sampler(
    policy: object,
    *,
    transition_strategy: FlowTransitionStrategy,
    trajectory_dtype: torch.dtype | None,
    forward_batch_size: int | None,
    audio_joint_sde: bool,
    init_same_noise: bool,
) -> LTXAudioConditionedTrajectorySampler:
    """Construct the actor-local LTX sampler used by Ray rollout workers."""

    if trajectory_dtype is None:
        raise ValueError("LTX Ray rollout requires an explicit trajectory dtype")
    return LTXAudioConditionedTrajectorySampler(
        policy,
        transition_strategy=transition_strategy,
        trajectory_dtype=trajectory_dtype,
        audio_joint_sde=audio_joint_sde,
        init_same_noise=init_same_noise,
        forward_batch_size=forward_batch_size,
    )


class LTXAudioConditionedTrajectoryReplay:
    """Replay video-only or joint AV likelihoods from the stored audio trajectory."""

    def __init__(self, policy: object, *, audio_joint_sde: bool) -> None:
        self.policy = policy
        self.module = policy.module
        self.audio_joint_sde = bool(audio_joint_sde)

    def replay(
        self,
        trajectory: FlowTrajectory | FlowTrajectoryReplayBatch,
        *,
        training: bool,
    ) -> FlowReplayResult:
        audio_trajectory = trajectory.conditioning.get(LTX_AUDIO_TRAJECTORY)
        if not isinstance(audio_trajectory, torch.Tensor):
            raise ValueError("LTX replay requires the stored audio trajectory")
        transition_strategy = flow_transition_strategy_from_identity(trajectory.transition_identity)
        log_probs: list[torch.Tensor] = []
        means: list[torch.Tensor] = []
        scales: list[torch.Tensor] = []
        velocities: list[torch.Tensor] = []
        sqrt_dts: list[torch.Tensor] = []
        audio_means: list[torch.Tensor] = []
        audio_scales: list[torch.Tensor] = []

        for index in trajectory.step_indices:
            video = trajectory.latents[:, index]
            next_video = trajectory.latents[:, index + 1]
            audio = audio_trajectory[:, index]
            next_audio = audio_trajectory[:, index + 1]
            sigma = _sigma_at(trajectory.sigmas, index, trajectory.batch_size)
            sigma_next = _sigma_at(trajectory.sigmas, index + 1, trajectory.batch_size)
            video_velocity, audio_velocity = self.policy.predict_joint_velocity(
                video,
                audio,
                sigma,
                sample_ids=trajectory.sample_ids,
                conditioning=trajectory.conditioning,
                training=training,
            )
            video_transition = transition_strategy.step(
                video_velocity,
                video,
                sigma,
                sigma_next,
                next_sample=next_video,
                trajectory_dtype=trajectory.latents.dtype,
            )
            if video_transition.log_prob is None:
                raise RuntimeError("LTX video replay produced no log-probability")

            log_prob = video_transition.log_prob
            if self.audio_joint_sde:
                audio_transition = transition_strategy.step(
                    audio_velocity,
                    audio,
                    sigma,
                    sigma_next,
                    next_sample=next_audio,
                    trajectory_dtype=trajectory.latents.dtype,
                )
                if audio_transition.log_prob is None:
                    raise RuntimeError("LTX joint audio replay produced no log-probability")
                audio_means.append(audio_transition.mean)
                audio_scales.append(audio_transition.scale)
                log_prob = _combine_modality_log_prob(
                    log_prob,
                    audio_transition.log_prob,
                    video=video,
                    audio=audio,
                )
            mean = video_transition.mean
            scale = video_transition.scale
            velocity = video_velocity

            log_probs.append(log_prob)
            means.append(mean)
            scales.append(scale)
            velocities.append(velocity)
            sqrt_dts.append(torch.sqrt(sigma - sigma_next))

        replay_scales = torch.stack(scales, dim=1)
        sqrt_dt = torch.stack(sqrt_dts, dim=1).to(replay_scales)
        sqrt_dt_broadcast = sqrt_dt.reshape(
            *sqrt_dt.shape,
            *((1,) * (replay_scales.ndim - 2)),
        )
        return LTXFlowReplayResult(
            log_probs=torch.stack(log_probs, dim=1),
            transition_means=torch.stack(means, dim=1),
            transition_scales=replay_scales,
            velocities=torch.stack(velocities, dim=1),
            std_dev_t=replay_scales / sqrt_dt_broadcast,
            sqrt_dt=sqrt_dt,
            audio_transition_means=(torch.stack(audio_means, dim=1) if self.audio_joint_sde else None),
            audio_transition_scales=(torch.stack(audio_scales, dim=1) if self.audio_joint_sde else None),
        )


__all__ = [
    "LTXAudioConditionedTrajectoryReplay",
    "LTXAudioConditionedTrajectorySampler",
    "LTXFlowReplayResult",
    "LTX_AUDIO_TRAJECTORY",
    "LTX_AUDIO_TRANSITION_MEANS",
    "LTX_AUDIO_TRANSITION_SCALES",
    "build_ltx_ray_trajectory_sampler",
]

"""Rewarded distribution matching over native causal student rollouts."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from worldfoundry.training.objectives.flow_matching import (
    flow_interpolate,
    flow_matching_denominator,
    flow_matching_mse,
    flow_velocity_target,
)

from ...shared.contracts import FlowPredictionAdapter
from ..dmd.objective import (
    DMDLossResult,
    DMDStudentSampler,
    FlowDMDLossAdapter,
    dmd_distribution_gradient,
    dmd_teacher_guidance,
    sample_dmd_score_sigmas,
)
from .config import RewardForcingConfig
from .contracts import (
    MotionQualityRewardAdapter,
    RewardForcingDecoderAdapter,
    RewardForcingTrainingBatch,
)
from .math import reward_forcing_multiplier, rewarded_dmd_proxy_loss


def _reward_batch(value: object) -> RewardForcingTrainingBatch:
    if not isinstance(value, RewardForcingTrainingBatch):
        raise TypeError("Reward-Forcing requires RewardForcingTrainingBatch")
    return value


def _prediction(
    adapter: FlowPredictionAdapter,
    noisy: Tensor,
    sigmas: Tensor,
    batch: RewardForcingTrainingBatch,
    *,
    conditioning: Mapping[str, object],
    branch: str,
) -> Tensor:
    clean = adapter.predict_clean(
        noisy,
        sigmas,
        sample_ids=batch.sample_ids,
        conditioning=conditioning,
        training=False,
        branch=branch,
    )
    if not isinstance(clean, Tensor) or clean.shape != noisy.shape:
        raise ValueError("Reward-Forcing score prediction must preserve latent shape")
    if not clean.is_floating_point():
        raise TypeError("Reward-Forcing score prediction must be floating point")
    return clean


class NativeRewardForcingLossAdapter:
    """Re-DMD generator weighting plus the ordinary fake-score flow loss."""

    def __init__(
        self,
        real_score: FlowPredictionAdapter,
        fake_score: FlowPredictionAdapter,
        student_sampler: DMDStudentSampler,
        reward_decoder: RewardForcingDecoderAdapter,
        motion_reward: MotionQualityRewardAdapter,
        config: RewardForcingConfig,
    ) -> None:
        if not isinstance(real_score, FlowPredictionAdapter):
            raise TypeError("real_score must implement FlowPredictionAdapter")
        if not isinstance(fake_score, FlowPredictionAdapter):
            raise TypeError("fake_score must implement FlowPredictionAdapter")
        if not isinstance(student_sampler, DMDStudentSampler):
            raise TypeError("student_sampler must implement DMDStudentSampler")
        if not isinstance(reward_decoder, RewardForcingDecoderAdapter):
            raise TypeError("reward_decoder must implement RewardForcingDecoderAdapter")
        if not isinstance(motion_reward, MotionQualityRewardAdapter):
            raise TypeError("motion_reward must implement MotionQualityRewardAdapter")
        if not isinstance(config, RewardForcingConfig):
            raise TypeError("config must be RewardForcingConfig")
        self.real_score = real_score
        self.fake_score = fake_score
        self.student_sampler = student_sampler
        self.reward_decoder = reward_decoder
        self.motion_reward = motion_reward
        self.config = config
        self.base_dmd = FlowDMDLossAdapter(
            None,
            real_score,
            fake_score,
            config.dmd_config,
            student_sampler=student_sampler,
        )
    def _validate_geometry(self, batch: RewardForcingTrainingBatch) -> Tensor:
        latents = batch.clean_latents
        if not isinstance(latents, Tensor) or not latents.is_floating_point():
            raise TypeError("Reward-Forcing clean latents must be a floating torch.Tensor")
        frame_dim = self.config.frame_dim % latents.ndim
        if frame_dim == 0:
            raise ValueError("resolved Reward-Forcing frame_dim cannot be the batch dimension")
        if int(latents.shape[frame_dim]) != self.config.training_frames:
            raise ValueError("Reward-Forcing latent frame count differs from training_frames")
        return latents

    def loss_denominator(
        self,
        batch: RewardForcingTrainingBatch,
        *,
        role: str,
    ) -> Tensor:
        resolved = _reward_batch(batch)
        self._validate_geometry(resolved)
        if role not in {"generator", "fake-score"}:
            raise ValueError(f"unsupported Reward-Forcing role: {role!r}")
        return flow_matching_denominator(
            resolved.clean_latents,
            loss_mask=resolved.loss_mask,
            sample_weights=resolved.sample_weights,
        )

    def generator_loss(
        self,
        batch: RewardForcingTrainingBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> DMDLossResult:
        resolved = _reward_batch(batch)
        reference = self._validate_geometry(resolved)
        generated = self.base_dmd.sample_student(
            resolved,
            generator=generator,
            training=True,
        )
        generated_clean = generated.clean_latents
        if not isinstance(generated_clean, Tensor) or generated_clean.shape != reference.shape:
            raise ValueError("Reward-Forcing student rollout must preserve latent shape")

        with torch.no_grad():
            videos = self.reward_decoder.decode_reward_videos(
                generated_clean.detach(),
                sample_ids=resolved.sample_ids,
                conditioning=resolved.conditioning,
            )
            if not isinstance(videos, Tensor) or videos.ndim != 5:
                raise TypeError("Reward-Forcing decoder must return [B,C,T,H,W] video")
            if tuple(videos.shape[:2]) != (resolved.batch_size, 3):
                raise ValueError("Reward-Forcing decoded videos must preserve batch size and have three channels")
            if not videos.is_floating_point() or not bool(torch.isfinite(videos).all()):
                raise ValueError("Reward-Forcing decoded videos must be finite floating tensors")
            rewards = self.motion_reward.score_motion_quality(videos, resolved)
            if not isinstance(rewards, Tensor) or tuple(rewards.shape) != (resolved.batch_size,):
                raise ValueError("motion reward adapter must return shape [B]")
            rewards = rewards.to(device=generated_clean.device, dtype=torch.float32).detach()
            multipliers = reward_forcing_multiplier(rewards, self.config.reward_beta)

            score_sigmas = sample_dmd_score_sigmas(
                generated_clean,
                self.config.dmd_config,
                generator=generator,
            )
            noise = torch.randn(
                generated_clean.shape,
                device=generated_clean.device,
                dtype=generated_clean.dtype,
                generator=generator,
            )
            noisy = flow_interpolate(generated_clean.detach(), noise, score_sigmas)
            fake_clean = _prediction(
                self.fake_score,
                noisy,
                score_sigmas,
                resolved,
                conditioning=resolved.conditioning,
                branch="positive",
            )
            real_conditional = _prediction(
                self.real_score,
                noisy,
                score_sigmas,
                resolved,
                conditioning=resolved.conditioning,
                branch="positive",
            )
            real_unconditional = _prediction(
                self.real_score,
                noisy,
                score_sigmas,
                resolved,
                conditioning=resolved.unconditional_conditioning,
                branch="negative",
            )
            guided_real = dmd_teacher_guidance(
                real_conditional,
                real_unconditional,
                self.config.teacher_guidance_scale,
            )
            gradient, normalizer = dmd_distribution_gradient(
                generated_clean,
                fake_clean,
                guided_real,
                normalization_epsilon=self.config.normalization_epsilon,
                per_sample_normalization=True,
            )

        reduced = rewarded_dmd_proxy_loss(
            generated_clean,
            gradient,
            multipliers,
            loss_mask=resolved.loss_mask,
            sample_weights=resolved.sample_weights,
        )
        return DMDLossResult(
            loss=reduced.loss,
            metrics={
                "loss_numerator": reduced.numerator.detach(),
                "loss_denominator": reduced.denominator.detach(),
                "reward_motion_quality": rewards,
                "reward_multiplier": multipliers,
                "reward_motion_quality_mean": rewards.mean(),
                "reward_multiplier_mean": multipliers.mean(),
                "dmd_normalizer": normalizer.detach(),
                "dmd_gradient_abs_mean": gradient.detach().abs().mean(),
                "score_sigma_mean": score_sigmas.detach().float().mean(),
                "student_target_index": torch.tensor(
                    generated.target_index,
                    device=generated_clean.device,
                ),
                "student_timestep": torch.tensor(
                    generated.timestep,
                    device=generated_clean.device,
                ),
            },
        )

    def fake_score_loss(
        self,
        batch: RewardForcingTrainingBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> DMDLossResult:
        resolved = _reward_batch(batch)
        reference = self._validate_geometry(resolved)
        # Reward-Forcing's released trainer keeps every role in eval mode even
        # while differentiating the critic.  ``training=False`` therefore
        # controls dropout/checkpointing only; autograd remains enabled for the
        # fake-score prediction below.
        with torch.no_grad():
            generated = self.base_dmd.sample_student(
                resolved,
                generator=generator,
                training=False,
            )
            generated_clean = generated.clean_latents
            if not isinstance(generated_clean, Tensor) or generated_clean.shape != reference.shape:
                raise ValueError("Reward-Forcing fake-score rollout must preserve latent shape")
            score_sigmas = sample_dmd_score_sigmas(
                generated_clean,
                self.config.dmd_config,
                generator=generator,
            )
            noise = torch.randn(
                generated_clean.shape,
                device=generated_clean.device,
                dtype=generated_clean.dtype,
                generator=generator,
            )
            noisy = flow_interpolate(generated_clean, noise, score_sigmas)
            target = flow_velocity_target(generated_clean, noise)
        prediction = self.fake_score.predict_velocity(
            noisy,
            score_sigmas,
            sample_ids=resolved.sample_ids,
            conditioning=resolved.conditioning,
            training=False,
            branch="positive",
        )
        if not isinstance(prediction, Tensor) or prediction.shape != reference.shape:
            raise ValueError("Reward-Forcing fake-score velocity must preserve latent shape")
        if not prediction.is_floating_point():
            raise TypeError("Reward-Forcing fake-score velocity must be floating point")
        reduced = flow_matching_mse(
            prediction,
            target,
            loss_mask=resolved.loss_mask,
            sample_weights=resolved.sample_weights,
        )
        return DMDLossResult(
            loss=reduced.loss,
            metrics={
                "loss_numerator": reduced.numerator.detach(),
                "loss_denominator": reduced.denominator.detach(),
                "score_sigma_mean": score_sigmas.detach().float().mean(),
                "student_target_index": torch.tensor(
                    generated.target_index,
                    device=reduced.loss.device,
                ),
                "student_timestep": torch.tensor(
                    generated.timestep,
                    device=reduced.loss.device,
                ),
            },
        )


__all__ = ["NativeRewardForcingLossAdapter"]

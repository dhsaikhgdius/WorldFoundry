"""Official-source teacher/self-forcing Causal-rCM execution contracts.

The executable oracle is ``rcm/models/t2v_model_causal.py`` in NVlabs/rCM at
commit ``ed3cb14dd936f92cdc9f9381af7369991509b41f`` (Apache-2.0).  Attention
layout is provided by :mod:`worldfoundry.core.attention.block_pattern`; this
module does not import the synthesis rCM runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Literal, Protocol, runtime_checkable

import torch

from worldfoundry.core.attention.block_pattern import AttnMaskSpec, BlockPattern

from ..consistency.math import (
    batch_coefficients,
    classifier_free_guidance,
    sample_lognormal_rf_time,
    shift_rf_time,
)
from .contracts import RCMLossResult, RCMTrainingBatch
from .math import (
    causal_scm_loss,
    discrete_consistency_loss,
    exact_dmd_proxy_loss,
    sample_discrete_rf_path,
    sum_scaled_losses,
)
from .synchronization import RCMTensorSynchronizer, synchronize_rcm_tensor


@dataclass(frozen=True, slots=True)
class CausalRCMConfig:
    """Consumed execution controls for TF consistency and SF-DMD."""

    consistency_mode: Literal["continuous", "discrete"] = "discrete"
    tangent_warmup_steps: int = 0
    student_update_frequency: int = 5
    causal_teacher_guidance_scale: float = 3.0
    bidirectional_teacher_guidance_scale: float = 5.0
    consistency_loss_scale: float = 100.0
    dmd_loss_scale: float = 1.0
    max_rollout_steps: int = 4
    generator_time_mean: float = -0.8
    generator_time_std: float = 1.6
    score_timestep_shift: float = 5.0
    tangent_normalization_constant: float = 0.1
    dcm_total_steps: int = 48
    dcm_skipping_interval_steps: int = 1
    dcm_timestep_shift: float = 3.0
    first_chunk_frames: int = 1
    chunk_frames: int = 1
    spatial_patch_area: int = 4
    rollout_timesteps: tuple[float, ...] = (15 / 16, 5 / 6, 5 / 8)

    def __post_init__(self) -> None:
        if self.consistency_mode not in {"continuous", "discrete"}:
            raise ValueError("consistency_mode must be continuous or discrete")
        for name, value, allow_zero in (
            ("tangent_warmup_steps", self.tangent_warmup_steps, True),
            ("student_update_frequency", self.student_update_frequency, False),
            ("max_rollout_steps", self.max_rollout_steps, False),
            ("dcm_total_steps", self.dcm_total_steps, False),
            ("dcm_skipping_interval_steps", self.dcm_skipping_interval_steps, False),
            ("first_chunk_frames", self.first_chunk_frames, False),
            ("chunk_frames", self.chunk_frames, False),
            ("spatial_patch_area", self.spatial_patch_area, False),
        ):
            if isinstance(value, bool) or int(value) != value or int(value) < (0 if allow_zero else 1):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be a {qualifier} integer")
        if self.dcm_skipping_interval_steps >= self.dcm_total_steps:
            raise ValueError("dcm_skipping_interval_steps must be smaller than dcm_total_steps")
        for name, value in (
            ("causal_teacher_guidance_scale", self.causal_teacher_guidance_scale),
            ("bidirectional_teacher_guidance_scale", self.bidirectional_teacher_guidance_scale),
            ("consistency_loss_scale", self.consistency_loss_scale),
            ("dmd_loss_scale", self.dmd_loss_scale),
            ("generator_time_mean", self.generator_time_mean),
            ("generator_time_std", self.generator_time_std),
            ("score_timestep_shift", self.score_timestep_shift),
            ("tangent_normalization_constant", self.tangent_normalization_constant),
            ("dcm_timestep_shift", self.dcm_timestep_shift),
        ):
            if not isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.consistency_loss_scale < 0 or self.dmd_loss_scale < 0:
            raise ValueError("loss scales must be non-negative")
        if self.consistency_loss_scale == 0 and self.dmd_loss_scale == 0:
            raise ValueError("at least one Causal-rCM loss must be enabled")
        if self.consistency_loss_scale == 0 and self.tangent_warmup_steps != 0:
            raise ValueError("tangent warmup must be zero without consistency loss")
        if self.generator_time_std <= 0:
            raise ValueError("generator_time_std must be positive")
        if (
            self.score_timestep_shift <= 0
            or self.tangent_normalization_constant <= 0
            or self.dcm_timestep_shift <= 0
        ):
            raise ValueError("time shifts and tangent normalization must be positive")
        rollout = tuple(float(value) for value in self.rollout_timesteps)
        if len(rollout) != self.max_rollout_steps - 1:
            raise ValueError("rollout_timesteps must exactly cover max_rollout_steps - 1")
        if any(not isfinite(value) or not 0 < value < 1 for value in rollout):
            raise ValueError("causal rollout RF times must be finite and in (0,1)")
        if any(left <= right for left, right in zip(rollout, rollout[1:])):
            raise ValueError("causal rollout RF times must be strictly descending")
        object.__setattr__(self, "rollout_timesteps", rollout)

    @property
    def dmd_enabled(self) -> bool:
        return self.dmd_loss_scale > 0


@runtime_checkable
class CausalTeacherForcingAdapter(Protocol):
    """Packed ``[context | noisy]`` causal velocity predictor."""

    module: object

    def predict_velocity(
        self,
        packed_latents: object,
        packed_rf_timesteps: object,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        attention_mask: AttnMaskSpec,
        training: bool,
        branch: str = "positive",
    ) -> object: ...


@runtime_checkable
class CausalExactJVPAdapter(CausalTeacherForcingAdapter, Protocol):
    """Verified forward-mode kernel required by TF-sCM."""

    supports_exact_jvp: bool

    def predict_velocity_with_directional_derivative(
        self,
        packed_latents: object,
        packed_rf_timesteps: object,
        tangent_latents: object,
        tangent_timesteps: object,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        attention_mask: AttnMaskSpec,
    ) -> tuple[object, object]: ...


@dataclass(frozen=True, slots=True)
class CausalRolloutRequest:
    """Block pattern and denoising times for one fresh SF rollout."""

    pattern: BlockPattern
    steps_per_block: tuple[int, ...]
    timesteps_per_block: tuple[tuple[float, ...], ...]


@runtime_checkable
class CausalSelfForcingAdapter(Protocol):
    """Native causal model seam that owns block KV-cache calls."""

    module: object

    def rollout(
        self,
        batch: RCMTrainingBatch,
        request: CausalRolloutRequest,
        *,
        training: bool,
        generator: object | None,
    ) -> object: ...


@runtime_checkable
class RFScoreAdapter(Protocol):
    """Bidirectional RF velocity seam used by the real and fake scores."""

    module: object

    def predict_velocity(
        self,
        noisy_latents: object,
        rf_timesteps: object,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> object: ...


def causal_block_pattern(
    latents: torch.Tensor,
    config: CausalRCMConfig,
) -> tuple[BlockPattern, int]:
    """Validate and construct the one shared training/inference chunk pattern."""

    if not isinstance(latents, torch.Tensor) or latents.ndim != 5:
        raise TypeError("Causal-rCM latents must have shape [B,C,T,H,W]")
    frames, height, width = map(int, latents.shape[-3:])
    if config.first_chunk_frames > frames:
        raise ValueError("first_chunk_frames exceeds the latent frame count")
    if (frames - config.first_chunk_frames) % config.chunk_frames != 0:
        raise ValueError("latent frames do not fit the configured causal chunk pattern")
    spatial_tokens = height * width
    if spatial_tokens % config.spatial_patch_area:
        raise ValueError("latent spatial tokens are not divisible by spatial_patch_area")
    pattern = BlockPattern(
        frame_tokens=spatial_tokens // config.spatial_patch_area,
        first_chunk_frames=config.first_chunk_frames,
        chunk_frames=config.chunk_frames,
    )
    blocks = 1 + (frames - config.first_chunk_frames) // config.chunk_frames
    return pattern, blocks


def _normal_like(
    reference: torch.Tensor,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    return torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


def _repeat_time(time: torch.Tensor, frames: int) -> torch.Tensor:
    if time.ndim == 2:
        if time.shape[1] != 1:
            raise ValueError("per-sample RF time must have shape [B] or [B,1]")
        time = time[:, 0]
    if time.ndim != 1:
        raise ValueError("per-sample RF time must have shape [B] or [B,1]")
    return time[:, None].expand(time.shape[0], frames)


def _causal_prediction(
    adapter: CausalTeacherForcingAdapter,
    packed_latents: torch.Tensor,
    packed_times: torch.Tensor,
    batch: RCMTrainingBatch,
    *,
    conditioning: Mapping[str, object],
    attention_mask: AttnMaskSpec,
    training: bool,
    branch: str = "positive",
) -> torch.Tensor:
    velocity = adapter.predict_velocity(
        packed_latents,
        packed_times,
        sample_ids=batch.sample_ids,
        conditioning=conditioning,
        attention_mask=attention_mask,
        training=training,
        branch=branch,
    )
    if not isinstance(velocity, torch.Tensor) or velocity.shape != packed_latents.shape:
        raise ValueError("causal velocity must match the packed latent tensor")
    return velocity


def _score_prediction(
    adapter: RFScoreAdapter,
    noisy: torch.Tensor,
    time: torch.Tensor,
    batch: RCMTrainingBatch,
    *,
    conditioning: Mapping[str, object],
    training: bool,
    branch: str = "positive",
) -> torch.Tensor:
    velocity = adapter.predict_velocity(
        noisy,
        time,
        sample_ids=batch.sample_ids,
        conditioning=conditioning,
        training=training,
        branch=branch,
    )
    if not isinstance(velocity, torch.Tensor) or velocity.shape != noisy.shape:
        raise ValueError("RF score velocity must match the latent tensor")
    return velocity


class NativeCausalRCMLossAdapter:
    """Own TF-sCM/dCM and SF-DMD formulas while adapters own model kernels."""

    def __init__(
        self,
        student: CausalTeacherForcingAdapter,
        causal_teacher: CausalTeacherForcingAdapter | None,
        rollout: CausalSelfForcingAdapter | None,
        bidirectional_teacher: RFScoreAdapter | None,
        fake_score: RFScoreAdapter | None,
        config: CausalRCMConfig,
        *,
        tensor_synchronizer: RCMTensorSynchronizer | None = None,
    ) -> None:
        if not isinstance(student, CausalTeacherForcingAdapter):
            raise TypeError("student must implement CausalTeacherForcingAdapter")
        if causal_teacher is not None and not isinstance(
            causal_teacher,
            CausalTeacherForcingAdapter,
        ):
            raise TypeError("causal_teacher must implement CausalTeacherForcingAdapter")
        if rollout is not None and not isinstance(rollout, CausalSelfForcingAdapter):
            raise TypeError("rollout must implement CausalSelfForcingAdapter")
        if bidirectional_teacher is not None and not isinstance(
            bidirectional_teacher,
            RFScoreAdapter,
        ):
            raise TypeError("bidirectional_teacher must implement RFScoreAdapter")
        if fake_score is not None and not isinstance(fake_score, RFScoreAdapter):
            raise TypeError("fake_score must implement RFScoreAdapter")
        if not isinstance(config, CausalRCMConfig):
            raise TypeError("config must be CausalRCMConfig")
        if tensor_synchronizer is not None and not isinstance(
            tensor_synchronizer,
            RCMTensorSynchronizer,
        ):
            raise TypeError("tensor_synchronizer must implement RCMTensorSynchronizer")
        if config.consistency_loss_scale > 0 and causal_teacher is None:
            raise ValueError("Causal-rCM consistency requires a causal teacher")
        if config.dmd_enabled and (
            rollout is None or bidirectional_teacher is None or fake_score is None
        ):
            raise ValueError("Causal-rCM DMD requires rollout, bidirectional teacher, and fake score")
        if rollout is not None and rollout.module is not student.module:
            raise ValueError("Causal-rCM rollout must execute the student module")
        if config.consistency_mode == "continuous" and (
            not isinstance(student, CausalExactJVPAdapter)
            or student.supports_exact_jvp is not True
        ):
            raise RuntimeError(
                "Causal TF-sCM requires a verified native exact-JVP model kernel"
            )
        self.student = student
        self.causal_teacher = causal_teacher
        self.rollout_adapter = rollout
        self.bidirectional_teacher = bidirectional_teacher
        self.fake_score = fake_score
        self.config = config
        self.tensor_synchronizer = tensor_synchronizer

    def _synchronize(self, value: torch.Tensor) -> torch.Tensor:
        return synchronize_rcm_tensor(value, self.tensor_synchronizer)

    def _normal(
        self,
        reference: torch.Tensor,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        return self._synchronize(_normal_like(reference, generator=generator))

    def loss_denominator(
        self,
        batch: RCMTrainingBatch,
        *,
        role: str,
    ) -> int:
        if not isinstance(batch, RCMTrainingBatch):
            raise TypeError("batch must be RCMTrainingBatch")
        if role not in {"student", "fake-score"}:
            raise ValueError("Causal-rCM role must be student or fake-score")
        return batch.batch_size

    def _mask(self, clean: torch.Tensor) -> tuple[BlockPattern, int, AttnMaskSpec]:
        pattern, blocks = causal_block_pattern(clean, self.config)
        return pattern, blocks, AttnMaskSpec(
            mode="teacher_forcing",
            pattern=pattern,
            clean_blocks=blocks,
        )

    def _guided_causal_teacher(
        self,
        packed: torch.Tensor,
        packed_times: torch.Tensor,
        batch: RCMTrainingBatch,
        mask: AttnMaskSpec,
    ) -> torch.Tensor:
        teacher = self.causal_teacher
        if teacher is None:
            raise RuntimeError("causal teacher is unavailable")
        with torch.no_grad():
            conditional = _causal_prediction(
                teacher,
                packed,
                packed_times,
                batch,
                conditioning=batch.conditioning,
                attention_mask=mask,
                training=False,
            )
            if self.config.causal_teacher_guidance_scale <= 1.0:
                return conditional
            unconditional = _causal_prediction(
                teacher,
                packed,
                packed_times,
                batch,
                conditioning=batch.unconditional_conditioning,
                attention_mask=mask,
                training=False,
                branch="negative",
            )
            return classifier_free_guidance(
                conditional,
                unconditional,
                self.config.causal_teacher_guidance_scale,
            )

    def _guided_bidirectional_teacher(
        self,
        noisy: torch.Tensor,
        time: torch.Tensor,
        batch: RCMTrainingBatch,
    ) -> torch.Tensor:
        teacher = self.bidirectional_teacher
        if teacher is None:
            raise RuntimeError("bidirectional teacher is unavailable")
        with torch.no_grad():
            conditional = _score_prediction(
                teacher,
                noisy,
                time,
                batch,
                conditioning=batch.conditioning,
                training=False,
            )
            if self.config.bidirectional_teacher_guidance_scale <= 1.0:
                return conditional
            unconditional = _score_prediction(
                teacher,
                noisy,
                time,
                batch,
                conditioning=batch.unconditional_conditioning,
                training=False,
                branch="negative",
            )
            return classifier_free_guidance(
                conditional,
                unconditional,
                self.config.bidirectional_teacher_guidance_scale,
            )

    def _discrete_consistency(
        self,
        batch: RCMTrainingBatch,
        *,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        clean = batch.clean_latents
        assert isinstance(clean, torch.Tensor)
        _, _, mask = self._mask(clean)
        frames = clean.shape[2]
        sampled_times = sample_discrete_rf_path(
            clean,
            total_steps=self.config.dcm_total_steps,
            skipping_interval_steps=self.config.dcm_skipping_interval_steps,
            timestep_shift=self.config.dcm_timestep_shift,
            generator=generator,
        )
        times = tuple(self._synchronize(torch.stack(sampled_times)).unbind())
        start, end = times[0].float(), times[-1].float()
        noisy = (1.0 - batch_coefficients(start, clean)) * clean + batch_coefficients(
            start,
            clean,
        ) * self._normal(clean, generator=generator)
        start_frames = _repeat_time(start, frames)
        packed = torch.cat((clean, noisy), dim=2)
        packed_times = torch.cat((torch.zeros_like(start_frames), start_frames), dim=1)
        velocity = _causal_prediction(
            self.student,
            packed,
            packed_times,
            batch,
            conditioning=batch.conditioning,
            attention_mask=mask,
            training=True,
        )[:, :, frames:]
        current = noisy - batch_coefficients(start, noisy) * velocity
        with torch.no_grad():
            integrated = noisy
            for current_time, next_time in zip(times[:-1], times[1:], strict=True):
                current_time = current_time.float()
                next_time = next_time.float()
                time_frames = _repeat_time(current_time, frames)
                teacher_packed = torch.cat((clean, integrated), dim=2)
                teacher_times = torch.cat((torch.zeros_like(time_frames), time_frames), dim=1)
                teacher_velocity = self._guided_causal_teacher(
                    teacher_packed,
                    teacher_times,
                    batch,
                    mask,
                )[:, :, frames:]
                integrated = integrated - batch_coefficients(
                    current_time - next_time,
                    integrated,
                ) * teacher_velocity
            end_frames = _repeat_time(end, frames)
            target_packed = torch.cat((clean, integrated), dim=2)
            target_times = torch.cat((torch.zeros_like(end_frames), end_frames), dim=1)
            target_velocity = _causal_prediction(
                self.student,
                target_packed,
                target_times,
                batch,
                conditioning=batch.conditioning,
                attention_mask=mask,
                training=False,
            )[:, :, frames:]
            target = integrated - batch_coefficients(end, integrated) * target_velocity
        loss = discrete_consistency_loss(
            current,
            target,
            causal_video_reduction=True,
        )
        return loss, {
            "consistency_loss": loss.detach(),
            "consistency_timestep_mean": start.detach().mean(),
            "consistency_target_timestep_mean": end.detach().mean(),
        }

    def _continuous_consistency(
        self,
        batch: RCMTrainingBatch,
        *,
        iteration: int,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        clean = batch.clean_latents
        assert isinstance(clean, torch.Tensor)
        _, _, mask = self._mask(clean)
        frames = clean.shape[2]
        time = self._synchronize(sample_lognormal_rf_time(
            clean,
            mean=self.config.generator_time_mean,
            std=self.config.generator_time_std,
            generator=generator,
        )).float()
        time_frames = _repeat_time(time, frames)
        noisy = (1.0 - batch_coefficients(time, clean)) * clean + batch_coefficients(
            time,
            clean,
        ) * self._normal(clean, generator=generator)
        packed = torch.cat((clean, noisy), dim=2)
        packed_times = torch.cat((torch.zeros_like(time_frames), time_frames), dim=1)
        teacher = self._guided_causal_teacher(packed, packed_times, batch, mask)[
            :,
            :,
            frames:,
        ]
        tangent_latents = torch.cat((torch.zeros_like(clean), teacher), dim=2)
        tangent_times = torch.cat((torch.zeros_like(time_frames), torch.ones_like(time_frames)), dim=1)
        student = self.student
        assert isinstance(student, CausalExactJVPAdapter)
        with torch.no_grad():
            stopped_packed, tangent_packed = student.predict_velocity_with_directional_derivative(
                packed,
                packed_times,
                tangent_latents,
                tangent_times,
                sample_ids=batch.sample_ids,
                conditioning=batch.conditioning,
                attention_mask=mask,
            )
        if not isinstance(stopped_packed, torch.Tensor) or stopped_packed.shape != packed.shape:
            raise ValueError("causal exact-JVP primal must match packed latents")
        if not isinstance(tangent_packed, torch.Tensor) or tangent_packed.shape != packed.shape:
            raise ValueError("causal exact-JVP tangent must match packed latents")
        current = _causal_prediction(
            self.student,
            packed,
            packed_times,
            batch,
            conditioning=batch.conditioning,
            attention_mask=mask,
            training=True,
        )[:, :, frames:]
        warmup = (
            1.0
            if self.config.tangent_warmup_steps == 0
            else min(1.0, float(iteration) / float(self.config.tangent_warmup_steps))
        )
        loss, tangent_norm, bad = causal_scm_loss(
            current,
            stopped_packed[:, :, frames:].detach(),
            teacher.detach(),
            tangent_packed[:, :, frames:].detach(),
            time,
            warmup_ratio=warmup,
            normalization_constant=self.config.tangent_normalization_constant,
        )
        return loss, {
            "consistency_loss": loss.detach(),
            "consistency_timestep_mean": time.detach().mean(),
            "tangent_norm": tangent_norm.detach().mean(),
            "tangent_warmup_ratio": torch.tensor(warmup, device=clean.device),
            "invalid_consistency_samples": bad.detach().sum(),
        }

    def _rollout_request(
        self,
        clean: torch.Tensor,
        *,
        steps: int,
    ) -> CausalRolloutRequest:
        pattern, blocks = causal_block_pattern(clean, self.config)
        if not 1 <= steps <= self.config.max_rollout_steps:
            raise ValueError("causal rollout steps fall outside the configured cycle")
        trajectory = (1.0, *self.config.rollout_timesteps[: steps - 1])
        return CausalRolloutRequest(
            pattern=pattern,
            steps_per_block=(steps,) * blocks,
            timesteps_per_block=(tuple(trajectory),) * blocks,
        )

    def _rollout(
        self,
        batch: RCMTrainingBatch,
        *,
        steps: int,
        training: bool,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        clean = batch.clean_latents
        assert isinstance(clean, torch.Tensor)
        adapter = self.rollout_adapter
        if adapter is None:
            raise RuntimeError("causal self-forcing rollout is unavailable")
        generated = adapter.rollout(
            batch,
            self._rollout_request(clean, steps=steps),
            training=training,
            generator=generator,
        )
        if not isinstance(generated, torch.Tensor) or generated.shape != clean.shape:
            raise ValueError("causal rollout must return a clean tensor matching the batch")
        return generated

    def _score_time(
        self,
        reference: torch.Tensor,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        uniform = torch.rand(
            (reference.shape[0],),
            device=reference.device,
            dtype=torch.float64,
            generator=generator,
        )
        return shift_rf_time(
            self._synchronize(uniform),
            self.config.score_timestep_shift,
        ).float()

    def _dmd(
        self,
        batch: RCMTrainingBatch,
        *,
        effective_student_iteration: int,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        fake_score = self.fake_score
        if fake_score is None:
            raise RuntimeError("causal DMD fake score is unavailable")
        steps = effective_student_iteration % self.config.max_rollout_steps + 1
        generated = self._rollout(
            batch,
            steps=steps,
            training=True,
            generator=generator,
        )
        time = self._score_time(generated, generator=generator)
        noise = self._normal(generated, generator=generator)
        noisy = (1.0 - batch_coefficients(time, generated)) * generated + batch_coefficients(
            time,
            generated,
        ) * noise
        with torch.no_grad():
            fake_velocity = _score_prediction(
                fake_score,
                noisy,
                time,
                batch,
                conditioning=batch.conditioning,
                training=False,
            )
            teacher_velocity = self._guided_bidirectional_teacher(noisy, time, batch)
            fake_clean = noisy - batch_coefficients(time, noisy) * fake_velocity
            teacher_clean = noisy - batch_coefficients(time, noisy) * teacher_velocity
        loss, denominator, bad = exact_dmd_proxy_loss(
            generated,
            fake_clean,
            teacher_clean,
        )
        return loss, {
            "dmd_loss": loss.detach(),
            "dmd_normalizer": denominator.detach().mean(),
            "dmd_timestep_mean": time.detach().mean(),
            "invalid_dmd_samples": bad.detach().sum(),
            "student_rollout_steps": torch.tensor(steps, device=generated.device),
        }

    def student_loss(
        self,
        batch: RCMTrainingBatch,
        *,
        iteration: int,
        effective_student_iteration: int,
        include_dmd: bool,
        generator: object | None = None,
    ) -> RCMLossResult:
        if not isinstance(batch, RCMTrainingBatch):
            raise TypeError("batch must be RCMTrainingBatch")
        clean = batch.clean_latents
        if not isinstance(clean, torch.Tensor) or clean.ndim != 5:
            raise TypeError("Causal-rCM requires [B,C,T,H,W] torch latents")
        if isinstance(iteration, bool) or int(iteration) < 0:
            raise ValueError("iteration must be non-negative")
        if isinstance(effective_student_iteration, bool) or effective_student_iteration < 0:
            raise ValueError("effective_student_iteration must be non-negative")
        if not isinstance(include_dmd, bool) or (include_dmd and not self.config.dmd_enabled):
            raise ValueError("include_dmd is inconsistent with the Causal-rCM config")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be torch.Generator or None")
        terms: list[tuple[torch.Tensor, float]] = []
        metrics: dict[str, object] = {}
        if self.config.consistency_loss_scale > 0:
            if self.config.consistency_mode == "continuous":
                consistency, values = self._continuous_consistency(
                    batch,
                    iteration=iteration,
                    generator=generator,
                )
            else:
                consistency, values = self._discrete_consistency(
                    batch,
                    generator=generator,
                )
            terms.append((consistency, self.config.consistency_loss_scale))
            metrics.update(values)
        if include_dmd:
            dmd, values = self._dmd(
                batch,
                effective_student_iteration=effective_student_iteration,
                generator=generator,
            )
            terms.append((dmd, self.config.dmd_loss_scale))
            metrics.update(values)
        total = sum_scaled_losses(terms)
        metrics["loss_denominator"] = torch.tensor(
            batch.batch_size,
            device=total.device,
            dtype=torch.float32,
        )
        metrics["joint_dmd"] = torch.tensor(include_dmd, device=total.device)
        return RCMLossResult(loss=total, metrics=metrics)

    def fake_score_loss(
        self,
        batch: RCMTrainingBatch,
        *,
        effective_fake_iteration: int,
        generator: object | None = None,
    ) -> RCMLossResult:
        if not isinstance(batch, RCMTrainingBatch):
            raise TypeError("batch must be RCMTrainingBatch")
        clean = batch.clean_latents
        if not isinstance(clean, torch.Tensor) or clean.ndim != 5:
            raise TypeError("Causal-rCM requires [B,C,T,H,W] torch latents")
        if isinstance(effective_fake_iteration, bool) or effective_fake_iteration < 0:
            raise ValueError("effective_fake_iteration must be non-negative")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be torch.Generator or None")
        fake_score = self.fake_score
        if fake_score is None:
            raise RuntimeError("causal fake score is unavailable")
        steps = effective_fake_iteration % self.config.max_rollout_steps + 1
        with torch.no_grad():
            generated = self._rollout(
                batch,
                steps=steps,
                training=False,
                generator=generator,
            )
        time = self._score_time(generated, generator=generator)
        noise = self._normal(generated, generator=generator)
        noisy = (1.0 - batch_coefficients(time, generated)) * generated + batch_coefficients(
            time,
            generated,
        ) * noise
        velocity = _score_prediction(
            fake_score,
            noisy,
            time,
            batch,
            conditioning=batch.conditioning,
            training=True,
        )
        loss = (velocity - (noise - generated)).square().mean()
        return RCMLossResult(
            loss=loss,
            metrics={
                "loss_denominator": torch.tensor(
                    batch.batch_size,
                    device=loss.device,
                    dtype=torch.float32,
                ),
                "fake_score_loss": loss.detach(),
                "fake_score_timestep_mean": time.detach().mean(),
                "fake_score_rollout_steps": torch.tensor(steps, device=loss.device),
            },
        )


__all__ = [
    "CausalExactJVPAdapter",
    "CausalRCMConfig",
    "CausalRolloutRequest",
    "CausalSelfForcingAdapter",
    "CausalTeacherForcingAdapter",
    "NativeCausalRCMLossAdapter",
    "RFScoreAdapter",
    "causal_block_pattern",
]

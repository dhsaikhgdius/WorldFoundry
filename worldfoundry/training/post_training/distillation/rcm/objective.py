"""Native bidirectional rCM consistency, DMD, and fake-score execution."""

from __future__ import annotations

from collections.abc import Mapping
from math import pi

import torch

from ..consistency.math import (
    batch_coefficients,
    classifier_free_guidance,
    rf_to_trigflow_time,
    sample_lognormal_rf_time,
    trigflow_interpolate,
)
from .config import RCMConfig
from .contracts import (
    RCMExactJVPAdapter,
    RCMLossResult,
    RCMPrediction,
    RCMPredictionAdapter,
    RCMTrainingBatch,
)
from .math import (
    bidirectional_scm_loss,
    discrete_consistency_loss,
    exact_dmd_proxy_loss,
    sample_discrete_trigflow_path,
    sum_scaled_losses,
    trigflow_fake_score_loss,
)
from .synchronization import RCMTensorSynchronizer, synchronize_rcm_tensor


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


def _prediction(
    adapter: RCMPredictionAdapter,
    noisy_latents: torch.Tensor,
    trig_timesteps: torch.Tensor,
    batch: RCMTrainingBatch,
    *,
    conditioning: Mapping[str, object],
    training: bool,
    branch: str = "positive",
) -> RCMPrediction:
    prediction = adapter.predict(
        noisy_latents,
        trig_timesteps,
        sample_ids=batch.sample_ids,
        conditioning=conditioning,
        training=training,
        branch=branch,
    )
    if not isinstance(prediction, RCMPrediction):
        raise TypeError("rCM prediction adapters must return RCMPrediction")
    for name, value in (
        ("clean_latents", prediction.clean_latents),
        ("velocity", prediction.velocity),
    ):
        if not isinstance(value, torch.Tensor) or value.shape != noisy_latents.shape:
            raise ValueError(f"rCM {name} must match the noisy latent tensor")
    return prediction


class NativeRCMLossAdapter:
    """Own the complete fixed-source bidirectional rCM objective graph."""

    def __init__(
        self,
        student: RCMPredictionAdapter,
        teacher: RCMPredictionAdapter,
        fake_score: RCMPredictionAdapter | None,
        config: RCMConfig,
        *,
        tensor_synchronizer: RCMTensorSynchronizer | None = None,
    ) -> None:
        if not isinstance(student, RCMPredictionAdapter):
            raise TypeError("student must implement RCMPredictionAdapter")
        if not isinstance(teacher, RCMPredictionAdapter):
            raise TypeError("teacher must implement RCMPredictionAdapter")
        if fake_score is not None and not isinstance(fake_score, RCMPredictionAdapter):
            raise TypeError("fake_score must implement RCMPredictionAdapter")
        if not isinstance(config, RCMConfig):
            raise TypeError("config must be RCMConfig")
        if tensor_synchronizer is not None and not isinstance(
            tensor_synchronizer,
            RCMTensorSynchronizer,
        ):
            raise TypeError("tensor_synchronizer must implement RCMTensorSynchronizer")
        if config.dmd_enabled and fake_score is None:
            raise ValueError("rCM DMD requires a fake-score model")
        if config.consistency_mode == "continuous":
            if not isinstance(student, RCMExactJVPAdapter) or student.supports_exact_jvp is not True:
                raise RuntimeError(
                    "continuous rCM requires a native adapter with verified exact JVP support"
                )
        self.student = student
        self.teacher = teacher
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
            raise ValueError("rCM role must be student or fake-score")
        return batch.batch_size

    def _sample_trig_time(
        self,
        reference: torch.Tensor,
        *,
        score: bool,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        mean = self.config.score_time_mean if score else self.config.generator_time_mean
        std = self.config.score_time_std if score else self.config.generator_time_std
        return rf_to_trigflow_time(
            self._synchronize(sample_lognormal_rf_time(
                reference,
                mean=mean,
                std=std,
                generator=generator,
            ))
        ).to(dtype=torch.float32)

    def _guided_teacher_prediction(
        self,
        noisy_latents: torch.Tensor,
        trig_timesteps: torch.Tensor,
        batch: RCMTrainingBatch,
    ) -> RCMPrediction:
        with torch.no_grad():
            conditional = _prediction(
                self.teacher,
                noisy_latents,
                trig_timesteps,
                batch,
                conditioning=batch.conditioning,
                training=False,
            )
            if self.config.teacher_guidance_scale <= 1.0:
                return conditional
            unconditional = _prediction(
                self.teacher,
                noisy_latents,
                trig_timesteps,
                batch,
                conditioning=batch.unconditional_conditioning,
                training=False,
                branch="negative",
            )
            return RCMPrediction(
                clean_latents=classifier_free_guidance(
                    conditional.clean_latents,
                    unconditional.clean_latents,
                    self.config.teacher_guidance_scale,
                ),
                velocity=classifier_free_guidance(
                    conditional.velocity,
                    unconditional.velocity,
                    self.config.teacher_guidance_scale,
                ),
            )

    def _continuous_consistency_loss(
        self,
        batch: RCMTrainingBatch,
        *,
        iteration: int,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        clean = batch.clean_latents
        assert isinstance(clean, torch.Tensor)
        time = self._sample_trig_time(clean, score=False, generator=generator)
        noisy = trigflow_interpolate(clean, self._normal(clean, generator=generator), time)
        teacher = self._guided_teacher_prediction(noisy, time, batch)
        coefficient = torch.cos(time) * torch.sin(time)
        tangent_latents = batch_coefficients(coefficient, noisy) * teacher.velocity
        tangent_timesteps = coefficient
        student = self.student
        assert isinstance(student, RCMExactJVPAdapter)
        with torch.no_grad():
            stopped, directional_derivative = student.predict_with_directional_derivative(
                noisy,
                time,
                tangent_latents,
                tangent_timesteps,
                sample_ids=batch.sample_ids,
                conditioning=batch.conditioning,
            )
        if not isinstance(stopped, RCMPrediction):
            raise TypeError("exact JVP adapter must return RCMPrediction as its primal")
        if not isinstance(directional_derivative, torch.Tensor):
            raise TypeError("exact JVP adapter must return a tensor directional derivative")
        if stopped.velocity.shape != noisy.shape or directional_derivative.shape != noisy.shape:
            raise ValueError("exact JVP primal and tangent must match the noisy latent tensor")
        current = _prediction(
            self.student,
            noisy,
            time,
            batch,
            conditioning=batch.conditioning,
            training=True,
        )
        warmup = (
            1.0
            if self.config.tangent_warmup_steps == 0
            else min(1.0, float(iteration) / float(self.config.tangent_warmup_steps))
        )
        loss, tangent_norm, bad = bidirectional_scm_loss(
            current.velocity,
            stopped.velocity.detach(),
            teacher.velocity.detach(),
            directional_derivative.detach(),
            noisy,
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

    def _discrete_consistency_loss(
        self,
        batch: RCMTrainingBatch,
        *,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        clean = batch.clean_latents
        assert isinstance(clean, torch.Tensor)
        sampled_times = sample_discrete_trigflow_path(
            clean,
            total_steps=self.config.dcm_total_steps,
            skipping_interval_steps=self.config.dcm_skipping_interval_steps,
            timestep_shift=self.config.dcm_timestep_shift,
            generator=generator,
        )
        times = tuple(self._synchronize(torch.stack(sampled_times)).unbind())
        start, end = times[0].float(), times[-1].float()
        noisy = trigflow_interpolate(clean, self._normal(clean, generator=generator), start)
        current = _prediction(
            self.student,
            noisy,
            start,
            batch,
            conditioning=batch.conditioning,
            training=True,
        ).clean_latents
        with torch.no_grad():
            integrated = noisy
            for current_time, next_time in zip(times[:-1], times[1:], strict=True):
                current_time = current_time.float()
                next_time = next_time.float()
                teacher = self._guided_teacher_prediction(integrated, current_time, batch)
                delta = current_time - next_time
                integrated = integrated - batch_coefficients(delta, integrated) * teacher.velocity
            target = _prediction(
                self.student,
                integrated,
                end,
                batch,
                conditioning=batch.conditioning,
                training=False,
            ).clean_latents
        loss = discrete_consistency_loss(current, target)
        return loss, {
            "consistency_loss": loss.detach(),
            "consistency_timestep_mean": start.detach().mean(),
            "consistency_target_timestep_mean": end.detach().mean(),
        }

    def _rollout(
        self,
        batch: RCMTrainingBatch,
        *,
        steps: int,
        with_grad: bool,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        clean = batch.clean_latents
        assert isinstance(clean, torch.Tensor)
        if not 1 <= steps <= self.config.max_rollout_steps:
            raise ValueError("rollout steps fall outside the configured cycle")
        current_time = torch.full(
            (clean.shape[0],),
            pi / 2,
            device=clean.device,
            dtype=torch.float32,
        )
        trajectory = [current_time]
        if self.config.fixed_rollout_timesteps:
            if len(self.config.fixed_rollout_timesteps) < steps - 1:
                raise ValueError("fixed_rollout_timesteps does not cover this rollout length")
            trajectory.extend(
                torch.full_like(current_time, self.config.fixed_rollout_timesteps[index])
                for index in range(steps - 1)
            )
        else:
            for _ in range(steps - 1):
                sampled = self._sample_trig_time(clean, score=True, generator=generator)
                current_time = torch.minimum(sampled, current_time)
                trajectory.append(current_time)
        trajectory.append(torch.zeros_like(current_time))
        latents = self._normal(clean, generator=generator)
        for index, (time, next_time) in enumerate(
            zip(trajectory[:-1], trajectory[1:], strict=True)
        ):
            final = index + 1 == steps
            context = torch.enable_grad() if with_grad and final else torch.no_grad()
            with context:
                generated = _prediction(
                    self.student,
                    latents,
                    time,
                    batch,
                    conditioning=batch.conditioning,
                    training=with_grad and final,
                ).clean_latents.float()
            if not final:
                latents = trigflow_interpolate(
                    generated,
                    self._normal(generated, generator=generator),
                    next_time,
                )
            else:
                latents = generated
        return latents

    def _dmd_loss(
        self,
        batch: RCMTrainingBatch,
        *,
        effective_student_iteration: int,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        fake_score = self.fake_score
        if fake_score is None:
            raise RuntimeError("rCM DMD requires a fake-score model")
        steps = effective_student_iteration % self.config.max_rollout_steps + 1
        generated = self._rollout(
            batch,
            steps=steps,
            with_grad=True,
            generator=generator,
        )
        time = self._sample_trig_time(generated, score=True, generator=generator)
        noisy = trigflow_interpolate(generated, self._normal(generated, generator=generator), time)
        with torch.no_grad():
            fake = _prediction(
                fake_score,
                noisy,
                time,
                batch,
                conditioning=batch.conditioning,
                training=False,
            ).clean_latents
            teacher = self._guided_teacher_prediction(noisy, time, batch).clean_latents
        loss, denominator, bad = exact_dmd_proxy_loss(generated, fake, teacher)
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
        if isinstance(iteration, bool) or int(iteration) < 0:
            raise ValueError("iteration must be a non-negative integer")
        if isinstance(effective_student_iteration, bool) or int(effective_student_iteration) < 0:
            raise ValueError("effective_student_iteration must be non-negative")
        if not isinstance(include_dmd, bool):
            raise TypeError("include_dmd must be a bool")
        if include_dmd and not self.config.dmd_enabled:
            raise ValueError("include_dmd cannot be true when DMD is disabled")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be torch.Generator or None")
        clean = batch.clean_latents
        if not isinstance(clean, torch.Tensor):
            raise TypeError("clean_latents must be a torch.Tensor")
        terms: list[tuple[torch.Tensor, float]] = []
        metrics: dict[str, object] = {}
        if self.config.consistency_loss_scale > 0:
            if self.config.consistency_mode == "continuous":
                consistency, consistency_metrics = self._continuous_consistency_loss(
                    batch,
                    iteration=iteration,
                    generator=generator,
                )
            else:
                consistency, consistency_metrics = self._discrete_consistency_loss(
                    batch,
                    generator=generator,
                )
            terms.append((consistency, self.config.consistency_loss_scale))
            metrics.update(consistency_metrics)
        if include_dmd:
            dmd, dmd_metrics = self._dmd_loss(
                batch,
                effective_student_iteration=effective_student_iteration,
                generator=generator,
            )
            terms.append((dmd, self.config.dmd_loss_scale))
            metrics.update(dmd_metrics)
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
        if isinstance(effective_fake_iteration, bool) or int(effective_fake_iteration) < 0:
            raise ValueError("effective_fake_iteration must be non-negative")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be torch.Generator or None")
        fake_score = self.fake_score
        if fake_score is None:
            raise RuntimeError("rCM fake-score phase requires a fake-score model")
        steps = effective_fake_iteration % self.config.max_rollout_steps + 1
        with torch.no_grad():
            generated = self._rollout(
                batch,
                steps=steps,
                with_grad=False,
                generator=generator,
            )
        time = self._sample_trig_time(generated, score=True, generator=generator)
        noisy = trigflow_interpolate(generated, self._normal(generated, generator=generator), time)
        fake = _prediction(
            fake_score,
            noisy,
            time,
            batch,
            conditioning=batch.conditioning,
            training=True,
        ).clean_latents
        loss = trigflow_fake_score_loss(generated, fake, time)
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


__all__ = ["NativeRCMLossAdapter"]

"""Native DMD, real-data regression, and temporal-loss composition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from worldfoundry.training.objectives.flow_matching import (
    flow_interpolate,
    flow_matching_mse,
    flow_velocity_target,
)

from ...shared.contracts import FlowPredictionAdapter
from ..dmd.contracts import DMDTrainingBatch
from ..dmd.objective import (
    DMDLossResult,
    DMDStudentSampler,
    FlowDMDLossAdapter,
)
from .config import AdaptiveVideoConfig
from .contracts import AdaptiveVideoTrainingBatch
from .math import (
    AdaptiveRegressionObservation,
    adaptive_regression_weights,
    temporal_variance_regularization,
)
from .state import AdaptiveRegressionEMA

ADAPTIVE_VIDEO_OBJECTIVE_STATE_SCHEMA = "worldfoundry-adaptive-video-objective"


@dataclass(frozen=True, slots=True)
class AdaptiveVideoLossResult(DMDLossResult):
    regression_observation: AdaptiveRegressionObservation | None = None


class FlowAdaptiveVideoLossAdapter:
    """Share one student rollout across all adaptive-video generator losses."""

    def __init__(
        self,
        student: FlowPredictionAdapter,
        real_score: FlowPredictionAdapter,
        fake_score: FlowPredictionAdapter,
        config: AdaptiveVideoConfig,
        *,
        student_sampler: DMDStudentSampler | None = None,
        config_digest: str | None = None,
    ) -> None:
        if not isinstance(config, AdaptiveVideoConfig):
            raise TypeError("config must be AdaptiveVideoConfig")
        self.base = FlowDMDLossAdapter(
            student,
            real_score,
            fake_score,
            config.dmd,
            student_sampler=student_sampler,
        )
        if not isinstance(student, FlowPredictionAdapter):
            raise TypeError(
                "adaptive real-data regression requires a FlowPredictionAdapter student"
            )
        self.student = student
        self.config = config
        self.config_digest = str(config_digest or config.digest)
        if not self.config_digest.strip():
            raise ValueError("config_digest must be non-empty")
        self.schedule_digest = self.config_digest
        self.regression_ema = AdaptiveRegressionEMA(
            len(config.dmd.schedule.sigmas),
            decay=config.regression_ema_decay,
        )

    @staticmethod
    def _generator_batch(batch: object) -> AdaptiveVideoTrainingBatch:
        if not isinstance(batch, AdaptiveVideoTrainingBatch):
            raise TypeError(
                "adaptive video objective requires AdaptiveVideoTrainingBatch"
            )
        if not torch.is_tensor(batch.real_latents):
            raise TypeError("real_latents must be a torch.Tensor")
        return batch

    @staticmethod
    def _dmd_batch(batch: object) -> DMDTrainingBatch:
        if not isinstance(batch, DMDTrainingBatch):
            raise TypeError("adaptive video fake-score requires DMDTrainingBatch")
        return batch

    def loss_denominator(
        self,
        batch: DMDTrainingBatch,
        *,
        role: str,
    ) -> object:
        resolved = (
            self._generator_batch(batch)
            if role == "generator"
            else self._dmd_batch(batch)
        )
        return self.base.loss_denominator(resolved, role=role)

    @staticmethod
    def _randn_like(
        reference: torch.Tensor,
        *,
        generator: object | None,
    ) -> torch.Tensor:
        return torch.randn(
            reference.shape,
            device=reference.device,
            dtype=reference.dtype,
            generator=generator,
        )

    def _regression_loss(
        self,
        batch: AdaptiveVideoTrainingBatch,
        *,
        generator: object | None,
    ) -> tuple[torch.Tensor, AdaptiveRegressionObservation, dict[str, object]]:
        real = batch.real_latents
        assert isinstance(real, torch.Tensor)
        indices = torch.randint(
            0,
            len(self.config.dmd.schedule.sigmas),
            (batch.batch_size,),
            device=real.device,
            generator=generator,
        )
        schedule_sigmas = torch.tensor(
            self.config.dmd.schedule.sigmas,
            device=real.device,
            dtype=torch.float32,
        )
        sigmas = schedule_sigmas[indices]
        noise = self._randn_like(real, generator=generator)
        noisy = flow_interpolate(real, noise, sigmas)
        prediction = self.student.predict_velocity(
            noisy,
            sigmas,
            sample_ids=batch.real_sample_ids,
            conditioning=batch.real_conditioning,
            training=True,
        )
        target = flow_velocity_target(real, noise)
        unweighted = flow_matching_mse(
            prediction,
            target,
            loss_mask=batch.real_loss_mask,
        )
        adaptive = adaptive_regression_weights(
            unweighted.per_sample,
            indices,
            self.regression_ema.values,
            self.regression_ema.initialized,
            decay=self.config.regression_ema_decay,
            sensitivity=self.config.regression_sensitivity,
        )
        sample_weights = adaptive.weights
        if batch.real_sample_weights is not None:
            if not torch.is_tensor(batch.real_sample_weights):
                raise TypeError("real_sample_weights must be a torch.Tensor")
            user_weights = batch.real_sample_weights.to(
                device=real.device,
                dtype=torch.float32,
            )
            if not bool(torch.isfinite(user_weights).all()) or not bool(
                (user_weights >= 0).all()
            ):
                raise ValueError(
                    "real_sample_weights must be finite and non-negative"
                )
            sample_weights = sample_weights * user_weights
        reduced = flow_matching_mse(
            prediction,
            target,
            loss_mask=batch.real_loss_mask,
            sample_weights=sample_weights,
        )
        return (
            reduced.loss,
            adaptive.observation,
            {
                "regression_loss": reduced.loss.detach(),
                "regression_loss_denominator": reduced.denominator.detach(),
                "regression_weight_mean": adaptive.weights.detach().mean(),
                "regression_schedule_indices": indices.detach(),
                "regression_tentative_ema": adaptive.tentative_ema.detach(),
            },
        )

    def generator_loss(
        self,
        batch: AdaptiveVideoTrainingBatch,
        *,
        generator: object | None = None,
    ) -> AdaptiveVideoLossResult:
        resolved = self._generator_batch(batch)
        generated = self.base.sample_student(
            resolved,
            generator=generator,
            training=True,
        )
        dmd_result = self.base.generator_loss_from_prediction(
            resolved,
            generated,
            generator=generator,
        )
        generated_clean = generated.clean_latents
        if not isinstance(generated_clean, torch.Tensor):
            raise TypeError("adaptive student must return a torch.Tensor")
        temporal = temporal_variance_regularization(
            generated_clean,
            frame_axis=1,
            epsilon=self.config.temporal_epsilon,
            cutoff=self.config.temporal_loss_cutoff,
        )
        regression_loss, observation, regression_metrics = self._regression_loss(
            resolved,
            generator=generator,
        )
        total = (
            dmd_result.loss.double()
            + self.config.regression_loss_weight * regression_loss.double()
            + self.config.temporal_regularization_weight
            * temporal.applied_loss.double()
        )
        denominator = torch.as_tensor(
            dmd_result.metrics["loss_denominator"],
            device=total.device,
            dtype=torch.float32,
        ).reshape(())
        metrics = dict(dmd_result.metrics)
        metrics.update(regression_metrics)
        metrics.update(
            {
                "loss_numerator": total.detach().float() * denominator,
                "loss_denominator": denominator,
                "dmd_loss": dmd_result.loss.detach(),
                "temporal_motion_metric": temporal.motion_metric.detach(),
                "temporal_raw_loss": temporal.raw_loss.detach(),
                "temporal_applied_loss": temporal.applied_loss.detach(),
            }
        )
        return AdaptiveVideoLossResult(
            loss=total,
            metrics=metrics,
            regression_observation=observation,
        )

    def fake_score_loss(
        self,
        batch: DMDTrainingBatch,
        *,
        generator: object | None = None,
    ) -> DMDLossResult:
        return self.base.fake_score_loss(
            self._dmd_batch(batch),
            generator=generator,
        )

    def commit_generator_step(self, results: tuple[object, ...]) -> None:
        observations: list[AdaptiveRegressionObservation] = []
        for result in results:
            if not isinstance(result, AdaptiveVideoLossResult):
                raise TypeError(
                    "adaptive generator commit received a foreign loss result"
                )
            if result.regression_observation is None:
                raise RuntimeError("adaptive generator result lacks regression statistics")
            observations.append(result.regression_observation)
        self.regression_ema.commit(observations)

    def adaptive_state_dict(self) -> dict[str, object]:
        return {
            "schema": ADAPTIVE_VIDEO_OBJECTIVE_STATE_SCHEMA,
            "config_digest": self.config_digest,
            "regression_ema": self.regression_ema.state_dict(),
        }

    def load_adaptive_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("adaptive objective state must be a mapping")
        if set(state_dict) != {"schema", "config_digest", "regression_ema"}:
            raise ValueError("adaptive objective state fields differ from the active schema")
        if state_dict["schema"] != ADAPTIVE_VIDEO_OBJECTIVE_STATE_SCHEMA:
            raise ValueError("unsupported adaptive objective state schema")
        if state_dict["config_digest"] != self.config_digest:
            raise ValueError("saved adaptive objective configuration differs")
        regression_state = state_dict["regression_ema"]
        if not isinstance(regression_state, Mapping):
            raise TypeError("saved adaptive regression EMA must be a mapping")
        self.regression_ema.load_state_dict(regression_state)


__all__ = [
    "ADAPTIVE_VIDEO_OBJECTIVE_STATE_SCHEMA",
    "AdaptiveVideoLossResult",
    "FlowAdaptiveVideoLossAdapter",
]

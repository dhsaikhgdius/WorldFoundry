"""Compound DMD objective used by native diagonal distillation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from worldfoundry.core.checkpoint import validate_state_dict_compatibility
from worldfoundry.training.objectives.flow_matching import flow_interpolate

from ...shared.contracts import FlowPredictionAdapter
from ..causal_consistency.ema import FrozenModuleEMA
from ..dmd.contracts import DMDTrainingBatch
from ..dmd.objective import (
    DMDLossResult,
    FlowDMDLossAdapter,
    dmd_teacher_guidance,
    sample_dmd_score_sigmas,
)
from .config import DiagonalObjectiveConfig
from .contracts import DiagonalFewStepPrediction
from .math import (
    diagonal_distribution_gradients,
    diagonal_flow_regression_loss,
    diagonal_proxy_losses,
    diagonal_regression_loss,
)
from .rollout import DiagonalFixedTeacherSampler, DiagonalRolloutSampler

DIAGONAL_OBJECTIVE_STATE_SCHEMA = "worldfoundry-diagonal-objective-state"


@dataclass(frozen=True, slots=True)
class DiagonalDMDLossResult(DMDLossResult):
    """Typed result used to gate the post-optimizer motion-head EMA commit."""


def _normal_like(reference: Tensor, *, generator: torch.Generator | None) -> Tensor:
    return torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


class DiagonalDMDLossAdapter:
    """Share one diagonal student rollout across all official loss branches."""

    def __init__(
        self,
        real_score: FlowPredictionAdapter,
        fake_score: FlowPredictionAdapter,
        config: DiagonalObjectiveConfig,
        *,
        student_sampler: DiagonalRolloutSampler,
        fixed_teacher_sampler: DiagonalFixedTeacherSampler | None = None,
        motion_head_student: nn.Module | None = None,
        motion_head_teacher: nn.Module | None = None,
        initialize_motion_teacher: bool = True,
    ) -> None:
        if not isinstance(config, DiagonalObjectiveConfig):
            raise TypeError("config must be DiagonalObjectiveConfig")
        if not isinstance(student_sampler, DiagonalRolloutSampler):
            raise TypeError("student_sampler must be DiagonalRolloutSampler")
        if not isinstance(initialize_motion_teacher, bool):
            raise TypeError("initialize_motion_teacher must be bool")
        if student_sampler.config.base_schedule != config.dmd.schedule:
            raise ValueError("diagonal rollout and DMD base schedules differ")
        if student_sampler.config.frame_dim != config.frame_dim:
            raise ValueError("diagonal rollout and objective frame dimensions differ")
        if config.use_teacher_regression:
            if not isinstance(fixed_teacher_sampler, DiagonalFixedTeacherSampler):
                raise TypeError("teacher regression requires DiagonalFixedTeacherSampler")
            if fixed_teacher_sampler.adapter.module is student_sampler.adapter.module:
                raise ValueError("fixed regression teacher must be distinct from the student")
            student_parallel = student_sampler.parallel_context
            teacher_parallel = fixed_teacher_sampler.parallel_context
            if (
                teacher_parallel.rank != student_parallel.rank
                or teacher_parallel.world_size != student_parallel.world_size
                or teacher_parallel.process_group is not student_parallel.process_group
            ):
                raise ValueError("fixed regression teacher and student must share a data-parallel group")
            if (
                fixed_teacher_sampler.config.frames_per_block
                != student_sampler.config.frames_per_block
                or fixed_teacher_sampler.config.frame_dim != student_sampler.config.frame_dim
                or fixed_teacher_sampler.config.context_timestep
                != student_sampler.config.context_timestep
                or fixed_teacher_sampler.config.context_sigma
                != student_sampler.config.context_sigma
            ):
                raise ValueError("fixed regression teacher and student causal geometry differ")
        elif fixed_teacher_sampler is not None:
            raise ValueError("fixed_teacher_sampler is unused when teacher regression is disabled")
        if config.use_flow_reg_loss:
            if not isinstance(motion_head_student, nn.Module) or not isinstance(motion_head_teacher, nn.Module):
                raise TypeError("flow regression requires student and teacher motion heads")
            if motion_head_student is motion_head_teacher:
                raise ValueError("motion-head student and teacher must be distinct modules")
            if not any(parameter.requires_grad for parameter in motion_head_student.parameters()):
                raise ValueError("motion-head student has no trainable parameters")
            module_parameters = {id(parameter) for parameter in student_sampler.adapter.module.parameters()}
            if not {id(parameter) for parameter in motion_head_student.parameters()} <= module_parameters:
                raise ValueError("motion-head student must be registered inside the student module")
            motion_ema = FrozenModuleEMA(
                motion_head_student,
                motion_head_teacher,
                decay=config.flow_reg_ema_decay,
                initialize_target=initialize_motion_teacher,
            )
        else:
            if motion_head_student is not None or motion_head_teacher is not None:
                raise ValueError("motion heads are unused when flow regression is disabled")
            motion_ema = None
        self.base = FlowDMDLossAdapter(
            None,
            real_score,
            fake_score,
            config.dmd,
            student_sampler=student_sampler,
        )
        self.real_score = real_score
        self.fake_score = fake_score
        self.config = config
        self.student_sampler = student_sampler
        self.fixed_teacher_sampler = fixed_teacher_sampler
        self.motion_head_student = motion_head_student
        self.motion_head_teacher = motion_head_teacher
        self.motion_ema = motion_ema
        self.motion_ema_updates = 0

    def loss_denominator(self, batch: DMDTrainingBatch, *, role: str) -> object:
        return self.base.loss_denominator(batch, role=role)

    def _owned_generator(self, generator: object | None) -> torch.Generator:
        if generator is not None and generator is not self.student_sampler.generator:
            raise ValueError("diagonal objective randomness must use the sampler generator")
        return self.student_sampler.generator

    def _regression_target(
        self,
        batch: DMDTrainingBatch,
        *,
        generator: object | None,
    ) -> tuple[DiagonalFewStepPrediction, Tensor | None]:
        if self.fixed_teacher_sampler is None:
            generated = self.base.sample_student(batch, generator=generator, training=True)
            if not isinstance(generated, DiagonalFewStepPrediction):
                raise TypeError("diagonal student sampler returned an incompatible prediction")
            return generated, None
        if not isinstance(generator, torch.Generator):
            raise TypeError("fixed teacher replay requires the sampler's torch.Generator")
        rng_state = generator.get_state().clone()
        generated = self.base.sample_student(batch, generator=generator, training=True)
        if not isinstance(generated, DiagonalFewStepPrediction):
            raise TypeError("diagonal student sampler returned an incompatible prediction")
        generator.set_state(rng_state)
        with torch.no_grad():
            target = self.fixed_teacher_sampler.sample(batch, generator=generator)
        return generated, target

    def generator_loss(
        self,
        batch: DMDTrainingBatch,
        *,
        generator: object | None = None,
    ) -> DiagonalDMDLossResult:
        if not isinstance(batch, DMDTrainingBatch):
            raise TypeError("diagonal objective requires DMDTrainingBatch")
        rng = self._owned_generator(generator)
        generated, fixed_target = self._regression_target(batch, generator=rng)
        clean = generated.clean_latents
        if not isinstance(clean, Tensor) or clean.shape != batch.clean_latents.shape:
            raise ValueError("diagonal student output must preserve the latent shape")
        gradient_mask = generated.rollout.gradient_mask

        with torch.no_grad():
            score_sigmas = sample_dmd_score_sigmas(clean, self.config.dmd, generator=rng)
            noisy = flow_interpolate(
                clean.detach(),
                _normal_like(clean, generator=rng),
                score_sigmas,
            )
            fake_clean = self.fake_score.predict_clean(
                noisy,
                score_sigmas,
                sample_ids=batch.sample_ids,
                conditioning=batch.conditioning,
                training=False,
            )
            real_conditional = self.real_score.predict_clean(
                noisy,
                score_sigmas,
                sample_ids=batch.sample_ids,
                conditioning=batch.conditioning,
                training=False,
            )
            real_unconditional = self.real_score.predict_clean(
                noisy,
                score_sigmas,
                sample_ids=batch.sample_ids,
                conditioning=batch.unconditional_conditioning,
                training=False,
                branch="negative",
            )
            guided_real = dmd_teacher_guidance(
                real_conditional,
                real_unconditional,
                self.config.dmd.teacher_guidance_scale,
            )
            gradients = diagonal_distribution_gradients(
                clean,
                fake_clean,
                guided_real,
                frame_dim=self.config.frame_dim,
                normalization_epsilon=self.config.dmd.normalization_epsilon,
            )
        proxy = diagonal_proxy_losses(
            clean,
            gradients,
            gradient_mask=gradient_mask,
            frame_dim=self.config.frame_dim,
        )
        regression_target = proxy.spatial_target if fixed_target is None else fixed_target.double().detach()
        regression = diagonal_regression_loss(
            clean.double(),
            regression_target,
            gradient_mask=gradient_mask,
            loss_type=self.config.regression_loss_type,
            epsilon=self.config.regression_epsilon,
            cauchy_scale=self.config.regression_cauchy_scale,
        )
        flow_regression = proxy.spatial * 0.0
        if self.config.use_flow_reg_loss:
            assert self.motion_head_student is not None and self.motion_head_teacher is not None
            flow_regression = diagonal_flow_regression_loss(
                clean,
                regression_target,
                self.motion_head_student,
                self.motion_head_teacher,
                gradient_mask=gradient_mask,
                frame_dim=self.config.frame_dim,
            ).double()
        motion_term = proxy.motion if self.config.use_motion_loss else proxy.spatial * 0.0
        total = (
            self.config.lambda_spatial_dmd * proxy.spatial
            + self.config.lambda_reg * regression
            + self.config.gamma_temporal
            * (self.config.lambda_flow_dmd * motion_term + flow_regression)
        )
        denominator = torch.tensor(clean.numel(), device=total.device, dtype=torch.float32)
        return DiagonalDMDLossResult(
            loss=total,
            metrics={
                "loss_numerator": total.detach().float() * denominator,
                "loss_denominator": denominator,
                "spatial_dmd_loss": proxy.spatial.detach(),
                "motion_dmd_loss": proxy.motion.detach(),
                "flow_regression_loss": flow_regression.detach(),
                "regression_loss": regression.detach(),
                "spatial_gradient_abs_mean": gradients.spatial.detach().abs().mean(),
                "motion_gradient_abs_mean": gradients.motion.detach().abs().mean(),
                "score_sigma_mean": score_sigmas.detach().float().mean(),
                "student_target_index": torch.tensor(generated.target_index, device=total.device),
                "student_timestep": torch.tensor(generated.timestep, device=total.device),
                "motion_weights": proxy.motion_weights.detach(),
            },
        )

    def fake_score_loss(
        self,
        batch: DMDTrainingBatch,
        *,
        generator: object | None = None,
    ) -> DMDLossResult:
        return self.base.fake_score_loss(batch, generator=self._owned_generator(generator))

    def commit_generator_step(self, results: tuple[object, ...]) -> None:
        if not results or not all(isinstance(result, DiagonalDMDLossResult) for result in results):
            raise TypeError("diagonal generator commit requires its own non-empty loss results")
        if self.motion_ema is not None:
            self.motion_ema.update()
            self.motion_ema_updates += 1

    def state_dict(self) -> dict[str, object]:
        teacher_state = None
        if self.motion_head_teacher is not None:
            teacher_state = {
                name: value.detach().clone()
                for name, value in self.motion_head_teacher.state_dict().items()
            }
        return {
            "schema": DIAGONAL_OBJECTIVE_STATE_SCHEMA,
            "motion_ema_updates": self.motion_ema_updates,
            "motion_head_teacher": teacher_state,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("diagonal objective state must be a mapping")
        expected = {
            "schema",
            "motion_ema_updates",
            "motion_head_teacher",
        }
        if set(state_dict) != expected:
            raise ValueError("diagonal objective state fields differ from the active schema")
        if state_dict["schema"] != DIAGONAL_OBJECTIVE_STATE_SCHEMA:
            raise ValueError(f"unsupported diagonal objective schema: {state_dict['schema']!r}")
        updates = state_dict["motion_ema_updates"]
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
            raise ValueError("saved diagonal motion EMA update count is invalid")
        saved_teacher = state_dict["motion_head_teacher"]
        if self.motion_head_teacher is None:
            if saved_teacher is not None or updates != 0:
                raise ValueError("saved diagonal objective unexpectedly contains a motion teacher")
        else:
            if not isinstance(saved_teacher, Mapping):
                raise TypeError("saved diagonal motion teacher state must be a mapping")
            validate_state_dict_compatibility(
                self.motion_head_teacher,
                saved_teacher,
                label="diagonal motion-head teacher",
            )
            incompatible = self.motion_head_teacher.load_state_dict(dict(saved_teacher), strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError("strict diagonal motion-head restore returned incompatible keys")
            self.motion_head_teacher.requires_grad_(False)
            self.motion_head_teacher.eval()
        self.motion_ema_updates = updates


__all__ = [
    "DIAGONAL_OBJECTIVE_STATE_SCHEMA",
    "DiagonalDMDLossAdapter",
    "DiagonalDMDLossResult",
]

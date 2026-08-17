"""Model-neutral SenseFlow objective with a single rollout shared across roles."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .config import SenseFlowConfig
from .contracts import (
    SenseFlowDiscriminatorAdapter,
    SenseFlowFakeScoreAdapter,
    SenseFlowGeneratorPhase,
    SenseFlowLossResult,
    SenseFlowPredictionAdapter,
    SenseFlowPreparedBatch,
    SenseFlowTeacherAdapter,
    SenseFlowTrainingBatch,
)
from .math import (
    flow_euler_step,
    flow_isg_paths,
    flow_velocity_from_clean,
    isg_loss_per_sample,
    sample_isg_midpoint,
    sample_score_sigmas,
    senseflow_adversarial_time_weight,
    senseflow_discriminator_hinge_loss,
    senseflow_distribution_gradient,
    senseflow_generator_hinge_loss,
    senseflow_proxy_loss_per_sample,
    senseflow_sigma_at_timestep,
)
from .rollout import SenseFlowAnchorRollout, simulate_senseflow_anchor


def _adapter_module(adapter: object, *, role: str) -> nn.Module:
    module = getattr(adapter, "module", None)
    if not isinstance(module, nn.Module):
        raise TypeError(f"SenseFlow {role} adapter must expose an nn.Module")
    return module


def _sample_weights(batch: SenseFlowTrainingBatch, *, device: torch.device) -> Tensor:
    if batch.sample_weights is None:
        return torch.ones((batch.batch_size,), device=device, dtype=torch.float32)
    if not isinstance(batch.sample_weights, Tensor):
        raise TypeError("SenseFlow sample_weights must be a torch.Tensor")
    weights = batch.sample_weights.to(device=device, dtype=torch.float32)
    if not bool(torch.isfinite(weights).all()) or not bool((weights >= 0).all()):
        raise ValueError("SenseFlow sample_weights must be finite and non-negative")
    if not bool(weights.sum() > 0):
        raise ValueError("SenseFlow batch must contain positive sample weight")
    return weights


def _weighted_mean(per_sample: Tensor, weights: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(per_sample, Tensor) or per_sample.shape != weights.shape:
        raise ValueError("SenseFlow objective must return one loss per weighted sample")
    numerator = (per_sample.float() * weights).sum()
    denominator = weights.sum()
    return numerator / denominator, numerator, denominator


def _normal_like(reference: Tensor, *, generator: torch.Generator) -> Tensor:
    return torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


def _sample_guidance(
    limits: tuple[float, float],
    *,
    device: torch.device,
    generator: torch.Generator,
) -> float:
    lower, upper = limits
    if lower == upper:
        return lower
    value = torch.rand((), device=device, generator=generator)
    return float((lower + (upper - lower) * value).item())


def _validate_prediction(value: object, reference: Tensor, *, role: str) -> Tensor:
    if not isinstance(value, Tensor) or value.shape != reference.shape:
        raise ValueError(f"SenseFlow {role} prediction must preserve the latent shape")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"non-finite SenseFlow {role} prediction")
    return value


class NativeSenseFlowLossAdapter:
    """Execute released flow IDA/ISG semantics without owning optimizer state."""

    def __init__(
        self,
        student: SenseFlowPredictionAdapter,
        teacher: SenseFlowTeacherAdapter,
        fake_score: SenseFlowFakeScoreAdapter,
        discriminator: SenseFlowDiscriminatorAdapter,
        config: SenseFlowConfig,
    ) -> None:
        if not isinstance(student, SenseFlowPredictionAdapter):
            raise TypeError("student must implement SenseFlowPredictionAdapter")
        if not isinstance(teacher, SenseFlowTeacherAdapter):
            raise TypeError("teacher must implement SenseFlowTeacherAdapter")
        if not isinstance(fake_score, SenseFlowFakeScoreAdapter):
            raise TypeError("fake_score must implement SenseFlowFakeScoreAdapter")
        if not isinstance(discriminator, SenseFlowDiscriminatorAdapter):
            raise TypeError("discriminator must implement SenseFlowDiscriminatorAdapter")
        if not isinstance(config, SenseFlowConfig):
            raise TypeError("config must be SenseFlowConfig")
        modules = (
            _adapter_module(student, role="student"),
            _adapter_module(teacher, role="teacher"),
            _adapter_module(fake_score, role="fake score"),
            _adapter_module(discriminator, role="discriminator"),
        )
        if len({id(module) for module in modules}) != len(modules):
            raise ValueError("SenseFlow model roles must be independently materialized")
        if any(parameter.requires_grad for parameter in modules[1].parameters()):
            raise ValueError("SenseFlow teacher parameters must be frozen")
        process_kinds = tuple(
            str(adapter.noise_process_kind).strip().lower().replace("_", "-")
            for adapter in (student, teacher, fake_score)
        )
        if set(process_kinds) != {"flow-matching"}:
            raise ValueError("native SenseFlow adapters must expose a flow-matching process")
        self.student = student
        self.teacher = teacher
        self.fake_score = fake_score
        self.discriminator = discriminator
        self.config = config
        self.generator_update_interval = config.generator_update_interval
        self.ida_decay = config.ida_decay
        self.ida_enabled = config.ida_enabled

    def loss_denominator(self, batch: SenseFlowTrainingBatch, *, role: str) -> Tensor:
        if role not in {"generator", "fake-score", "discriminator"}:
            raise ValueError(f"unsupported SenseFlow loss role: {role!r}")
        if not isinstance(batch, SenseFlowTrainingBatch):
            raise TypeError("SenseFlow loss denominator requires SenseFlowTrainingBatch")
        return _sample_weights(batch, device=batch.real_latents.device).sum()

    def _teacher_clean(
        self,
        noisy: Tensor,
        sigmas: Tensor,
        batch: SenseFlowTrainingBatch,
        *,
        guidance_scale: float,
    ) -> Tensor:
        return _validate_prediction(
            self.teacher.predict_guided_clean(
                noisy,
                sigmas,
                sample_ids=batch.sample_ids,
                conditioning=batch.conditioning,
                unconditional_conditioning=batch.unconditional_conditioning,
                guidance_scale=guidance_scale,
            ),
            noisy,
            role="teacher",
        )

    def _isg_loss(
        self,
        rollout: SenseFlowAnchorRollout,
        batch: SenseFlowTrainingBatch,
        *,
        generator: torch.Generator,
    ) -> tuple[Tensor, Mapping[str, object]]:
        schedule = self.config.schedule
        index = rollout.anchor_index
        next_sigma = schedule.next_sigma(index)
        current_timestep = schedule.timesteps[index]
        next_timestep = schedule.next_timestep(index)
        midpoint_timestep = sample_isg_midpoint(
            current_timestep,
            next_timestep,
            margin=schedule.isg_margin,
            device=rollout.anchor_sample.device,
            generator=generator,
        )
        midpoint_sigma = senseflow_sigma_at_timestep(
            midpoint_timestep,
            num_train_timesteps=schedule.num_train_timesteps,
            flow_shift=schedule.flow_shift,
            timestep_index_offset=schedule.timestep_index_offset,
        )
        midpoint_sigmas = midpoint_sigma.expand(batch.batch_size)
        next_sigmas = torch.full_like(rollout.anchor_sigmas, next_sigma)
        with torch.no_grad():
            guidance = _sample_guidance(
                self.config.isg_teacher_guidance,
                device=rollout.anchor_sample.device,
                generator=generator,
            )
            teacher_clean = self._teacher_clean(
                rollout.anchor_sample,
                rollout.anchor_sigmas,
                batch,
                guidance_scale=guidance,
            )
            teacher_velocity = flow_velocity_from_clean(
                rollout.anchor_sample,
                teacher_clean,
                rollout.anchor_sigmas,
            )
            midpoint_sample = flow_euler_step(
                rollout.anchor_sample,
                teacher_velocity,
                rollout.anchor_sigmas,
                midpoint_sigmas,
            )
            midpoint_clean = _validate_prediction(
                self.student.predict_clean(
                    midpoint_sample,
                    midpoint_sigmas,
                    sample_ids=batch.sample_ids,
                    conditioning=batch.conditioning,
                    training=False,
                ),
                midpoint_sample,
                role="ISG midpoint student",
            )
            midpoint_velocity = flow_velocity_from_clean(
                midpoint_sample,
                midpoint_clean,
                midpoint_sigmas,
            )
        anchor_velocity = flow_velocity_from_clean(
            rollout.anchor_sample,
            rollout.generated_clean,
            rollout.anchor_sigmas,
        )
        paths = flow_isg_paths(
            rollout.anchor_sample,
            teacher_velocity,
            midpoint_velocity,
            anchor_velocity,
            anchor_sigmas=rollout.anchor_sigmas,
            midpoint_sigmas=midpoint_sigmas,
            next_sigmas=next_sigmas,
        )
        losses = isg_loss_per_sample(
            paths.direct_next,
            paths.target_next,
            loss_type=self.config.isg_loss,
            epsilon=self.config.isg_epsilon,
        )
        return losses, {
            "isg_midpoint_timestep": midpoint_timestep.detach(),
            "isg_midpoint_sigma": midpoint_sigma.detach(),
            "isg_teacher_guidance": torch.tensor(
                guidance,
                device=losses.device,
                dtype=torch.float32,
            ),
            "isg_direct_target_mae": (
                paths.direct_next.detach().float() - paths.target_next.detach().float()
            ).abs().mean(),
        }

    def _distribution_matching(
        self,
        generated: Tensor,
        batch: SenseFlowTrainingBatch,
        *,
        generator: torch.Generator,
    ) -> tuple[Tensor, Mapping[str, object]]:
        score_sigmas = sample_score_sigmas(
            generated,
            sampling=self.config.score_sampling,
            minimum_timestep_fraction=self.config.score_min_timestep_fraction,
            maximum_timestep_fraction=self.config.score_max_timestep_fraction,
            flow_shift=self.config.score_flow_shift,
            generator=generator,
            num_train_timesteps=self.config.schedule.num_train_timesteps,
        )
        noise = _normal_like(generated, generator=generator)
        with torch.no_grad():
            noisy = self.student.add_noise(generated.detach(), noise, score_sigmas)
            if not isinstance(noisy, Tensor) or noisy.shape != generated.shape:
                raise ValueError("SenseFlow student add_noise must preserve the latent shape")
            fake_clean = _validate_prediction(
                self.fake_score.predict_clean(
                    noisy,
                    score_sigmas,
                    sample_ids=batch.sample_ids,
                    conditioning=batch.conditioning,
                    training=False,
                ),
                generated,
                role="fake-score",
            )
            guidance = _sample_guidance(
                self.config.dmd_teacher_guidance,
                device=generated.device,
                generator=generator,
            )
            teacher_clean = self._teacher_clean(
                noisy,
                score_sigmas,
                batch,
                guidance_scale=guidance,
            )
            gradient, normalizer = senseflow_distribution_gradient(
                generated.detach(),
                fake_clean,
                teacher_clean,
                normalization_epsilon=self.config.normalization_epsilon,
            )
        return senseflow_proxy_loss_per_sample(generated, gradient), {
            "score_sigma_mean": score_sigmas.detach().mean(),
            "dmd_normalizer_mean": normalizer.detach().mean(),
            "dmd_teacher_guidance": torch.tensor(
                guidance,
                device=generated.device,
                dtype=torch.float32,
            ),
        }

    def generator_phase(
        self,
        batch: SenseFlowTrainingBatch,
        *,
        update: bool,
        generator: torch.Generator,
    ) -> SenseFlowGeneratorPhase:
        if not isinstance(batch, SenseFlowTrainingBatch):
            raise TypeError("SenseFlow generator phase requires SenseFlowTrainingBatch")
        if not isinstance(update, bool):
            raise TypeError("update must be bool")
        rollout = simulate_senseflow_anchor(
            self.student,
            batch,
            self.config,
            generator=generator,
            training=update,
        )
        prepared = SenseFlowPreparedBatch(
            batch=batch,
            generated_clean=rollout.generated_clean.detach(),
            anchor_sigmas=rollout.anchor_sigmas.detach(),
            anchor_index=rollout.anchor_index,
            anchor_timestep=rollout.anchor_timestep,
            backward_simulation=rollout.backward_simulation,
        )
        if not update:
            return SenseFlowGeneratorPhase(prepared=prepared, loss_result=None)

        generated = rollout.generated_clean
        weights = _sample_weights(batch, device=generated.device)
        zero = torch.zeros((batch.batch_size,), device=generated.device, dtype=torch.float32)
        dmd = zero
        dmd_metrics: Mapping[str, object] = {}
        if self.config.distribution_matching_weight > 0:
            dmd, dmd_metrics = self._distribution_matching(
                generated,
                batch,
                generator=generator,
            )
        isg = zero
        isg_metrics: Mapping[str, object] = {}
        if self.config.isg_weight > 0:
            isg, isg_metrics = self._isg_loss(rollout, batch, generator=generator)
        adversarial = zero
        adversarial_proxy = torch.zeros((), device=generated.device, dtype=torch.float32)
        if self.config.generator_adversarial_weight > 0:
            fake_logits = self.discriminator.logits(
                generated,
                sample_ids=batch.sample_ids,
                conditioning=batch.conditioning,
                reference_latents=batch.real_latents,
                training=False,
            )
            if not isinstance(fake_logits, Tensor):
                raise TypeError("SenseFlow discriminator logits must be a torch.Tensor")
            adversarial = senseflow_generator_hinge_loss(fake_logits)
            adversarial_scales = torch.full_like(
                rollout.anchor_sigmas,
                self.config.schedule.adversarial_scale(rollout.anchor_index),
            )
            adversarial = adversarial * senseflow_adversarial_time_weight(adversarial_scales)
            adversarial_mean = _weighted_mean(adversarial, weights)[0]
            adversarial_gradient = torch.autograd.grad(
                adversarial_mean,
                generated,
                create_graph=False,
                retain_graph=False,
            )[0]
            adversarial_proxy = (generated.float() * adversarial_gradient.detach()).sum()

        combined = (
            self.config.distribution_matching_weight * dmd
            + self.config.isg_weight * isg
            + self.config.generator_adversarial_weight * adversarial
        )
        actual_loss, numerator, denominator = _weighted_mean(combined, weights)
        dmd_mean = _weighted_mean(dmd, weights)[0] * self.config.distribution_matching_weight
        isg_mean = _weighted_mean(isg, weights)[0] * self.config.isg_weight
        # The reported scalar is exact; only student-owned paths contribute gradients.
        loss = (
            actual_loss.detach()
            + dmd_mean
            - dmd_mean.detach()
            + isg_mean
            - isg_mean.detach()
            + self.config.generator_adversarial_weight * adversarial_proxy
            - (self.config.generator_adversarial_weight * adversarial_proxy).detach()
        )
        metrics = {
            "loss_numerator": numerator.detach(),
            "loss_denominator": denominator.detach(),
            "distribution_matching": _weighted_mean(dmd, weights)[0].detach(),
            "isg": _weighted_mean(isg, weights)[0].detach(),
            "generator_adversarial": _weighted_mean(adversarial, weights)[0].detach(),
            "anchor_index": torch.tensor(rollout.anchor_index, device=loss.device),
            "anchor_timestep": torch.tensor(rollout.anchor_timestep, device=loss.device),
            "anchor_sigma": rollout.anchor_sigmas[0].detach(),
            "backward_simulation": torch.tensor(
                int(rollout.backward_simulation),
                device=loss.device,
                dtype=torch.int64,
            ),
            **dmd_metrics,
            **isg_metrics,
        }
        return SenseFlowGeneratorPhase(
            prepared=prepared,
            loss_result=SenseFlowLossResult(loss=loss, metrics=metrics),
        )

    def fake_score_loss(
        self,
        prepared: SenseFlowPreparedBatch,
        *,
        generator: torch.Generator,
    ) -> SenseFlowLossResult:
        if not isinstance(prepared, SenseFlowPreparedBatch):
            raise TypeError("prepared must be SenseFlowPreparedBatch")
        generated = prepared.generated_clean
        batch = prepared.batch
        sigmas = sample_score_sigmas(
            generated,
            sampling=self.config.fake_score_sampling,
            minimum_timestep_fraction=self.config.fake_score_min_timestep_fraction,
            maximum_timestep_fraction=self.config.fake_score_max_timestep_fraction,
            flow_shift=self.config.score_flow_shift,
            generator=generator,
            num_train_timesteps=self.config.schedule.num_train_timesteps,
        )
        noise = _normal_like(generated, generator=generator)
        noisy = self.fake_score.add_noise(generated, noise, sigmas)
        if not isinstance(noisy, Tensor) or noisy.shape != generated.shape:
            raise ValueError("SenseFlow fake-score add_noise must preserve the latent shape")
        per_sample = self.fake_score.denoising_loss_per_sample(
            generated,
            noisy,
            noise,
            sigmas,
            sample_ids=batch.sample_ids,
            conditioning=batch.conditioning,
            training=True,
        )
        if not isinstance(per_sample, Tensor) or per_sample.shape != (batch.batch_size,):
            raise ValueError("fake-score denoising loss must return shape [B]")
        per_sample = per_sample.float() * self.config.fake_score_weight
        weights = _sample_weights(batch, device=generated.device)
        loss, numerator, denominator = _weighted_mean(per_sample, weights)
        return SenseFlowLossResult(
            loss=loss,
            metrics={
                "loss_numerator": numerator.detach(),
                "loss_denominator": denominator.detach(),
                "denoising": _weighted_mean(per_sample, weights)[0].detach(),
                "score_sigma_mean": sigmas.detach().mean(),
                "anchor_index": torch.tensor(prepared.anchor_index, device=loss.device),
            },
        )

    def discriminator_loss(self, prepared: SenseFlowPreparedBatch) -> SenseFlowLossResult:
        if not isinstance(prepared, SenseFlowPreparedBatch):
            raise TypeError("prepared must be SenseFlowPreparedBatch")
        batch = prepared.batch
        generated = prepared.generated_clean
        fake_logits = self.discriminator.logits(
            generated,
            sample_ids=batch.sample_ids,
            conditioning=batch.conditioning,
            reference_latents=batch.real_latents,
            training=True,
        )
        real_logits = self.discriminator.logits(
            batch.real_latents,
            sample_ids=batch.real_sample_ids,
            conditioning=batch.real_conditioning,
            reference_latents=batch.real_latents,
            training=True,
        )
        if not isinstance(fake_logits, Tensor) or not isinstance(real_logits, Tensor):
            raise TypeError("SenseFlow discriminator logits must be torch.Tensor values")
        per_sample = senseflow_discriminator_hinge_loss(real_logits, fake_logits)
        per_sample = (
            per_sample
            * senseflow_adversarial_time_weight(
                torch.full_like(
                    prepared.anchor_sigmas,
                    self.config.schedule.adversarial_scale(prepared.anchor_index),
                )
            )
            * self.config.discriminator_weight
        )
        weights = _sample_weights(batch, device=generated.device)
        loss, numerator, denominator = _weighted_mean(per_sample, weights)
        return SenseFlowLossResult(
            loss=loss,
            metrics={
                "loss_numerator": numerator.detach(),
                "loss_denominator": denominator.detach(),
                "hinge": _weighted_mean(per_sample, weights)[0].detach(),
                "real_logit_mean": real_logits.detach().float().mean(),
                "fake_logit_mean": fake_logits.detach().float().mean(),
                "anchor_index": torch.tensor(prepared.anchor_index, device=loss.device),
            },
        )


__all__ = ["NativeSenseFlowLossAdapter"]

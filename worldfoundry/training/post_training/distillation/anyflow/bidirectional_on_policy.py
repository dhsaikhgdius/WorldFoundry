"""Native full-video AnyFlow on-policy distillation objectives."""

from __future__ import annotations

from dataclasses import asdict

import torch
from torch import Tensor

from ...shared.distributed import PostTrainingParallelContext
from .bidirectional_pretrain import NativeAnyFlowBidirectionalPretrainLossAdapter
from .config import (
    AnyFlowBidirectionalOnPolicyConfig,
    AnyFlowBidirectionalPretrainConfig,
)
from .contracts import (
    AnyFlowBidirectionalAdapter,
    AnyFlowLossResult,
    AnyFlowScoreAdapter,
    AnyFlowTrainingBatch,
)
from .dmd_objective import anyflow_dmd_loss, anyflow_fake_score_loss
from .rollout import anyflow_bidirectional_rollout, sample_rollout_choice
from .synchronization import AnyFlowDecisionRNG


class NativeAnyFlowBidirectionalOnPolicyLossAdapter:
    """Full-video rollout DMD, FlowMap cotraining, and fresh fake score."""

    fake_score_decision_draws = 2

    def __init__(
        self,
        student: AnyFlowBidirectionalAdapter,
        real_score: AnyFlowScoreAdapter,
        fake_score: AnyFlowScoreAdapter,
        config: AnyFlowBidirectionalOnPolicyConfig,
        decisions: AnyFlowDecisionRNG,
        *,
        parallel_context: PostTrainingParallelContext | None = None,
    ) -> None:
        if not isinstance(student, AnyFlowBidirectionalAdapter):
            raise TypeError("student must implement AnyFlowBidirectionalAdapter")
        if not isinstance(real_score, AnyFlowScoreAdapter):
            raise TypeError("real_score must implement AnyFlowScoreAdapter")
        if not isinstance(fake_score, AnyFlowScoreAdapter):
            raise TypeError("fake_score must implement AnyFlowScoreAdapter")
        if not isinstance(config, AnyFlowBidirectionalOnPolicyConfig):
            raise TypeError("config must be AnyFlowBidirectionalOnPolicyConfig")
        modules = (student.module, real_score.module, fake_score.module)
        if len({id(module) for module in modules}) != 3:
            raise ValueError("AnyFlow student, real score, and fake score must be distinct")
        if not isinstance(decisions, AnyFlowDecisionRNG):
            raise TypeError("decisions must be AnyFlowDecisionRNG")
        self.student = student
        self.real_score = real_score
        self.fake_score = fake_score
        self.config = config
        self.decisions = decisions
        self.discriminator_update_ratio = config.discriminator_update_ratio
        self.parallel_context = parallel_context or PostTrainingParallelContext.current()
        self.pixel = NativeAnyFlowBidirectionalPretrainLossAdapter(
            student,
            AnyFlowBidirectionalPretrainConfig(
                flow_map=config.flow_map,
                image_conditioning_probability=(config.image_conditioning_probability),
                conditioning_dropout_probability=(config.conditioning_dropout_probability),
            ),
            decisions,
            parallel_context=self.parallel_context,
        )
        self.generator_decision_draws = 2 + (
            self.pixel.decision_draws_per_student_loss if config.cotrain_flowmap else 0
        )
        self.config_state = asdict(config)

    def loss_denominator(self, batch: AnyFlowTrainingBatch, *, role: str) -> Tensor:
        if role not in {"generator", "fake-score"}:
            raise ValueError(f"unsupported AnyFlow on-policy role: {role!r}")
        if batch.batch_size < self.config.dmd_batch_size:
            raise ValueError("AnyFlow local batch is smaller than the configured DMD batch size")
        denominator = batch.batch_size if role == "generator" else self.config.dmd_batch_size
        return torch.tensor(
            float(denominator),
            device=batch.clean_latents.device,
            dtype=torch.float32,
        )

    def _rollout(
        self,
        batch: AnyFlowTrainingBatch,
        *,
        generator: torch.Generator | None,
        differentiable: bool,
    ) -> tuple[Tensor, int, int]:
        reference = batch.clean_latents
        if not isinstance(reference, Tensor):
            raise TypeError("AnyFlow clean_latents must be torch.Tensor")
        choice = sample_rollout_choice(
            self.config,
            self.decisions,
            reference=reference,
        )
        initial_noise = torch.randn(
            reference.shape,
            device=reference.device,
            dtype=reference.dtype,
            generator=generator,
        )
        generated = anyflow_bidirectional_rollout(
            self.student,
            batch,
            initial_noise,
            choice,
            self.config,
            differentiable=differentiable,
        )
        return generated, choice.step_count, choice.gradient_interval

    def generator_loss(
        self,
        batch: AnyFlowTrainingBatch,
        *,
        generator: object | None = None,
    ) -> AnyFlowLossResult:
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be torch.Generator or None")
        dmd_batch = batch.head(self.config.dmd_batch_size)
        generated, step_count, gradient_interval = self._rollout(
            dmd_batch,
            generator=generator,
            differentiable=True,
        )
        dmd = anyflow_dmd_loss(
            generated,
            dmd_batch,
            self.real_score,
            self.fake_score,
            self.config.flow_map,
            dmd_weight=self.config.dmd_weight,
            real_guidance_scale=self.config.real_guidance_scale,
            minimum_timestep=self.config.dmd_min_timestep,
            maximum_timestep=self.config.dmd_max_timestep,
            generator=generator,
        )
        loss = dmd.loss
        metrics: dict[str, object] = {
            "loss_denominator": torch.tensor(
                float(batch.batch_size),
                device=loss.device,
                dtype=torch.float32,
            ),
            "dmd_loss": dmd.loss.detach().float(),
            "rollout_steps": step_count,
            "gradient_interval": gradient_interval,
            "dmd_time_mean": dmd.raw_time_mean,
            "normalizer_mean": dmd.normalizer_mean,
            "dmd_batch_size": dmd_batch.batch_size,
        }
        if self.config.cotrain_flowmap:
            pixel = self.pixel.student_loss(batch, generator=generator)
            loss = loss + pixel.loss
            metrics["pixel_loss"] = pixel.loss.detach().float()
            metrics["pixel"] = dict(pixel.metrics)
        if not bool(torch.isfinite(loss.detach())):
            raise FloatingPointError("non-finite AnyFlow generator loss")
        return AnyFlowLossResult(loss=loss, metrics=metrics)

    def fake_score_loss(
        self,
        batch: AnyFlowTrainingBatch,
        *,
        generator: object | None = None,
    ) -> AnyFlowLossResult:
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be torch.Generator or None")
        dmd_batch = batch.head(self.config.dmd_batch_size)
        with torch.no_grad():
            generated, step_count, gradient_interval = self._rollout(
                dmd_batch,
                generator=generator,
                differentiable=False,
            )
        score = anyflow_fake_score_loss(
            generated,
            dmd_batch,
            self.fake_score,
            self.config.flow_map,
            logit_mean=self.config.fake_score_logit_mean,
            logit_std=self.config.fake_score_logit_std,
            minimum_timestep=self.config.dmd_min_timestep,
            maximum_timestep=self.config.dmd_max_timestep,
            generator=generator,
        )
        if not bool(torch.isfinite(score.loss.detach())):
            raise FloatingPointError("non-finite AnyFlow fake-score loss")
        return AnyFlowLossResult(
            loss=score.loss,
            metrics={
                "loss_denominator": torch.tensor(
                    float(dmd_batch.batch_size),
                    device=score.loss.device,
                    dtype=torch.float32,
                ),
                "rollout_steps": step_count,
                "sampled_gradient_interval": gradient_interval,
                "fake_score_time_mean": score.raw_time_mean,
                "dmd_batch_size": dmd_batch.batch_size,
            },
        )


__all__ = ["NativeAnyFlowBidirectionalOnPolicyLossAdapter"]

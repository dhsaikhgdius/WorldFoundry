"""Native FAR FlowMap pretraining objective."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from ...shared.distributed import PostTrainingParallelContext
from .conditioning import apply_conditioning_dropout
from .config import AnyFlowPretrainConfig
from .contracts import (
    AnyFlowFARAdapter,
    AnyFlowLossResult,
    AnyFlowTrainingBatch,
)
from .flowmap_objective import flowmap_regression_loss
from .math import (
    flowmap_interpolate,
    fused_guidance_prediction,
    sample_logit_normal_time,
    shift_flowmap_time,
)
from .synchronization import AnyFlowDecisionRNG


def _randn_like(reference: Tensor, *, generator: torch.Generator | None) -> Tensor:
    return torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


def _prediction(
    student: AnyFlowFARAdapter,
    noisy: Tensor,
    timesteps: Tensor,
    destinations: Tensor,
    *,
    clean: Tensor,
    context: Tensor,
    config: AnyFlowPretrainConfig,
    sampled_chunk_count: int,
    batch: AnyFlowTrainingBatch,
    conditioning: Mapping[str, object],
    training: bool,
    branch: str,
) -> Tensor:
    value = student.predict_flow_map(
        noisy,
        timesteps,
        destinations,
        clean_latents=clean,
        context_latents=context,
        partition=config.far.partition,
        sampled_chunk_count=sampled_chunk_count,
        sample_ids=batch.sample_ids,
        conditioning=conditioning,
        training=training,
        branch=branch,
    )
    if not isinstance(value, Tensor) or value.shape != noisy.shape:
        raise ValueError("AnyFlow FAR predictions must match the target latent shape")
    return value


class NativeAnyFlowPretrainLossAdapter:
    """FlowMap interval mixture, central target, beta08, and global balancing."""

    decision_draws_per_student_loss = 3

    def __init__(
        self,
        student: AnyFlowFARAdapter,
        config: AnyFlowPretrainConfig,
        decisions: AnyFlowDecisionRNG,
        *,
        parallel_context: PostTrainingParallelContext | None = None,
    ) -> None:
        if not isinstance(student, AnyFlowFARAdapter):
            raise TypeError("student must implement AnyFlowFARAdapter")
        if not isinstance(config, AnyFlowPretrainConfig):
            raise TypeError("config must be AnyFlowPretrainConfig")
        if not isinstance(decisions, AnyFlowDecisionRNG):
            raise TypeError("decisions must be AnyFlowDecisionRNG")
        self.student = student
        self.config = config
        self.decisions = decisions
        self.parallel_context = parallel_context or PostTrainingParallelContext.current()
        self.config_digest = config.digest

    def loss_denominator(self, batch: AnyFlowTrainingBatch, *, role: str) -> Tensor:
        if role != "student":
            raise ValueError(f"unsupported AnyFlow pretrain role: {role!r}")
        return torch.tensor(
            float(batch.batch_size),
            device=batch.clean_latents.device,
            dtype=torch.float32,
        )

    def _sampled_chunk_count(self, reference: Tensor) -> int:
        partition = self.config.far.partition
        long_context = self.decisions.bernoulli(
            self.config.far.long_context_training_ratio,
            reference=reference,
        )
        final = partition.chunk_count if long_context else partition.full_chunk_limit
        return self.decisions.choice(
            tuple(range(partition.full_chunk_limit, final + 1)),
            reference=reference,
        )

    def _bidirectional_loss(
        self,
        batch: AnyFlowTrainingBatch,
        clean: Tensor,
        *,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Released FAR auxiliary: full-video logit-normal RF regression."""

        batch_size, frames = int(clean.shape[0]), int(clean.shape[2])
        noise = _randn_like(clean, generator=generator)
        raw_time = sample_logit_normal_time(
            batch_size,
            device=clean.device,
            mean=0.0,
            std=1.0,
            generator=generator,
        )
        shifted = shift_flowmap_time(
            raw_time,
            self.config.flow_map.timestep_shift,
        )
        model_time = shifted[:, None].expand(batch_size, frames) * float(self.config.flow_map.num_train_timesteps)
        noisy = flowmap_interpolate(clean, noise, shifted)
        conditional = self.student.predict_bidirectional_velocity(
            noisy,
            model_time,
            sample_ids=batch.sample_ids,
            conditioning=batch.conditioning,
            training=True,
            branch="positive",
        )
        if not isinstance(conditional, Tensor) or conditional.shape != noisy.shape:
            raise ValueError("AnyFlow bidirectional prediction must preserve the latent shape")
        guidance = self.config.flow_map.fused_guidance_scale
        if guidance == 1.0:
            prediction = conditional
        else:
            with torch.no_grad():
                unconditional = self.student.predict_bidirectional_velocity(
                    noisy,
                    model_time,
                    sample_ids=batch.sample_ids,
                    conditioning=batch.unconditional_conditioning,
                    training=False,
                    branch="negative",
                )
            if not isinstance(unconditional, Tensor) or unconditional.shape != noisy.shape:
                raise ValueError("AnyFlow bidirectional prediction must preserve the latent shape")
            prediction = fused_guidance_prediction(
                conditional,
                unconditional,
                guidance,
            )
        return (prediction.float() - (noise - clean).float()).square().flatten(1).mean()

    def student_loss(
        self,
        batch: AnyFlowTrainingBatch,
        *,
        generator: object | None = None,
    ) -> AnyFlowLossResult:
        if not isinstance(batch, AnyFlowTrainingBatch):
            raise TypeError("batch must be AnyFlowTrainingBatch")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be torch.Generator or None")
        full_clean_video = batch.clean_latents
        if not isinstance(full_clean_video, Tensor):
            raise TypeError("AnyFlow clean_latents must be torch.Tensor")
        dropout = apply_conditioning_dropout(
            batch,
            self.config.conditioning_dropout_probability,
            generator=generator,
        )
        training_batch = dropout.batch
        sampled_chunks = self._sampled_chunk_count(full_clean_video)
        selected = self.config.far.partition.prefix(sampled_chunks)
        sampled_frames = sum(selected)
        if int(full_clean_video.shape[2]) < sampled_frames:
            raise ValueError("AnyFlow batch has fewer frames than the sampled FAR prefix")
        clean_video = full_clean_video[:, :, :sampled_frames]
        context_frames, target_frames = self.config.far.partition.context_target_frames(sampled_chunks)
        context = clean_video[:, :, :context_frames]
        clean = clean_video[:, :, context_frames : context_frames + target_frames]

        def causal_prediction(
            noisy: Tensor,
            timesteps: Tensor,
            destinations: Tensor,
            conditioning: Mapping[str, object],
            training: bool,
            branch: str,
        ) -> Tensor:
            return _prediction(
                self.student,
                noisy,
                timesteps,
                destinations,
                clean=clean,
                context=context,
                config=self.config,
                sampled_chunk_count=sampled_chunks,
                batch=training_batch,
                conditioning=conditioning,
                training=training,
                branch=branch,
            )

        causal = flowmap_regression_loss(
            clean,
            training_batch,
            self.config.flow_map,
            causal_prediction,
            parallel_context=self.parallel_context,
            generator=generator,
        )
        causal_loss = causal.loss
        use_bidirectional = self.decisions.bernoulli(
            self.config.bidirectional_modeling_probability,
            reference=full_clean_video,
        )
        if use_bidirectional:
            bidirectional_loss = self._bidirectional_loss(
                training_batch,
                full_clean_video,
                generator=generator,
            )
        else:
            bidirectional_loss = causal_loss.new_zeros(())
        loss = causal_loss + bidirectional_loss
        if not bool(torch.isfinite(loss.detach())):
            raise FloatingPointError("non-finite AnyFlow pretraining loss")
        return AnyFlowLossResult(
            loss=loss,
            metrics={
                "loss_denominator": torch.tensor(
                    float(batch.batch_size),
                    device=loss.device,
                    dtype=torch.float32,
                ),
                "causal_loss": causal_loss.detach().float(),
                "bidirectional_loss": bidirectional_loss.detach().float(),
                "bidirectional_applied": use_bidirectional,
                "conditioning_dropped_samples": dropout.mask.sum().detach(),
                "sampled_chunk_count": sampled_chunks,
                "context_frames": context_frames,
                "target_frames": target_frames,
                **dict(causal.metrics),
            },
        )


__all__ = ["NativeAnyFlowPretrainLossAdapter"]

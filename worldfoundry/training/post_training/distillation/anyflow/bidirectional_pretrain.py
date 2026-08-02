"""Native full-video AnyFlow FlowMap pretraining."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from ...shared.distributed import PostTrainingParallelContext
from .conditioning import apply_conditioning_dropout
from .config import AnyFlowBidirectionalPretrainConfig
from .contracts import (
    AnyFlowBidirectionalAdapter,
    AnyFlowLossResult,
    AnyFlowTrainingBatch,
)
from .flowmap_objective import flowmap_regression_loss
from .synchronization import AnyFlowDecisionRNG


class NativeAnyFlowBidirectionalPretrainLossAdapter:
    """Full-video interval-mixture objective from the released Wan trainer."""

    decision_draws_per_student_loss = 1

    def __init__(
        self,
        student: AnyFlowBidirectionalAdapter,
        config: AnyFlowBidirectionalPretrainConfig,
        decisions: AnyFlowDecisionRNG,
        *,
        parallel_context: PostTrainingParallelContext | None = None,
    ) -> None:
        if not isinstance(student, AnyFlowBidirectionalAdapter):
            raise TypeError("student must implement AnyFlowBidirectionalAdapter")
        if not isinstance(config, AnyFlowBidirectionalPretrainConfig):
            raise TypeError("config must be AnyFlowBidirectionalPretrainConfig")
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
        clean = batch.clean_latents
        if not isinstance(clean, Tensor):
            raise TypeError("AnyFlow clean_latents must be torch.Tensor")
        dropout = apply_conditioning_dropout(
            batch,
            self.config.conditioning_dropout_probability,
            generator=generator,
        )
        training_batch = dropout.batch
        condition_first_frame = self.decisions.bernoulli(
            self.config.image_conditioning_probability,
            reference=clean,
        )

        def prediction(
            noisy: Tensor,
            timesteps: Tensor,
            destinations: Tensor,
            conditioning: Mapping[str, object],
            training: bool,
            branch: str,
        ) -> Tensor:
            value = self.student.predict_flow_map(
                noisy,
                timesteps,
                destinations,
                sample_ids=training_batch.sample_ids,
                conditioning=conditioning,
                training=training,
                branch=branch,
            )
            if not isinstance(value, Tensor) or value.shape != noisy.shape:
                raise ValueError("AnyFlow bidirectional prediction must preserve the latent shape")
            return value

        result = flowmap_regression_loss(
            clean,
            training_batch,
            self.config.flow_map,
            prediction,
            parallel_context=self.parallel_context,
            generator=generator,
            condition_first_frame=condition_first_frame,
        )
        return AnyFlowLossResult(
            loss=result.loss,
            metrics={
                "loss_denominator": torch.tensor(
                    float(batch.batch_size),
                    device=result.loss.device,
                    dtype=torch.float32,
                ),
                "conditioning_dropped_samples": dropout.mask.sum().detach(),
                **dict(result.metrics),
            },
        )


__all__ = ["NativeAnyFlowBidirectionalPretrainLossAdapter"]

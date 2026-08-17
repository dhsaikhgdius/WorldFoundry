"""AnyFlow checkpoint selection, native materialization, and DDP wrapping."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.optimizations import (
    AttentionBackend,
    RuntimePolicy,
)
from worldfoundry.core.attention.chunk_partition import TemporalChunkPartition
from worldfoundry.training.distributed.parallel import DistributedTrainingContext
from worldfoundry.training.models.anyflow import (
    ANYFLOW_BIDIRECTIONAL_WAN_SMALL_CHECKPOINT,
    ANYFLOW_FAR_WAN_SMALL_CHECKPOINT,
    NativeAnyFlowModelMaterializer,
)
from worldfoundry.training.post_training.distillation.anyflow.adapters import (
    NativeAnyFlowBidirectionalAdapter,
    NativeAnyFlowFARAdapter,
    NativeAnyFlowScoreAdapter,
)
from worldfoundry.training.post_training.shared.building import resolve_tensor_dtype
from worldfoundry.training.post_training.shared.role_checkpoints import (
    ResolvedRoleCheckpoint,
    resolve_role_checkpoint,
)
from worldfoundry.training.recipes.post_training.algorithms.anyflow import (
    AnyFlowBidirectionalOnPolicyAlgorithmSpec,
    AnyFlowBidirectionalPretrainAlgorithmSpec,
    AnyFlowFAROnPolicyAlgorithmSpec,
    AnyFlowFARPretrainAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

AnyFlowStudent = NativeAnyFlowFARAdapter | NativeAnyFlowBidirectionalAdapter
AnyFlowAlgorithm = (
    AnyFlowFARPretrainAlgorithmSpec
    | AnyFlowBidirectionalPretrainAlgorithmSpec
    | AnyFlowFAROnPolicyAlgorithmSpec
    | AnyFlowBidirectionalOnPolicyAlgorithmSpec
)


def far_partition(algorithm: AnyFlowAlgorithm) -> TemporalChunkPartition | None:
    if not isinstance(
        algorithm,
        (AnyFlowFARPretrainAlgorithmSpec, AnyFlowFAROnPolicyAlgorithmSpec),
    ):
        return None
    return TemporalChunkPartition(
        chunks=algorithm.far.chunk_partition,
        full_chunk_limit=algorithm.far.full_chunk_limit,
        patch_size=algorithm.far.patch_size,
        compressed_patch_size=algorithm.far.compressed_patch_size,
    )


class AnyFlowTrainableRoles(nn.Module):
    """The optimizer-owned role set stored by DCP."""

    def __init__(self, student: nn.Module, fake_score: nn.Module | None = None) -> None:
        super().__init__()
        self.student = student
        if fake_score is not None:
            self.fake_score = fake_score


@dataclass(frozen=True, slots=True)
class AnyFlowRoleBundle:
    student: AnyFlowStudent
    student_checkpoint: ResolvedRoleCheckpoint
    real_score: NativeAnyFlowScoreAdapter | None = None
    real_score_checkpoint: ResolvedRoleCheckpoint | None = None
    fake_score: NativeAnyFlowScoreAdapter | None = None
    fake_score_checkpoint: ResolvedRoleCheckpoint | None = None

    def checkpoint_identity(self) -> dict[str, object]:
        result = {"student": self.student_checkpoint.to_dict()}
        if self.real_score_checkpoint is not None:
            result["real_score"] = self.real_score_checkpoint.to_dict()
        if self.fake_score_checkpoint is not None:
            result["fake_score"] = self.fake_score_checkpoint.to_dict()
        return result

    def trainable_model(self) -> AnyFlowTrainableRoles:
        fake_module = None if self.fake_score is None else self.fake_score.module
        return AnyFlowTrainableRoles(self.student.module, fake_module)


def _resolved_checkpoint(
    *,
    role: str,
    reference: str,
    default: CheckpointSpec,
    overrides: Mapping[str, CheckpointSpec],
) -> ResolvedRoleCheckpoint:
    return resolve_role_checkpoint(
        role=role,
        reference=reference,
        native_default=default,
        local_override=overrides.get(role),
    )


def _ddp(
    module: nn.Module,
    context: DistributedTrainingContext | None,
) -> nn.Module:
    if context is None:
        return module
    return DistributedDataParallel(
        module,
        device_ids=[context.local_rank],
        output_device=context.local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True,
    )


def materialize_anyflow_roles(
    recipe: PostTrainingRecipe,
    *,
    algorithm: AnyFlowAlgorithm,
    device: torch.device,
    distributed_context: DistributedTrainingContext | None,
    checkpoint_overrides: Mapping[str, CheckpointSpec] | None = None,
    force_torch_attention: bool = True,
) -> AnyFlowRoleBundle:
    """Load the independent roles required by one of the four AnyFlow modes."""

    overrides = dict(checkpoint_overrides or {})
    dtype = resolve_tensor_dtype(recipe.runtime.param_dtype)
    policy = RuntimePolicy(
        device=device,
        dtype=dtype,
        attention=(AttentionBackend.TORCH if force_torch_attention else AttentionBackend.AUTO),
    )
    materializer = NativeAnyFlowModelMaterializer()
    partition = far_partition(algorithm)
    student_default = (
        ANYFLOW_FAR_WAN_SMALL_CHECKPOINT
        if partition is not None
        else ANYFLOW_BIDIRECTIONAL_WAN_SMALL_CHECKPOINT
    )
    student_checkpoint = _resolved_checkpoint(
        role="student",
        reference=recipe.model.checkpoint,
        default=student_default,
        overrides=overrides,
    )
    if partition is None:
        student: AnyFlowStudent = materializer.bidirectional_student(
            student_checkpoint.checkpoint,
            checkpoint_identity=recipe.model.checkpoint,
            policy=policy,
            gradient_checkpointing=recipe.runtime.activation_checkpoint == "full",
        )
    else:
        student = materializer.far_student(
            student_checkpoint.checkpoint,
            checkpoint_identity=recipe.model.checkpoint,
            partition=partition,
            policy=policy,
            gradient_checkpointing=recipe.runtime.activation_checkpoint == "full",
        )
    student.module = _ddp(student.module, distributed_context)

    if not isinstance(
        algorithm,
        (AnyFlowFAROnPolicyAlgorithmSpec, AnyFlowBidirectionalOnPolicyAlgorithmSpec),
    ):
        return AnyFlowRoleBundle(
            student=student,
            student_checkpoint=student_checkpoint,
        )

    real_checkpoint = _resolved_checkpoint(
        role="real-score",
        reference=algorithm.real_score_checkpoint,
        default=ANYFLOW_BIDIRECTIONAL_WAN_SMALL_CHECKPOINT,
        overrides=overrides,
    )
    fake_checkpoint = _resolved_checkpoint(
        role="fake-score",
        reference=algorithm.fake_score_checkpoint,
        default=ANYFLOW_BIDIRECTIONAL_WAN_SMALL_CHECKPOINT,
        overrides=overrides,
    )
    real_score = materializer.real_score(
        real_checkpoint.checkpoint,
        checkpoint_identity=algorithm.real_score_checkpoint,
        policy=policy,
    )
    fake_score = materializer.fake_score(
        fake_checkpoint.checkpoint,
        checkpoint_identity=algorithm.fake_score_checkpoint,
        policy=policy,
        gradient_checkpointing=recipe.runtime.activation_checkpoint == "full",
    )
    fake_score.module = _ddp(fake_score.module, distributed_context)
    return AnyFlowRoleBundle(
        student=student,
        student_checkpoint=student_checkpoint,
        real_score=real_score,
        real_score_checkpoint=real_checkpoint,
        fake_score=fake_score,
        fake_score_checkpoint=fake_checkpoint,
    )


__all__ = [
    "AnyFlowAlgorithm",
    "AnyFlowRoleBundle",
    "AnyFlowStudent",
    "AnyFlowTrainableRoles",
    "far_partition",
    "materialize_anyflow_roles",
]

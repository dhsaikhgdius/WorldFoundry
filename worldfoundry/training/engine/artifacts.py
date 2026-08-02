"""Distributed-safe run-directory creation and trained-model export."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.tensor import DTensor

from worldfoundry.training.distributed.parallel import DistributedTrainingContext
from worldfoundry.training.tuning.full_model import FullModelArtifact, save_full_model
from worldfoundry.training.tuning.peft import (
    PeftAdapterArtifact,
    PeftLoraApplication,
    save_peft_adapter,
)


def create_run_directory(
    destination: Path,
    context: DistributedTrainingContext | None,
) -> None:
    """Create an exclusive output directory consistently across all ranks."""

    if context is None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            destination.mkdir()
        except FileExistsError as error:
            raise FileExistsError(f"post-training run output already exists: {destination}") from error
        return
    result: list[object] = [None]
    if context.is_coordinator:
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.mkdir()
        except Exception as error:  # noqa: BLE001 - broadcast coordinator failure.
            result[0] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(result, src=0)
    if result[0] is not None:
        raise FileExistsError(f"post-training run output could not be created: {destination}: {result[0]}")
    context.barrier()


def export_peft_application(
    application: PeftLoraApplication,
    destination: Path,
    *,
    metadata: Mapping[str, object],
    distributed_context: DistributedTrainingContext | None,
    role: str,
) -> PeftAdapterArtifact:
    """Export PEFT state, gathering a full CPU state on distributed runs."""

    if distributed_context is None:
        return save_peft_adapter(application, destination, metadata=metadata)
    model_state = get_model_state_dict(
        application.model,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            ignore_frozen_params=True,
            strict=False,
        ),
    )
    result: list[object] = [None]
    if distributed_context.is_coordinator:
        try:
            if any(isinstance(value, DTensor) for value in model_state.values()):
                raise RuntimeError(f"FSDP2 {role} export left DTensor values on rank zero")
            artifact = save_peft_adapter(
                application,
                destination,
                metadata=metadata,
                model_state_dict=model_state,
            )
            result[0] = {
                "path": str(artifact.path),
                "manifest_sha256": artifact.manifest_sha256,
                "file_digests": dict(artifact.file_digests),
                "metadata": dict(artifact.metadata),
            }
        except Exception as error:  # noqa: BLE001 - broadcast rank-zero failure.
            result[0] = {"error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(result, src=0)
    payload = result[0]
    if not isinstance(payload, Mapping) or "error" in payload:
        raise RuntimeError(f"FSDP2 {role} export failed: {payload}")
    return PeftAdapterArtifact(
        path=Path(str(payload["path"])),
        manifest_sha256=str(payload["manifest_sha256"]),
        file_digests=dict(payload["file_digests"]),
        metadata=dict(payload["metadata"]),
    )


def export_full_model(
    model: nn.Module,
    destination: Path,
    *,
    metadata: Mapping[str, object],
    distributed_context: DistributedTrainingContext | None,
    role: str,
    max_shard_size_bytes: int,
) -> FullModelArtifact:
    """Export full native state, gathering a full CPU state when distributed."""

    if distributed_context is None:
        return save_full_model(
            model,
            destination,
            metadata=metadata,
            max_shard_size_bytes=max_shard_size_bytes,
        )
    model_state = get_model_state_dict(
        model,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            ignore_frozen_params=False,
            strict=False,
        ),
    )
    result: list[object] = [None]
    if distributed_context.is_coordinator:
        try:
            if any(isinstance(value, DTensor) for value in model_state.values()):
                raise RuntimeError(f"FSDP2 {role} full export left DTensor values on rank zero")
            artifact = save_full_model(
                model,
                destination,
                metadata=metadata,
                model_state_dict=model_state,
                max_shard_size_bytes=max_shard_size_bytes,
            )
            result[0] = {
                "path": str(artifact.path),
                "manifest_sha256": artifact.manifest_sha256,
                "file_digests": dict(artifact.file_digests),
                "metadata": dict(artifact.metadata),
                "tensor_count": artifact.tensor_count,
                "tensor_element_count": artifact.tensor_element_count,
                "parameter_count": artifact.parameter_count,
                "trainable_parameter_count": artifact.trainable_parameter_count,
            }
        except Exception as error:  # noqa: BLE001 - broadcast rank-zero failure.
            result[0] = {"error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(result, src=0)
    payload = result[0]
    if not isinstance(payload, Mapping) or "error" in payload:
        raise RuntimeError(f"FSDP2 {role} full export failed: {payload}")
    return FullModelArtifact(
        path=Path(str(payload["path"])),
        manifest_sha256=str(payload["manifest_sha256"]),
        file_digests=dict(payload["file_digests"]),
        metadata=dict(payload["metadata"]),
        tensor_count=int(payload["tensor_count"]),
        tensor_element_count=int(payload["tensor_element_count"]),
        parameter_count=int(payload["parameter_count"]),
        trainable_parameter_count=int(payload["trainable_parameter_count"]),
    )


__all__ = ["create_run_directory", "export_full_model", "export_peft_application"]

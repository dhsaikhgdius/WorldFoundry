"""Immutable asynchronous staging for exact-resume checkpoints."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from concurrent.futures import Future
from pathlib import Path
from typing import Protocol

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.tensor import DTensor

from .artifacts import TrainingCheckpointArtifact


class StagedCheckpointFinalizer(Protocol):
    """Commit a completed DCP staging directory."""

    def finalize_staged_checkpoint(
        self,
        *,
        staging_path: Path,
        final_path: Path,
        global_step: int,
        identity: Mapping[str, object],
        gradient_accumulation_phase: int,
        world_size: int,
        staging_strategy: str,
        optional_state_presence: Mapping[str, bool],
    ) -> TrainingCheckpointArtifact: ...


def _snapshot_for_async_save(value: object) -> object:
    if isinstance(value, DTensor):
        local = value.to_local().detach().clone()
        return DTensor.from_local(
            local,
            device_mesh=value.device_mesh,
            placements=value.placements,
            shape=value.shape,
            stride=value.stride(),
            run_check=False,
        )
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _snapshot_for_async_save(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot_for_async_save(item) for item in value)
    if isinstance(value, list):
        return [_snapshot_for_async_save(item) for item in value]
    return copy.deepcopy(value)


class ImmutableTrainingFileSystemWriter(dcp.FileSystemWriter):
    """Fence GPU-to-CPU staging before the asynchronous disk thread starts.

    PyTorch's blocking async stager can leave DTensor values aliased. A later
    optimizer step would then create a mixed-step checkpoint. Each local shard
    is cloned while preserving its mesh and placements; plain tensors are
    staged on CPU so the disk thread owns an immutable snapshot.
    """

    def stage(self, state_dict: dict[str, object]) -> dict[str, object]:
        staged = _snapshot_for_async_save(state_dict)
        if torch.cuda.is_available():
            torch.cuda.synchronize(torch.cuda.current_device())
        return staged


class PendingTrainingCheckpoint:
    """One asynchronous DCP write that must be joined before process exit."""

    def __init__(
        self,
        *,
        manager: StagedCheckpointFinalizer,
        future: Future[object],
        staging_path: Path,
        final_path: Path,
        global_step: int,
        identity: Mapping[str, object],
        gradient_accumulation_phase: int,
        world_size: int,
        staging_strategy: str,
        optional_state_presence: Mapping[str, bool],
    ) -> None:
        self._manager = manager
        self._future = future
        self._staging_path = staging_path
        self._final_path = final_path
        self._global_step = global_step
        self._identity = copy.deepcopy(dict(identity))
        self._gradient_accumulation_phase = gradient_accumulation_phase
        self._world_size = world_size
        self._staging_strategy = staging_strategy
        self._optional_state_presence = dict(optional_state_presence)
        self._artifact: TrainingCheckpointArtifact | None = None

    def done(self) -> bool:
        return self._artifact is not None or self._future.done()

    def wait(self, timeout: float | None = None) -> TrainingCheckpointArtifact:
        if self._artifact is None:
            self._future.result(timeout=timeout)
            self._artifact = self._manager.finalize_staged_checkpoint(
                staging_path=self._staging_path,
                final_path=self._final_path,
                global_step=self._global_step,
                identity=self._identity,
                gradient_accumulation_phase=self._gradient_accumulation_phase,
                world_size=self._world_size,
                staging_strategy=self._staging_strategy,
                optional_state_presence=self._optional_state_presence,
            )
        return self._artifact


__all__ = [
    "PendingTrainingCheckpoint",
]

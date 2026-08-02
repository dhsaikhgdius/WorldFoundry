"""Lazy public API for exact-resume checkpointing."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "IMMUTABLE_DTENSOR_ASYNC_STAGING": ".artifacts",
    "SYNCHRONOUS_DCP_STAGING": ".artifacts",
    "TrainingCheckpointArtifact": ".artifacts",
    "TRAINING_CHECKPOINT_COMMIT_SCHEMA": ".checkpointer",
    "TRAINING_CHECKPOINT_MANIFEST_SCHEMA": ".checkpointer",
    "TRAINING_CHECKPOINT_POINTER_SCHEMA": ".checkpointer",
    "TrainingCheckpointer": ".checkpointer",
    "IncompleteTrainingCheckpointError": ".errors",
    "TrainingCheckpointCompatibilityError": ".errors",
    "TrainingCheckpointError": ".errors",
    "PendingTrainingCheckpoint": ".staging",
    "TRAINING_PROGRESS_SCHEMA": ".state",
    "TRAINING_RUNTIME_STATE_SCHEMA": ".state",
    "TrainingProgress": ".state",
    "TrainingState": ".state",
    "NAMED_STATEFUL_COLLECTION_SCHEMA": ".stateful",
    "NamedStatefulCollection": ".stateful",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

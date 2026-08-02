"""Functional errors raised by exact-resume checkpointing."""


class TrainingCheckpointError(RuntimeError):
    """Base error for checkpoint validation or I/O failures."""


class IncompleteTrainingCheckpointError(TrainingCheckpointError):
    """Raised when a checkpoint has no valid atomic commit."""


class TrainingCheckpointCompatibilityError(TrainingCheckpointError):
    """Raised when exact-resume identities or topology differ."""


__all__ = [
    "IncompleteTrainingCheckpointError",
    "TrainingCheckpointCompatibilityError",
    "TrainingCheckpointError",
]

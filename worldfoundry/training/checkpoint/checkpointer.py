"""Atomic commit, inspection, and loading of PyTorch DCP checkpoints."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path

import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from worldfoundry.core.io.integrity import canonical_json as _core_canonical_json
from worldfoundry.core.io.integrity import replace_json_atomic, sync_directory

from .artifacts import (
    CHECKPOINT_STAGING_STRATEGIES,
    IMMUTABLE_DTENSOR_ASYNC_STAGING,
    OPTIONAL_TRAINING_STATE_NAMES,
    SYNCHRONOUS_DCP_STAGING,
    TrainingCheckpointArtifact,
    normalize_non_negative_int,
)
from .errors import (
    IncompleteTrainingCheckpointError,
    TrainingCheckpointCompatibilityError,
    TrainingCheckpointError,
)
from .staging import ImmutableTrainingFileSystemWriter, PendingTrainingCheckpoint
from .state import TrainingState

TRAINING_CHECKPOINT_MANIFEST_SCHEMA = "worldfoundry-training-checkpoint"
TRAINING_CHECKPOINT_COMMIT_SCHEMA = "worldfoundry-training-checkpoint-commit"
TRAINING_CHECKPOINT_POINTER_SCHEMA = "worldfoundry-training-checkpoint-pointer"

_CHECKPOINT_NAME_PATTERN = re.compile(r"step-[0-9]{8,}")
_STAGING_NAME_PATTERN = re.compile(r"\.step-[0-9]{8,}\.[0-9a-f]{32}\.staging")
_MANIFEST_NAME = "checkpoint-manifest.json"
_COMMIT_NAME = "_SUCCESS"
_LATEST_NAME = "latest.json"

_KEEP_LAST_ENV = "WORLDFOUNDRY_TRAINING_CHECKPOINT_KEEP_LAST"
_CLEAN_STAGING_ENV = "WORLDFOUNDRY_TRAINING_CHECKPOINT_CLEAN_STAGING"

logger = logging.getLogger(__name__)


def _keep_last_from_env() -> int | None:
    raw = os.environ.get(_KEEP_LAST_ENV)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw.strip())
    except ValueError as error:
        raise ValueError(f"{_KEEP_LAST_ENV} must be an integer >= 1, got {raw!r}") from error
    if value < 1:
        raise ValueError(f"{_KEEP_LAST_ENV} must be an integer >= 1, got {raw!r}")
    return value


def _clean_staging_from_env() -> bool:
    raw = os.environ.get(_CLEAN_STAGING_ENV)
    if raw is None or not raw.strip():
        return True
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "on", "yes"}:
        return True
    if normalized in {"0", "false", "off", "no"}:
        return False
    raise ValueError(f"{_CLEAN_STAGING_ENV} must be a boolean flag, got {raw!r}")


def _canonical_mapping(value: Mapping[str, object], *, field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    try:
        encoded = _core_canonical_json({str(key): item for key, item in value.items()})
    except (TypeError, ValueError) as error:
        raise TypeError("training checkpoint metadata must be canonical JSON") from error
    normalized = json.loads(encoded)
    if not isinstance(normalized, dict):
        raise TypeError(f"{field_name} must resolve to a JSON object")
    return normalized


def _atomic_write_json(path: Path, value: object) -> None:
    replace_json_atomic(path, value, root=path.parent)


def _read_json(path: Path, *, field_name: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IncompleteTrainingCheckpointError(f"invalid {field_name}: {path}") from error
    if not isinstance(value, dict):
        raise IncompleteTrainingCheckpointError(f"{field_name} must be a JSON object: {path}")
    return value


def _distributed_context() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


class TrainingCheckpointer:
    """Save and load atomically committed DCP state.

    Retention (``keep_last``) is opt-in: by default every committed
    checkpoint is kept, matching the historical behavior.  It can be enabled
    per instance via the constructor or globally via the
    ``WORLDFOUNDRY_TRAINING_CHECKPOINT_KEEP_LAST`` environment variable
    (an integer >= 1 counting committed checkpoints to retain).

    Orphaned staging directories (``.step-*.<token>.staging`` residue from a
    crashed or interrupted write) are removed once before the first save of
    this instance; disable via ``clean_orphaned_staging=False`` or
    ``WORLDFOUNDRY_TRAINING_CHECKPOINT_CLEAN_STAGING=0``.  Cleanup is tied to
    saving on purpose: a checkpointer used only for loading may point at a
    root owned by a live run whose in-flight staging must not be touched,
    whereas the root being saved to belongs to this run alone.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        keep_last: int | None = None,
        clean_orphaned_staging: bool | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        if keep_last is None:
            keep_last = _keep_last_from_env()
        elif not isinstance(keep_last, int) or isinstance(keep_last, bool) or keep_last < 1:
            raise ValueError("keep_last must be an integer >= 1 or None")
        self.keep_last = keep_last
        if clean_orphaned_staging is None:
            clean_orphaned_staging = _clean_staging_from_env()
        elif not isinstance(clean_orphaned_staging, bool):
            raise TypeError("clean_orphaned_staging must be a bool or None")
        self._staging_cleanup_pending = clean_orphaned_staging

    def _paths(self, global_step: int) -> tuple[Path, Path]:
        step = normalize_non_negative_int(global_step, field_name="global_step")
        final_path = self.root / f"step-{step:08d}"
        if final_path.exists():
            raise FileExistsError(f"training checkpoint already exists: {final_path}")
        rank, world_size = _distributed_context()
        token: object = uuid.uuid4().hex if rank == 0 else None
        if world_size > 1:
            values = [token]
            dist.broadcast_object_list(values, src=0)
            token = values[0]
        if not isinstance(token, str) or not token:
            raise RuntimeError("failed to coordinate checkpoint staging identity")
        staging_path = self.root / f".{final_path.name}.{token}.staging"
        if staging_path.exists():
            raise FileExistsError(f"checkpoint staging path already exists: {staging_path}")
        return staging_path, final_path

    def _remove_orphaned_staging(self) -> None:
        """Delete staging residue left in this run's write root by dead processes."""

        for candidate in sorted(self.root.iterdir()):
            if not candidate.is_dir() or _STAGING_NAME_PATTERN.fullmatch(candidate.name) is None:
                continue
            try:
                shutil.rmtree(candidate)
            except OSError:
                logger.warning("failed to remove orphaned checkpoint staging directory: %s", candidate, exc_info=True)
            else:
                logger.warning("removed orphaned checkpoint staging directory: %s", candidate)

    def _apply_retention(self, *, keep_last: int, newest_path: Path) -> None:
        """Keep the ``keep_last`` highest-step committed checkpoints (rank0 only).

        Only fully committed ``step-*`` directories (manifest and commit
        marker present) are candidates; anything unrecognized or partially
        written is never touched.  The checkpoint committed by the current
        call is always retained even if resuming from an older step left
        higher-numbered checkpoints in the root.
        """

        committed: list[tuple[int, Path]] = []
        for candidate in self.root.iterdir():
            if not candidate.is_dir() or _CHECKPOINT_NAME_PATTERN.fullmatch(candidate.name) is None:
                continue
            if not (candidate / _COMMIT_NAME).is_file() or not (candidate / _MANIFEST_NAME).is_file():
                continue
            committed.append((int(candidate.name.removeprefix("step-")), candidate))
        committed.sort(key=lambda item: item[0], reverse=True)
        for _, stale in committed[keep_last:]:
            if stale == newest_path:
                continue
            try:
                shutil.rmtree(stale)
            except OSError:
                logger.warning("failed to remove stale training checkpoint: %s", stale, exc_info=True)
            else:
                logger.info("removed stale training checkpoint (keep_last=%d): %s", keep_last, stale)

    def save(
        self,
        state: TrainingState,
        *,
        asynchronous: bool = False,
    ) -> TrainingCheckpointArtifact | PendingTrainingCheckpoint:
        if not isinstance(state, TrainingState):
            raise TypeError("state must be TrainingState")
        if not isinstance(asynchronous, bool):
            raise TypeError("asynchronous must be a bool")
        if self._staging_cleanup_pending:
            self._staging_cleanup_pending = False
            rank, _ = _distributed_context()
            if rank == 0:
                self._remove_orphaned_staging()
            _barrier()
        global_step = state.progress.optimizer_steps
        gradient_accumulation_phase = state.progress.gradient_accumulation_phase
        identity = dict(state.identity)
        optional_state_presence = state.optional_state_presence
        _, world_size = _distributed_context()
        staging_path, final_path = self._paths(global_step)
        payload = {"trainer": state}
        if asynchronous:
            staging_strategy = IMMUTABLE_DTENSOR_ASYNC_STAGING
            writer = ImmutableTrainingFileSystemWriter(
                staging_path,
                overwrite=False,
            )
            future = dcp.async_save(payload, storage_writer=writer)
            return PendingTrainingCheckpoint(
                manager=self,
                future=future,
                staging_path=staging_path,
                final_path=final_path,
                global_step=global_step,
                identity=identity,
                gradient_accumulation_phase=gradient_accumulation_phase,
                world_size=world_size,
                staging_strategy=staging_strategy,
                optional_state_presence=optional_state_presence,
            )
        staging_strategy = SYNCHRONOUS_DCP_STAGING
        dcp.save(payload, checkpoint_id=staging_path)
        return self.finalize_staged_checkpoint(
            staging_path=staging_path,
            final_path=final_path,
            global_step=global_step,
            identity=identity,
            gradient_accumulation_phase=gradient_accumulation_phase,
            world_size=world_size,
            staging_strategy=staging_strategy,
            optional_state_presence=optional_state_presence,
        )

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
    ) -> TrainingCheckpointArtifact:
        """Atomically commit a completed synchronous or asynchronous write."""

        if staging_strategy not in CHECKPOINT_STAGING_STRATEGIES:
            raise TrainingCheckpointError(f"unsupported checkpoint staging strategy: {staging_strategy!r}")
        resolved_optional_presence = dict(optional_state_presence)
        if set(resolved_optional_presence) != set(OPTIONAL_TRAINING_STATE_NAMES) or any(
            not isinstance(value, bool) for value in resolved_optional_presence.values()
        ):
            raise TrainingCheckpointError("optional training-state presence is invalid")
        _barrier()
        rank, active_world_size = _distributed_context()
        if active_world_size != world_size:
            raise TrainingCheckpointError("world size changed while checkpoint save was pending")
        if rank == 0:
            payload_files: dict[str, int] = {}
            for candidate in sorted(staging_path.rglob("*")):
                if candidate.is_file():
                    relative = candidate.relative_to(staging_path).as_posix()
                    payload_files[relative] = candidate.stat().st_size
            if not payload_files or ".metadata" not in payload_files:
                raise TrainingCheckpointError("DCP did not produce a complete payload")
            manifest = {
                "schema": TRAINING_CHECKPOINT_MANIFEST_SCHEMA,
                "checkpoint_name": final_path.name,
                "global_step": global_step,
                "identity": dict(identity),
                "world_size": world_size,
                "exact_same_topology": True,
                "gradient_accumulation_phase": gradient_accumulation_phase,
                "staging_strategy": staging_strategy,
                "optional_state_presence": resolved_optional_presence,
                "files": payload_files,
            }
            manifest_path = staging_path / _MANIFEST_NAME
            _atomic_write_json(manifest_path, manifest)
            _atomic_write_json(
                staging_path / _COMMIT_NAME,
                {
                    "schema": TRAINING_CHECKPOINT_COMMIT_SCHEMA,
                    "checkpoint_name": final_path.name,
                },
            )
            sync_directory(staging_path)
            os.replace(staging_path, final_path)
            sync_directory(self.root)
            _atomic_write_json(
                self.root / _LATEST_NAME,
                {
                    "schema": TRAINING_CHECKPOINT_POINTER_SCHEMA,
                    "checkpoint_name": final_path.name,
                    "global_step": global_step,
                },
            )
            # Retention runs only after the new checkpoint is fully committed
            # (os.replace + root sync + latest pointer) so a crash inside the
            # deletion loop can never leave the root without a valid latest.
            if self.keep_last is not None:
                self._apply_retention(keep_last=self.keep_last, newest_path=final_path)
        _barrier()
        return self.inspect(final_path)

    def _resolve(self, checkpoint: str | Path | None) -> Path:
        if checkpoint is None or str(checkpoint) == "latest":
            pointer = _read_json(
                self.root / _LATEST_NAME,
                field_name="latest checkpoint pointer",
            )
            expected = {"schema", "checkpoint_name", "global_step"}
            if set(pointer) != expected or pointer["schema"] != TRAINING_CHECKPOINT_POINTER_SCHEMA:
                raise IncompleteTrainingCheckpointError("latest checkpoint pointer is invalid")
            name = str(pointer["checkpoint_name"])
            if _CHECKPOINT_NAME_PATTERN.fullmatch(name) is None:
                raise IncompleteTrainingCheckpointError("latest checkpoint name is invalid")
            path = self.root / name
            artifact = self.inspect(path)
            if artifact.global_step != int(pointer["global_step"]):
                raise IncompleteTrainingCheckpointError("latest checkpoint pointer differs from its committed artifact")
            return path
        candidate = Path(checkpoint)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        return candidate.expanduser().resolve()

    def inspect(self, checkpoint: str | Path) -> TrainingCheckpointArtifact:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_dir():
            raise IncompleteTrainingCheckpointError(f"training checkpoint is not a directory: {path}")
        commit_path = path / _COMMIT_NAME
        manifest_path = path / _MANIFEST_NAME
        if not commit_path.is_file() or not manifest_path.is_file():
            raise IncompleteTrainingCheckpointError(f"training checkpoint has no valid atomic commit: {path}")
        commit = _read_json(commit_path, field_name="checkpoint commit")
        commit_expected = {"schema", "checkpoint_name"}
        if set(commit) != commit_expected or commit["schema"] != TRAINING_CHECKPOINT_COMMIT_SCHEMA:
            raise IncompleteTrainingCheckpointError("checkpoint commit fields are invalid")
        if commit["checkpoint_name"] != path.name:
            raise IncompleteTrainingCheckpointError("checkpoint commit does not match its manifest")
        manifest = _read_json(manifest_path, field_name="checkpoint manifest")
        manifest_expected = {
            "schema",
            "checkpoint_name",
            "global_step",
            "identity",
            "world_size",
            "exact_same_topology",
            "gradient_accumulation_phase",
            "staging_strategy",
            "optional_state_presence",
            "files",
        }
        if set(manifest) != manifest_expected or manifest["schema"] != TRAINING_CHECKPOINT_MANIFEST_SCHEMA:
            raise IncompleteTrainingCheckpointError("checkpoint manifest fields are invalid")
        if manifest["checkpoint_name"] != path.name:
            raise IncompleteTrainingCheckpointError("checkpoint manifest name differs from its directory")
        identity = _canonical_mapping(manifest["identity"], field_name="checkpoint identity")
        if manifest["exact_same_topology"] is not True:
            raise IncompleteTrainingCheckpointError("checkpoint does not declare exact same-topology resume")
        if int(manifest["gradient_accumulation_phase"]) != 0:
            raise IncompleteTrainingCheckpointError("checkpoint was not committed at an optimizer boundary")
        staging_strategy = str(manifest["staging_strategy"])
        if staging_strategy not in CHECKPOINT_STAGING_STRATEGIES:
            raise IncompleteTrainingCheckpointError("checkpoint manifest staging strategy is invalid")
        optional_state_presence = manifest["optional_state_presence"]
        if (
            not isinstance(optional_state_presence, Mapping)
            or set(optional_state_presence) != set(OPTIONAL_TRAINING_STATE_NAMES)
            or any(not isinstance(value, bool) for value in optional_state_presence.values())
        ):
            raise IncompleteTrainingCheckpointError("checkpoint optional training-state presence is invalid")
        files = manifest["files"]
        if not isinstance(files, Mapping) or not files:
            raise IncompleteTrainingCheckpointError("checkpoint manifest has no DCP payload files")
        actual_payload = {
            candidate.relative_to(path).as_posix()
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.name not in {_MANIFEST_NAME, _COMMIT_NAME}
        }
        if actual_payload != set(files):
            raise IncompleteTrainingCheckpointError("checkpoint payload file set differs from its manifest")
        file_sizes: dict[str, int] = {}
        for relative, raw_size in files.items():
            if not isinstance(relative, str):
                raise IncompleteTrainingCheckpointError("checkpoint file descriptor is invalid")
            candidate = path / relative
            if not candidate.is_file():
                raise IncompleteTrainingCheckpointError(f"checkpoint payload file is invalid: {relative}")
            size = int(raw_size)
            if size < 0 or candidate.stat().st_size != size:
                raise IncompleteTrainingCheckpointError(f"checkpoint payload was modified: {relative}")
            file_sizes[relative] = size
        return TrainingCheckpointArtifact(
            path=path,
            global_step=manifest["global_step"],
            staging_strategy=staging_strategy,
            optional_state_presence=optional_state_presence,
            identity=identity,
            file_size_bytes=file_sizes,
        )

    def load(
        self,
        state: TrainingState,
        checkpoint: str | Path | None = None,
    ) -> TrainingCheckpointArtifact:
        if not isinstance(state, TrainingState):
            raise TypeError("state must be TrainingState")
        path = self._resolve(checkpoint)
        artifact = self.inspect(path)
        if dict(artifact.identity) != dict(state.identity):
            raise TrainingCheckpointCompatibilityError(
                "checkpoint identity differs from the active recipe/data/model/runtime"
            )
        if dict(artifact.optional_state_presence) != state.optional_state_presence:
            differences = sorted(
                name
                for name in OPTIONAL_TRAINING_STATE_NAMES
                if artifact.optional_state_presence[name] != state.optional_state_presence[name]
            )
            raise TrainingCheckpointCompatibilityError(
                "optional training-state presence differs for " + ", ".join(differences)
            )
        dcp.load({"trainer": state}, checkpoint_id=path)
        if state.progress.optimizer_steps != artifact.global_step:
            raise TrainingCheckpointCompatibilityError("loaded progress step differs from the checkpoint manifest")
        return artifact


__all__ = [
    "TRAINING_CHECKPOINT_COMMIT_SCHEMA",
    "TRAINING_CHECKPOINT_MANIFEST_SCHEMA",
    "TRAINING_CHECKPOINT_POINTER_SCHEMA",
    "TrainingCheckpointer",
]

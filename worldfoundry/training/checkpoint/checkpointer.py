"""Atomic commit, inspection, and loading of PyTorch DCP checkpoints."""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Mapping
from pathlib import Path

import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from worldfoundry.core.io.file_utils import file_sha256 as _file_sha256
from worldfoundry.core.io.integrity import canonical_json as _core_canonical_json
from worldfoundry.core.io.integrity import canonical_sha256 as _core_canonical_sha256
from worldfoundry.core.io.integrity import replace_json_atomic, sync_directory

from .artifacts import (
    CHECKPOINT_STAGING_STRATEGIES,
    IMMUTABLE_DTENSOR_ASYNC_STAGING,
    OPTIONAL_TRAINING_STATE_NAMES,
    SHA256_PATTERN,
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
_MANIFEST_NAME = "checkpoint-manifest.json"
_COMMIT_NAME = "_SUCCESS"
_LATEST_NAME = "latest.json"


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


def _canonical_sha256(value: object) -> str:
    try:
        return _core_canonical_sha256(value)
    except (TypeError, ValueError) as error:
        raise TypeError("training checkpoint metadata must be canonical JSON") from error


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
    """Save and load checksum-audited, atomically committed DCP state."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)

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
        global_step = state.progress.optimizer_steps
        gradient_accumulation_phase = state.progress.gradient_accumulation_phase
        identity = dict(state.identity)
        identity_digest = state.identity_digest
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
                identity_digest=identity_digest,
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
            identity_digest=identity_digest,
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
        identity_digest: str,
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
            payload_files: dict[str, dict[str, object]] = {}
            for candidate in sorted(staging_path.rglob("*")):
                if candidate.is_symlink():
                    raise TrainingCheckpointError(f"training checkpoint cannot contain symlinks: {candidate}")
                if candidate.is_file():
                    relative = candidate.relative_to(staging_path).as_posix()
                    payload_files[relative] = {
                        "sha256": _file_sha256(candidate),
                        "size_bytes": candidate.stat().st_size,
                    }
            if not payload_files or ".metadata" not in payload_files:
                raise TrainingCheckpointError("DCP did not produce a complete payload")
            manifest = {
                "schema": TRAINING_CHECKPOINT_MANIFEST_SCHEMA,
                "checkpoint_name": final_path.name,
                "global_step": global_step,
                "identity": dict(identity),
                "identity_digest": identity_digest,
                "world_size": world_size,
                "exact_same_topology": True,
                "gradient_accumulation_phase": gradient_accumulation_phase,
                "staging_strategy": staging_strategy,
                "optional_state_presence": resolved_optional_presence,
                "files": payload_files,
            }
            manifest_path = staging_path / _MANIFEST_NAME
            _atomic_write_json(manifest_path, manifest)
            manifest_sha256 = _file_sha256(manifest_path)
            _atomic_write_json(
                staging_path / _COMMIT_NAME,
                {
                    "schema": TRAINING_CHECKPOINT_COMMIT_SCHEMA,
                    "checkpoint_name": final_path.name,
                    "manifest_sha256": manifest_sha256,
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
                    "manifest_sha256": manifest_sha256,
                },
            )
        _barrier()
        return self.inspect(final_path)

    def _resolve(self, checkpoint: str | Path | None) -> Path:
        if checkpoint is None or str(checkpoint) == "latest":
            pointer = _read_json(
                self.root / _LATEST_NAME,
                field_name="latest checkpoint pointer",
            )
            expected = {"schema", "checkpoint_name", "global_step", "manifest_sha256"}
            if set(pointer) != expected or pointer["schema"] != TRAINING_CHECKPOINT_POINTER_SCHEMA:
                raise IncompleteTrainingCheckpointError("latest checkpoint pointer is invalid")
            name = str(pointer["checkpoint_name"])
            if _CHECKPOINT_NAME_PATTERN.fullmatch(name) is None:
                raise IncompleteTrainingCheckpointError("latest checkpoint name is invalid")
            path = self.root / name
            artifact = self.inspect(path)
            if (
                artifact.global_step != int(pointer["global_step"])
                or artifact.manifest_sha256 != pointer["manifest_sha256"]
            ):
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
        for candidate in path.rglob("*"):
            if candidate.is_symlink():
                raise IncompleteTrainingCheckpointError(f"training checkpoint cannot contain symlinks: {candidate}")
        commit_path = path / _COMMIT_NAME
        manifest_path = path / _MANIFEST_NAME
        if not commit_path.is_file() or not manifest_path.is_file():
            raise IncompleteTrainingCheckpointError(f"training checkpoint has no valid atomic commit: {path}")
        commit = _read_json(commit_path, field_name="checkpoint commit")
        commit_expected = {"schema", "checkpoint_name", "manifest_sha256"}
        if set(commit) != commit_expected or commit["schema"] != TRAINING_CHECKPOINT_COMMIT_SCHEMA:
            raise IncompleteTrainingCheckpointError("checkpoint commit fields are invalid")
        manifest_sha256 = _file_sha256(manifest_path)
        if commit["checkpoint_name"] != path.name or commit["manifest_sha256"] != manifest_sha256:
            raise IncompleteTrainingCheckpointError("checkpoint commit does not match its manifest")
        manifest = _read_json(manifest_path, field_name="checkpoint manifest")
        manifest_expected = {
            "schema",
            "checkpoint_name",
            "global_step",
            "identity",
            "identity_digest",
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
        identity_digest = str(manifest["identity_digest"])
        if identity_digest != _canonical_sha256(identity):
            raise IncompleteTrainingCheckpointError("checkpoint identity digest is invalid")
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
        file_digests: dict[str, str] = {}
        for relative, raw_descriptor in files.items():
            if not isinstance(relative, str) or not isinstance(raw_descriptor, Mapping):
                raise IncompleteTrainingCheckpointError("checkpoint file descriptor is invalid")
            if set(raw_descriptor) != {"sha256", "size_bytes"}:
                raise IncompleteTrainingCheckpointError("checkpoint file descriptor fields are invalid")
            candidate = path / relative
            if candidate.is_symlink() or not candidate.is_file():
                raise IncompleteTrainingCheckpointError(f"checkpoint payload file is invalid: {relative}")
            digest = str(raw_descriptor["sha256"])
            size = int(raw_descriptor["size_bytes"])
            if SHA256_PATTERN.fullmatch(digest) is None:
                raise IncompleteTrainingCheckpointError(f"checkpoint file digest is invalid: {relative}")
            if candidate.stat().st_size != size or _file_sha256(candidate) != digest:
                raise IncompleteTrainingCheckpointError(f"checkpoint payload was modified: {relative}")
            file_digests[relative] = digest
        return TrainingCheckpointArtifact(
            path=path,
            global_step=manifest["global_step"],
            staging_strategy=staging_strategy,
            optional_state_presence=optional_state_presence,
            manifest_sha256=manifest_sha256,
            identity_digest=identity_digest,
            file_sha256=file_digests,
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
        if artifact.identity_digest != state.identity_digest:
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

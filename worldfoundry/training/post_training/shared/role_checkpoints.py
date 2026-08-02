"""Strict checkpoint identities for independently materialized model roles."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType

from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.core.io.integrity import canonical_sha256

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_ROLE_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]*")


def _role_name(value: object) -> str:
    role = str(value).strip().lower().replace("_", "-")
    if _ROLE_PATTERN.fullmatch(role) is None:
        raise ValueError(f"checkpoint role contains unsupported characters: {value!r}")
    return role


def _checkpoint_payload(checkpoint: CheckpointSpec) -> dict[str, object]:
    return {
        "sources": list(checkpoint.sources),
        "repo_id": checkpoint.repo_id,
        "revision": checkpoint.revision,
        "files": list(checkpoint.files),
        "allow_patterns": list(checkpoint.allow_patterns),
        "file_sha256": dict(checkpoint.file_sha256),
        "file_size_bytes": dict(checkpoint.file_size_bytes),
        "resource_sha256": dict(checkpoint.resource_sha256),
        "resource_size_bytes": dict(checkpoint.resource_size_bytes),
    }


def _audit_default(checkpoint: CheckpointSpec) -> None:
    if checkpoint.sources:
        _audit_local(checkpoint)
        return
    if checkpoint.repo_id is None or _COMMIT_PATTERN.fullmatch(str(checkpoint.revision or "").lower()) is None:
        raise ValueError("native default role checkpoint must use an immutable 40-hex Hub revision")
    if not checkpoint.files:
        raise ValueError("native default role checkpoint must declare loaded files")


def _audit_local(checkpoint: CheckpointSpec) -> None:
    if not checkpoint.sources or checkpoint.repo_id is not None:
        raise ValueError("local role checkpoint must use only explicit local sources")
    declared = set(checkpoint.files)
    if not declared:
        raise ValueError("local role checkpoint must declare its loaded files")
    if declared != set(checkpoint.file_sha256) or declared != set(checkpoint.file_size_bytes):
        raise ValueError("local role checkpoint requires SHA-256 and byte size for every loaded file")


def _parse_pinned_reference(reference: str) -> tuple[str, str]:
    repo_id, separator, revision = reference.rpartition("@")
    revision = revision.lower()
    if not separator or not repo_id.strip() or "/" not in repo_id or _COMMIT_PATTERN.fullmatch(revision) is None:
        raise ValueError(
            "role checkpoint references must be 'default' or 'REPO@40_HEX_COMMIT'; "
            "audited local paths must be supplied as CheckpointSpec overrides"
        )
    return repo_id.strip(), revision


def _audit_local_mirror(
    local: CheckpointSpec,
    native_default: CheckpointSpec,
    *,
    requested_repo_id: str,
    requested_revision: str,
) -> None:
    if requested_repo_id != native_default.repo_id or requested_revision != str(native_default.revision).lower():
        raise ValueError("an audited local role mirror can only satisfy the native default repository and revision")
    if (
        local.files != native_default.files
        or dict(local.file_sha256) != dict(native_default.file_sha256)
        or dict(local.file_size_bytes) != dict(native_default.file_size_bytes)
    ):
        raise ValueError("audited local role mirror bytes differ from the explicitly requested native checkpoint")


@dataclass(frozen=True, slots=True)
class ResolvedRoleCheckpoint:
    """One role's materialization spec and immutable resume identity."""

    role: str
    requested_reference: str
    checkpoint: CheckpointSpec
    source_kind: str

    def __post_init__(self) -> None:
        role = _role_name(self.role)
        reference = str(self.requested_reference).strip()
        if not reference:
            raise ValueError("requested_reference cannot be empty")
        if not isinstance(self.checkpoint, CheckpointSpec):
            raise TypeError("checkpoint must be a native CheckpointSpec")
        if self.source_kind not in {
            "native-default",
            "pinned-hub",
            "audited-local",
        }:
            raise ValueError(f"unsupported role checkpoint source_kind: {self.source_kind!r}")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "requested_reference", reference)

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "requested_reference": self.requested_reference,
            "source_kind": self.source_kind,
            "checkpoint": _checkpoint_payload(self.checkpoint),
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


def resolve_role_checkpoint(
    *,
    role: str,
    reference: str,
    native_default: CheckpointSpec,
    audited_local_override: CheckpointSpec | None = None,
) -> ResolvedRoleCheckpoint:
    """Resolve a role without guessing between Hub ids and filesystem paths.

    Recipe strings accept only ``default`` or ``REPO@40_HEX_COMMIT``.  Local
    weights enter through an already constructed :class:`CheckpointSpec` so
    their complete byte audit cannot be omitted accidentally.
    """

    resolved_role = _role_name(role)
    if not isinstance(native_default, CheckpointSpec):
        raise TypeError("native_default must be a CheckpointSpec")
    _audit_default(native_default)
    requested = str(reference).strip()
    if not requested:
        raise ValueError("role checkpoint reference cannot be empty")
    if audited_local_override is not None:
        if not isinstance(audited_local_override, CheckpointSpec):
            raise TypeError("audited_local_override must be a CheckpointSpec")
        _audit_local(audited_local_override)
        resolved_reference = "audited-local"
        if requested != "default":
            repo_id, revision = _parse_pinned_reference(requested)
            _audit_local_mirror(
                audited_local_override,
                native_default,
                requested_repo_id=repo_id,
                requested_revision=revision,
            )
            resolved_reference = requested
        return ResolvedRoleCheckpoint(
            role=resolved_role,
            requested_reference=resolved_reference,
            checkpoint=audited_local_override,
            source_kind="audited-local",
        )
    if requested == "default":
        return ResolvedRoleCheckpoint(
            role=resolved_role,
            requested_reference=requested,
            checkpoint=native_default,
            source_kind="native-default",
        )

    repo_id, revision = _parse_pinned_reference(requested)
    if repo_id == native_default.repo_id and revision == str(native_default.revision).lower():
        checkpoint = native_default
        source_kind = "native-default"
    else:
        metadata = dict(native_default.metadata)
        metadata.update(
            {
                "role_checkpoint_reference": requested,
                "integrity_mode": "immutable-hub-revision",
            }
        )
        checkpoint = CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=native_default.files,
            allow_patterns=native_default.allow_patterns,
            metadata=MappingProxyType(metadata),
        )
        source_kind = "pinned-hub"
    return ResolvedRoleCheckpoint(
        role=resolved_role,
        requested_reference=requested,
        checkpoint=checkpoint,
        source_kind=source_kind,
    )


__all__ = ["ResolvedRoleCheckpoint", "resolve_role_checkpoint"]

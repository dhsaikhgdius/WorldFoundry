"""Explicit checkpoint asset identities used by training caches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def checkpoint_asset_identity(
    *,
    repo_id: object,
    revision: object,
    files: Sequence[str],
    file_size_bytes: Mapping[str, int] | None = None,
    sources: Sequence[str] = (),
) -> dict[str, object]:
    repository = str(repo_id).strip()
    resolved_revision = str(revision).strip()
    resolved_files = tuple(str(name).strip() for name in files)
    if not repository or not resolved_revision or not resolved_files or any(not name for name in resolved_files):
        raise ValueError("checkpoint identity requires repository, revision, and files")
    sizes = {str(name): int(size) for name, size in dict(file_size_bytes or {}).items()}
    identity: dict[str, object] = {
        "repo_id": repository,
        "revision": resolved_revision,
        "files": list(resolved_files),
        "file_size_bytes": sizes,
    }
    resolved_sources = [str(source) for source in sources]
    if resolved_sources:
        identity["sources"] = resolved_sources
    return identity


__all__ = ["checkpoint_asset_identity"]

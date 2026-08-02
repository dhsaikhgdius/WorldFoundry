"""Content identity for model components used during feature precomputation."""

from __future__ import annotations

import re
from collections.abc import Mapping

from worldfoundry.core.io.integrity import canonical_sha256

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def checkpoint_asset_digest(
    *,
    repository: str,
    revision: str,
    file_sha256: Mapping[str, str],
) -> str:
    """Combine a pinned repository revision and all relevant file hashes."""

    resolved_repository = str(repository).strip()
    resolved_revision = str(revision).strip()
    if not resolved_repository:
        raise ValueError("checkpoint repository cannot be empty")
    if not resolved_revision:
        raise ValueError("checkpoint revision cannot be empty")
    if not file_sha256:
        raise ValueError("checkpoint asset digest requires at least one file")
    files: dict[str, str] = {}
    for raw_name, raw_digest in file_sha256.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("checkpoint file names cannot be empty")
        if name in files:
            raise ValueError(f"checkpoint file name is duplicated after normalization: {name!r}")
        digest = str(raw_digest).strip().lower()
        if _SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"checkpoint file {name!r} must have a SHA-256 digest")
        files[name] = digest
    return canonical_sha256(
        {
            "repository": resolved_repository,
            "revision": resolved_revision,
            "file_sha256": {name: files[name] for name in sorted(files)},
        }
    )


__all__ = ["checkpoint_asset_digest"]

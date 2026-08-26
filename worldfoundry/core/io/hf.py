# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hugging Face helpers shared across encoders.

Remote repos are preloaded before ``from_pretrained(..., local_files_only=True)``
so multi-rank jobs do not race to download the same snapshot or treat a partial
cache entry as complete.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import torch.distributed as dist
from filelock import FileLock

from worldfoundry.core.distributed import get_global_rank, is_distributed_initialized
from worldfoundry.core.distributed.generic_collectives import is_local_master
from worldfoundry.core.io.disk import (
    CACHE_MIN_FREE_ENV,
    DiskSpaceError,
    cache_min_free_bytes,
    disk_space_error_from_exception,
    ensure_free_disk,
)

HF_DOWNLOAD_COORDINATOR_ENV = "WORLDFOUNDRY_HF_DOWNLOAD_COORDINATOR"
_DEFAULT_DOWNLOAD_COORDINATOR = "local"


def _str2bool(v: str | bool) -> bool:
    """Parse the usual yes/no/true/false/1/0 strings into a bool."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise ValueError(f"Boolean value expected, got {v!r}")


def _hub_cache_dir(cache_dir: str | os.PathLike[str] | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

    return Path(HUGGINGFACE_HUB_CACHE).expanduser()


def _snapshot_download(*args, **kwargs) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(*args, **kwargs)


def _lock_path(
    repo_id: str,
    revision: str | None,
    cache_dir: str | os.PathLike[str] | None,
) -> Path:
    cache_root = _hub_cache_dir(cache_dir)
    lock_key = f"{repo_id}@{revision or 'main'}"
    lock_digest = hashlib.sha256(lock_key.encode("utf-8")).hexdigest()[:16]
    safe_name = repo_id.replace("/", "--")
    locks_dir = cache_root / ".worldfoundry_locks"
    return locks_dir / f"{safe_name}-{lock_digest}.lock"


def _normalize_patterns(
    patterns: str | Sequence[str] | None,
) -> str | list[str] | None:
    if patterns is None or isinstance(patterns, str):
        return patterns
    return list(patterns)


def download_coordinator_mode() -> str:
    """Return ``local`` (per-node local-rank0) or ``global`` (world rank0) download mode."""

    raw = os.getenv(HF_DOWNLOAD_COORDINATOR_ENV, _DEFAULT_DOWNLOAD_COORDINATOR)
    mode = str(raw).strip().lower()
    if mode in ("local", "local-rank0", "node"):
        return "local"
    if mode in ("global", "global-rank0", "rank0"):
        return "global"
    raise ValueError(
        f"{HF_DOWNLOAD_COORDINATOR_ENV} must be 'local' or 'global', got {raw!r}"
    )


def _should_download_hf_snapshot() -> bool:
    """Whether this process should perform the Hub download.

    Default ``local`` mode uses each node's local-rank0 so node-local HF caches
    still populate when ranks do not share a filesystem. ``global`` mode keeps
    the historical world-rank0 download and **requires** a shared HF cache
    visible to every rank after the broadcast.
    """

    if download_coordinator_mode() == "global":
        return get_global_rank() == 0
    if not is_distributed_initialized():
        return True
    return is_local_master()


HF_URI_SCHEME = "hf://"


def _parse_hf_uri(path: str) -> tuple[str, str]:
    """Parse ``hf://<owner>/<repo>[/<subpath>]`` into ``(repo_id, subpath)``."""

    parts = path[len(HF_URI_SCHEME) :].split("/", 2)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid HF path {path!r}; expected hf://<owner>/<repo>[/<subpath>].")
    repo_id = f"{parts[0]}/{parts[1]}"
    subpath = parts[2] if len(parts) > 2 else ""
    return repo_id, subpath


def _allow_patterns_for_subpath(subpath: str) -> list[str] | None:
    if not subpath:
        return None
    return [subpath, f"{subpath}/*", f"{subpath}/**"]


def resolve_hf_path(path: str | PathLike[str] | None) -> str | Any:
    """Resolve a possibly ``hf://``-prefixed path to a local filesystem path.

    Accepts either:

    * a local path (returned unchanged if it exists), or
    * ``hf://<owner>/<repo>[/<subpath>]`` — reuses an already materialized
      WorldFoundry-local snapshot, otherwise downloads only the requested
      subtree through the configured Hugging Face endpoint.

    Set ``HF_HUB_OFFLINE=1`` when runtime I/O must remain strictly offline.
    """
    if not isinstance(path, str) or not path:
        return path
    if os.path.exists(path):
        return path
    if not path.startswith(HF_URI_SCHEME):
        return path

    repo_id, subpath = _parse_hf_uri(path)
    from worldfoundry.core.io.paths import resolve_local_hf_model_path

    try:
        local_root = resolve_local_hf_model_path(repo_id)
    except FileNotFoundError:
        local_root = Path(
            _snapshot_download(
                repo_id=repo_id,
                allow_patterns=_allow_patterns_for_subpath(subpath),
            )
        )
    resolved = local_root / subpath if subpath else local_root
    return str(resolved.resolve())


def hf_download_or_fpath(path: str | PathLike[str] | None) -> str | Any:
    """Backwards-compatible alias for :func:`resolve_hf_path`."""
    return resolve_hf_path(path)


def _is_probable_hf_repo_id(value: str) -> bool:
    text = value.strip()
    if not text or text.startswith((".", "~", "/")):
        return False
    return "/" in text and not Path(text).expanduser().exists()


def _required_files_present(directory: Path, required_files: Sequence[str]) -> bool:
    return all((directory / filename).exists() for filename in required_files)


def _snapshot_candidates(cache_root: Path) -> list[Path]:
    snapshots_root = cache_root / "snapshots"
    if not snapshots_root.is_dir():
        return []
    candidates: list[Path] = []
    ref_path = cache_root / "refs" / "main"
    if ref_path.is_file():
        ref = ref_path.read_text(encoding="utf-8").strip()
        if ref:
            candidates.append(snapshots_root / ref)
    candidates.append(snapshots_root / "worldfoundry-local")
    candidates.extend(sorted(snapshots_root.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate
        if candidate.is_dir() and resolved not in seen:
            deduped.append(candidate)
            seen.add(resolved)
    return deduped


def resolve_hf_snapshot_path(
    value: str | os.PathLike[str],
    required_files: Sequence[str] = (),
    *,
    local_files_only_env: str = "WORLDFOUNDRY_HF_LOCAL_FILES_ONLY",
    local_files_only: bool | None = None,
) -> Path:
    """Resolve a repo id, HF cache repo root, or local path to a usable snapshot."""

    text = str(value)
    path = Path(text).expanduser()
    if path.exists():
        if path.is_dir():
            candidates = _snapshot_candidates(path)
            for candidate in candidates:
                if _required_files_present(candidate, required_files):
                    return candidate
            if candidates and not required_files:
                return candidates[0]
        return path

    if _is_probable_hf_repo_id(text):
        if local_files_only is None:
            local_files_only = _str2bool(os.getenv(local_files_only_env, "false"))
        return Path(_snapshot_download(repo_id=text, local_files_only=local_files_only)).expanduser()
    return path


def _download_snapshot(
    repo_id: str,
    *,
    revision: str | None,
    cache_dir: str | os.PathLike[str] | None,
    allow_patterns: str | Sequence[str] | None,
    ignore_patterns: str | Sequence[str] | None,
    token: str | bool | None = None,
) -> None:
    lock_file = _lock_path(repo_id, revision, cache_dir)
    cache_root = _hub_cache_dir(cache_dir)
    min_bytes = cache_min_free_bytes()
    settings: dict[str, object] = {"repo": repo_id}
    if cache_dir is not None:
        settings["cache_dir"] = Path(cache_dir).expanduser()
    ensure_free_disk(
        cache_root,
        required_bytes=min_bytes,
        label="Hugging Face cache",
        env_vars=("HF_HOME", "HF_HUB_CACHE", CACHE_MIN_FREE_ENV),
        settings=settings,
    )
    try:
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(lock_file)):
            _snapshot_download(
                repo_id,
                revision=revision,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                local_files_only=False,
                allow_patterns=_normalize_patterns(allow_patterns),
                ignore_patterns=_normalize_patterns(ignore_patterns),
                token=token,
            )
    except Exception as exc:
        disk_error = disk_space_error_from_exception(
            exc,
            path=cache_root,
            label="Hugging Face cache",
            required_bytes=min_bytes,
            env_vars=("HF_HOME", "HF_HUB_CACHE", CACHE_MIN_FREE_ENV),
            settings=settings,
        )
        if disk_error is not None:
            raise disk_error from exc
        raise


def maybe_download_hf_repo_on_rank0(
    repo_id_or_path: str,
    *,
    revision: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    allow_patterns: str | Sequence[str] | None = None,
    ignore_patterns: str | Sequence[str] | None = None,
    token: str | bool | None = None,
) -> None:
    """Download a remote HF repo snapshot from a coordinator rank when allowed.

    Local paths and explicit offline/local-only modes are no-ops. For remote
    repositories a coordinator rank preloads the snapshot while other ranks wait.

    Coordinator selection is controlled by
    ``WORLDFOUNDRY_HF_DOWNLOAD_COORDINATOR`` (default ``local``):

    * ``local`` — each node's local-rank0 downloads into that node's HF cache.
      Use this when caches live on node-local disks (the common torchrun case).
    * ``global`` — only world rank0 downloads, then broadcasts success/failure.
      **Requires a shared filesystem** so every rank can read the same HF cache
      path after the broadcast; otherwise non-rank0 nodes see missing files.

    A filesystem lock still serializes independent processes that share one HF
    cache directory.
    """
    if (
        os.path.isdir(repo_id_or_path)
        or _str2bool(os.getenv("HF_HUB_OFFLINE", "false"))
        or _str2bool(os.getenv("LOCAL_FILES_ONLY", "false"))
    ):
        return

    mode = download_coordinator_mode()
    payload: list[dict[str, str | None]] = [{"error": None}]
    download_error: BaseException | None = None
    if _should_download_hf_snapshot():
        try:
            _download_snapshot(
                repo_id_or_path,
                revision=revision,
                cache_dir=cache_dir,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                token=token,
            )
        except DiskSpaceError as exc:
            download_error = exc
            payload = [{"error": str(exc), "disk_error": "1"}]
        except Exception as exc:
            download_error = exc
            payload = [{"error": f"{type(exc).__name__}: {exc}"}]

    if is_distributed_initialized():
        if mode == "global":
            dist.broadcast_object_list(payload, src=0)
        else:
            # Wait for every node's local-rank0 download before local-only reopen.
            dist.barrier()

    error = payload[0]["error"]
    if error is not None:
        if payload[0].get("disk_error"):
            raise DiskSpaceError(error)
        if download_error is not None and not isinstance(download_error, DiskSpaceError):
            raise RuntimeError(
                f"Coordinator rank failed to download Hugging Face repo {repo_id_or_path!r}: {error}"
            ) from download_error
        raise RuntimeError(
            f"Coordinator rank failed to download Hugging Face repo {repo_id_or_path!r}: {error}"
        )


def materialize_hf_snapshot(
    repo_id_or_path: str | os.PathLike[str],
    *,
    revision: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    allow_patterns: str | Sequence[str] | None = None,
    ignore_patterns: str | Sequence[str] | None = None,
    required_files: Sequence[str] = (),
    local_files_only: bool | None = None,
    token: str | bool | None = None,
) -> Path:
    """Return a local snapshot for either a path or Hugging Face repo id.

    Remote downloads are coordinated by ``maybe_download_hf_repo_on_rank0``
    (default: per-node local-rank0) and then reopened in local-only mode,
    avoiding partial-cache races in multi-GPU jobs.
    """

    value = str(repo_id_or_path)
    path = Path(value).expanduser()
    if path.exists():
        resolved = resolve_hf_snapshot_path(path, required_files=required_files, local_files_only=True)
        if required_files and not _required_files_present(resolved, required_files):
            raise FileNotFoundError(f"snapshot {resolved} is missing required files: {list(required_files)}")
        return resolved.resolve()
    if not _is_probable_hf_repo_id(value):
        return path.resolve()

    if local_files_only is None:
        local_files_only = _str2bool(os.getenv("WORLDFOUNDRY_HF_LOCAL_FILES_ONLY", "false"))
    if not local_files_only:
        maybe_download_hf_repo_on_rank0(
            value,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            token=token,
        )
    snapshot = Path(
        _snapshot_download(
            repo_id=value,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=True,
            allow_patterns=_normalize_patterns(allow_patterns),
            ignore_patterns=_normalize_patterns(ignore_patterns),
            token=token,
        )
    ).expanduser()
    if required_files and not _required_files_present(snapshot, required_files):
        raise FileNotFoundError(f"snapshot {snapshot} is missing required files: {list(required_files)}")
    return snapshot.resolve()


__all__ = [
    "HF_DOWNLOAD_COORDINATOR_ENV",
    "HF_URI_SCHEME",
    "download_coordinator_mode",
    "hf_download_or_fpath",
    "materialize_hf_snapshot",
    "maybe_download_hf_repo_on_rank0",
    "resolve_hf_path",
    "resolve_hf_snapshot_path",
]

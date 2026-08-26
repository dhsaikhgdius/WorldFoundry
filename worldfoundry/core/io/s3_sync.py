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

"""Multi-rank S3 → local-cache sync utility used by examples and recipes."""

import base64
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import torch.distributed as dist
import tqdm

from worldfoundry.core.io.disk import (
    CACHE_MIN_FREE_ENV,
    cache_min_free_bytes,
    ensure_free_disk,
    raise_if_disk_space_error,
)
from worldfoundry.core.io.s3_filesystem import S3FileSystem

# Sidecar stamp written after a successful size/checksum validation so warm
# syncs can skip HEAD + SHA256 when the local bytes are unchanged (CA-05).
_S3_SYNC_STAMP_SUFFIX = ".wf-s3-stamp.json"


class ValidationError(RuntimeError):
    """Raised when downloaded file validation fails."""


def _shorten_path(path: str, max_len: int = 72) -> str:
    """Truncate a path to max_len characters, keeping head and tail with ' ... ' in the middle."""
    sep = " ... "
    if len(path) <= max_len:
        return path
    head_len = (max_len - len(sep)) // 2
    tail_len = max_len - head_len - len(sep)
    return f"{path[:head_len]}{sep}{path[-tail_len:]}"


def _compute_file_sha256_b64(file_path: str) -> str:
    """Compute SHA256 hash of a file and return base64-encoded digest."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as file:
        while True:
            chunk = file.read(8 * 1024 * 1024)
            if not chunk:
                break
            sha256.update(chunk)
    return base64.b64encode(sha256.digest()).decode("ascii")


def _stamp_path_for(local_path: str) -> str:
    return f"{local_path}{_S3_SYNC_STAMP_SUFFIX}"


def _local_identity(local_path: str) -> dict[str, int]:
    stat = os.stat(local_path)
    return {
        "local_size": int(stat.st_size),
        "local_mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
    }


def _read_validation_stamp(local_path: str) -> dict[str, object] | None:
    stamp_path = _stamp_path_for(local_path)
    if not os.path.isfile(stamp_path):
        return None
    try:
        with open(stamp_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_validation_stamp(
    local_path: str,
    *,
    content_length: int,
    checksum_sha256: str | None,
    etag: str | None,
) -> None:
    identity = _local_identity(local_path)
    payload = {
        **identity,
        "content_length": int(content_length),
        "checksum_sha256": checksum_sha256,
        "etag": etag,
    }
    stamp_path = _stamp_path_for(local_path)
    tmp_path = f"{stamp_path}.{os.getpid()}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
    os.replace(tmp_path, stamp_path)


def _validation_stamp_matches(local_path: str, stamp: dict[str, object]) -> bool:
    identity = _local_identity(local_path)
    try:
        return (
            int(stamp.get("local_size", -1)) == identity["local_size"]
            and int(stamp.get("local_mtime_ns", -1)) == identity["local_mtime_ns"]
            and int(stamp.get("content_length", -1)) == identity["local_size"]
        )
    except (TypeError, ValueError):
        return False


def _get_world_rank_robust() -> int:
    """Return the current torch-distributed rank, or 0 when distributed is not initialized."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _barrier_robust() -> None:
    """Issue a torch-distributed barrier, no-op when distributed is not initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def sync_s3_dir_to_local(
    s3_dir: str,
    s3_credential_path: str,
    cache_dir: str,
    max_workers: int = 32,
    show_progress: bool = True,
    verify_checksum: bool = True,
    desc: str = "Syncing from S3",
) -> None:
    """Mirror an S3 prefix to a local directory.

    Only rank 0 downloads; other ranks block on a barrier so the cache is
    fully populated before they read it. Local paths are a no-op.

    Warm syncs reuse a sidecar validation stamp (size/mtime + remote length /
    checksum) so unchanged local files skip HEAD + SHA256 (CA-05).

    Args:
        s3_dir: ``s3://`` prefix to mirror, or a local path (no-op).
        s3_credential_path: S3 credentials JSON.
        cache_dir: Local destination directory.
        max_workers: Max parallel downloads on rank 0.
        show_progress: Show a tqdm bar.
        verify_checksum: Validate size and (when available) FULL_OBJECT
            SHA256 of each downloaded file; one retry on mismatch.
        desc: Progress-bar label.

    Examples:

      >>> sync_s3_dir_to_local(
      ...     s3_dir="s3://bucket/assets",
      ...     s3_credential_path="credentials/s3_checkpoint.secret",
      ...     cache_dir="cache/WorldFoundry/assets",
      ... )
    """
    if not s3_dir.startswith("s3://"):
        assert os.path.exists(s3_dir), f"{s3_dir} is not a S3 path or a local path."
        return

    world_rank = _get_world_rank_robust()
    parsed_url = urlparse(s3_dir)
    bucket = parsed_url.netloc
    obj_prefix = parsed_url.path.lstrip("/").removesuffix("/")

    cache_dir = os.path.expanduser(cache_dir)
    min_bytes = cache_min_free_bytes()
    ensure_free_disk(
        cache_dir,
        required_bytes=min_bytes,
        label="S3 local cache",
        env_vars=("WORLDFOUNDRY_CACHE_DIR", CACHE_MIN_FREE_ENV),
        settings={"s3_dir": s3_dir, "cache_dir": cache_dir},
    )

    should_download = world_rank == 0
    s3_fs = S3FileSystem(credential_path=s3_credential_path) if should_download else None

    def _validate_local_file(local_path: str, key: str) -> None:
        """Validate local file using remote size and optional FULL_OBJECT SHA256 checksum."""
        if not verify_checksum:
            return
        stamp = _read_validation_stamp(local_path)
        if stamp is not None and _validation_stamp_matches(local_path, stamp):
            return
        assert s3_fs is not None
        metadata = s3_fs.head_object(s3_uri=f"s3://{bucket}/{key}", checksum_mode=True)

        remote_size = int(metadata["ContentLength"])
        local_size = os.path.getsize(local_path)
        if local_size != remote_size:
            raise ValidationError(f"File size mismatch for {local_path}")

        checksum_type = metadata.get("ChecksumType")
        remote_sha256 = metadata.get("ChecksumSHA256")
        if remote_sha256 and checksum_type == "FULL_OBJECT":
            local_sha256 = _compute_file_sha256_b64(local_path)
            if local_sha256 != remote_sha256:
                raise ValidationError(
                    f"SHA256 checksum mismatch for {local_path}, expected {remote_sha256}, got {local_sha256}"
                )
        _write_validation_stamp(
            local_path,
            content_length=remote_size,
            checksum_sha256=remote_sha256 if checksum_type == "FULL_OBJECT" else None,
            etag=metadata.get("ETag"),
        )

    def _download_one(obj_suffix: str, retries_left: int = 1) -> None:
        """Download one object and validate. Retry once on ValidationError."""
        assert s3_fs is not None
        dest_path = os.path.join(cache_dir, obj_suffix)
        key = f"{obj_prefix}/{obj_suffix}" if obj_prefix else obj_suffix
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        if not os.path.exists(dest_path):
            s3_obj = f"{s3_dir.removesuffix('/')}/{obj_suffix}"
            tqdm.tqdm.write(f"Downloading: {_shorten_path(s3_obj)}")
            try:
                s3_fs.download_to_local(s3_uri=s3_obj, local_path=dest_path)
            except Exception as exc:
                raise_if_disk_space_error(
                    exc,
                    path=dest_path,
                    label="S3 local cache",
                    required_bytes=min_bytes,
                    env_vars=("WORLDFOUNDRY_CACHE_DIR", CACHE_MIN_FREE_ENV),
                    settings={"s3_dir": s3_dir, "cache_dir": cache_dir},
                )
                raise

        try:
            _validate_local_file(local_path=dest_path, key=key)
        except ValidationError as exc:
            stamp_path = _stamp_path_for(dest_path)
            if os.path.exists(stamp_path):
                os.remove(stamp_path)
            if retries_left > 0:
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                _download_one(obj_suffix=obj_suffix, retries_left=retries_left - 1)
            else:
                raise exc

    try:
        if should_download:
            assert s3_fs is not None
            object_suffixes = s3_fs.list_files_recursive(s3_dir=s3_dir)
            if object_suffixes:
                worker_count = min(max(1, max_workers), len(object_suffixes))
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    futures = [executor.submit(_download_one, obj_suffix) for obj_suffix in object_suffixes]
                    with tqdm.tqdm(
                        total=len(object_suffixes),
                        desc=desc,
                        disable=not show_progress,
                    ) as pbar:
                        for future in as_completed(futures):
                            future.result()
                            pbar.update(1)
    finally:
        if s3_fs is not None:
            s3_fs.close()

    _barrier_robust()

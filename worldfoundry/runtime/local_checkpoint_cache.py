"""Optional checkpoint staging for latency-sensitive resident runtimes.

Deployments may explicitly stage immutable model files into a faster storage
tier before loading. Rank zero performs the copy once and publishes the
resolved path to its peers. The open-source runtime does not guess storage
topology from filesystem paths and staging is disabled by default.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

logger = logging.getLogger(__name__)

_READY_FILE = ".worldfoundry-local-cache.json"


@contextmanager
def _publish_lock(cache_root: Path, target_name: str) -> Iterator[None]:
    """Serialize the publish step across unrelated processes via ``flock``.

    Staging copies may run concurrently, but replacing the published target
    directory must be exclusive: without the lock, one process could delete a
    directory that a concurrent process just published and started reading.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX platforms
        yield
        return
    lock_path = cache_root / f".{target_name}.lock"
    with open(lock_path, "w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _enabled(source: Path) -> bool:
    del source
    raw = os.getenv("WORLDFOUNDRY_REALTIME_STAGE_CHECKPOINT", "0").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        "WORLDFOUNDRY_REALTIME_STAGE_CHECKPOINT must be a boolean value."
    )


def _selected_files(source: Path, include_paths: tuple[str, ...]) -> list[Path]:
    if not include_paths:
        return [item for item in source.rglob("*") if item.is_file() and not item.is_symlink()]
    selected: set[Path] = set()
    for relative in include_paths:
        item = source / relative
        if item.is_file() and not item.is_symlink():
            selected.add(item)
        elif item.is_dir():
            selected.update(
                child for child in item.rglob("*") if child.is_file() and not child.is_symlink()
            )
    return list(selected)


def _directory_size(path: Path, include_paths: tuple[str, ...] = ()) -> int:
    return sum(item.stat().st_size for item in _selected_files(path, include_paths))


def _cache_target(
    source: Path,
    cache_root: Path,
    include_paths: tuple[str, ...] = (),
) -> Path:
    identity = json.dumps([str(source), sorted(include_paths)], separators=(",", ":"))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    safe_name = "-".join(part for part in source.name.split() if part) or "checkpoint"
    return cache_root / f"{safe_name}-{digest}"


def _copy_tree_parallel(
    source: Path,
    target: Path,
    *,
    include_paths: tuple[str, ...] = (),
) -> None:
    """Copy independent checkpoint shards concurrently from distributed storage."""

    files = _selected_files(source, include_paths)
    directories = sorted(
        {
            parent
            for item in files
            for parent in item.parents
            if parent != source and source in parent.parents
        },
        key=lambda item: len(item.parts),
    )
    target.mkdir(parents=True, exist_ok=False)
    for directory in directories:
        (target / directory.relative_to(source)).mkdir(parents=True, exist_ok=True)

    links = [] if include_paths else [item for item in source.rglob("*") if item.is_symlink()]
    for link in links:
        destination = target / link.relative_to(source)
        destination.symlink_to(os.readlink(link), target_is_directory=link.is_dir())

    files.sort(key=lambda item: item.stat().st_size, reverse=True)
    workers = max(
        int(os.getenv("WORLDFOUNDRY_REALTIME_CHECKPOINT_COPY_WORKERS", "4") or "4"),
        1,
    )

    def copy_file(item: Path) -> None:
        shutil.copy2(item, target / item.relative_to(source))

    with ThreadPoolExecutor(max_workers=min(workers, max(len(files), 1))) as executor:
        list(executor.map(copy_file, files))


def _is_ready(target: Path, source: Path, required_paths: tuple[str, ...]) -> bool:
    ready = target / _READY_FILE
    if not ready.is_file():
        return False
    try:
        payload = json.loads(ready.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if payload.get("source") != str(source):
        return False
    if not all((target / relative).exists() for relative in required_paths):
        return False
    size_bytes = payload.get("size_bytes")
    if isinstance(size_bytes, int) and size_bytes > 0:
        try:
            actual = _directory_size(target)
        except OSError:
            return False
        # Allow small metadata overhead from the ready file itself.
        if actual + 4096 < size_bytes:
            return False
    return True


def _stage_rank_zero(
    source: Path,
    *,
    cache_root: Path,
    required_paths: tuple[str, ...],
    include_paths: tuple[str, ...],
) -> Path:
    target = _cache_target(source, cache_root, include_paths)
    if _is_ready(target, source, required_paths):
        return target

    cache_root.mkdir(parents=True, exist_ok=True)
    source_bytes = _directory_size(source, include_paths)
    free_bytes = shutil.disk_usage(cache_root).free
    if source_bytes > int(free_bytes * 0.9):
        raise OSError(
            f"Not enough node-local space to stage {source} "
            f"({source_bytes / 1024**3:.1f} GiB required, {free_bytes / 1024**3:.1f} GiB free)."
        )

    temporary = cache_root / f".{target.name}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    logger.info(
        "staging %.1f GiB checkpoint from %s to %s",
        source_bytes / 1024**3,
        source,
        target,
    )
    try:
        _copy_tree_parallel(source, temporary, include_paths=include_paths)
        (temporary / _READY_FILE).write_text(
            json.dumps(
                {
                    "source": str(source),
                    "size_bytes": source_bytes,
                    "include_paths": list(include_paths),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        with _publish_lock(cache_root, target.name):
            if _is_ready(target, source, required_paths):
                # A concurrent process published a valid tree while we were
                # copying. Reuse it and discard the duplicate; deleting the
                # published tree here would break readers already loading it.
                shutil.rmtree(temporary, ignore_errors=True)
                return target
            if target.exists():
                shutil.rmtree(target)
            temporary.rename(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def stage_checkpoint_for_realtime(
    source: str | Path,
    *,
    required_paths: Iterable[str] = (),
    include_paths: Iterable[str] = (),
    distributed: Any = None,
) -> Path:
    """Return an immutable node-local view of ``source`` when configured.

    ``distributed`` may be ``torch.distributed``.  All ranks call this
    function in the same order; only rank zero performs I/O and broadcasts the
    resulting path or error.
    """

    resolved = Path(source).expanduser().resolve()
    if not _enabled(resolved):
        return resolved
    cache_root_value = os.getenv("WORLDFOUNDRY_REALTIME_LOCAL_CHECKPOINT_CACHE")
    if not cache_root_value:
        raise ValueError(
            "Set WORLDFOUNDRY_REALTIME_LOCAL_CHECKPOINT_CACHE to an explicit "
            "staging directory when checkpoint staging is enabled."
        )
    cache_root = Path(cache_root_value).expanduser()
    required = tuple(str(item) for item in required_paths)
    included = tuple(sorted(str(item) for item in include_paths))
    missing_source = [relative for relative in required if not (resolved / relative).exists()]
    if missing_source:
        raise FileNotFoundError(
            f"Checkpoint is incomplete at {resolved}; missing: {', '.join(missing_source)}"
        )
    is_distributed = bool(
        distributed is not None
        and distributed.is_available()
        and distributed.is_initialized()
        and distributed.get_world_size() > 1
    )
    rank = int(distributed.get_rank()) if is_distributed else 0
    status: list[dict[str, str] | None] = [None]
    if rank == 0:
        try:
            target = _stage_rank_zero(
                resolved,
                cache_root=cache_root,
                required_paths=required,
                include_paths=included,
            )
            status[0] = {"path": str(target), "error": ""}
        except Exception as exc:
            status[0] = {"path": "", "error": f"{type(exc).__name__}: {exc}"}
    if is_distributed:
        distributed.broadcast_object_list(status, src=0)
    payload = status[0] or {}
    if payload.get("error"):
        raise RuntimeError(f"Realtime checkpoint staging failed: {payload['error']}")
    target = Path(payload.get("path") or resolved)
    if not all((target / relative).exists() for relative in required):
        raise FileNotFoundError(f"Staged checkpoint is incomplete: {target}")
    return target


def resolve_local_checkpoint_cache_root(environ: Mapping[str, str] | None = None) -> Path | None:
    """Return the configured staging root, or ``None`` when staging is unset."""

    env = os.environ if environ is None else environ
    raw = str(env.get("WORLDFOUNDRY_REALTIME_LOCAL_CHECKPOINT_CACHE") or "").strip()
    return Path(raw).expanduser() if raw else None


def _publish_lock_held(cache_root: Path, target_name: str) -> bool:
    """Return True when another process currently holds the publish flock."""

    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX platforms
        return False
    lock_path = cache_root / f".{target_name}.lock"
    if not lock_path.exists():
        return False
    try:
        with open(lock_path, "w", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            fcntl.flock(handle, fcntl.LOCK_UN)
    except OSError:
        return True
    return False


def gc_local_checkpoint_cache(
    *,
    cache_root: Path | None = None,
    keep_newest: int | None = None,
    dry_run: bool = False,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Remove stale staged checkpoint trees, oldest first (LRU by mtime).

    ``keep_newest`` retains the N most recently modified trees (default from
    ``WORLDFOUNDRY_LOCAL_CHECKPOINT_CACHE_KEEP`` or 2). Dot-entries (ready /
    lock / tmp sidecars) are never candidates, and trees whose publish flock
    is currently held by another process are skipped rather than deleted.
    """

    env = os.environ if environ is None else environ
    root = cache_root or resolve_local_checkpoint_cache_root(env)
    if keep_newest is None:
        try:
            keep_newest = max(int(env.get("WORLDFOUNDRY_LOCAL_CHECKPOINT_CACHE_KEEP", "2") or "2"), 0)
        except ValueError:
            keep_newest = 2
    report: dict[str, Any] = {
        "cache_root": None if root is None else str(root),
        "kept": [],
        "removed": [],
        "locked": [],
        "errors": [],
        "dry_run": dry_run,
        "keep_newest": keep_newest,
    }
    if root is None or not root.is_dir():
        return report

    candidates: list[tuple[float, Path]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        try:
            mtime = child.stat().st_mtime
        except OSError as exc:
            report["errors"].append(f"{child}: {exc}")
            continue
        candidates.append((mtime, child))
    candidates.sort(key=lambda item: item[0], reverse=True)
    for index, (_, path) in enumerate(candidates):
        if index < keep_newest:
            report["kept"].append(str(path))
            continue
        if _publish_lock_held(root, path.name):
            report["locked"].append(str(path))
            continue
        if dry_run:
            report["removed"].append(str(path))
            continue
        try:
            shutil.rmtree(path)
            report["removed"].append(str(path))
        except OSError as exc:
            report["errors"].append(f"{path}: {exc}")
    return report


def _main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="WorldFoundry local checkpoint cache utilities")
    parser.add_argument("--gc", action="store_true", help="Garbage-collect stale staged checkpoint trees")
    parser.add_argument("--dry-run", action="store_true", help="Report candidates without deleting")
    parser.add_argument("--keep-newest", type=int, default=None, help="Retain the N newest trees")
    parser.add_argument("--json", action="store_true", help="Emit a JSON report")
    args = parser.parse_args(argv)
    if not args.gc:
        parser.error("specify --gc")
    report = gc_local_checkpoint_cache(keep_newest=args.keep_newest, dry_run=args.dry_run)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"cache root: {report['cache_root']}")
        for label in ("kept", "removed", "locked"):
            print(f"{label} ({len(report[label])}):")
            for path in report[label]:
                print(f"  - {path}")
        for err in report["errors"]:
            print(f"error: {err}", file=sys.stderr)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "gc_local_checkpoint_cache",
    "resolve_local_checkpoint_cache_root",
    "stage_checkpoint_for_realtime",
]

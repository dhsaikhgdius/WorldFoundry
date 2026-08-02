"""Small subprocess helpers for model-owned in-tree inference runtimes."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from worldfoundry.core.io import file_sha256

MEDIA_SUFFIXES = frozenset({".mp4", ".mov", ".webm", ".gif", ".png", ".jpg", ".jpeg"})


def require_path(value: Any, label: str, *, kind: str | None = None) -> Path:
    """Resolve an existing path with a model-specific diagnostic."""
    if value is None or str(value).strip() == "":
        raise ValueError(f"missing required runtime path: {label}")
    path = Path(str(value)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    if kind == "dir" and not path.is_dir():
        raise FileNotFoundError(f"{label} is not a directory: {path}")
    return path


def ensure_in_tree_runtime(path: Path, *, package_file: str | Path) -> Path:
    """Reject external source checkouts; model code must be owned by this tree."""
    package_root = Path(package_file).resolve().parent
    resolved = path.resolve()
    if resolved != package_root and package_root not in resolved.parents:
        raise ValueError(f"runtime source must be in-tree under {package_root}, got {resolved}")
    return resolved


def newest_media(
    roots: Iterable[str | Path],
    *,
    since: float,
    preferred_names: Sequence[str] = (),
) -> Path | None:
    """Return the newest fresh media artifact from model-owned output roots."""
    candidates: list[Path] = []
    for raw_root in roots:
        root = Path(raw_root).expanduser()
        if not root.exists():
            continue
        if root.is_file():
            candidates.append(root)
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in MEDIA_SUFFIXES:
                continue
            try:
                if path.stat().st_mtime >= since - 1.0:
                    candidates.append(path)
            except OSError:
                continue
    candidates.sort(key=lambda item: (item.stat().st_mtime, item.stat().st_size), reverse=True)
    for name in preferred_names:
        for path in candidates:
            if path.name == name:
                return path
    return candidates[0] if candidates else None


def execute_in_tree(
    command: Sequence[str | Path],
    *,
    cwd: str | Path,
    output_path: str | Path,
    search_roots: Sequence[str | Path] = (),
    env: Mapping[str, Any] | None = None,
    python_paths: Sequence[str | Path] = (),
    preferred_names: Sequence[str] = (),
) -> dict[str, Any]:
    """Run one model-owned CLI and normalize its generated artifact."""
    workdir = require_path(cwd, "in-tree runtime root", kind="dir")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_output = None
    if output.is_file():
        previous_stat = output.stat()
        previous_output = (previous_stat.st_mtime_ns, previous_stat.st_size)
    log_path = output.with_suffix(output.suffix + ".log")
    process_env = os.environ.copy()
    if python_paths:
        process_env["PYTHONPATH"] = os.pathsep.join(
            [*(str(Path(item).resolve()) for item in python_paths), process_env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
    if env:
        process_env.update({str(key): str(value) for key, value in env.items()})
    rendered = [str(item) for item in command]
    started = time.time()
    completed = subprocess.run(
        rendered,
        cwd=workdir,
        env=process_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    log_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        return {
            "status": "failed",
            "error": f"in-tree model CLI exited with code {completed.returncode}; see {log_path}",
            "artifact_path": str(output),
            "metadata": {"command": rendered, "cwd": str(workdir), "log_path": str(log_path)},
        }
    output_is_fresh = False
    if output.is_file():
        current_stat = output.stat()
        current_output = (current_stat.st_mtime_ns, current_stat.st_size)
        output_is_fresh = (
            current_stat.st_mtime >= started - 1.0
            and (previous_output is None or current_output != previous_output)
        )
    produced = output if output_is_fresh else newest_media(
        search_roots, since=started, preferred_names=preferred_names
    )
    if produced is None:
        return {
            "status": "failed",
            "error": f"in-tree model CLI completed but produced no media artifact; see {log_path}",
            "artifact_path": str(output),
            "metadata": {"command": rendered, "cwd": str(workdir), "log_path": str(log_path)},
        }
    if produced.resolve() != output:
        shutil.copy2(produced, output)
    return {
        "status": "succeeded",
        "video": str(output),
        "artifact_path": str(output),
        "artifact_sha256": file_sha256(output),
        "backend_quality": "in_tree_official_runtime",
        "metadata": {
            "command": rendered,
            "cwd": str(workdir),
            "source_artifact": str(produced),
            "log_path": str(log_path),
        },
    }


__all__ = [
    "ensure_in_tree_runtime",
    "execute_in_tree",
    "newest_media",
    "require_path",
]

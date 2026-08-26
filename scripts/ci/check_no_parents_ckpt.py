#!/usr/bin/env python3
"""SY-04: refuse ``Path(...).parents[N] / "ckpt"`` hardcoding outside vendored runtimes.

Checkpoint roots must come from ``WORLDFOUNDRY_CKPT_DIR`` / ``checkpoint_root_path`` /
``resolve_profile_checkpoint``, not from walking ``__file__`` parents into a
sibling ``ckpt`` directory (layout drifts across install modes).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOT = REPO_ROOT / "worldfoundry"
# ``foo_runtime/`` trees are upstream vendors; leave their layout alone.
_RUNTIME_DIR = re.compile(r"(^|/)[^/]+_runtime(/|$)")
# parents[N] / "ckpt"  or  parents[N].resolve() / "ckpt"
_PARENTS_CKPT = re.compile(
    r"""parents\s*\[\s*\d+\s*\](?:\s*\.\s*resolve\s*\(\s*\))?\s*/\s*["']ckpt["']"""
)


def _iter_python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if _RUNTIME_DIR.search(rel):
            continue
        files.append(path)
    return files


def find_parents_ckpt_hits(repo_root: Path | None = None) -> list[tuple[str, int, str]]:
    root = Path(repo_root or REPO_ROOT)
    scan = root / "worldfoundry"
    hits: list[tuple[str, int, str]] = []
    for path in _iter_python_files(scan):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _PARENTS_CKPT.search(line):
                hits.append((path.relative_to(root).as_posix(), lineno, line.strip()))
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    hits = find_parents_ckpt_hits(args.repo_root)
    if not hits:
        print("ok: no parents[N]/ckpt hardcoding under worldfoundry/ (excluding *_runtime/)")
        return 0
    print("forbidden parents[N]/ckpt hardcoding (use checkpoint_root_path / resolve_profile_checkpoint):", file=sys.stderr)
    for rel, lineno, line in hits:
        print(f"  {rel}:{lineno}: {line}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

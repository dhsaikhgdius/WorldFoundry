#!/usr/bin/env python3
"""DA-01: refuse growth of tracked ``worldfoundry/data/test_cases`` blobs.

The directory is already ~500 git-tracked files / ~338 MiB and is also listed in
``.gitignore`` (which does not untrack existing paths). Rewriting history to
drop the blobs is deferred to a later LFS/HF migration; until then CI pins the
current baseline so PRs cannot silently enlarge the checkout.

Baselines (round7 DA-01):
  - max tracked files: 500
  - max tracked bytes: 338 MiB
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_CASES_PREFIX = "worldfoundry/data/test_cases"
MAX_TRACKED_FILES = 500
MAX_TRACKED_BYTES = 338 * 1024 * 1024


def _git_ls_files(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z", "--", TEST_CASES_PREFIX],
        check=True,
        capture_output=True,
    )
    return [item for item in result.stdout.decode("utf-8", errors="surrogateescape").split("\0") if item]


def measure_tracked_test_cases(repo_root: Path | None = None) -> tuple[int, int]:
    """Return ``(file_count, total_bytes)`` for tracked test_cases paths."""

    root = Path(repo_root or REPO_ROOT)
    paths = _git_ls_files(root)
    total_bytes = 0
    for rel in paths:
        path = root / rel
        if path.is_file():
            total_bytes += path.stat().st_size
    return len(paths), total_bytes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--max-files", type=int, default=MAX_TRACKED_FILES)
    parser.add_argument("--max-bytes", type=int, default=MAX_TRACKED_BYTES)
    args = parser.parse_args(argv)

    file_count, total_bytes = measure_tracked_test_cases(args.repo_root)
    print(
        f"tracked {TEST_CASES_PREFIX}: {file_count} files / {total_bytes} bytes "
        f"(limits {args.max_files} files / {args.max_bytes} bytes)"
    )
    errors: list[str] = []
    if file_count > args.max_files:
        errors.append(
            f"tracked file count {file_count} exceeds baseline {args.max_files}; "
            "do not add more blobs under worldfoundry/data/test_cases "
            "(migrate to LFS/HF instead)."
        )
    if total_bytes > args.max_bytes:
        errors.append(
            f"tracked byte size {total_bytes} exceeds baseline {args.max_bytes}; "
            "do not enlarge worldfoundry/data/test_cases (migrate to LFS/HF instead)."
        )
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

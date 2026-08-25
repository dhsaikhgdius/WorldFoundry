#!/usr/bin/env python3
"""Ensure a single runner_common import at module top."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "worldfoundry/evaluation/tasks/execution/runners"
IMPORT_RE = re.compile(
    r"^from worldfoundry\.evaluation\.tasks\.execution\.framework\.runner_common import .+\n",
    re.M,
)
STANDARD_IMPORT = (
    "from worldfoundry.evaluation.tasks.execution.framework.runner_common import "
    "SCORECARD_SCHEMA_VERSION, VIDEO_SUFFIXES\n"
)


def header_end(lines: list[str]) -> int:
    i = 0
    if lines and lines[0].startswith("#!"):
        i = 1
    while i < len(lines):
        s = lines[i].strip()
        if not s or s.startswith("from __future__"):
            i += 1
            continue
        if s.startswith(('"""', "'''")):
            q = s[:3]
            i += 1
            while i < len(lines) and q not in lines[i]:
                i += 1
            i += 1
            continue
        break
    return i


def fix(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if "SCORECARD_SCHEMA_VERSION" not in text and "VIDEO_SUFFIXES" not in text:
        return False
    stripped = IMPORT_RE.sub("", text)
    lines = stripped.splitlines(keepends=True)
    i = header_end(lines)
    lines = lines[:i] + [STANDARD_IMPORT, "\n"] + lines[i:]
    new_text = "".join(lines)
    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
        return True
    return False


def main() -> None:
    n = sum(1 for p in ROOT.rglob("*.py") if "/runtime/" not in p.as_posix() and fix(p))
    print(n)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Remove stray runner_common imports and keep a single top-level import when needed."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "worldfoundry/evaluation/tasks/execution/runners"
IMPORT_LINE = (
    "from worldfoundry.evaluation.tasks.execution.framework.runner_common import "
    "SCORECARD_SCHEMA_VERSION, VIDEO_SUFFIXES\n"
)
IMPORT_RE = re.compile(
    r"^from worldfoundry\.evaluation\.tasks\.execution\.framework\.runner_common import .+\n",
    re.M,
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


def needs_runner_common(text: str) -> bool:
    if "from worldfoundry.evaluation.reporting.scorecard import SCORECARD_SCHEMA_VERSION" in text:
        uses_scorecard = False
    else:
        uses_scorecard = "SCORECARD_SCHEMA_VERSION" in text
    uses_video = "VIDEO_SUFFIXES" in text
    return uses_scorecard or uses_video


def fix(path: Path) -> bool:
    if "/runtime/" in path.as_posix():
        return False
    original = path.read_text(encoding="utf-8")
    text = IMPORT_RE.sub("", original)
    if not needs_runner_common(text):
        if text != original:
            path.write_text(text, encoding="utf-8")
            return True
        return False
    lines = text.splitlines(keepends=True)
    insert_at = header_end(lines)
    lines = lines[:insert_at] + [IMPORT_LINE, "\n"] + lines[insert_at:]
    text = "".join(lines)
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    changed = sum(1 for path in ROOT.rglob("*.py") if fix(path))
    print(f"fixed {changed} files")


if __name__ == "__main__":
    main()

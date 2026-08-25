#!/usr/bin/env python3
"""Remove mid-module runner_common import fragments left by the codemod."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNERS_ROOT = REPO_ROOT / "worldfoundry/evaluation/tasks/execution/runners"
IMPORT_START = "from worldfoundry.evaluation.tasks.execution.framework.runner_common import"
ORPHAN_LINE = re.compile(
    r"^\s+(SCORECARD_SCHEMA_VERSION|VIDEO_SUFFIXES|resolve_env_path|build_import_metric_rows|build_video_coverage),?\s*$"
)


def header_end_index(lines: list[str]) -> int:
    insert_at = 0
    if lines and lines[0].startswith("#!"):
        insert_at = 1
    while insert_at < len(lines):
        stripped = lines[insert_at].strip()
        if stripped == "" or stripped.startswith("from __future__"):
            insert_at += 1
            continue
        if stripped.startswith(('"""', "'''")):
            quote = stripped[:3]
            insert_at += 1
            while insert_at < len(lines) and quote not in lines[insert_at]:
                insert_at += 1
            insert_at += 1
            continue
        if stripped.startswith("import ") or stripped.startswith("from "):
            insert_at += 1
            continue
        break
    return insert_at


def clean_file(path: Path) -> bool:
    if "/runtime/" in path.as_posix():
        return False
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    header_end = header_end_index(lines)
    cleaned: list[str] = []
    skip_until_close = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if skip_until_close:
            if stripped.endswith(")"):
                skip_until_close = False
            continue
        if idx >= header_end and stripped.startswith(IMPORT_START):
            if stripped.endswith("("):
                skip_until_close = True
            continue
        if idx >= header_end and ORPHAN_LINE.match(line):
            continue
        if idx >= header_end and stripped == ")":
            continue
        cleaned.append(line)
    text = "".join(cleaned)
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    changed = sum(1 for path in RUNNERS_ROOT.rglob("*.py") if clean_file(path))
    print(f"cleaned {changed} files")


if __name__ == "__main__":
    main()

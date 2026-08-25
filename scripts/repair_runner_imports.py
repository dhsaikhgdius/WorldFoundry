#!/usr/bin/env python3
"""Re-apply runner_common imports at module top and remove stray copies."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNERS_ROOT = REPO_ROOT / "worldfoundry/evaluation/tasks/execution/runners"
IMPORT_LINE = (
    "from worldfoundry.evaluation.tasks.execution.framework.runner_common import "
    "SCORECARD_SCHEMA_VERSION, VIDEO_SUFFIXES, resolve_env_path"
)
IMPORT_RE = re.compile(
    r"^from worldfoundry\.evaluation\.tasks\.execution\.framework\.runner_common import .+\n",
    re.M,
)


def insert_import(lines: list[str]) -> list[str]:
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
        break
    return lines[:insert_at] + [IMPORT_LINE + "\n", "\n"] + lines[insert_at:]


def needs_import(text: str) -> bool:
    return any(
        token in text
        for token in ("SCORECARD_SCHEMA_VERSION", "VIDEO_SUFFIXES", "resolve_env_path(")
    )


def patch_file(path: Path) -> bool:
    if "/runtime/" in path.as_posix():
        return False
    original = path.read_text(encoding="utf-8")
    if not needs_import(original):
        return False
    text = IMPORT_RE.sub("", original)
    lines = text.splitlines(keepends=True)
    lines = insert_import(lines)
    text = "".join(lines)
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    changed = 0
    for path in sorted(RUNNERS_ROOT.rglob("*.py")):
        if patch_file(path):
            changed += 1
    print(f"repaired imports in {changed} files")


if __name__ == "__main__":
    main()

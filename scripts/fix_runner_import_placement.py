#!/usr/bin/env python3
"""Move misplaced runner_common imports to the module import section."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNERS_ROOT = REPO_ROOT / "worldfoundry/evaluation/tasks/execution/runners"
IMPORT_LINE = (
    "from worldfoundry.evaluation.tasks.execution.framework.runner_common import "
    "SCORECARD_SCHEMA_VERSION, VIDEO_SUFFIXES, resolve_env_path"
)


def fix_file(path: Path) -> bool:
    lines = path.read_text(encoding="utf-8").splitlines()
    indices = [i for i, line in enumerate(lines) if line.strip() == IMPORT_LINE]
    if not indices:
        return False
    if len(indices) == 1 and indices[0] < 40:
        return False

    # Remove all existing copies.
    filtered = [line for line in lines if line.strip() != IMPORT_LINE]
    text = "\n".join(filtered)
    if text and not text.endswith("\n"):
        text += "\n"
    lines = text.splitlines(keepends=True)

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

    new_lines = lines[:insert_at] + [IMPORT_LINE + "\n", "\n"] + lines[insert_at:]
    path.write_text("".join(new_lines), encoding="utf-8")
    return True


def main() -> None:
    changed = 0
    for path in sorted(RUNNERS_ROOT.rglob("*.py")):
        if fix_file(path):
            changed += 1
    print(f"fixed {changed} files")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Safe codemod: centralize SCORECARD_SCHEMA_VERSION and VIDEO_SUFFIXES only."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNERS_ROOT = REPO_ROOT / "worldfoundry/evaluation/tasks/execution/runners"
FRAMEWORK_ROOT = REPO_ROOT / "worldfoundry/evaluation/tasks/execution/framework"
IMPORT_LINE = (
    "from worldfoundry.evaluation.tasks.execution.framework.runner_common import "
    "SCORECARD_SCHEMA_VERSION, VIDEO_SUFFIXES\n"
)

SCORECARD_RE = re.compile(r'^SCORECARD_SCHEMA_VERSION = "worldfoundry-scorecard"\n', re.M)
VIDEO_SUFFIXES_RE = re.compile(
    r'^VIDEO_SUFFIXES = frozenset\(\{[^}]+\}\)\n',
    re.M,
)


def insert_import(text: str) -> str:
    if "framework.runner_common import" in text:
        return text
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
    return "".join(lines[:insert_at] + [IMPORT_LINE, "\n"] + lines[insert_at:])


def patch_file(path: Path) -> bool:
    if "/runtime/" in path.as_posix():
        return False
    original = path.read_text(encoding="utf-8")
    if not SCORECARD_RE.search(original) and not VIDEO_SUFFIXES_RE.search(original):
        return original
    text = SCORECARD_RE.sub("", original)
    text = VIDEO_SUFFIXES_RE.sub("", text)
    if "SCORECARD_SCHEMA_VERSION" in text or "VIDEO_SUFFIXES" in text:
        text = insert_import(text)
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_official_runner() -> bool:
    path = FRAMEWORK_ROOT / "official_runner.py"
    text = path.read_text(encoding="utf-8")
    if "from worldfoundry.evaluation.reporting.scorecard import SCORECARD_SCHEMA_VERSION" in text:
        return False
    text = SCORECARD_RE.sub("", text)
    anchor = "from worldfoundry.evaluation.tasks.catalog.zoo_registry import load_benchmark_zoo_registry\n"
    if anchor not in text:
        return False
    text = text.replace(
        anchor,
        "from worldfoundry.evaluation.reporting.scorecard import SCORECARD_SCHEMA_VERSION\n" + anchor,
        1,
    )
    path.write_text(text, encoding="utf-8")
    return True


def main() -> None:
    changed = sum(1 for path in RUNNERS_ROOT.rglob("*.py") if patch_file(path))
    if patch_official_runner():
        changed += 1
    print(f"patched {changed} files")


if __name__ == "__main__":
    main()

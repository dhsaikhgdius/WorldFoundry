#!/usr/bin/env python3
"""Remove duplicate runner_common imports (keep only the first)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "worldfoundry/evaluation/tasks/execution/runners"
IMPORT_RE = re.compile(
    r"^from worldfoundry\.evaluation\.tasks\.execution\.framework\.runner_common import .+\n",
    re.M,
)


def fix(path: Path) -> bool:
    if "/runtime/" in path.as_posix():
        return False
    text = path.read_text(encoding="utf-8")
    matches = list(IMPORT_RE.finditer(text))
    if len(matches) <= 1:
        return False
    # Keep first, remove rest
    new = text
    for match in reversed(matches[1:]):
        new = new[: match.start()] + new[match.end() :]
    path.write_text(new, encoding="utf-8")
    return True


def main() -> None:
    n = sum(1 for p in ROOT.rglob("*.py") if fix(p))
    print(f"removed duplicate imports from {n} files")


if __name__ == "__main__":
    main()

"""EX-04 docs: contributors are pointed at the [eval_core] extra."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_PATHS = (
    REPO_ROOT / "CONTRIBUTING.md",
    REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "maintainers" / "contributing.mdx",
    REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "maintainers" / "contributing.zh.mdx",
)


def test_contributing_docs_mention_eval_core_extra() -> None:
    for path in DOC_PATHS:
        text = path.read_text(encoding="utf-8")
        assert ".[eval_core]" in text or '".[eval_core]"' in text, path
        assert "test-eval-core" in text, path
        assert "download.pytorch.org/whl/cpu" in text, path

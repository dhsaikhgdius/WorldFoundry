from __future__ import annotations

from pathlib import Path


def _lock_path(tier: str) -> Path:
    return Path(__file__).resolve().parents[2] / "requirements" / "lock" / f"worldfoundry-unified.{tier}.lock.txt"


def _resolved_requirement_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("--"):
            continue
        lines.append(line)
    return lines


def test_cu128_unified_lock_is_populated_with_torch() -> None:
    """I-05: cu128 lock must be a real uv compile, not a placeholder stub."""

    text = _lock_path("cu128").read_text(encoding="utf-8")
    assert "Do not invent" not in text
    assert "--index-url https://download.pytorch.org/whl/cu128" in text
    resolved = _resolved_requirement_lines(text)
    assert len(resolved) > 50
    assert any(line.startswith("torch==") for line in resolved)
    assert "clip @ git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6" in text

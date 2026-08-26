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


def test_cu121_and_cu124_unified_locks_are_populated_with_index_torch() -> None:
    """I-05 follow-up: older CUDA tiers now have real compiles (not invented stubs)."""

    expected = {
        "cu121": ("https://download.pytorch.org/whl/cu121", "torch==2.5.1"),
        "cu124": ("https://download.pytorch.org/whl/cu124", "torch==2.6.0"),
    }
    for tier, (index, torch_pin) in expected.items():
        text = _lock_path(tier).read_text(encoding="utf-8")
        assert "Do not invent" not in text, tier
        assert f"--index-url {index}" in text, tier
        resolved = _resolved_requirement_lines(text)
        assert len(resolved) > 50, tier
        assert any(line.startswith(torch_pin) for line in resolved), (tier, torch_pin)
        assert "clip @ git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6" in text

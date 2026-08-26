"""Contract: flash-attn installer prefers wheels before source builds (I-06)."""

from __future__ import annotations

from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "setup" / "install_flash_attn.sh"


def test_flash_attn_installer_prefers_wheels_and_pins_buckets() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "WORLDFOUNDRY_FLASH_ATTN_FORCE_BUILD" in text
    assert "--only-binary=:all:" in text
    assert "flash-attn>=2.5,<2.6" in text
    assert "flash-attn>=2.8,<2.9" in text
    assert "resolve_cuda_home" in text
    assert "install_wheel" in text
    assert "install_from_source" in text

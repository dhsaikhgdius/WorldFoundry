"""I-02: conda_install second pip pass constrains the CUDA-index torch stack."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "setup" / "conda_install.sh"


def test_i02_second_pip_uses_torch_constraint() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "TORCH_CONSTRAINT_FILE=" in text
    assert '--constraint "$TORCH_CONSTRAINT_FILE"' in text
    assert 'lines.append(f"{name}=={version}")' in text
    # Post-install CUDA tier assertion against torch.version.cuda.
    assert "torch.version.cuda" in text
    assert "does not match selected tier" in text

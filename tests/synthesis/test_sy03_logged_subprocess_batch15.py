"""SY-03 batch15: model_zoo / embodied asset scripts use run_logged_subprocess."""

from __future__ import annotations

from pathlib import Path


def test_sy03_batch15_scripts_wire_run_logged_subprocess() -> None:
    """Source contract: download/parity scripts must wire run_logged_subprocess."""
    root = Path(__file__).resolve().parents[2]
    for rel in (
        "scripts/model_zoo/matrix_game2_demo_common.py",
        "scripts/model_zoo/matrix_game_parity_worker.py",
        "scripts/model_zoo/materialize_base_model_assets.py",
        "scripts/model_zoo/download_checkpoints.py",
        "scripts/embodied/prepare_official_assets.py",
        "scripts/setup/download_embodied_action_official_assets.py",
    ):
        text = (root / rel).read_text(encoding="utf-8")
        assert "run_logged_subprocess" in text, rel
        assert "from worldfoundry.core.process import run_logged_subprocess" in text, rel
        assert "subprocess.run(" not in text, rel

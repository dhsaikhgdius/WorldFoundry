"""SY-03 batch16: in_tree_cli + docker mirror use run_logged_subprocess."""

from __future__ import annotations

from pathlib import Path


def test_sy03_batch16_in_tree_and_mirror_wire_run_logged_subprocess() -> None:
    root = Path(__file__).resolve().parents[2]
    for rel in (
        "worldfoundry/runtime/in_tree_cli.py",
        "scripts/embodied/mirror_docker_images.py",
    ):
        text = (root / rel).read_text(encoding="utf-8")
        assert "run_logged_subprocess" in text, rel
        assert "from worldfoundry.core.process import run_logged_subprocess" in text, rel
        assert "subprocess.run(" not in text, rel
        assert "subprocess.Popen(" not in text, rel

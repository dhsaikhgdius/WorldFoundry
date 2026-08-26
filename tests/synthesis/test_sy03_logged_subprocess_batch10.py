"""SY-03 batch10: CLI downloads and World Explorer cmake use run_logged_subprocess."""

from __future__ import annotations

from pathlib import Path


def test_sy03_batch10_cli_and_explorer_wire_run_logged_subprocess() -> None:
    root = Path(__file__).resolve().parents[2]
    cli = (root / "worldfoundry/cli/models.py").read_text(encoding="utf-8")
    assert "run_logged_subprocess" in cli
    assert "from worldfoundry.core.process import run_logged_subprocess" in cli
    assert "subprocess.run(" not in cli

    explorer = (root / "worldfoundry/studio/native/world_explorer/__main__.py").read_text(
        encoding="utf-8"
    )
    assert "run_logged_subprocess" in explorer
    assert "from worldfoundry.core.process import run_logged_subprocess" in explorer
    assert "subprocess.run(" not in explorer
    assert "cmake_configure.stdout.log" in explorer
    assert "cmake_build.stdout.log" in explorer

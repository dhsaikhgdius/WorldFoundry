"""DX: project_codex_env.sh must not embed host-specific paths."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "dev" / "project_codex_env.sh"


def test_project_codex_env_has_no_host_specific_paths() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "/mnt/cpfs" not in text
    assert "yangboxue" not in text
    assert "juanxi" not in text


def test_project_codex_env_removes_external_absolute_symlink(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    # Run the script against a temporary fake repo root by copying the script
    # into a temp tree that mirrors scripts/dev/../..
    fake_root = tmp_path / "repo"
    script_dir = fake_root / "scripts" / "dev"
    script_dir.mkdir(parents=True)
    script_path = script_dir / "project_codex_env.sh"
    script_path.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR)

    codex_home = fake_root / ".codex"
    codex_home.mkdir()
    external = tmp_path / "external-codex" / "sessions"
    external.mkdir(parents=True)
    link = codex_home / "sessions"
    link.symlink_to(external)

    monkeypatch.delenv("CODEX_PERSIST_HOME", raising=False)
    completed = subprocess.run(
        ["bash", str(script_path)],
        cwd=fake_root,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        env={**os.environ, "HOME": str(tmp_path / "home")},
    )
    assert completed.returncode == 0, completed.stderr
    assert not link.is_symlink()
    assert (codex_home / "sessions").is_dir()

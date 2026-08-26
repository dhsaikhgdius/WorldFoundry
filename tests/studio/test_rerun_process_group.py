from __future__ import annotations

import signal
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from worldfoundry.studio.visualization.backends import frontends as frontends_mod


def test_require_rerun_executable_accepts_path_or_which(tmp_path) -> None:
    binary = tmp_path / "rerun"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    frontends_mod._require_rerun_executable(str(binary))
    with patch.object(frontends_mod.shutil, "which", return_value="/usr/bin/rerun"):
        frontends_mod._require_rerun_executable("rerun")


def test_require_rerun_executable_hints_studio_rerun_extra() -> None:
    with (
        patch.object(frontends_mod.shutil, "which", return_value=None),
        pytest.raises(SystemExit, match="studio_rerun"),
    ):
        frontends_mod._require_rerun_executable("missing-rerun-cli")


def test_stop_rerun_process_group_escalates_to_sigkill() -> None:
    process = MagicMock()
    process.pid = 4242
    process.wait.side_effect = [subprocess.TimeoutExpired(cmd="rerun", timeout=5), None]
    with patch.object(frontends_mod.os, "killpg") as killpg:
        frontends_mod._stop_rerun_process_group(process)
    assert killpg.call_args_list[0].args == (4242, signal.SIGTERM)
    assert killpg.call_args_list[1].args == (4242, signal.SIGKILL)


def test_serve_rerun_frontend_uses_new_session(monkeypatch, tmp_path) -> None:
    asset = tmp_path / "scene.rrd"
    asset.write_bytes(b"rrd")
    entry = MagicMock(model_id="demo")
    launch = MagicMock(
        simulator_url="",
        asset_path=str(asset),
        host="127.0.0.1",
        port=9876,
    )
    process = MagicMock()
    process.wait.return_value = 0

    def _env_first(key: str, *args, **kwargs):
        if key == "WORLDFOUNDRY_STUDIO_RERUN_COMMAND":
            return "rerun {asset}"
        if key == "WORLDFOUNDRY_STUDIO_RERUN_URL":
            return ""
        return ""

    with (
        patch.object(frontends_mod, "host_for_frontend", return_value="127.0.0.1"),
        patch.object(frontends_mod, "port_for_frontend", return_value=9876),
        patch.object(frontends_mod, "env_first", side_effect=_env_first),
        patch.object(frontends_mod, "print_remote_access"),
        patch.object(frontends_mod, "_require_rerun_executable"),
        patch.object(frontends_mod.subprocess, "Popen", return_value=process) as popen,
    ):
        frontends_mod.serve_rerun_frontend(entry, launch)
    assert popen.call_args.kwargs.get("start_new_session") is True

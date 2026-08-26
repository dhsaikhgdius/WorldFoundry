from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from worldfoundry.core.io import hf as hf_mod


def test_download_coordinator_mode_defaults_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(hf_mod.HF_DOWNLOAD_COORDINATOR_ENV, raising=False)
    assert hf_mod.download_coordinator_mode() == "local"
    monkeypatch.setenv(hf_mod.HF_DOWNLOAD_COORDINATOR_ENV, "global")
    assert hf_mod.download_coordinator_mode() == "global"
    monkeypatch.setenv(hf_mod.HF_DOWNLOAD_COORDINATOR_ENV, "local-rank0")
    assert hf_mod.download_coordinator_mode() == "local"


def test_download_coordinator_mode_rejects_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(hf_mod.HF_DOWNLOAD_COORDINATOR_ENV, "weird")
    with pytest.raises(ValueError, match="local"):
        hf_mod.download_coordinator_mode()


def test_should_download_local_mode_uses_local_master(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(hf_mod.HF_DOWNLOAD_COORDINATOR_ENV, "local")
    with (
        patch.object(hf_mod, "is_distributed_initialized", return_value=True),
        patch.object(hf_mod, "is_local_master", return_value=False) as local_master,
        patch.object(hf_mod, "get_global_rank", return_value=0),
    ):
        assert hf_mod._should_download_hf_snapshot() is False
        local_master.assert_called_once()


def test_should_download_global_mode_uses_world_rank0(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(hf_mod.HF_DOWNLOAD_COORDINATOR_ENV, "global")
    with (
        patch.object(hf_mod, "is_distributed_initialized", return_value=True),
        patch.object(hf_mod, "get_global_rank", return_value=3),
        patch.object(hf_mod, "is_local_master", return_value=True),
    ):
        assert hf_mod._should_download_hf_snapshot() is False
    with patch.object(hf_mod, "get_global_rank", return_value=0):
        assert hf_mod._should_download_hf_snapshot() is True


def test_maybe_download_local_mode_barriers_not_broadcast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(hf_mod.HF_DOWNLOAD_COORDINATOR_ENV, "local")
    dist = MagicMock()
    with (
        patch.object(hf_mod, "dist", dist),
        patch.object(hf_mod, "is_distributed_initialized", return_value=True),
        patch.object(hf_mod, "_should_download_hf_snapshot", return_value=True),
        patch.object(hf_mod, "_download_snapshot") as download,
    ):
        hf_mod.maybe_download_hf_repo_on_rank0("owner/repo")
    download.assert_called_once()
    dist.barrier.assert_called_once()
    dist.broadcast_object_list.assert_not_called()


def test_maybe_download_global_mode_broadcasts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(hf_mod.HF_DOWNLOAD_COORDINATOR_ENV, "global")
    dist = MagicMock()
    with (
        patch.object(hf_mod, "dist", dist),
        patch.object(hf_mod, "is_distributed_initialized", return_value=True),
        patch.object(hf_mod, "_should_download_hf_snapshot", return_value=False),
        patch.object(hf_mod, "_download_snapshot") as download,
    ):
        hf_mod.maybe_download_hf_repo_on_rank0("owner/repo")
    download.assert_not_called()
    dist.broadcast_object_list.assert_called_once()
    dist.barrier.assert_not_called()

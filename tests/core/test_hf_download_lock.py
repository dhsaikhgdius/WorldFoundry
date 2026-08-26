from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from worldfoundry.core.io import hf as hf_mod


def test_download_lock_timeout_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(hf_mod.HF_DOWNLOAD_LOCK_TIMEOUT_ENV, raising=False)
    assert hf_mod.download_lock_timeout_seconds() == 7200.0
    monkeypatch.setenv(hf_mod.HF_DOWNLOAD_LOCK_TIMEOUT_ENV, "12.5")
    assert hf_mod.download_lock_timeout_seconds() == 12.5
    monkeypatch.setenv(hf_mod.HF_DOWNLOAD_LOCK_TIMEOUT_ENV, "-1")
    assert hf_mod.download_lock_timeout_seconds() == -1.0


def test_hf_download_lock_logs_while_waiting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv(hf_mod.HF_DOWNLOAD_LOCK_TIMEOUT_ENV, "2")
    monkeypatch.setenv(hf_mod.HF_DOWNLOAD_LOCK_WAIT_LOG_ENV, "0.2")
    lock_file = tmp_path / "repo.lock"
    held = threading.Event()
    release = threading.Event()

    def _hold_lock() -> None:
        blocker = FileLock(str(lock_file), thread_local=False)
        blocker.acquire()
        held.set()
        release.wait(timeout=5)
        blocker.release()

    holder = threading.Thread(target=_hold_lock, daemon=True)
    holder.start()
    assert held.wait(timeout=2)
    errors: list[BaseException] = []

    def _waiter() -> None:
        try:
            with hf_mod._hf_download_lock(lock_file):
                pass
        except BaseException as exc:  # noqa: BLE001 — surface in parent thread
            errors.append(exc)

    with caplog.at_level(logging.INFO, logger=hf_mod.__name__):
        waiter = threading.Thread(target=_waiter, daemon=True)
        waiter.start()
        time.sleep(0.5)
        release.set()
        waiter.join(timeout=2)
        holder.join(timeout=2)
    assert not errors
    assert any("Waiting for Hugging Face download lock" in msg for msg in caplog.messages)


def test_hf_download_lock_times_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(hf_mod.HF_DOWNLOAD_LOCK_TIMEOUT_ENV, "0.3")
    monkeypatch.setenv(hf_mod.HF_DOWNLOAD_LOCK_WAIT_LOG_ENV, "0.1")
    lock_file = tmp_path / "repo.lock"
    held = threading.Event()
    release = threading.Event()

    def _hold_lock() -> None:
        blocker = FileLock(str(lock_file), thread_local=False)
        blocker.acquire()
        held.set()
        release.wait(timeout=5)
        blocker.release()

    thread = threading.Thread(target=_hold_lock, daemon=True)
    thread.start()
    assert held.wait(timeout=2)
    try:
        with pytest.raises(FileLockTimeout):
            with hf_mod._hf_download_lock(lock_file):
                pass
    finally:
        release.set()
        thread.join(timeout=2)


def test_download_snapshot_uses_timed_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(hf_mod.HF_DOWNLOAD_LOCK_TIMEOUT_ENV, "5")
    seen: dict[str, object] = {}

    @contextmanager
    def _fake_lock(lock_file: Path):
        seen["lock_file"] = lock_file
        yield object()

    monkeypatch.setattr(hf_mod, "_hf_download_lock", _fake_lock)
    monkeypatch.setattr(hf_mod, "ensure_free_disk", lambda *args, **kwargs: None)
    monkeypatch.setattr(hf_mod, "_snapshot_download", lambda *args, **kwargs: "/cache")
    monkeypatch.setattr(hf_mod, "_hub_cache_dir", lambda cache_dir: tmp_path)

    hf_mod._download_snapshot(
        "owner/repo",
        revision=None,
        cache_dir=tmp_path,
        allow_patterns=None,
        ignore_patterns=None,
    )
    assert isinstance(seen["lock_file"], Path)
    assert seen["lock_file"].name.endswith(".lock")

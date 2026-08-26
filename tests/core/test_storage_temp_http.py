from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from worldfoundry.core.io import storage as storage_mod


def test_local_path_for_uri_uses_default_temp_and_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    seen: dict[str, object] = {}

    def _fake_ensure(path, *, required_bytes, label, env_vars=(), settings=None, mkdir=True):
        seen["path"] = Path(path)
        seen["required_bytes"] = required_bytes
        seen["label"] = label

    class _Source(io.BytesIO):
        pass

    with (
        patch.object(storage_mod, "ensure_free_disk", side_effect=_fake_ensure),
        patch.object(storage_mod, "cache_min_free_bytes", return_value=42),
        patch.object(storage_mod, "open_uri") as open_uri,
    ):
        open_uri.return_value.__enter__.return_value = _Source(b"hello-remote")
        open_uri.return_value.__exit__.return_value = False
        with storage_mod.local_path_for_uri("https://example.test/a.bin") as local:
            assert local.parent == tmp_path
            assert local.read_bytes() == b"hello-remote"
    assert seen["path"] == tmp_path
    assert seen["required_bytes"] == 42
    assert seen["label"] == "WorldFoundry temp download"


def test_open_uri_http_streams_without_bytesio(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"streamed-body" * 100
    response = MagicMock()
    response.read = MagicMock(side_effect=[payload, b""])
    response.__enter__ = MagicMock(return_value=response)
    response.__exit__ = MagicMock(return_value=False)

    # Force the no-fsspec HTTP fallback path.
    monkeypatch.setattr(storage_mod, "_optional_fsspec", lambda: None)

    with patch.object(storage_mod, "urlopen", return_value=response) as urlopen:
        with storage_mod.open_uri("https://example.test/blob.bin", "rb") as handle:
            assert handle is response
            assert "BytesIO" not in type(handle).__name__
    request = urlopen.call_args.args[0]
    assert request.get_header("User-agent") == "worldfoundry"


def test_exists_uri_http_head_sends_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage_mod, "_optional_fsspec", lambda: None)
    response = MagicMock()
    response.__enter__ = MagicMock(return_value=response)
    response.__exit__ = MagicMock(return_value=False)
    with patch.object(storage_mod, "urlopen", return_value=response) as urlopen:
        assert storage_mod.exists_uri("https://example.test/x") is True
    request = urlopen.call_args.args[0]
    assert request.get_method() == "HEAD"
    assert request.get_header("User-agent") == "worldfoundry"

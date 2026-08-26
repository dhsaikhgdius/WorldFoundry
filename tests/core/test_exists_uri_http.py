from __future__ import annotations

from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from worldfoundry.core.io import storage as storage_mod


def test_exists_uri_http_head_sends_user_agent(monkeypatch) -> None:
    monkeypatch.setattr(storage_mod, "_optional_fsspec", lambda: None)
    response = MagicMock()
    response.__enter__ = MagicMock(return_value=response)
    response.__exit__ = MagicMock(return_value=False)
    with patch.object(storage_mod, "urlopen", return_value=response) as urlopen:
        assert storage_mod.exists_uri("https://example.test/x") is True
    request = urlopen.call_args.args[0]
    assert request.get_method() == "HEAD"
    assert request.get_header("User-agent") == "worldfoundry"


def test_exists_uri_http_head_405_falls_back_to_ranged_get(monkeypatch) -> None:
    monkeypatch.setattr(storage_mod, "_optional_fsspec", lambda: None)
    head_error = HTTPError("https://example.test/x", 405, "Method Not Allowed", hdrs=None, fp=None)
    get_response = MagicMock()
    get_response.__enter__ = MagicMock(return_value=get_response)
    get_response.__exit__ = MagicMock(return_value=False)

    with patch.object(storage_mod, "urlopen", side_effect=[head_error, get_response]) as urlopen:
        assert storage_mod.exists_uri("https://example.test/x") is True
    assert urlopen.call_count == 2
    assert urlopen.call_args_list[0].args[0].get_method() == "HEAD"
    get_request = urlopen.call_args_list[1].args[0]
    assert get_request.get_method() == "GET"
    assert get_request.get_header("Range") == "bytes=0-0"
    assert get_request.get_header("User-agent") == "worldfoundry"


def test_exists_uri_http_head_404_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(storage_mod, "_optional_fsspec", lambda: None)
    missing = HTTPError("https://example.test/missing", 404, "Not Found", hdrs=None, fp=None)
    with patch.object(storage_mod, "urlopen", side_effect=missing):
        assert storage_mod.exists_uri("https://example.test/missing") is False

from __future__ import annotations

import pytest

from worldfoundry.evaluation.tasks.embodied.model_server import serve as serve_mod


def test_default_host_is_loopback() -> None:
    assert serve_mod.DEFAULT_HOST == "127.0.0.1"


def test_loopback_host_requires_no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(serve_mod.EMBODIED_SERVER_TOKEN_ENV, raising=False)
    assert serve_mod.require_embodied_auth_token_for_host("127.0.0.1") == ""
    assert serve_mod.require_embodied_auth_token_for_host("localhost") == ""
    assert serve_mod.require_embodied_auth_token_for_host("::1") == ""


def test_non_loopback_without_token_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(serve_mod.EMBODIED_SERVER_TOKEN_ENV, raising=False)
    with pytest.raises(SystemExit, match=serve_mod.EMBODIED_SERVER_TOKEN_ENV):
        serve_mod.require_embodied_auth_token_for_host("0.0.0.0")


def test_non_loopback_with_token_returns_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(serve_mod.EMBODIED_SERVER_TOKEN_ENV, "shared-secret")
    assert serve_mod.require_embodied_auth_token_for_host("0.0.0.0") == "shared-secret"


def test_token_from_headers_parses_bearer_only() -> None:
    assert serve_mod._token_from_headers({"Authorization": "Bearer abc"}) == "abc"
    assert serve_mod._token_from_headers({"authorization": "bearer xyz "}) == "xyz"
    assert serve_mod._token_from_headers({"Authorization": b"Bearer bin"}) == "bin"
    assert serve_mod._token_from_headers({"Authorization": "Basic abc"}) is None
    assert serve_mod._token_from_headers({}) is None
    assert serve_mod._token_from_headers(None) is None

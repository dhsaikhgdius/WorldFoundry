from __future__ import annotations

import pytest

from worldfoundry.evaluation.tasks.embodied.model_server.protocol import (
    PROTOCOL_VERSION,
    MessageType,
    assert_compatible_protocol_version,
    hello_payload,
)


def test_hello_payload_includes_protocol_version() -> None:
    payload = hello_payload(role="server", ready=True)
    assert payload["protocol_version"] == PROTOCOL_VERSION
    assert payload["ready"] is True


def test_assert_compatible_protocol_version_accepts_match() -> None:
    assert_compatible_protocol_version({"protocol_version": PROTOCOL_VERSION}, peer="server")


def test_assert_compatible_protocol_version_rejects_mismatch() -> None:
    with pytest.raises(RuntimeError, match="protocol_version mismatch"):
        assert_compatible_protocol_version({"protocol_version": PROTOCOL_VERSION + 1}, peer="client")


def test_protocol_defines_ping_pong_message_types() -> None:
    assert MessageType.PING.value == "ping"
    assert MessageType.PONG.value == "pong"

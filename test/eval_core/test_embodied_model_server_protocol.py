from __future__ import annotations

import pytest

from worldfoundry.evaluation.tasks.embodied.model_server.protocol import (
    PROTOCOL_VERSION,
    Message,
    MessageType,
    assert_compatible_protocol_version,
    hello_payload,
    pack_message,
    unpack_message,
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


def test_ping_pong_roundtrip_msgpack() -> None:
    ping = Message(MessageType.PING, {"echo": "ready-check"}, seq=7)
    raw = pack_message(ping)
    decoded = unpack_message(raw)
    assert decoded.type is MessageType.PING
    assert decoded.payload["echo"] == "ready-check"
    assert decoded.seq == 7

    pong = Message(MessageType.PONG, {"ready": True, "protocol_version": PROTOCOL_VERSION}, seq=7)
    assert unpack_message(pack_message(pong)).type is MessageType.PONG

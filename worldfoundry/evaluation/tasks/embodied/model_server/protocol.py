"""Msgpack WebSocket protocol shared with embodied policy servers."""

from __future__ import annotations

from dataclasses import dataclass, field
import enum
import time
from typing import Any, Mapping


PROTOCOL_VERSION = 1


class MessageType(str, enum.Enum):
    HELLO = "hello"
    PING = "ping"
    PONG = "pong"
    OBSERVATION = "observation"
    ACTION = "action"
    EPISODE_START = "episode_start"
    EPISODE_END = "episode_end"
    ERROR = "error"


@dataclass(frozen=True)
class Message:
    type: MessageType
    payload: dict[str, Any]
    seq: int = 0
    timestamp: float = field(default_factory=time.time)


def _encode_object(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return {
                "__ndarray__": True,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data": value.tobytes(),
            }
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    raise TypeError(f"cannot msgpack-encode object of type {type(value).__name__}")


def _decode_object(value: dict[str, Any]) -> Any:
    if value.get("__ndarray__") is True:
        try:
            import numpy as np

            array = np.frombuffer(value["data"], dtype=value["dtype"])
            return array.reshape(tuple(value["shape"]))
        except Exception:
            return value
    return value


def pack_message(message: Message) -> bytes:
    """Serialize a protocol message to msgpack bytes."""
    import msgpack

    raw = {
        "type": message.type.value,
        "payload": message.payload,
        "seq": message.seq,
        "timestamp": message.timestamp,
    }
    return msgpack.packb(raw, default=_encode_object, use_bin_type=True)


def unpack_message(data: bytes | bytearray | memoryview) -> Message:
    """Deserialize msgpack bytes to a typed message."""
    import msgpack

    raw = msgpack.unpackb(data, object_hook=_decode_object, raw=False)
    if not isinstance(raw, dict):
        raise ValueError(f"expected protocol message dict, got {type(raw).__name__}")
    missing = [key for key in ("type", "payload", "seq", "timestamp") if key not in raw]
    if missing:
        raise ValueError(f"message missing fields: {missing}")
    return Message(
        type=MessageType(raw["type"]),
        payload=dict(raw["payload"] or {}),
        seq=int(raw["seq"]),
        timestamp=float(raw["timestamp"]),
    )


def hello_payload(**extra: Any) -> dict[str, Any]:
    """Build a HELLO payload."""
    return {
        "worldfoundry_component": "embodied_model_server",
        "protocol_version": PROTOCOL_VERSION,
        **extra,
    }


def assert_compatible_protocol_version(payload: Mapping[str, Any] | None, *, peer: str) -> None:
    """Raise ``RuntimeError`` when a peer advertises an incompatible protocol version."""

    if not isinstance(payload, Mapping):
        return
    raw = payload.get("protocol_version")
    if raw is None:
        return
    try:
        version = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid {peer} protocol_version: {raw!r}") from exc
    if version != int(PROTOCOL_VERSION):
        raise RuntimeError(
            f"protocol_version mismatch: {peer}={version} local={PROTOCOL_VERSION}"
        )


__all__ = [
    "PROTOCOL_VERSION",
    "Message",
    "MessageType",
    "assert_compatible_protocol_version",
    "hello_payload",
    "pack_message",
    "unpack_message",
]

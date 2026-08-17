"""JSON-safe artifact codec used by the reward HTTP transport."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from pathlib import Path

import torch

_DTYPES = {
    str(dtype).removeprefix("torch."): dtype
    for dtype in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
}


def encode_wire_value(value: object) -> object:
    """Encode nested reward inputs without pickle or framework-specific files."""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return {"kind": "path", "value": str(value)}
    if isinstance(value, bytes):
        return {"kind": "bytes", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu").contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {
            "kind": "tensor",
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "data": base64.b64encode(raw).decode("ascii"),
        }
    if isinstance(value, Mapping):
        return {str(key): encode_wire_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [encode_wire_value(item) for item in value]
    raise TypeError(f"reward HTTP codec does not support {type(value).__name__}")


def decode_wire_value(value: object) -> object:
    """Decode a value produced by :func:`encode_wire_value`."""

    if isinstance(value, list):
        return [decode_wire_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("kind")
    if kind == "path" and set(value) == {"kind", "value"}:
        return Path(str(value["value"]))
    if kind == "bytes" and set(value) == {"kind", "data"}:
        return base64.b64decode(str(value["data"]), validate=True)
    if kind == "tensor" and set(value) == {"kind", "dtype", "shape", "data"}:
        dtype_name = str(value["dtype"])
        if dtype_name not in _DTYPES:
            raise ValueError(f"unsupported tensor dtype {dtype_name!r}")
        raw = bytearray(base64.b64decode(str(value["data"]), validate=True))
        byte_tensor = torch.frombuffer(raw, dtype=torch.uint8)
        tensor = byte_tensor.view(_DTYPES[dtype_name])
        return tensor.reshape(tuple(int(size) for size in value["shape"])).clone()
    return {str(key): decode_wire_value(item) for key, item in value.items()}


__all__ = ["decode_wire_value", "encode_wire_value"]

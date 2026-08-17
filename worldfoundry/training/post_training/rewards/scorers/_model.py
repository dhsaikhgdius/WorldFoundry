"""Small helpers shared by local model-backed reward scorers."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def move_inputs(inputs: Mapping[str, object], device: torch.device) -> dict[str, object]:
    return {
        name: value.to(device=device) if isinstance(value, torch.Tensor) else value for name, value in inputs.items()
    }


def feature_tensor(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    pooler_output = getattr(output, "pooler_output", None)
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output
    last_hidden_state = getattr(output, "last_hidden_state", None)
    if isinstance(last_hidden_state, torch.Tensor):
        return last_hidden_state[:, 0]
    if isinstance(output, tuple | list) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"unsupported reward model feature output {type(output).__name__}")


__all__ = ["feature_tensor", "move_inputs", "resolve_device"]

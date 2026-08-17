"""Training-time exponential moving averages shared by native model families."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from worldfoundry.core.distributed import get_local_tensor_if_dtensor


def power_ema_exponent(rate: float) -> float:
    """Return the EDM2 power-profile exponent used by Cosmos trainers."""

    resolved = float(rate)
    if not 0.0 < resolved <= 1.0:
        raise ValueError("PowerEMA rate must be in (0, 1]")
    return float(np.roots([1.0, 7.0, 16.0 - resolved**-2, 12.0 - resolved**-2]).real.max())


def power_ema_beta(iteration: int, *, rate: float = 0.1, iteration_shift: int = 0) -> float:
    """Return the author-trainer decay for a zero-based optimizer iteration."""

    current = int(iteration) + int(iteration_shift)
    if current < 1:
        return 0.0
    exponent = power_ema_exponent(rate)
    return float((1.0 - 1.0 / (current + 1)) ** (exponent + 1.0))


class PowerEMA(nn.Module):
    """FP32 parameter shadows following the Cosmos/EDM2 power EMA schedule."""

    def __init__(
        self,
        model: nn.Module,
        *,
        rate: float = 0.1,
        iteration_shift: int = 0,
    ) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("PowerEMA model must be an nn.Module")
        tracked = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not tracked:
            raise ValueError("PowerEMA requires trainable parameters")

        control_device = get_local_tensor_if_dtensor(tracked[0][1]).device
        self.register_buffer("rate", torch.tensor(float(rate), dtype=torch.float64, device=control_device))
        self.register_buffer(
            "exponent",
            torch.tensor(power_ema_exponent(rate), dtype=torch.float64, device=control_device),
        )
        self.register_buffer(
            "iteration_shift",
            torch.tensor(int(iteration_shift), dtype=torch.int64, device=control_device),
        )
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.int64, device=control_device))
        self.register_buffer("last_beta", torch.zeros((), dtype=torch.float64, device=control_device))
        self._shadow_names: dict[str, str] = {}
        for index, (name, parameter) in enumerate(tracked):
            shadow_name = f"shadow_{index:06d}"
            self._shadow_names[name] = shadow_name
            self.register_buffer(shadow_name, parameter.detach().to(dtype=torch.float32).clone())
        self._stored_parameters: list[tuple[int, torch.Tensor]] = []

    def beta(self, iteration: int) -> float:
        current = int(iteration) + int(self.iteration_shift.item())
        if current < 1:
            return 0.0
        return float((1.0 - 1.0 / (current + 1)) ** (float(self.exponent.item()) + 1.0))

    def _tracked_parameters(self, model: nn.Module, *, action: str) -> list[tuple[str, nn.Parameter]]:
        """Resolve every tracked shadow to a live parameter or fail loudly.

        Shadows are keyed by the ``named_parameters()`` names captured at
        construction.  If the module is re-wrapped afterwards (container
        modules, PEFT injection, FSDP1 flattening) those names gain prefixes,
        silently detaching the EMA from the model; the exported weights would
        stay near their initialization and the error would only surface in
        final evaluations.  Validation happens before any mutation.
        """

        tracked = [(name, parameter) for name, parameter in model.named_parameters() if name in self._shadow_names]
        if len(tracked) != len(self._shadow_names):
            missing = sorted(set(self._shadow_names) - {name for name, _ in tracked})
            preview = ", ".join(missing[:5])
            raise RuntimeError(
                f"PowerEMA cannot {action}: {len(missing)} of {len(self._shadow_names)} tracked "
                f"parameters are missing from the module (e.g. {preview}); the module was "
                "likely re-wrapped or renamed after EMA construction"
            )
        return tracked

    @torch.no_grad()
    def forward(self, model: nn.Module) -> None:
        """Update shadows after one optimizer step."""

        tracked = self._tracked_parameters(model, action="update shadows")
        iteration = int(self.num_updates.item())
        beta = self.beta(iteration)
        for name, parameter in tracked:
            shadow = get_local_tensor_if_dtensor(getattr(self, self._shadow_names[name]))
            source = get_local_tensor_if_dtensor(parameter).detach()
            shadow.mul_(beta).add_(source, alpha=1.0 - beta)
        self.last_beta.fill_(beta)
        self.num_updates.add_(1)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        tracked = self._tracked_parameters(model, action="export shadows")
        for name, parameter in tracked:
            target = get_local_tensor_if_dtensor(parameter)
            shadow = get_local_tensor_if_dtensor(getattr(self, self._shadow_names[name]))
            target.copy_(shadow.to(dtype=target.dtype))

    @torch.no_grad()
    def store(self, parameters: Iterable[nn.Parameter]) -> None:
        """Store only parameters that EMA export will temporarily replace."""

        values = tuple(parameters)
        self._stored_parameters = [
            (index, get_local_tensor_if_dtensor(parameter).detach().clone())
            for index, parameter in enumerate(values)
            if parameter.requires_grad
        ]

    @torch.no_grad()
    def restore(self, parameters: Iterable[nn.Parameter]) -> None:
        values = tuple(parameters)
        for index, stored in self._stored_parameters:
            target = get_local_tensor_if_dtensor(values[index])
            target.copy_(stored.to(dtype=target.dtype))
        self._stored_parameters = []


__all__ = ["PowerEMA", "power_ema_beta", "power_ema_exponent"]

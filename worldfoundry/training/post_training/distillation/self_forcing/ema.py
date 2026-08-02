"""Self-Forcing name for the shared start-delayed module EMA."""

from __future__ import annotations

from ...shared.ema import DelayedModuleEMA


class DelayedSelfForcingEMA(DelayedModuleEMA):
    """Backward-compatible semantic name for Self-Forcing callers."""


__all__ = ["DelayedSelfForcingEMA"]

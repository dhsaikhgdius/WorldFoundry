"""Wan Self-Forcing run lifecycle over the shared two-optimizer runner."""

from __future__ import annotations

from .dmd_run import WanDMDTrainingRun

WAN_SELF_FORCING_RUN_SCHEMA = "worldfoundry-wan-self-forcing-run"


class WanSelfForcingTrainingRun(WanDMDTrainingRun):
    """Export and checkpoint lifecycle for the causal Self-Forcing student."""

    run_schema = WAN_SELF_FORCING_RUN_SCHEMA
    algorithm_label = "Wan Self-Forcing"
    export_role_label = "Self-Forcing student"


__all__ = ["WAN_SELF_FORCING_RUN_SCHEMA", "WanSelfForcingTrainingRun"]

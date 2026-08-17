"""Wan naming and defaults for the shared student-distillation lifecycle."""

from __future__ import annotations

from ..student_distillation import StudentDistillationTrainingRun

WAN_DMD_RUN_SCHEMA = "worldfoundry-wan-dmd-run"


class WanDMDTrainingRun(StudentDistillationTrainingRun):
    """Wan DMD specialization of the shared distillation run."""

    run_schema = WAN_DMD_RUN_SCHEMA
    algorithm_label = "Wan DMD"
    export_role_label = "DMD student"


__all__ = ["WAN_DMD_RUN_SCHEMA", "WanDMDTrainingRun"]

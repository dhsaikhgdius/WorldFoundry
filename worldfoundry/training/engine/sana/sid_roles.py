"""Mutable role tree for a materialized SANA SiD run."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from worldfoundry.training.distributed.fsdp import FSDP2Application
from worldfoundry.training.models.sana import SanaTrainAdapter
from worldfoundry.training.models.sana_sid import SanaSIDPredictionAdapter


class SanaSIDTrainableRoles(nn.Module):
    """One DCP model tree containing exactly student and fake-score parameters."""

    def __init__(self, student: nn.Module, fake_score: nn.Module) -> None:
        super().__init__()
        if not isinstance(student, nn.Module) or not isinstance(fake_score, nn.Module):
            raise TypeError("SANA SiD trainable roles must be nn.Module values")
        if student is fake_score:
            raise ValueError("SANA SiD mutable roles must be distinct")
        self.student = student
        self.fake_score = fake_score


@dataclass(frozen=True, slots=True)
class SanaSIDRoleBundle:
    student_preparation: SanaTrainAdapter
    fake_score_preparation: SanaTrainAdapter
    teacher_preparation: SanaTrainAdapter
    student: SanaSIDPredictionAdapter
    teacher: SanaSIDPredictionAdapter
    fake_score: SanaSIDPredictionAdapter
    asset_identity: dict[str, object]
    student_fsdp: FSDP2Application | None = None
    teacher_fsdp: FSDP2Application | None = None
    fake_score_fsdp: FSDP2Application | None = None


__all__ = ["SanaSIDRoleBundle", "SanaSIDTrainableRoles"]

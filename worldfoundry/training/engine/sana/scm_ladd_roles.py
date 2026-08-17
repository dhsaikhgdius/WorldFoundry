"""Mutable role tree and materialized-role contracts for SANA SCM-LADD."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from worldfoundry.training.distributed.fsdp import FSDP2Application
from worldfoundry.training.models.sana import SanaTrainAdapter
from worldfoundry.training.models.sana_scm_ladd import (
    SanaLADDDiscriminatorAdapter,
    SanaSCMVelocityAdapter,
)
from worldfoundry.training.post_training.shared.role_checkpoints import (
    ResolvedRoleCheckpoint,
)
from worldfoundry.training.tuning.peft import PeftLoraApplication


class SanaSCMLADDTrainableRoles(nn.Module):
    """One DCP model tree containing exactly the two mutable roles."""

    def __init__(self, student: nn.Module, discriminator_heads: nn.Module) -> None:
        super().__init__()
        if not isinstance(student, nn.Module) or not isinstance(discriminator_heads, nn.Module):
            raise TypeError("SANA SCM-LADD trainable roles must be nn.Module values")
        if student is discriminator_heads:
            raise ValueError("student and discriminator heads must be distinct")
        self.student = student
        self.discriminator_heads = discriminator_heads


@dataclass(frozen=True, slots=True)
class SanaSCMLADDRoleBundle:
    student_preparation: SanaTrainAdapter
    student: SanaSCMVelocityAdapter
    teacher: SanaSCMVelocityAdapter
    discriminator: SanaLADDDiscriminatorAdapter
    student_checkpoint: ResolvedRoleCheckpoint
    teacher_checkpoint: ResolvedRoleCheckpoint
    student_peft: PeftLoraApplication | None
    student_fsdp: FSDP2Application | None
    teacher_fsdp: FSDP2Application | None
    discriminator_fsdp: FSDP2Application | None

    def runtime_identity(self) -> dict[str, object]:
        peft = None
        if self.student_peft is not None:
            peft = {
                "target_audit": self.student_peft.target_audit.to_dict(),
                "trainable_parameter_names": list(self.student_peft.trainable_parameter_names),
                "trainable_parameter_count": self.student_peft.trainable_parameter_count,
            }

        def fsdp(value: FSDP2Application | None) -> dict[str, object] | None:
            if value is None:
                return None
            return value.to_dict()

        return {
            "student_checkpoint": self.student_checkpoint.to_dict(),
            "teacher_checkpoint": self.teacher_checkpoint.to_dict(),
            "student": {"peft": peft, "fsdp2": fsdp(self.student_fsdp)},
            "teacher": {"fsdp2": fsdp(self.teacher_fsdp)},
            "discriminator_heads": {"fsdp2": fsdp(self.discriminator_fsdp)},
        }


__all__ = ["SanaSCMLADDRoleBundle", "SanaSCMLADDTrainableRoles"]

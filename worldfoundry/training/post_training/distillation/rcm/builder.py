"""Construction of native bidirectional and causal rCM training stacks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal

import torch
from torch import nn

from worldfoundry.training.checkpoint.stateful import NamedStatefulCollection
from worldfoundry.training.recipes.post_training.algorithms.rcm import (
    CausalRCMAlgorithmSpec,
    RCMAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
from worldfoundry.training.recipes.spec import OptimizerSpec

from ...shared.building import (
    build_post_training_optimizer,
    named_stateful_collection,
    require_checkpoint_identity,
    require_independent_modules,
    validate_post_training_recipe,
)
from ...shared.distributed import PostTrainingParallelContext
from .causal import (
    CausalRCMConfig,
    CausalSelfForcingAdapter,
    CausalTeacherForcingAdapter,
    NativeCausalRCMLossAdapter,
    RFScoreAdapter,
)
from .config import RCMConfig
from .contracts import RCMPredictionAdapter
from .engine import NativeRCMTrainEngine
from .objective import NativeRCMLossAdapter
from .synchronization import RCMTensorSynchronizer


def _module(adapter: object, *, role: str) -> nn.Module:
    module = getattr(adapter, "module", None)
    if not isinstance(module, nn.Module):
        raise TypeError(f"{role}.module must be an nn.Module")
    return module


class _FrozenTeacherRoles(nn.Module):
    """One eval/freeze handle over the functional teacher inventory."""

    def __init__(self, **roles: nn.Module) -> None:
        super().__init__()
        if not roles:
            raise ValueError("rCM requires at least one teacher module")
        if len({id(module) for module in roles.values()}) != len(roles):
            raise ValueError("rCM teacher roles must use distinct modules")
        self.roles = nn.ModuleDict(roles)


@dataclass(frozen=True, slots=True)
class NativeRCMTrainingStack:
    recipe: PostTrainingRecipe
    config: RCMConfig | CausalRCMConfig
    loss_adapter: NativeRCMLossAdapter | NativeCausalRCMLossAdapter
    student_optimizer: torch.optim.Optimizer
    fake_score_optimizer: torch.optim.Optimizer | None
    engine: NativeRCMTrainEngine
    scheduler_state: NamedStatefulCollection | None
    ema_state: NamedStatefulCollection | None

    @property
    def optimizers(self) -> tuple[torch.optim.Optimizer, ...]:
        if self.fake_score_optimizer is None:
            return (self.student_optimizer,)
        return self.student_optimizer, self.fake_score_optimizer

    def checkpoint_state_kwargs(self) -> dict[str, object | None]:
        return {
            "lr_scheduler": self.scheduler_state,
            "ema": self.ema_state,
            "algorithm_state": None,
        }


def rcm_config_from_algorithm(algorithm: RCMAlgorithmSpec) -> RCMConfig:
    """Consume every bidirectional rCM recipe field except its dispatch tag."""

    if not isinstance(algorithm, RCMAlgorithmSpec):
        raise TypeError("algorithm must be RCMAlgorithmSpec")
    values = asdict(algorithm)
    if values.pop("type") != "rcm":
        raise RuntimeError("validated rCM dispatch tag changed unexpectedly")
    values.pop("teacher_checkpoint")
    values.pop("fake_score_checkpoint")
    return RCMConfig(**values)


def causal_rcm_config_from_algorithm(
    algorithm: CausalRCMAlgorithmSpec,
) -> CausalRCMConfig:
    """Consume every Causal-rCM recipe field except its dispatch tag."""

    if not isinstance(algorithm, CausalRCMAlgorithmSpec):
        raise TypeError("algorithm must be CausalRCMAlgorithmSpec")
    values = asdict(algorithm)
    if values.pop("type") != "causal-rcm":
        raise RuntimeError("validated Causal-rCM dispatch tag changed unexpectedly")
    values.pop("causal_teacher_checkpoint")
    values.pop("bidirectional_teacher_checkpoint")
    values.pop("fake_score_checkpoint")
    return CausalRCMConfig(**values)


def _build_engine(
    *,
    recipe: PostTrainingRecipe,
    config: RCMConfig | CausalRCMConfig,
    loss_adapter: NativeRCMLossAdapter | NativeCausalRCMLossAdapter,
    student_module: nn.Module,
    teacher_module: nn.Module,
    fake_score_module: nn.Module | None,
    student_optimizer_spec: OptimizerSpec,
    fake_score_optimizer_spec: OptimizerSpec | None,
    student_scheduler: object | None,
    fake_score_scheduler: object | None,
    student_ema: object | None,
    parallel_context: PostTrainingParallelContext | None,
    fused_adamw: bool | Literal["auto"],
) -> NativeRCMTrainingStack:
    if not isinstance(student_optimizer_spec, OptimizerSpec):
        raise TypeError("student_optimizer_spec must be OptimizerSpec")
    accumulation = student_optimizer_spec.gradient_accumulation_steps
    if config.dmd_enabled:
        if not isinstance(fake_score_optimizer_spec, OptimizerSpec):
            raise ValueError("rCM DMD requires fake_score_optimizer_spec")
        if fake_score_optimizer_spec.gradient_accumulation_steps != accumulation:
            raise ValueError("rCM student and fake-score accumulation steps must match")
    elif fake_score_optimizer_spec is not None:
        raise ValueError("fake_score_optimizer_spec requires DMD")
    student_optimizer = build_post_training_optimizer(
        replace(student_optimizer_spec, gradient_accumulation_steps=1),
        student_module,
        fused=fused_adamw,
        role="rCM student",
    )
    fake_score_optimizer = (
        None
        if fake_score_optimizer_spec is None or fake_score_module is None
        else build_post_training_optimizer(
            replace(fake_score_optimizer_spec, gradient_accumulation_steps=1),
            fake_score_module,
            fused=fused_adamw,
            role="rCM fake-score",
        )
    )
    engine = NativeRCMTrainEngine(
        student_module=student_module,
        teacher_module=teacher_module,
        fake_score_module=fake_score_module,
        loss_adapter=loss_adapter,
        student_optimizer=student_optimizer,
        fake_score_optimizer=fake_score_optimizer,
        tangent_warmup_steps=config.tangent_warmup_steps,
        student_update_frequency=config.student_update_frequency,
        dmd_enabled=config.dmd_enabled,
        student_max_grad_norm=student_optimizer_spec.max_grad_norm,
        fake_score_max_grad_norm=(
            None
            if fake_score_optimizer_spec is None
            else fake_score_optimizer_spec.max_grad_norm
        ),
        gradient_accumulation_steps=accumulation,
        student_scheduler=student_scheduler,
        fake_score_scheduler=fake_score_scheduler,
        student_ema=student_ema,
        parallel_context=parallel_context,
    )
    return NativeRCMTrainingStack(
        recipe=recipe,
        config=config,
        loss_adapter=loss_adapter,
        student_optimizer=student_optimizer,
        fake_score_optimizer=fake_score_optimizer,
        engine=engine,
        scheduler_state=named_stateful_collection(
            student=student_scheduler,
            fake_score=fake_score_scheduler,
        ),
        ema_state=named_stateful_collection(student=student_ema),
    )


def build_native_rcm_training_stack(
    recipe: PostTrainingRecipe,
    *,
    student: RCMPredictionAdapter,
    teacher: RCMPredictionAdapter,
    fake_score: RCMPredictionAdapter | None,
    student_scheduler: object | None = None,
    fake_score_scheduler: object | None = None,
    student_ema: object | None = None,
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
    tensor_synchronizer: RCMTensorSynchronizer | None = None,
) -> NativeRCMTrainingStack:
    """Build the complete bidirectional rCM optimizer plane."""

    validate_post_training_recipe(recipe)
    if not isinstance(recipe.algorithm, RCMAlgorithmSpec):
        raise TypeError("rCM stack requires RCMAlgorithmSpec")
    config = rcm_config_from_algorithm(recipe.algorithm)
    if not isinstance(student, RCMPredictionAdapter):
        raise TypeError("student must implement RCMPredictionAdapter")
    if not isinstance(teacher, RCMPredictionAdapter):
        raise TypeError("teacher must implement RCMPredictionAdapter")
    if fake_score is not None and not isinstance(fake_score, RCMPredictionAdapter):
        raise TypeError("fake_score must implement RCMPredictionAdapter")
    student_module = _module(student, role="student")
    teacher_module = _module(teacher, role="teacher")
    fake_score_module = None if fake_score is None else _module(fake_score, role="fake_score")
    roles = {"student": student_module, "teacher": teacher_module}
    if fake_score_module is not None:
        roles["fake-score"] = fake_score_module
    require_independent_modules(roles)
    require_checkpoint_identity(
        student,
        recipe.model.checkpoint,
        role="rCM student",
    )
    require_checkpoint_identity(
        teacher,
        recipe.algorithm.teacher_checkpoint,
        role="rCM teacher",
    )
    if fake_score is not None:
        assert recipe.algorithm.fake_score_checkpoint is not None
        require_checkpoint_identity(
            fake_score,
            recipe.algorithm.fake_score_checkpoint,
            role="rCM fake score",
        )
    if any(parameter.requires_grad for parameter in teacher_module.parameters()):
        raise ValueError("rCM teacher must be frozen before stack construction")
    teacher_module.eval()
    loss_adapter = NativeRCMLossAdapter(
        student,
        teacher,
        fake_score,
        config,
        tensor_synchronizer=tensor_synchronizer,
    )
    return _build_engine(
        recipe=recipe,
        config=config,
        loss_adapter=loss_adapter,
        student_module=student_module,
        teacher_module=teacher_module,
        fake_score_module=fake_score_module,
        student_optimizer_spec=recipe.optimizer,
        fake_score_optimizer_spec=recipe.fake_score_optimizer,
        student_scheduler=student_scheduler,
        fake_score_scheduler=fake_score_scheduler,
        student_ema=student_ema,
        parallel_context=parallel_context,
        fused_adamw=fused_adamw,
    )


def build_native_causal_rcm_training_stack(
    recipe: PostTrainingRecipe,
    *,
    student: CausalTeacherForcingAdapter,
    causal_teacher: CausalTeacherForcingAdapter | None,
    rollout: CausalSelfForcingAdapter | None,
    bidirectional_teacher: RFScoreAdapter | None,
    fake_score: RFScoreAdapter | None,
    student_scheduler: object | None = None,
    fake_score_scheduler: object | None = None,
    student_ema: object | None = None,
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
    tensor_synchronizer: RCMTensorSynchronizer | None = None,
) -> NativeRCMTrainingStack:
    """Build TF consistency plus optional SF-DMD over native causal roles."""

    validate_post_training_recipe(recipe)
    if not isinstance(recipe.algorithm, CausalRCMAlgorithmSpec):
        raise TypeError("Causal-rCM stack requires CausalRCMAlgorithmSpec")
    config = causal_rcm_config_from_algorithm(recipe.algorithm)
    if not isinstance(student, CausalTeacherForcingAdapter):
        raise TypeError("student must implement CausalTeacherForcingAdapter")
    student_module = _module(student, role="student")
    teachers: dict[str, nn.Module] = {}
    if causal_teacher is not None:
        if not isinstance(causal_teacher, CausalTeacherForcingAdapter):
            raise TypeError("causal_teacher must implement CausalTeacherForcingAdapter")
        teachers["causal"] = _module(causal_teacher, role="causal_teacher")
    if bidirectional_teacher is not None:
        if not isinstance(bidirectional_teacher, RFScoreAdapter):
            raise TypeError("bidirectional_teacher must implement RFScoreAdapter")
        teachers["bidirectional"] = _module(
            bidirectional_teacher,
            role="bidirectional_teacher",
        )
    teacher_module = _FrozenTeacherRoles(**teachers)
    fake_score_module = None
    if fake_score is not None:
        if not isinstance(fake_score, RFScoreAdapter):
            raise TypeError("fake_score must implement RFScoreAdapter")
        fake_score_module = _module(fake_score, role="fake_score")
    role_modules = {"student": student_module, **teachers}
    if fake_score_module is not None:
        role_modules["fake_score"] = fake_score_module
    if len({id(module) for module in role_modules.values()}) != len(role_modules):
        raise ValueError("Causal-rCM student, teacher, and fake-score role modules must be distinct")
    require_independent_modules(role_modules)
    require_checkpoint_identity(
        student,
        recipe.model.checkpoint,
        role="Causal-rCM student",
    )
    if causal_teacher is not None:
        assert recipe.algorithm.causal_teacher_checkpoint is not None
        require_checkpoint_identity(
            causal_teacher,
            recipe.algorithm.causal_teacher_checkpoint,
            role="Causal-rCM causal teacher",
        )
    if bidirectional_teacher is not None:
        assert recipe.algorithm.bidirectional_teacher_checkpoint is not None
        require_checkpoint_identity(
            bidirectional_teacher,
            recipe.algorithm.bidirectional_teacher_checkpoint,
            role="Causal-rCM bidirectional teacher",
        )
    if fake_score is not None:
        assert recipe.algorithm.fake_score_checkpoint is not None
        require_checkpoint_identity(
            fake_score,
            recipe.algorithm.fake_score_checkpoint,
            role="Causal-rCM fake score",
        )
    for name, module in teachers.items():
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise ValueError(f"Causal-rCM {name} teacher must be frozen")
        module.eval()
    loss_adapter = NativeCausalRCMLossAdapter(
        student,
        causal_teacher,
        rollout,
        bidirectional_teacher,
        fake_score,
        config,
        tensor_synchronizer=tensor_synchronizer,
    )
    return _build_engine(
        recipe=recipe,
        config=config,
        loss_adapter=loss_adapter,
        student_module=student_module,
        teacher_module=teacher_module,
        fake_score_module=fake_score_module,
        student_optimizer_spec=recipe.optimizer,
        fake_score_optimizer_spec=recipe.fake_score_optimizer,
        student_scheduler=student_scheduler,
        fake_score_scheduler=fake_score_scheduler,
        student_ema=student_ema,
        parallel_context=parallel_context,
        fused_adamw=fused_adamw,
    )


__all__ = [
    "NativeRCMTrainingStack",
    "build_native_causal_rcm_training_stack",
    "build_native_rcm_training_stack",
    "causal_rcm_config_from_algorithm",
    "rcm_config_from_algorithm",
]

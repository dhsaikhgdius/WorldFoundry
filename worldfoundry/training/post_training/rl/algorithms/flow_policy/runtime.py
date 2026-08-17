"""Central runtime registry for native flow-policy algorithms."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from worldfoundry.training.recipes.post_training.algorithms.bagel_flow_unigrpo import (
    BagelFlowUniGRPOAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.algorithms.dance_grpo import (
    DanceGRPOAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.algorithms.flow_dppo import (
    FlowDPPOAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.algorithms.flow_grpo import (
    FlowGRPOAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.algorithms.flow_policy import (
    FlowPolicyAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.algorithms.grpo_guard import (
    GRPOGuardAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.algorithms.mix_grpo import (
    MixGRPOAlgorithmSpec,
)

from ..bagel_flow_unigrpo.engine import build_native_bagel_flow_unigrpo_engine
from ..bagel_flow_unigrpo.session import NativeBagelFlowUniGRPOTrainingSession
from ..dance_grpo.engine import build_native_dance_grpo_engine
from ..dance_grpo.session import NativeDanceGRPOTrainingSession
from ..flow_dppo.engine import build_native_flow_dppo_engine
from ..flow_dppo.session import NativeFlowDPPOTrainingSession
from ..flow_grpo.engine import build_native_flow_grpo_engine
from ..flow_grpo.session import NativeFlowGRPOTrainingSession
from ..grpo_guard.engine import build_native_grpo_guard_engine
from ..grpo_guard.session import NativeGRPOGuardTrainingSession
from ..mix_grpo.engine import build_native_mix_grpo_engine
from ..mix_grpo.session import NativeMixGRPOTrainingSession
from .engine import NativeFlowPolicyEngine
from .session import NativeFlowPolicyTrainingSession


@dataclass(frozen=True, slots=True)
class FlowPolicyAlgorithmRuntime:
    """Executable bindings that differ between flow-policy objectives."""

    algorithm_type: type[FlowPolicyAlgorithmSpec]
    display_name: str
    engine_factory: Callable[..., NativeFlowPolicyEngine]
    session_type: type[NativeFlowPolicyTrainingSession]


_FLOW_POLICY_RUNTIMES = (
    FlowPolicyAlgorithmRuntime(
        algorithm_type=DanceGRPOAlgorithmSpec,
        display_name="DANCE",
        engine_factory=build_native_dance_grpo_engine,
        session_type=NativeDanceGRPOTrainingSession,
    ),
    FlowPolicyAlgorithmRuntime(
        algorithm_type=MixGRPOAlgorithmSpec,
        display_name="MixGRPO",
        engine_factory=build_native_mix_grpo_engine,
        session_type=NativeMixGRPOTrainingSession,
    ),
    FlowPolicyAlgorithmRuntime(
        algorithm_type=FlowGRPOAlgorithmSpec,
        display_name="Flow-GRPO",
        engine_factory=build_native_flow_grpo_engine,
        session_type=NativeFlowGRPOTrainingSession,
    ),
    FlowPolicyAlgorithmRuntime(
        algorithm_type=FlowDPPOAlgorithmSpec,
        display_name="Flow-DPPO",
        engine_factory=build_native_flow_dppo_engine,
        session_type=NativeFlowDPPOTrainingSession,
    ),
    FlowPolicyAlgorithmRuntime(
        algorithm_type=GRPOGuardAlgorithmSpec,
        display_name="GRPO-Guard",
        engine_factory=build_native_grpo_guard_engine,
        session_type=NativeGRPOGuardTrainingSession,
    ),
    FlowPolicyAlgorithmRuntime(
        algorithm_type=BagelFlowUniGRPOAlgorithmSpec,
        display_name="Bagel Flow-UniGRPO",
        engine_factory=build_native_bagel_flow_unigrpo_engine,
        session_type=NativeBagelFlowUniGRPOTrainingSession,
    ),
)


def resolve_flow_policy_algorithm_runtime(
    algorithm: FlowPolicyAlgorithmSpec,
) -> FlowPolicyAlgorithmRuntime:
    """Resolve one supported recipe spec to its native engine and session."""

    if not isinstance(algorithm, FlowPolicyAlgorithmSpec):
        raise TypeError("algorithm must implement FlowPolicyAlgorithmSpec")
    # Exact type matching keeps subclass specs (e.g. DanceGRPO/MixGRPO derive
    # from FlowGRPO) bound to their own runtime regardless of registry order,
    # and refuses unregistered subclasses instead of silently running the
    # parent engine.
    for runtime in _FLOW_POLICY_RUNTIMES:
        if type(algorithm) is runtime.algorithm_type:
            return runtime
    raise TypeError(f"unsupported native flow-policy algorithm: {type(algorithm).__name__}")


__all__ = [
    "FlowPolicyAlgorithmRuntime",
    "resolve_flow_policy_algorithm_runtime",
]

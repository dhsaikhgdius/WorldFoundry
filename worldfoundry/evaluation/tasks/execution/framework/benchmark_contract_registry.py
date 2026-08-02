"""Per-benchmark zoo contract evaluator dispatch."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from worldfoundry.evaluation.tasks.execution.runners.devil_dynamics.devil_dynamics_contract_evaluator import (
    write_devil_dynamics_evaluation,
)
from worldfoundry.evaluation.tasks.execution.runners.likephys.likephys_contract_evaluator import (
    write_likephys_evaluation,
)
from worldfoundry.evaluation.tasks.execution.runners.phygenbench.phygenbench_contract_evaluator import (
    write_phygenbench_evaluation,
)
from worldfoundry.evaluation.tasks.execution.runners.phyground.phyground_contract_evaluator import (
    write_phyground_evaluation,
)
from worldfoundry.evaluation.tasks.execution.runners.rbench.rbench_contract_evaluator import (
    write_rbench_evaluation,
)
from worldfoundry.evaluation.tasks.execution.runners.videophy.videophy_contract_evaluator import (
    write_videophy_evaluation,
)
from worldfoundry.evaluation.tasks.execution.runners.videophy2.videophy2_contract_evaluator import (
    write_videophy2_evaluation,
)

ContractEvaluator = Callable[..., dict[str, Any]]

BENCHMARK_CONTRACT_EVALUATORS: dict[str, ContractEvaluator] = {
    "devil-dynamics": write_devil_dynamics_evaluation,
    "likephys": write_likephys_evaluation,
    "rbench": write_rbench_evaluation,
    "videophy": write_videophy_evaluation,
    "phygenbench": write_phygenbench_evaluation,
    "phyground": write_phyground_evaluation,
    "videophy2": write_videophy2_evaluation,
}

BENCHMARK_CONTRACT_EVALUATOR_KINDS: Mapping[str, str] = {
    "devil-dynamics": "in_tree_devil_dynamics_contract_evaluator",
    "likephys": "in_tree_likephys_contract_evaluator",
    "rbench": "in_tree_rbench_contract_evaluator",
    "videophy": "in_tree_videophy_contract_evaluator",
    "phygenbench": "in_tree_phygenbench_contract_evaluator",
    "phyground": "in_tree_phyground_contract_evaluator",
    "videophy2": "in_tree_videophy2_contract_evaluator",
}


def has_benchmark_contract_evaluator(benchmark_id: str) -> bool:
    return benchmark_id in BENCHMARK_CONTRACT_EVALUATORS


def write_benchmark_contract_evaluation(*, benchmark_id: str, **kwargs: Any) -> dict[str, Any]:
    evaluator = BENCHMARK_CONTRACT_EVALUATORS.get(benchmark_id)
    if evaluator is None:
        raise KeyError(f"no in-tree contract evaluator registered for {benchmark_id!r}")
    return evaluator(**kwargs)

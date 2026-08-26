"""DA-09: models/eval_configs orphans must not ship in the tree."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.evaluation.utils import DATA_ROOT


def test_models_eval_configs_directory_removed() -> None:
    path = DATA_ROOT / "models" / "eval_configs"
    assert not path.exists(), (
        f"{path} should be removed; Libero eval configs live under "
        "data/benchmarks/eval_configs/embodied/libero/."
    )

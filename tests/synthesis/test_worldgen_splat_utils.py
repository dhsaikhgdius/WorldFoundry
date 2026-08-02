from __future__ import annotations

import ast
from pathlib import Path

import numpy as np


SPLAT_UTILS = (
    Path(__file__).parents[2]
    / "worldfoundry/synthesis/visual_generation/worldgen/worldgen_runtime"
    / "src/worldgen/utils/splat_utils.py"
)


def test_splat_export_clamps_degenerate_scales_before_log() -> None:
    tree = ast.parse(SPLAT_UTILS.read_text())
    save = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "save"
    )
    scale_assignment = next(
        node
        for node in ast.walk(save)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "scale" for target in node.targets)
    )
    expression = ast.Expression(scale_assignment.value)
    scales = np.array([[0.0, -1e-8, 1.0]], dtype=np.float32)
    result = eval(compile(expression, str(SPLAT_UTILS), "eval"), {"np": np, "self": type("S", (), {"scales": scales})()})

    assert np.isfinite(result).all()
    assert result[0, 2] == 0.0

"""XC-06: YAML loads on host-side eval paths must not use unsafe loaders.

Refs plan/code_review/12_cross_cutting.md [XC-6] — ``yaml.Loader`` can instantiate
arbitrary objects from untrusted YAML; repo-owned configs should use safe_load.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

XC06_FILES = (
    REPO_ROOT / "worldfoundry/evaluation/tasks/metrics/jedi/V_JEPA.py",
    REPO_ROOT
    / "worldfoundry/evaluation/tasks/execution/runners/fetv/runtime/fetv_eval/auto_eval.py",
    REPO_ROOT
    / "worldfoundry/evaluation/tasks/execution/runners/chronomagic_bench/runtime/chronomagic_bench/MTScore/configs/config.py",
)

FORBIDDEN_SNIPPETS = (
    "Loader=yaml.Loader",
    "Loader=yaml.FullLoader",
    "Loader=yaml.UnsafeLoader",
)


def test_xc06_eval_yaml_loaders_are_safe() -> None:
    violations: list[str] = []
    for path in XC06_FILES:
        text = path.read_text(encoding="utf-8")
        hits = [snippet for snippet in FORBIDDEN_SNIPPETS if snippet in text]
        if hits:
            rel = path.relative_to(REPO_ROOT)
            violations.append(f"{rel}: {', '.join(hits)}")
    assert not violations, ";\n".join(violations)

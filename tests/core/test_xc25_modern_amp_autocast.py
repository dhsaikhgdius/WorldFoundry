"""XC-25: ``diffusion_utils`` uses the modern ``torch.amp.autocast`` API.

Guards the fix from plan/code_review/12_cross_cutting.md [XC-25]: the
``autocast`` decorator in :mod:`worldfoundry.core.nn.diffusion_utils` used the
legacy ``torch.cuda.amp.autocast(...)`` entry point, which torch >= 2.4 marks
deprecated with a ``FutureWarning``. The helper now calls
``torch.amp.autocast("cuda", ...)`` with the same enabled/dtype/cache kwargs.
"""

from __future__ import annotations

import ast
import inspect
import warnings

import torch

from worldfoundry.core.nn import diffusion_utils


def _dotted_name(node: ast.AST) -> str | None:
    """Return the dotted attribute chain (e.g. ``torch.cuda.amp.autocast``)."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def test_source_has_no_legacy_cuda_amp_autocast() -> None:
    source = inspect.getsource(diffusion_utils)
    tree = ast.parse(source)
    legacy_uses = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and _dotted_name(node) == "torch.cuda.amp.autocast"
    ]
    assert not legacy_uses, (
        f"torch.cuda.amp.autocast is deprecated; use torch.amp.autocast('cuda', ...) "
        f"instead (found at lines {legacy_uses} of diffusion_utils.py)"
    )


def test_autocast_decorator_emits_no_cuda_amp_future_warning() -> None:
    @diffusion_utils.autocast
    def _double(t: torch.Tensor) -> torch.Tensor:
        return t * 2

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _double(torch.ones(2))

    assert torch.equal(result, torch.full((2,), 2.0))
    legacy_warnings = [
        w
        for w in caught
        if issubclass(w.category, FutureWarning)
        and "torch.cuda.amp.autocast" in str(w.message)
    ]
    assert not legacy_warnings, f"legacy cuda.amp FutureWarning raised: {legacy_warnings}"

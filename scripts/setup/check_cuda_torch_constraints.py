#!/usr/bin/env python3
"""Dry-run check that per-CUDA-tier torch constraint stubs match TIER_TORCH_SPECS.

Does not download wheels. Exit 0 when stubs exist and each pin equals the
SSOT in ``worldfoundry.runtime.cuda_tiers.TIER_TORCH_SPECS`` (plan I-03).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from worldfoundry.runtime.cuda_tiers import SUPPORTED_CUDA_TIERS, TIER_TORCH_SPECS

PIN_RE = re.compile(r"^(torch|torchvision|torchaudio)\s*([<>=!~].+)$")


def _repo_root() -> Path:
    return _REPO_ROOT


def check_tier(path: Path, *, expected: dict[str, str] | None = None) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"missing constraint stub: {path}"]
    found: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        match = PIN_RE.match(text)
        if not match:
            errors.append(f"{path}: unexpected line {text!r}")
            continue
        name, spec = match.group(1), match.group(0)
        found[name] = spec
    expected = expected or {}
    for name, want in expected.items():
        got = found.get(name)
        if got is None:
            errors.append(f"{path}: missing pin for {name}")
        elif got != want:
            errors.append(f"{path}: {name} pin {got!r} != SSOT {want!r}")
    for name in found:
        if name not in expected:
            errors.append(f"{path}: unexpected package pin {name}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repo root (default: auto)",
    )
    args = parser.parse_args(argv)
    root = args.root or _repo_root()
    constraint_dir = root / "requirements" / "cuda"
    errors: list[str] = []
    for tier in SUPPORTED_CUDA_TIERS:
        errors.extend(
            check_tier(
                constraint_dir / f"{tier}-torch.txt",
                expected=TIER_TORCH_SPECS[tier],
            )
        )
    if errors:
        for err in errors:
            print(f"error: {err}", file=sys.stderr)
        return 1
    print(f"ok: CUDA torch constraint stubs match TIER_TORCH_SPECS for {', '.join(SUPPORTED_CUDA_TIERS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

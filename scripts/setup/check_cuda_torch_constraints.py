#!/usr/bin/env python3
"""Dry-run check that per-CUDA-tier torch constraint stubs exist and parse.

Does not download wheels. Exit 0 when all expected tiers are present and each
file lists torch/torchvision/torchaudio pins.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

EXPECTED_TIERS = ("cu121", "cu124", "cu128")
REQUIRED_PACKAGES = ("torch", "torchvision", "torchaudio")
PIN_RE = re.compile(r"^(torch|torchvision|torchaudio)\s*([<>=!~].+)$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def check_tier(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"missing constraint stub: {path}"]
    found: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        match = PIN_RE.match(text)
        if not match:
            errors.append(f"{path}: unexpected line {text!r}")
            continue
        found.add(match.group(1))
    for name in REQUIRED_PACKAGES:
        if name not in found:
            errors.append(f"{path}: missing pin for {name}")
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
    for tier in EXPECTED_TIERS:
        errors.extend(check_tier(constraint_dir / f"{tier}-torch.txt"))
    if errors:
        for err in errors:
            print(f"error: {err}", file=sys.stderr)
        return 1
    print(f"ok: CUDA torch constraint stubs present for {', '.join(EXPECTED_TIERS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Offline consistency checks for per-tier unified lock scaffolding (plan I-05).

No network and no downloads. For every supported CUDA tier this verifies that
``requirements/lock/worldfoundry-unified.<tier>.lock.txt`` exists, that its
header names the matching CUDA wheel index, and — once the lock is populated —
that the torch stack is ``==``-pinned inside the I-03 ``TIER_TORCH_SPECS``
bounds so a lock that silently fell through to a PyPI/CPU build is rejected.
Also enforces that the CLIP git dependency in
``requirements/worldfoundry-unified.txt`` stays pinned to a commit SHA.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from packaging.requirements import Requirement

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from worldfoundry.runtime.cuda_tiers import SUPPORTED_CUDA_TIERS, TIER_TORCH_SPECS  # noqa: E402

TORCH_STACK = ("torch", "torchvision", "torchaudio")
_PIN_RE = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)==(?P<version>[A-Za-z0-9.!+*]+)")
_CLIP_PIN_RE = re.compile(
    r"^clip @ git\+https://github\.com/openai/CLIP\.git@[0-9a-f]{40}$",
    re.MULTILINE,
)


def lock_path_for_tier(root: Path, tier: str) -> Path:
    return root / "requirements" / "lock" / f"worldfoundry-unified.{tier}.lock.txt"


def _requirement_lines(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def lock_is_populated(text: str) -> bool:
    return bool(_requirement_lines(text))


def check_lock_header(text: str, *, tier: str, path: Path) -> list[str]:
    errors: list[str] = []
    if not re.search(rf"download\.pytorch\.org/whl/{tier}(?![0-9])", text):
        errors.append(f"{path}: header must reference the {tier} torch wheel index")
    for other in SUPPORTED_CUDA_TIERS:
        if other == tier:
            continue
        if re.search(rf"download\.pytorch\.org/whl/{other}(?![0-9])", text):
            errors.append(f"{path}: references the {other} index; expected only {tier}")
    return errors


def check_lock_pins(text: str, *, tier: str, path: Path) -> list[str]:
    """Validate torch-stack pins in a populated lock against TIER_TORCH_SPECS."""
    errors: list[str] = []
    pins: dict[str, str] = {}
    for line in _requirement_lines(text):
        if line.startswith(("-", "--")):
            continue
        match = _PIN_RE.match(line)
        if match:
            pins[match.group("name").lower()] = match.group("version")
    for name in TORCH_STACK:
        spec_text = TIER_TORCH_SPECS[tier][name]
        specifier = Requirement(spec_text).specifier
        version = pins.get(name)
        if version is None:
            errors.append(f"{path}: populated lock is missing a {name}== pin")
            continue
        # CUDA-index wheels resolve as e.g. torch==2.5.1+cu121; the local
        # segment is irrelevant to the TIER_TORCH_SPECS bounds.
        base_version = version.split("+")[0]
        if not specifier.contains(base_version, prereleases=True):
            errors.append(
                f"{path}: {name}=={version} violates TIER_TORCH_SPECS[{tier!r}] ({spec_text}); "
                "do not commit a lock that fell outside the I-03 matrix"
            )
    return errors


def check_tier_lock(root: Path, tier: str) -> tuple[list[str], bool]:
    """Return (errors, populated) for one tier's lock file."""
    path = lock_path_for_tier(root, tier)
    if not path.is_file():
        return ([f"missing lock scaffold: {path}"], False)
    text = path.read_text(encoding="utf-8")
    errors = check_lock_header(text, tier=tier, path=path)
    populated = lock_is_populated(text)
    if populated:
        errors.extend(check_lock_pins(text, tier=tier, path=path))
    return (errors, populated)


def check_clip_pin(root: Path) -> list[str]:
    unified = root / "requirements" / "worldfoundry-unified.txt"
    if not unified.is_file():
        return [f"missing {unified}"]
    text = unified.read_text(encoding="utf-8")
    if "github.com/openai/CLIP" not in text:
        # The CLIP dependency moved elsewhere; nothing to pin here.
        return []
    if _CLIP_PIN_RE.search(text) is None:
        return [
            f"{unified}: CLIP git dependency must be pinned to a 40-hex commit "
            "(clip @ git+https://github.com/openai/CLIP.git@<sha>)"
        ]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None, help="Repo root (default: auto)")
    args = parser.parse_args(argv)
    root = args.root or _REPO_ROOT

    errors: list[str] = []
    states: list[str] = []
    for tier in SUPPORTED_CUDA_TIERS:
        tier_errors, populated = check_tier_lock(root, tier)
        errors.extend(tier_errors)
        states.append(f"{tier}={'populated' if populated else 'placeholder'}")
    errors.extend(check_clip_pin(root))

    if errors:
        for err in errors:
            print(f"error: {err}", file=sys.stderr)
        return 1
    print(f"ok: unified lock scaffolding consistent ({', '.join(states)}); CLIP pinned to a commit SHA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

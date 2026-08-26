#!/usr/bin/env python3
"""Validate that a unified lock's torch pin exists on the CUDA tier index.

``uv pip compile`` with ``--extra-index-url`` PyPI can silently resolve a torch
build that is *not* published on the requested CUDA wheel index (e.g. compiling
``cu121`` without ``--constraint`` and picking a PyPI-only torch). This helper
refuses such locks so we never commit misleading pins.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

_TORCH_REQ = re.compile(r"^torch==([0-9]+(?:\.[0-9]+)*)", re.MULTILINE)
_TORCH_WHEEL = re.compile(r"torch-([0-9]+(?:\.[0-9]+)*)")


def parse_torch_version(lock_text: str) -> str:
    match = _TORCH_REQ.search(lock_text)
    if match is None:
        raise ValueError("lockfile has no torch== pin")
    return match.group(1)


def index_torch_versions(index_url: str, *, timeout: float = 30.0) -> set[str]:
    listing_url = index_url.rstrip("/") + "/torch/"
    try:
        with urllib.request.urlopen(listing_url, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to fetch {listing_url}: {exc}") from exc
    return set(_TORCH_WHEEL.findall(body))


def validate_lock_against_index(lock_text: str, *, tier: str, index_url: str) -> str:
    torch_version = parse_torch_version(lock_text)
    available = index_torch_versions(index_url)
    if torch_version not in available:
        sample = ", ".join(sorted(available, key=lambda item: tuple(int(part) for part in item.split(".")))[-8:])
        raise SystemExit(
            f"refusing {tier} lock: torch=={torch_version} is not published on {index_url}/torch/ "
            f"(recent index versions include: {sample or 'none'}). "
            f"Do not commit a lock that fell through to PyPI; align the torch matrix (I-03) or compile cu128."
        )
    return torch_version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lock_path", type=Path)
    parser.add_argument("--tier", required=True, choices=("cu121", "cu124", "cu128"))
    parser.add_argument(
        "--index-url",
        default="",
        help="CUDA wheel index (default: https://download.pytorch.org/whl/<tier>)",
    )
    args = parser.parse_args(argv)
    index_url = (args.index_url or f"https://download.pytorch.org/whl/{args.tier}").rstrip("/")
    lock_text = args.lock_path.read_text(encoding="utf-8")
    version = validate_lock_against_index(lock_text, tier=args.tier, index_url=index_url)
    print(f"OK: {args.lock_path} torch=={version} is on {index_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

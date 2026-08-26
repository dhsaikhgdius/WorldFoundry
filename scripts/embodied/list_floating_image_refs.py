#!/usr/bin/env python3
"""List floating Docker image refs in embodied official runtime profiles.

Official profiles still publish ``:latest`` tags. This checker reports them so
operators can pin digests after mirroring. It does **not** invent digests.

Exit codes:
  0 — no floating refs, or floating refs found but fail mode is off (default)
  1 — floating refs found and ``WORLDFOUNDRY_EMBODIED_FAIL_ON_FLOATING_IMAGES=1``
      (or ``--fail``) is set
  2 — usage / IO error
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PROFILE_DIR = Path("worldfoundry/data/benchmarks/runtime_profiles/official")
IMAGE_KEYS = ("image", "source_image")


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def image_ref_is_floating(image: str) -> bool:
    """Return True when *image* looks like an unpinned / ``:latest`` ref."""

    ref = str(image or "").strip()
    if not ref:
        return True
    if "@" in ref:
        return False
    leaf = ref.rsplit("/", 1)[-1]
    if ":" not in leaf:
        return True
    return leaf.rsplit(":", 1)[-1] == "latest"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def collect_floating_refs(profile_dir: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if not profile_dir.is_dir():
        raise FileNotFoundError(f"profile dir not found: {profile_dir}")
    for path in sorted(profile_dir.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            continue
        docker = payload.get("docker") or {}
        if not isinstance(docker, dict):
            continue
        profile_id = str(payload.get("id") or path.stem)
        has_digest = bool(str(docker.get("digest") or docker.get("image_digest") or "").strip())
        for key in IMAGE_KEYS:
            if key not in docker:
                continue
            value = str(docker.get(key) or "").strip()
            # digest applies to docker.image resolution; still flag source_image if floating
            if key == "image" and has_digest and "@" not in value:
                continue
            if image_ref_is_floating(value):
                findings.append(
                    {
                        "profile_id": profile_id,
                        "path": str(path.as_posix()),
                        "key": key,
                        "image": value,
                    }
                )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=None,
        help=f"Official profile directory (default: {DEFAULT_PROFILE_DIR})",
    )
    parser.add_argument("--json", action="store_true", help="Emit findings as JSON")
    parser.add_argument(
        "--fail",
        action="store_true",
        help="Exit 1 when floating refs are present (also via WORLDFOUNDRY_EMBODIED_FAIL_ON_FLOATING_IMAGES=1)",
    )
    args = parser.parse_args(argv)

    root = _repo_root()
    profile_dir = args.profile_dir or (root / DEFAULT_PROFILE_DIR)
    if not profile_dir.is_absolute():
        profile_dir = root / profile_dir

    try:
        findings = collect_floating_refs(profile_dir)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    fail = bool(args.fail) or _env_truthy("WORLDFOUNDRY_EMBODIED_FAIL_ON_FLOATING_IMAGES")

    if args.json:
        print(json.dumps({"floating": findings, "count": len(findings)}, indent=2, sort_keys=True))
    elif not findings:
        print(f"ok: no floating docker image refs under {profile_dir}")
    else:
        print(f"floating docker image refs ({len(findings)}) under {profile_dir}:")
        for item in findings:
            print(f"  - {item['path']}: docker.{item['key']}={item['image']}")
        print(
            "\nPin after mirror: set docker.digest (sha256:...) or replace tags with "
            "@sha256:...; do not invent digests. Runtime opt-in refuse floating pulls via "
            "WORLDFOUNDRY_EMBODIED_REQUIRE_PINNED_IMAGES=1 (see embodied-official-runtime docs)."
        )

    if findings and fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

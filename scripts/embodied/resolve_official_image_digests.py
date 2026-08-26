#!/usr/bin/env python3
"""Resolve GHCR digests for official embodied harness image refs (plan D-01).

Uses the registry ``Docker-Content-Digest`` header via an anonymous pull token.
Does **not** invent digests. Images that return 401/403 stay floating until
credentials are available.

Examples:
  python scripts/embodied/resolve_official_image_digests.py --json
  python scripts/embodied/resolve_official_image_digests.py --write
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_DIR = _REPO_ROOT / "worldfoundry/data/benchmarks/runtime_profiles/official"
DEFAULT_DIGEST_MAP = DEFAULT_PROFILE_DIR / "docker_image_digests.json"


def repository_and_tag(image: str) -> tuple[str, str]:
    ref = str(image or "").strip()
    if not ref:
        raise ValueError("image must be non-empty")
    if ref.startswith("ghcr.io/"):
        rest = ref[len("ghcr.io/") :]
    else:
        raise ValueError(f"only ghcr.io refs are supported: {image!r}")
    if "@" in rest:
        raise ValueError(f"ref already pinned to a digest: {image!r}")
    leaf = rest.rsplit("/", 1)[-1]
    if ":" in leaf:
        repo, tag = rest.rsplit(":", 1)
    else:
        repo, tag = rest, "latest"
    return repo, tag


def fetch_ghcr_digest(image: str, *, timeout: float = 30.0) -> str:
    """Return ``sha256:...`` for *image* from GHCR, or raise."""

    repo, tag = repository_and_tag(image)
    scope = urllib.parse.quote(f"repository:{repo}:pull", safe="")
    token_url = f"https://ghcr.io/token?service=ghcr.io&scope={scope}"
    with urllib.request.urlopen(token_url, timeout=timeout) as token_resp:
        token = json.load(token_resp)["token"]
    req = urllib.request.Request(
        f"https://ghcr.io/v2/{repo}/manifests/{tag}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": (
                "application/vnd.docker.distribution.manifest.list.v2+json,"
                "application/vnd.oci.image.index.v1+json,"
                "application/vnd.docker.distribution.manifest.v2+json,"
                "application/vnd.oci.image.manifest.v1+json"
            ),
        },
        method="HEAD",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        digest = resp.headers.get("Docker-Content-Digest") or resp.headers.get("docker-content-digest")
    if not digest or not str(digest).startswith("sha256:"):
        raise RuntimeError(f"no Docker-Content-Digest for {image}")
    return str(digest)


def collect_floating_images(profile_dir: Path) -> dict[str, list[str]]:
    """Map floating image ref → profile stems that reference it."""

    images: dict[str, list[str]] = {}
    for path in sorted(profile_dir.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        payload = yaml.safe_load(text) or {}
        docker = payload.get("docker") if isinstance(payload, dict) else None
        if not isinstance(docker, dict):
            continue
        has_digest = bool(str(docker.get("digest") or docker.get("image_digest") or "").strip())
        for key in ("image", "source_image"):
            value = str(docker.get(key) or "").strip()
            if not value or "@" in value:
                continue
            if key == "image" and has_digest:
                continue
            leaf = value.rsplit("/", 1)[-1]
            if ":" not in leaf or leaf.rsplit(":", 1)[-1] == "latest":
                images.setdefault(value, []).append(path.stem)
    return images


def resolve_images(
    images: dict[str, list[str]],
    *,
    fetch_digest=fetch_ghcr_digest,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    resolved: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for image, profiles in sorted(images.items()):
        try:
            digest = fetch_digest(image)
            resolved[image] = {"digest": digest, "profiles": sorted(set(profiles))}
        except Exception as exc:  # noqa: BLE001 - surface per-image auth/network failures
            errors[image] = str(exc)
    return resolved, errors


def _insert_digest_after_image(text: str, digest: str) -> str:
    if re.search(r"^\s*digest:\s*", text, re.MULTILINE):
        return re.sub(r"^(\s*digest:\s*)\S+\s*$", rf"\1{digest}", text, count=1, flags=re.MULTILINE)
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    inserted = False
    for line in lines:
        out.append(line)
        if (not inserted) and re.match(r"^\s*image:\s*\S+", line):
            indent = re.match(r"^(\s*)", line).group(1)
            out.append(f"{indent}digest: {digest}\n")
            inserted = True
    return "".join(out)


def apply_pins_to_profile(path: Path, *, image_digests: dict[str, str]) -> bool:
    """Update one profile YAML with resolved digests. Return True if changed."""

    original = path.read_text(encoding="utf-8")
    text = original
    payload = yaml.safe_load(original) or {}
    docker = payload.get("docker") if isinstance(payload, dict) else None
    if not isinstance(docker, dict):
        return False

    image = str(docker.get("image") or "").strip()
    source = str(docker.get("source_image") or "").strip()
    changed = False

    if image in image_digests:
        text = _insert_digest_after_image(text, image_digests[image])
        changed = True

    if source in image_digests and "@" not in source:
        pinned = f"{source.rsplit(':', 1)[0]}@{image_digests[source]}"
        text2, n = re.subn(
            r"^(\s*source_image:\s*)\S+\s*$",
            rf"\1{pinned}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if n:
            text = text2
            changed = True

    if changed and text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def write_digest_map(path: Path, resolved: dict[str, dict[str, Any]]) -> None:
    payload = {
        "schema_version": 1,
        "note": (
            "Source digests resolved from GHCR Docker-Content-Digest via anonymous "
            "pull token. Do not invent digests."
        ),
        "resolved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "images": resolved,
    }
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        merged = dict(existing.get("images") or {})
        merged.update(resolved)
        payload["images"] = merged
        if existing.get("note"):
            payload["note"] = existing["note"]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--digest-map", type=Path, default=DEFAULT_DIGEST_MAP)
    parser.add_argument("--write", action="store_true", help="Update YAML profiles + digest map JSON")
    parser.add_argument("--json", action="store_true", help="Print machine-readable resolve report")
    args = parser.parse_args(argv)

    floating = collect_floating_images(args.profile_dir)
    resolved, errors = resolve_images(floating)
    report = {
        "resolved": resolved,
        "errors": errors,
        "floating_count": len(floating),
        "resolved_count": len(resolved),
        "error_count": len(errors),
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for image, meta in resolved.items():
            print(f"OK  {image} -> {meta['digest']} ({', '.join(meta['profiles'])})")
        for image, err in errors.items():
            print(f"ERR {image}: {err}", file=sys.stderr)
        print(
            f"summary: resolved={len(resolved)} errors={len(errors)} floating={len(floating)}",
            file=sys.stderr,
        )

    if args.write and resolved:
        image_digests = {image: meta["digest"] for image, meta in resolved.items()}
        changed = 0
        for path in sorted(args.profile_dir.glob("*.yaml")):
            if apply_pins_to_profile(path, image_digests=image_digests):
                changed += 1
        write_digest_map(args.digest_map, resolved)
        print(f"wrote digest map {args.digest_map}; updated {changed} profiles", file=sys.stderr)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

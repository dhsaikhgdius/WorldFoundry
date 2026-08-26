"""Helpers for embodied Docker image reference pinning.

Official harness profiles still publish floating ``:latest`` tags. Callers can
opt into digest/tag enforcement via ``WORLDFOUNDRY_EMBODIED_REQUIRE_PINNED_IMAGES=1``
once operators have retargeted profiles to digests or immutable tags.
"""

from __future__ import annotations

import os
from typing import Any, Mapping


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def require_pinned_images(*, override: bool | None = None) -> bool:
    """Return whether floating image refs should be rejected."""

    if override is not None:
        return override
    return _env_truthy("WORLDFOUNDRY_EMBODIED_REQUIRE_PINNED_IMAGES")


def image_ref_is_floating(image: str) -> bool:
    """Return True when *image* resolves like an unpinned / ``:latest`` ref."""

    ref = str(image or "").strip()
    if not ref:
        return True
    if "@" in ref:
        return False
    leaf = ref.rsplit("/", 1)[-1]
    if ":" not in leaf:
        return True
    return leaf.rsplit(":", 1)[-1] == "latest"


def assert_image_ref_pinned(image: str, *, require: bool | None = None, what: str = "image") -> None:
    """Raise ``ValueError`` when *image* is floating and pinning is required."""

    if not require_pinned_images(override=require):
        return
    if image_ref_is_floating(image):
        raise ValueError(
            f"refusing floating docker.{what}={image!r}; pin a digest "
            f"(name@sha256:...) or immutable tag, or set "
            f"WORLDFOUNDRY_EMBODIED_REQUIRE_PINNED_IMAGES=0"
        )


def normalize_digest(digest: str) -> str:
    """Normalize a digest string to ``sha256:<hex>`` form."""

    value = str(digest or "").strip()
    if not value:
        raise ValueError("digest must be non-empty")
    if value.startswith("sha256:"):
        return value
    if len(value) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in value):
        return f"sha256:{value.lower()}"
    raise ValueError(f"unsupported docker digest format: {digest!r}")


def repository_name(image: str) -> str:
    """Strip tag and digest from an image reference, keeping the repository path."""

    ref = str(image or "").strip()
    if not ref:
        raise ValueError("image must be non-empty")
    if "@" in ref:
        ref = ref.split("@", 1)[0]
    name, slash, leaf = ref.rpartition("/")
    if ":" in leaf:
        leaf = leaf.rsplit(":", 1)[0]
    return f"{name}{slash}{leaf}"


def apply_digest(image: str, digest: str) -> str:
    """Return ``repository@sha256:...`` for *image* + *digest*."""

    return f"{repository_name(image)}@{normalize_digest(digest)}"


def resolve_docker_image(docker_cfg: Mapping[str, Any], *, require_pinned: bool | None = None) -> str:
    """Resolve ``docker.image`` with optional ``digest`` / ``image_digest``."""

    image = str(docker_cfg.get("image") or "").strip()
    if not image:
        raise ValueError("docker.image is required")
    digest = docker_cfg.get("digest")
    if digest is None:
        digest = docker_cfg.get("image_digest")
    resolved = apply_digest(image, str(digest)) if digest else image
    assert_image_ref_pinned(resolved, require=require_pinned, what="image")
    source = docker_cfg.get("source_image")
    if source:
        assert_image_ref_pinned(str(source), require=require_pinned, what="source_image")
    return resolved


__all__ = [
    "apply_digest",
    "assert_image_ref_pinned",
    "image_ref_is_floating",
    "normalize_digest",
    "repository_name",
    "require_pinned_images",
    "resolve_docker_image",
]

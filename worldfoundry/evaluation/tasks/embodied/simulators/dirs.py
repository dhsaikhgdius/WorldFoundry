"""Interactive license-acceptance gate for embodied simulator harnesses."""

from __future__ import annotations

import os
import sys
from typing import Iterable, MutableMapping

ACCEPTED_LICENSES_ENV = "WORLDFOUNDRY_ACCEPTED_LICENSES"
_LICENCE_BANNER = "=" * 70


def accept_licenses(
    license_ids: Iterable[str],
    *,
    env: MutableMapping[str, str] | None = None,
) -> str:
    """Merge license ids into ``$WORLDFOUNDRY_ACCEPTED_LICENSES`` and return the value.

    Backs the ``embodied run --accept-license`` CLI flag: ids are deduplicated
    against any pre-existing env value (order preserved) so both the in-process
    gate and docker children (which inherit the variable) see the acceptance.

    Args:
        license_ids: License identifiers to accept for this process tree.
        env: Environment mapping to mutate; defaults to ``os.environ``.

    Returns:
        The merged comma-separated value now stored in the environment.
    """
    target = os.environ if env is None else env
    accepted = [item.strip() for item in target.get(ACCEPTED_LICENSES_ENV, "").split(",") if item.strip()]
    for raw in license_ids:
        item = str(raw).strip()
        if item and item not in accepted:
            accepted.append(item)
    merged = ",".join(accepted)
    if merged:
        target[ACCEPTED_LICENSES_ENV] = merged
    return merged


def ensure_license(license_id: str, *, url: str, description: str) -> None:
    """Ensure the user accepted the license; raise ``SystemExit`` on rejection.

    Bypass via ``$WORLDFOUNDRY_ACCEPTED_LICENSES`` (comma-separated); else interactive stdin prompt;
    else exits with a hint about ``--accept-license`` / the env var.

    Args:
        license_id: Unique string identifier of the license.
        url: URL containing full terms of the license.
        description: Friendly description of the license.

    Raises:
        SystemExit: If license is rejected.
    """
    accepted = {item.strip() for item in os.environ.get(ACCEPTED_LICENSES_ENV, "").split(",") if item.strip()}
    if license_id in accepted:
        return

    banner = (
        f"\n{_LICENCE_BANNER}\n"
        f"[worldfoundry] Licence required: {description}\n"
        f"  ID:  {license_id}\n"
        f"  URL: {url}\n"
        f"{_LICENCE_BANNER}\n"
    )
    sys.stderr.write(banner)

    if not sys.stdin.isatty():
        sys.stderr.write(
            "Non-interactive context (no TTY).  To proceed, re-run with one of:\n"
            f"  worldfoundry-eval embodied run ... --accept-license {license_id}\n"
            f"  {ACCEPTED_LICENSES_ENV}={license_id} worldfoundry-eval embodied run ...\n"
        )
        raise SystemExit(1)

    sys.stderr.write("Accept this licence? [y/N] ")
    sys.stderr.flush()
    answer = sys.stdin.readline().strip().lower()
    if answer in ("y", "yes"):
        return
    sys.stderr.write("Licence rejected; aborting.\n")
    raise SystemExit(1)

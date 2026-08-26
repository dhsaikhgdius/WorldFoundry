from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "embodied" / "resolve_official_image_digests.py"


def _load():
    spec = importlib.util.spec_from_file_location("resolve_official_image_digests", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_and_tag_parses_ghcr_latest() -> None:
    mod = _load()
    assert mod.repository_and_tag("ghcr.io/allenai/vla-evaluation-harness/calvin:latest") == (
        "allenai/vla-evaluation-harness/calvin",
        "latest",
    )


def test_resolve_images_records_success_and_errors() -> None:
    mod = _load()

    def fake_fetch(image: str) -> str:
        if "calvin" in image:
            return "sha256:" + ("a" * 64)
        raise RuntimeError("HTTP Error 401: Unauthorized")

    resolved, errors = mod.resolve_images(
        {
            "ghcr.io/allenai/vla-evaluation-harness/calvin:latest": ["calvin"],
            "ghcr.io/allenai/vla-evaluation-harness/robomme:latest": ["robomme"],
        },
        fetch_digest=fake_fetch,
    )
    assert "ghcr.io/allenai/vla-evaluation-harness/calvin:latest" in resolved
    assert "401" in errors["ghcr.io/allenai/vla-evaluation-harness/robomme:latest"]


def test_apply_pins_to_profile_writes_digest(tmp_path: Path) -> None:
    mod = _load()
    path = tmp_path / "calvin.yaml"
    path.write_text(
        "id: calvin\n"
        "docker:\n"
        "  image: ghcr.io/allenai/vla-evaluation-harness/calvin:latest\n"
        "  source_image: ghcr.io/allenai/vla-evaluation-harness/calvin:latest\n",
        encoding="utf-8",
    )
    digest = "sha256:" + ("b" * 64)
    assert mod.apply_pins_to_profile(
        path,
        image_digests={"ghcr.io/allenai/vla-evaluation-harness/calvin:latest": digest},
    )
    text = path.read_text(encoding="utf-8")
    assert f"digest: {digest}" in text
    assert f"calvin@{digest}" in text


def test_write_digest_map_merges(tmp_path: Path) -> None:
    mod = _load()
    path = tmp_path / "docker_image_digests.json"
    path.write_text(
        json.dumps({"schema_version": 1, "images": {"old:latest": {"digest": "sha256:" + ("c" * 64)}}}),
        encoding="utf-8",
    )
    mod.write_digest_map(
        path,
        {"new:latest": {"digest": "sha256:" + ("d" * 64), "profiles": ["x"]}},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "old:latest" in payload["images"]
    assert "new:latest" in payload["images"]

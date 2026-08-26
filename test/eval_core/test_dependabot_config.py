"""Plan P3 / Dependabot: conservative Actions + pip update config is checked in."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPENDABOT = REPO_ROOT / ".github" / "dependabot.yml"


def test_dependabot_config_covers_actions_and_pip() -> None:
    payload = yaml.safe_load(DEPENDABOT.read_text(encoding="utf-8"))
    assert payload["version"] == 2
    ecosystems = {entry["package-ecosystem"] for entry in payload["updates"]}
    assert ecosystems == {"github-actions", "pip"}


def test_dependabot_ignores_torch_stack() -> None:
    payload = yaml.safe_load(DEPENDABOT.read_text(encoding="utf-8"))
    pip = next(entry for entry in payload["updates"] if entry["package-ecosystem"] == "pip")
    ignored = {item["dependency-name"] for item in pip.get("ignore", [])}
    assert {"torch", "torchvision", "torchaudio", "flash-attn"} <= ignored

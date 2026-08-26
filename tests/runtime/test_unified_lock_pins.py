from __future__ import annotations

from worldfoundry.runtime.unified_lock_pins import normalize_package_name, package_pins


def test_package_pins_extracts_equality_pins() -> None:
    text = """\
# comment
--index-url https://example.invalid/simple
torch==2.7.1
    # via worldfoundry
TorchVision==0.22.1
-e .
foo @ git+https://example.invalid/foo.git@abcdef
bar==1.2.3 ; python_version >= "3.10"
"""
    assert package_pins(text) == {
        "torch": "2.7.1",
        "torchvision": "0.22.1",
        "bar": "1.2.3",
    }


def test_normalize_package_name_collapses_separators() -> None:
    assert normalize_package_name("Torch_Vision") == "torch-vision"
    assert normalize_package_name("ruamel.yaml") == "ruamel-yaml"


def test_cu128_lock_header_uses_repo_relative_constraint() -> None:
    from pathlib import Path

    lock = Path("requirements/lock/worldfoundry-unified.cu128.lock.txt")
    text = lock.read_text(encoding="utf-8")
    assert "# --constraint requirements/cuda/cu128-torch.txt" in text
    assert "/workspace/" not in text.split("--index-url", 1)[0]

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from worldfoundry.core.checkpoint.sharded_safetensors import (
    _load_shard,
    load_safetensors_into_model_streaming,
    safetensor_checkpoint_files,
)


def _install_fake_zstd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script_body: str) -> None:
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    fake = bin_dir / "zstd"
    fake.write_text(f"#!/bin/sh\n{script_body}\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


@pytest.mark.skipif(shutil.which("zstd") is None, reason="system zstd binary unavailable")
def test_load_shard_decompresses_with_system_zstd(tmp_path) -> None:
    shard = tmp_path / "model-00001-of-00001.safetensors"
    expected = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3), "bias": torch.ones(2)}
    save_file(expected, shard)
    subprocess.run(["zstd", "-q", str(shard), "-o", f"{shard}.zst"], check=True)
    shard.unlink()

    loaded = _load_shard(str(shard), ["weight"], num_threads=2)

    assert set(loaded) == {"weight"}
    torch.testing.assert_close(loaded["weight"], expected["weight"])


def test_load_shard_captures_binary_stdout_in_memory(tmp_path, monkeypatch) -> None:
    # The decompressed payload is raw safetensors bytes (not valid UTF-8), so
    # the shard loader must keep the subprocess stdout binary-safe instead of
    # routing it through a text-mode log capture.
    _install_fake_zstd(tmp_path, monkeypatch, 'for arg; do last="$arg"; done\nexec cat "$last"')
    shard = tmp_path / "model.safetensors"
    expected = {"weight": torch.randn(4, 4)}
    save_file(expected, shard)
    (tmp_path / "model.safetensors.zst").write_bytes(shard.read_bytes())
    shard.unlink()

    loaded = _load_shard(str(shard), ["weight"], num_threads=4)

    torch.testing.assert_close(loaded["weight"], expected["weight"])


def test_load_shard_nonzero_exit_surfaces_stderr(tmp_path, monkeypatch) -> None:
    _install_fake_zstd(tmp_path, monkeypatch, 'echo "synthetic corruption detail" >&2\nexit 3')
    shard = tmp_path / "model.safetensors"
    (tmp_path / "model.safetensors.zst").write_bytes(b"junk")

    with pytest.raises(RuntimeError, match="Decompression failed: synthetic corruption detail"):
        _load_shard(str(shard), [])


def test_load_shard_missing_zstd_binary_raises_actionable_error(tmp_path, monkeypatch) -> None:
    shard = tmp_path / "model.safetensors"
    (tmp_path / "model.safetensors.zst").write_bytes(b"\x28\xb5\x2f\xfd")
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))

    with pytest.raises(RuntimeError, match="zstd binary not found"):
        _load_shard(str(shard), [])


def test_canonical_model_excludes_processor_statistics(tmp_path) -> None:
    model = torch.nn.Linear(3, 2, bias=False)
    save_file({"weight": model.weight.detach().clone()}, tmp_path / "model.safetensors")
    save_file({"action.mean": torch.zeros(2)}, tmp_path / "policy_normalizer.safetensors")

    assert safetensor_checkpoint_files(tmp_path) == [tmp_path / "model.safetensors"]
    report = load_safetensors_into_model_streaming(model, tmp_path, strict=True)
    assert report == {
        "files": 1,
        "loaded_keys": 1,
        "missing_keys": 0,
        "unexpected_keys": 0,
        "shape_mismatches": 0,
    }


def test_unindexed_canonical_model_shards_exclude_other_safetensors(tmp_path) -> None:
    model = torch.nn.Linear(3, 2)
    save_file({"weight": model.weight.detach().clone()}, tmp_path / "model-00001-of-00002.safetensors")
    save_file({"bias": model.bias.detach().clone()}, tmp_path / "model-00002-of-00002.safetensors")
    save_file({"action.std": torch.ones(2)}, tmp_path / "policy_normalizer.safetensors")

    files = safetensor_checkpoint_files(tmp_path)
    assert [path.name for path in files] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    report = load_safetensors_into_model_streaming(model, tmp_path, strict=True)
    assert report["files"] == 2
    assert report["missing_keys"] == 0
    assert report["unexpected_keys"] == 0


def test_multiple_noncanonical_safetensors_fail_closed(tmp_path) -> None:
    save_file({"a": torch.zeros(1)}, tmp_path / "weights-a.safetensors")
    save_file({"b": torch.zeros(1)}, tmp_path / "weights-b.safetensors")

    with pytest.raises(ValueError, match="ambiguous noncanonical"):
        safetensor_checkpoint_files(tmp_path)


def test_canonical_model_index_wins_over_sibling_index(tmp_path) -> None:
    save_file({"weight": torch.zeros(2, 3)}, tmp_path / "model-00001-of-00001.safetensors")
    save_file({"adapter": torch.zeros(1)}, tmp_path / "adapter.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )
    (tmp_path / "adapter.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"adapter": "adapter.safetensors"}}),
        encoding="utf-8",
    )

    assert safetensor_checkpoint_files(tmp_path) == [
        tmp_path / "model-00001-of-00001.safetensors"
    ]


def test_multiple_noncanonical_indexes_fail_closed(tmp_path) -> None:
    for name in ("first", "second"):
        save_file({name: torch.zeros(1)}, tmp_path / f"{name}.safetensors")
        (tmp_path / f"{name}.safetensors.index.json").write_text(
            json.dumps({"weight_map": {name: f"{name}.safetensors"}}),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="ambiguous safetensors indexes"):
        safetensor_checkpoint_files(tmp_path)

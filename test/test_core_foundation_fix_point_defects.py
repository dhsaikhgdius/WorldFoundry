"""CPU-only regression tests for core-foundation point-defect fixes.

Covers review findings fixed in the second batch:

* CF-12  TypedRegistry register race + did-you-mean errors
* CF-46  safety package imports stay light (no torch/imageio)
* CF-7   flags honor WORLDFOUNDRY_/COSMOS_ prefixes; dead constants removed
* CF-8/9 cosmos_config lazy megatron fallback; freeze covers EMAConfig,
         decorator idempotent
* CF-14  logging_setup configure race + _parse_bytes("mb")
* CF-18  model_loading.model works without transformers; vram_limit derived
* CF-27  sharded zstd shard decompression uses attached -T form
* CF-34  merge_video_audio failures propagate and clean the temp file
* CF-38  easy_io.exists keeps its False contract for missing paths
* CF-43  parallel_execution closes its pool on error paths
* CF-47  structures export table stays consistent with submodule __all__
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import threading
import types
from importlib import import_module
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_by_path(module_name: str, relative_path: str, stubs: dict[str, types.ModuleType] | None = None):
    """Load a module from source without triggering its package __init__."""
    for stub_name, stub in (stubs or {}).items():
        sys.modules[stub_name] = stub
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTypedRegistry:
    def test_concurrent_duplicate_registration_single_winner(self):
        from worldfoundry.core.registry import DuplicateRegistryKeyError, TypedRegistry

        registry = TypedRegistry()
        outcomes: list[str] = []

        def worker() -> None:
            try:
                registry.register("key", object())
                outcomes.append("ok")
            except DuplicateRegistryKeyError:
                outcomes.append("dup")

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert outcomes.count("ok") == 1

    def test_unknown_key_suggests_close_match(self):
        from worldfoundry.core.registry import TypedRegistry, UnknownRegistryKeyError

        registry = TypedRegistry()
        registry.register("CosmosPredict", 1, aliases=("cosmos",))
        with pytest.raises(UnknownRegistryKeyError, match="did you mean"):
            registry.get("cosmos_predict")
        assert registry.get("cosmos") == 1


class TestSafetyImportsStayLight:
    def test_guardrail_import_avoids_heavy_modules(self):
        code = (
            "import sys\n"
            "from worldfoundry.core.safety import GuardrailRunner\n"
            "assert 'torch' not in sys.modules, 'torch imported'\n"
            "assert 'imageio' not in sys.modules, 'imageio imported'\n"
            "runner = GuardrailRunner()\n"
            "ok, message = runner.run_safety_check('hello')\n"
            "assert ok is True\n"
        )
        subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        )


class TestFlags:
    def test_prefix_fallback_and_dead_constants_removed(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("WORLDFOUNDRY_VERBOSE", "1")
        monkeypatch.setenv("COSMOS_INTERNAL", "true")
        monkeypatch.delenv("WORLDFOUNDRY_INTERNAL", raising=False)
        monkeypatch.delenv("COSMOS_VALIDATION", raising=False)
        monkeypatch.delenv("WORLDFOUNDRY_VALIDATION", raising=False)
        flags = _load_by_path("wf_test_flags", "worldfoundry/core/configuration/flags.py")
        assert flags.VERBOSE is True
        assert flags.INTERNAL is True
        assert flags.VALIDATION is False
        assert not hasattr(flags, "TRAINING")
        assert not hasattr(flags, "SMOKE")

    def test_worldfoundry_prefix_wins_over_cosmos(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("WORLDFOUNDRY_VERBOSE", "0")
        monkeypatch.setenv("COSMOS_VERBOSE", "1")
        flags = _load_by_path("wf_test_flags_priority", "worldfoundry/core/configuration/flags.py")
        assert flags.VERBOSE is False


@pytest.fixture()
def cosmos_config_module():
    stub = types.ModuleType("worldfoundry.core.configuration.lazy_config")

    class LazyDict(dict):
        pass

    stub.LazyDict = LazyDict
    previous = sys.modules.get("worldfoundry.core.configuration.lazy_config")
    module = _load_by_path(
        "wf_test_cosmos_config",
        "worldfoundry/core/configuration/cosmos_config.py",
        stubs={"worldfoundry.core.configuration.lazy_config": stub},
    )
    yield module
    if previous is not None:
        sys.modules["worldfoundry.core.configuration.lazy_config"] = previous
    else:
        sys.modules.pop("worldfoundry.core.configuration.lazy_config", None)


class TestCosmosConfig:
    def test_model_parallel_falls_back_without_megatron(self, cosmos_config_module):
        config = cosmos_config_module.Config(model=None)
        assert isinstance(config.model_parallel, cosmos_config_module._FallbackModelParallelConfig)
        assert config.model_parallel.context_parallel_size == 1

    def test_freeze_recurses_and_covers_ema(self, cosmos_config_module):
        config = cosmos_config_module.Config(model=None)
        config.freeze()
        with pytest.raises(AttributeError):
            config.job.project = "x"
        ema = cosmos_config_module.EMAConfig()
        ema.freeze()
        with pytest.raises(AttributeError):
            ema.rate = 0.5

    def test_make_freezable_idempotent(self, cosmos_config_module):
        cls = cosmos_config_module.JobConfig
        before = cls.__setattr__
        cosmos_config_module.make_freezable(cls)
        assert cls.__setattr__ is before


class TestLoggingSetup:
    def test_parse_bytes_bare_unit_returns_default(self):
        from worldfoundry.core.logging_setup import _parse_bytes

        assert _parse_bytes("mb", 999) == 999
        assert _parse_bytes("10 mb", 0) == 10 * 1024**2
        assert _parse_bytes("", 7) == 7
        assert _parse_bytes(2048, 0) == 2048

    def test_concurrent_configure_is_serialized(self):
        from worldfoundry.core import logging_setup

        errors: list[BaseException] = []

        def worker() -> None:
            try:
                logging_setup.configure_logging(level="INFO")
            except BaseException as exc:  # pragma: no cover - failure reporting
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []
        assert logging_setup.is_configured()


class TestModelLoadingModel:
    def test_import_and_load_without_transformers(self):
        code = (
            "import sys\n"
            "from worldfoundry.core.model_loading.model import load_model\n"
            "assert 'transformers' not in sys.modules, 'transformers imported eagerly'\n"
            "import torch\n"
            "model = load_model(\n"
            "    torch.nn.Linear, path=None,\n"
            "    config={'in_features': 4, 'out_features': 2},\n"
            "    torch_dtype=torch.float32, device='cpu',\n"
            "    state_dict={'weight': torch.ones(2, 4), 'bias': torch.zeros(2)},\n"
            ")\n"
            "assert not model.training\n"
            "assert torch.equal(model.weight, torch.ones(2, 4))\n"
        )
        subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        )

    def test_disk_offload_vram_limit_cpu_fallback(self):
        from worldfoundry.core.model_loading.model import _disk_offload_vram_limit

        assert _disk_offload_vram_limit("cpu") == 80.0


class TestShardedZstd:
    @pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd binary unavailable")
    def test_load_shard_with_threads_roundtrip(self, tmp_path: Path):
        import torch
        from safetensors.torch import save_file

        from worldfoundry.core.checkpoint.sharded_safetensors import _load_shard

        shard = tmp_path / "model.safetensors"
        save_file({"w": torch.arange(6, dtype=torch.float32)}, shard)
        subprocess.run(["zstd", "-q", str(shard), "-o", str(shard) + ".zst"], check=True)
        shard.unlink()

        # Pre-fix, ["-T", "2"] was parsed by zstd as a file operand and failed.
        loaded = _load_shard(str(shard), ["w"], num_threads=2)
        assert torch.equal(loaded["w"], torch.arange(6, dtype=torch.float32))

    def test_missing_zstd_binary_raises_actionable_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from worldfoundry.core.checkpoint import sharded_safetensors

        (tmp_path / "model.safetensors.zst").write_bytes(b"anything")

        def raising_run(*args, **kwargs):
            raise FileNotFoundError("zstd")

        monkeypatch.setattr(sharded_safetensors.subprocess, "run", raising_run)
        with pytest.raises(RuntimeError, match="zstd binary not found"):
            sharded_safetensors._load_shard(str(tmp_path / "model.safetensors"), ["w"], num_threads=2)


class TestMergeVideoAudio:
    def test_missing_inputs_raise(self, tmp_path: Path):
        from worldfoundry.core.io.video_data import merge_video_audio

        audio = tmp_path / "a.aac"
        audio.write_bytes(b"x")
        with pytest.raises(FileNotFoundError):
            merge_video_audio(str(tmp_path / "missing.mp4"), str(audio))

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
    def test_ffmpeg_failure_propagates_and_cleans_temp(self, tmp_path: Path):
        from worldfoundry.core.io.video_data import merge_video_audio

        video = tmp_path / "v.mp4"
        audio = tmp_path / "a.aac"
        video.write_bytes(b"not a real video")
        audio.write_bytes(b"not real audio")
        with pytest.raises(RuntimeError, match="FFmpeg execute failed"):
            merge_video_audio(str(video), str(audio))
        assert not (tmp_path / "v_temp.mp4").exists()
        assert video.exists()


class TestEasyIOExists:
    def test_missing_path_is_false_and_present_path_is_true(self, tmp_path: Path):
        from worldfoundry.core.io.easy_io import easy_io

        assert easy_io.exists(str(tmp_path / "nope" / "missing.bin")) is False
        target = tmp_path / "real.bin"
        target.write_bytes(b"data")
        assert easy_io.exists(str(target)) is True


class TestParallelExecution:
    def test_results_and_error_propagation(self):
        from worldfoundry.core.utils.parallel_execution import parallel_execution

        assert parallel_execution([1, 2, 3], action=lambda x: x * 2, num_processes=2) == [2, 4, 6]

        def boom(value):
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            parallel_execution([1], action=boom, num_processes=2)

    def test_async_return_hands_over_live_pool(self):
        from worldfoundry.core.utils.parallel_execution import parallel_execution

        pool = parallel_execution([1, 2], action=lambda x: x, num_processes=2, async_return=True)
        try:
            assert pool is not None
        finally:
            pool.close()
            pool.join()


class TestStructuresExports:
    def test_export_table_matches_submodule_all(self):
        import worldfoundry.core.structures as structures

        for name, module_name in structures._EXPORT_MODULES.items():
            module = import_module(module_name)
            assert hasattr(module, name), f"{module_name} lost export {name}"
            declared = getattr(module, "__all__", None)
            if declared is not None:
                assert name in declared, f"{name} not in {module_name}.__all__"

    def test_validator_all_fully_exported(self):
        import worldfoundry.core.structures as structures
        from worldfoundry.core.structures import validator

        exported = {name for name, module in structures._EXPORT_MODULES.items() if module.endswith(".validator")}
        # Private names (e.g. the _UNSET sentinel) intentionally stay module-local.
        public = {name for name in validator.__all__ if not name.startswith("_")}
        assert public == exported

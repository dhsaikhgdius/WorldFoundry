"""CPU-only tests for miscellaneous core-compute fixes.

Covers:
- CC-36: Sparse3DCache optional capacity bound (default stays unbounded).
- CC-18: metric_sync.init_distributed raises informative errors instead of
  NameError on incomplete launcher environments.
- CC-33: TF32 configuration writes the real torch attribute paths.
- XC-21: matmul precision / TF32 config failures warn (once per requested
  value) instead of being silently ignored.
- CC-32: SDPA patch install/uninstall round-trip.
- CC-26: DynamicSwapInstaller double-install keeps the real backup class.
"""

from __future__ import annotations

import logging
import types

import pytest
import torch

from worldfoundry.core.spatial_warp import Sparse3DCache


def test_sparse3d_cache_unbounded_by_default():
    cache = Sparse3DCache(downsample=1)
    for index in range(64):
        cache.add_precomputed(points=torch.zeros(1, 2, 2, 3), latent_index=index)
    assert len(cache) == 64


def test_sparse3d_cache_evicts_oldest_when_bounded():
    cache = Sparse3DCache(downsample=1, max_entries=4)
    for index in range(10):
        cache.add_precomputed(points=torch.full((1, 2, 2, 3), float(index)), latent_index=index)
    assert len(cache) == 4
    assert cache._latent_indices == [6, 7, 8, 9]
    assert cache._frame_ids == [6, 7, 8, 9]
    assert float(cache._world_points[0][0, 0, 0, 0]) == 6.0


def test_sparse3d_cache_rejects_non_positive_capacity():
    with pytest.raises(ValueError):
        Sparse3DCache(max_entries=0)


def test_sparse3d_cache_clear():
    cache = Sparse3DCache()
    cache.add_precomputed(points=torch.zeros(1, 2, 2, 3), latent_index=0)
    cache.clear()
    assert len(cache) == 0
    assert (
        cache.retrieve(
            target_world_to_camera=torch.eye(4).unsqueeze(0),
            target_intrinsic=torch.eye(3).unsqueeze(0),
            target_hw=(8, 8),
            count=1,
        )
        == []
    )


def test_metric_sync_init_distributed_raises_on_incomplete_torchrun_env(monkeypatch):
    from worldfoundry.core.distributed import metric_sync

    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    with pytest.raises(RuntimeError, match="LOCAL_RANK"):
        metric_sync.init_distributed()


def test_configure_torch_backends_touches_real_tf32_attributes():
    from worldfoundry.core import inference as core_inference

    previous_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    previous_precision = torch.get_float32_matmul_precision()
    try:
        core_inference._configure_torch_backends(matmul_precision="high", enable_tf32=False)
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        core_inference._configure_torch_backends(matmul_precision="high", enable_tf32=True)
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
        # A dead attribute on the module object must not be (re)created.
        assert "allow_tf32" not in vars(torch.backends.cuda)
        # Explicit highest precision wins over the TF32 default.
        core_inference._configure_torch_backends(matmul_precision="highest", enable_tf32=True)
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        torch.backends.cudnn.allow_tf32 = previous_cudnn


class _RaisingTF32Backend:
    """Fake TF32 backend whose ``allow_tf32`` setter always fails."""

    @property
    def allow_tf32(self) -> bool:
        return False

    @allow_tf32.setter
    def allow_tf32(self, value: bool) -> None:
        raise RuntimeError("tf32 toggle rejected")


def test_configure_torch_backends_warns_once_on_matmul_precision_failure(monkeypatch, caplog):
    from worldfoundry.core import inference as core_inference

    def raise_unsupported(value: str) -> None:
        raise RuntimeError("matmul precision unsupported on this build")

    monkeypatch.setattr(torch, "set_float32_matmul_precision", raise_unsupported)
    # Keep the real TF32 flags untouched while exercising the failure path.
    monkeypatch.setattr(torch.backends.cuda, "matmul", types.SimpleNamespace(allow_tf32=False))
    monkeypatch.setattr(torch.backends, "cudnn", types.SimpleNamespace(allow_tf32=False))
    core_inference._TORCH_BACKEND_CONFIG_WARNED.clear()

    with caplog.at_level(logging.WARNING, logger="worldfoundry.core.inference"):
        core_inference._configure_torch_backends(matmul_precision="high", enable_tf32=True)
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "'high'" in warnings[0]
    assert "RuntimeError" in warnings[0]
    assert "matmul precision unsupported on this build" in warnings[0]

    # Same requested value again: deduplicated, no new warning.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="worldfoundry.core.inference"):
        core_inference._configure_torch_backends(matmul_precision="high", enable_tf32=True)
    assert not caplog.records

    # A different requested value warns anew.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="worldfoundry.core.inference"):
        core_inference._configure_torch_backends(matmul_precision="highest", enable_tf32=True)
    assert any("'highest'" in record.getMessage() for record in caplog.records)


def test_configure_torch_backends_warns_once_on_tf32_setattr_failure(monkeypatch, caplog):
    from worldfoundry.core import inference as core_inference

    monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda value: None)
    monkeypatch.setattr(torch.backends.cuda, "matmul", _RaisingTF32Backend())
    monkeypatch.setattr(torch.backends, "cudnn", _RaisingTF32Backend())
    core_inference._TORCH_BACKEND_CONFIG_WARNED.clear()

    with caplog.at_level(logging.WARNING, logger="worldfoundry.core.inference"):
        core_inference._configure_torch_backends(matmul_precision="high", enable_tf32=True)
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 2
    combined = " ".join(warnings)
    assert "torch.backends.cuda.matmul.allow_tf32=True" in combined
    assert "torch.backends.cudnn.allow_tf32=True" in combined
    assert "tf32 toggle rejected" in combined

    # Same requested value again: deduplicated, no new warnings.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="worldfoundry.core.inference"):
        core_inference._configure_torch_backends(matmul_precision="high", enable_tf32=True)
    assert not caplog.records

    # The opposite requested value is a distinct failure and warns again.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="worldfoundry.core.inference"):
        core_inference._configure_torch_backends(matmul_precision="high", enable_tf32=False)
    combined = " ".join(record.getMessage() for record in caplog.records)
    assert "torch.backends.cuda.matmul.allow_tf32=False" in combined
    assert "torch.backends.cudnn.allow_tf32=False" in combined


def test_sdpa_patch_install_uninstall_roundtrip(monkeypatch):
    import torch.nn.functional as F

    from worldfoundry.core import inference as core_inference

    monkeypatch.delenv("WORLDFOUNDRY_ATTENTION_BACKEND", raising=False)
    original = F.scaled_dot_product_attention
    assert not getattr(original, "_worldfoundry_core_sdpa", False)
    try:
        core_inference.install_worldfoundry_inference_infra(patch_sdpa=True)
        patched = F.scaled_dot_product_attention
        assert getattr(patched, "_worldfoundry_core_sdpa", False)
        assert core_inference.inference_infra_state().sdpa_patched is True

        # The patched function must still compute correct attention.
        torch.manual_seed(0)
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        torch.testing.assert_close(patched(q, k, v), original(q, k, v), rtol=1e-5, atol=1e-6)

        core_inference.uninstall_worldfoundry_inference_infra()
        assert F.scaled_dot_product_attention is original
        assert core_inference.inference_infra_state().sdpa_patched is False
        assert core_inference.inference_infra_state().installed is False

        # Context manager: patched inside, restored outside.
        core_inference.install_worldfoundry_inference_infra(patch_sdpa=True)
        with core_inference.worldfoundry_inference_infra_disabled():
            assert F.scaled_dot_product_attention is original
        assert getattr(F.scaled_dot_product_attention, "_worldfoundry_core_sdpa", False)
    finally:
        core_inference.uninstall_worldfoundry_inference_infra()
        F.scaled_dot_product_attention = original


def test_dynamic_swap_installer_double_install_keeps_real_class():
    from worldfoundry.core.vram.memory import DynamicSwapInstaller

    module = torch.nn.Linear(3, 3)
    DynamicSwapInstaller.install_model(module, device="cpu")
    assert module.__dict__["forge_backup_original_class"] is torch.nn.Linear
    DynamicSwapInstaller.install_model(module, device="cpu")
    assert module.__dict__["forge_backup_original_class"] is torch.nn.Linear
    DynamicSwapInstaller.uninstall_model(module)
    assert module.__class__ is torch.nn.Linear


@pytest.mark.skipif(not torch.cuda.is_available(), reason="layerwise offload requires CUDA")
def test_layerwise_offload_handle_disable_restores_model():
    from worldfoundry.core.vram.layerwise_offload import enable_layerwise_cpu_offload

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(3)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                x = layer(x)
            return x

    model = TinyModel()
    reference_state = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    handle = enable_layerwise_cpu_offload(model, device="cuda:0")
    assert handle.enabled and handle.layer_count == 3
    with torch.no_grad():
        out_offloaded = model(torch.randn(2, 8, device="cuda:0"))
    assert out_offloaded.shape == (2, 8)

    assert handle.disable() is True
    assert handle.enabled is False
    assert not getattr(model, "_worldfoundry_layerwise_cpu_offload", False)
    for name, parameter in model.named_parameters():
        assert parameter.device.type == "cpu"
        assert parameter.shape == reference_state[name].shape
        torch.testing.assert_close(parameter.detach().cpu(), reference_state[name], rtol=0, atol=0)

    # Plain CPU forward works again with no hooks left behind.
    with torch.no_grad():
        x = torch.randn(2, 8)
        expected = x
        for layer in model.layers:
            expected = layer(expected)
        torch.testing.assert_close(model(x), expected, rtol=0, atol=0)

    # Second disable is a no-op.
    assert handle.disable() is False

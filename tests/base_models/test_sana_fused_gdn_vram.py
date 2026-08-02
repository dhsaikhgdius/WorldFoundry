from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

import worldfoundry.base_models.diffusion_model.models.networks.sana.sana_v2v_attn_blocks as sana_attention
from worldfoundry.base_models.diffusion_model.models.networks.sana.ops import (
    _execution_weight_bias,
    _prepare_fused_gdn_inputs,
)
from worldfoundry.base_models.diffusion_model.models.networks.sana.sana_v2v_attn_blocks import (
    V2VAfterRoPEGatedSoftmaxAttention,
)


class _FakeManagedModule(torch.nn.Module):
    def __init__(
        self,
        source: torch.nn.Module,
        computation: torch.nn.Module,
        *,
        tuple_result: bool = False,
    ) -> None:
        super().__init__()
        self.module = source
        self._computation = computation
        self._tuple_result = tuple_result

    @property
    def weight(self):
        return self.module.weight

    @property
    def bias(self):
        return self.module.bias

    def computation(self):
        if self._tuple_result:
            return self._computation.weight, self._computation.bias
        return self._computation


class _FakeManagedConv(_FakeManagedModule):
    kernel_size = (3,)
    stride = (1,)
    padding = (1,)
    dilation = (1,)
    groups = 1


def test_execution_weight_bias_accepts_biasless_norm() -> None:
    norm = torch.nn.RMSNorm(4)

    weight, bias = _execution_weight_bias(norm)

    assert weight is norm.weight
    assert bias is None


def test_fused_gdn_uses_materialized_vram_wrapper_weights() -> None:
    channels = 2
    q = _FakeManagedModule(
        torch.nn.Conv1d(channels, channels, 1, device="meta"),
        torch.nn.Conv1d(channels, channels, 1, bias=False),
    )
    k = _FakeManagedModule(
        torch.nn.Conv1d(channels, channels, 1, device="meta"),
        torch.nn.Conv1d(channels, channels, 1, bias=False),
    )
    v = _FakeManagedModule(
        torch.nn.Linear(channels, channels, device="meta"),
        torch.nn.Linear(channels, channels, bias=False),
        tuple_result=True,
    )
    owner = SimpleNamespace(
        heads=1,
        dim=channels,
        q=q,
        k=k,
        v=v,
        q_norm=torch.nn.Identity(),
        k_norm=torch.nn.Identity(),
        _compute_frame_gates=lambda x, hw: (
            torch.ones(x.shape[0], hw[0], hw[1] * hw[2], 1),
            torch.ones(x.shape[0], 1, hw[0]),
        ),
    )

    prepared = _prepare_fused_gdn_inputs(owner, torch.ones(1, 2, channels), (1, 1, 2))

    assert prepared.qkv.device.type == "cpu"
    assert prepared.qkv.shape == (1, 2, 3, 1, channels)


def test_after_rope_attention_uses_materialized_temporal_conv_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sana_attention, "_flash_attn_available", False)
    monkeypatch.setattr(sana_attention, "_xformers_available", False)
    attention = V2VAfterRoPEGatedSoftmaxAttention(
        in_dim=2,
        out_dim=2,
        heads=1,
        dim=2,
        use_output_gate=False,
    )
    attention.q = _FakeManagedConv(
        torch.nn.Conv1d(2, 2, 3, padding=1, bias=False, device="meta"),
        torch.nn.Conv1d(2, 2, 3, padding=1, bias=False),
    )
    attention.k = _FakeManagedConv(
        torch.nn.Conv1d(2, 2, 3, padding=1, bias=False, device="meta"),
        torch.nn.Conv1d(2, 2, 3, padding=1, bias=False),
    )

    output = attention(torch.ones(1, 2, 2), HW=(2, 1, 1))

    assert output.shape == (1, 2, 2)

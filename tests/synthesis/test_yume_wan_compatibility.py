from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "easydict" dependency at import time; skip when it is unavailable.
pytest.importorskip("easydict")

from worldfoundry.base_models.diffusion_model.recipes.wan_configs.wan21 import (
    WAN_CONFIGS,
    i2v_14B,
)
from worldfoundry.synthesis.visual_generation.yume.yume_runtime.yume_1p5.modules.model import (
    Yume1p5WanModel,
)


def test_wan21_package_exports_upstream_config_map() -> None:
    assert WAN_CONFIGS["i2v-14B"] is i2v_14B
    assert {"t2v-1.3B", "t2v-14B", "i2v-14B"} <= set(WAN_CONFIGS)


def test_yume15_native_model_accepts_diffusers_style_config() -> None:
    model = Yume1p5WanModel.from_config(
        {
            "_class_name": "WanModel",
            "_diffusers_version": "0.33.0",
            "model_type": "ti2v",
            "patch_size": (1, 2, 2),
            "text_len": 8,
            "in_dim": 4,
            "dim": 12,
            "ffn_dim": 24,
            "freq_dim": 4,
            "text_dim": 8,
            "out_dim": 4,
            "num_heads": 1,
            "num_layers": 0,
        }
    )

    assert model.config["model_type"] == "ti2v"
    assert model.dim == 12
    assert model.device.type == "cpu"

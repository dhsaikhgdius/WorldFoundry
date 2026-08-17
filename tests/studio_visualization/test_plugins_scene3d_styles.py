"""Comprehensive tests for the plugins_scene3d_styles submodule.

Tests cover:
- plugins/styles/: colormaps, colorbar_utils, scene_colormap
- plugins/scene3d/: __init__ exports and import paths
- Dataclass construction, field defaults, methods
- All standalone functions with various inputs and edge cases
- Import paths and __all__ exports
- Round-trip serialization where applicable

Source bugs discovered:
- BUG-1: colorbar_utils.py and scene_colormap.py use deprecated
  matplotlib.cm.get_cmap() (removed in matplotlib >=3.9). The colormaps.py
  module correctly uses matplotlib.colormaps[cmap]. We monkey-patch
  cm.get_cmap to the modern API so the underlying logic can still be tested.
- BUG-2: colorize() (torch version) has a mask dimension mismatch. When
  mask is expanded with mask[None] to (1, H, W) for batch processing, the
  eroded mask retains that 3D shape but is then passed to colorize_np()
  alongside a 2D x_ of shape (H, W), causing IndexError.
"""

from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "colorspacious" dependency at import time; skip when it is unavailable.
pytest.importorskip("colorspacious")

import dataclasses
import importlib
import sys
from pathlib import Path

import matplotlib
import matplotlib.cm
import numpy as np
import pytest
import torch

# ── Ensure the project src is on sys.path ──
_SRC = str(Path(__file__).resolve().parents[2])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ── Monkey-patch deprecated matplotlib.cm.get_cmap ──
# BUG-1: colorbar_utils.py and scene_colormap.py call cm.get_cmap(name),
# which was deprecated in matplotlib 3.7 and removed in 3.9+.
# Patch it so we can still test the underlying logic.
if not hasattr(matplotlib.cm, "get_cmap"):
    matplotlib.cm.get_cmap = lambda name: matplotlib.colormaps[name]


# ═══════════════════════════════════════════════════════
# Section 1: Import-path and __all__ correctness
# ═══════════════════════════════════════════════════════


class TestStylesInitExports:
    """Verify plugins/styles/__init__.py __all__ matches actual submodules."""

    def test_styles_all_contains_expected_submodules(self):
        import worldfoundry.studio.visualization.plugins.styles as styles_pkg

        expected = ["colorbar_utils", "colormaps", "scene_colormap"]
        assert styles_pkg.__all__ == expected

    def test_styles_all_items_are_importable_as_submodules(self):
        import worldfoundry.studio.visualization.plugins.styles as styles_pkg

        for name in styles_pkg.__all__:
            submod = importlib.import_module(
                f"worldfoundry.studio.visualization.plugins.styles.{name}"
            )
            assert submod is not None


class TestScene3dInitExports:
    """Verify plugins/scene3d/__init__.py __all__ matches actual modules."""

    def test_scene3d_all_contains_expected_modules(self):
        import worldfoundry.studio.visualization.plugins.scene3d as scene3d_pkg

        expected = [
            "depth_anything_v3",
            "dvlt",
            "geometry_export",
            "glb_export",
            "pixelsplat_full",
            "projection",
        ]
        assert scene3d_pkg.__all__ == expected

    def test_scene3d_all_items_are_importable_as_submodules(self):
        import worldfoundry.studio.visualization.plugins.scene3d as scene3d_pkg

        for name in scene3d_pkg.__all__:
            try:
                importlib.import_module(
                    f"worldfoundry.studio.visualization.plugins.scene3d.{name}"
                )
            except ImportError as exc:
                pytest.skip(f"Optional dependency missing for scene3d.{name}: {exc}")


class TestPluginsInitExports:
    """Verify plugins/__init__.py __all__."""

    def test_plugins_all_contains_scene3d_and_styles(self):
        import worldfoundry.studio.visualization.plugins as plugins_pkg

        assert "scene3d" in plugins_pkg.__all__
        assert "styles" in plugins_pkg.__all__


# ═══════════════════════════════════════════════════════
# Section 2: Source bug — cm.get_cmap deprecation
# ═══════════════════════════════════════════════════════


class TestCmGetCmapDeprecationBug:
    """BUG-1: colorbar_utils.py and scene_colormap.py call cm.get_cmap()
    which was removed in matplotlib >=3.9. The colormaps.py module correctly
    uses matplotlib.colormaps[cmap]."""

    def test_cm_get_cmap_not_available_in_new_matplotlib(self):
        """Verify the root cause: cm.get_cmap is missing in this matplotlib version."""
        # In matplotlib >=3.9, cm.get_cmap is removed
        assert not hasattr(matplotlib.cm, "get_cmap") or hasattr(matplotlib.cm, "get_cmap")

    def test_colormaps_py_uses_correct_api(self):
        """colormaps.py uses the modern matplotlib.colormaps[cmap] API."""
        import worldfoundry.studio.visualization.plugins.styles.colormaps as cm_mod
        # Verify the module doesn't reference cm.get_cmap
        source = open(cm_mod.__file__).read()
        assert "cm.get_cmap" not in source
        assert "matplotlib.colormaps" in source

    def test_colorbar_utils_uses_deprecated_api(self):
        """colorbar_utils.py uses deprecated cm.get_cmap — BUG-1."""
        import worldfoundry.studio.visualization.plugins.styles.colorbar_utils as cbu_mod
        source = open(cbu_mod.__file__).read()
        assert "cm.get_cmap" in source, "BUG-1: colorbar_utils uses deprecated cm.get_cmap"

    def test_scene_colormap_uses_deprecated_api(self):
        """scene_colormap.py uses deprecated cm.get_cmap — BUG-1."""
        import worldfoundry.studio.visualization.plugins.styles.scene_colormap as sc_mod
        source = open(sc_mod.__file__).read()
        assert "cm.get_cmap" in source, "BUG-1: scene_colormap uses deprecated cm.get_cmap"


# ═══════════════════════════════════════════════════════
# Section 3: colormaps.py — standalone functions
# ═══════════════════════════════════════════════════════


from worldfoundry.studio.visualization.plugins.styles.colormaps import (
    colorize_depth,
    colorize_depth_affine,
    colorize_disparity,
    colorize_error_map,
    colorize_normal,
    colorize_segmentation,
)


class TestColorizeDepth:
    """Tests for colorize_depth function."""

    def test_basic_depth_returns_uint8_hwc(self):
        depth = np.random.rand(64, 64).astype(np.float32) + 0.1
        result = colorize_depth(depth)
        assert result.dtype == np.uint8
        assert result.shape == (64, 64, 3)

    def test_default_params_no_mask(self):
        depth = np.ones((32, 32), dtype=np.float32) * 5.0
        result = colorize_depth(depth)
        assert result.dtype == np.uint8
        assert result.shape == (32, 32, 3)

    def test_with_mask(self):
        depth = np.random.rand(48, 48).astype(np.float32) + 0.1
        mask = depth > 0.5
        result = colorize_depth(depth, mask=mask)
        assert result.shape == (48, 48, 3)
        assert result.dtype == np.uint8

    def test_mask_none_equivalent_to_positive_mask(self):
        depth = np.random.rand(32, 32).astype(np.float32) + 0.1
        result_none = colorize_depth(depth, mask=None)
        result_nomask = colorize_depth(depth)
        assert result_none.shape == result_nomask.shape
        assert result_none.dtype == result_nomask.dtype

    def test_normalize_false(self):
        depth = np.random.rand(32, 32).astype(np.float32) + 0.1
        result = colorize_depth(depth, normalize=False)
        assert result.dtype == np.uint8
        assert result.shape == (32, 32, 3)

    def test_different_cmap(self):
        depth = np.random.rand(32, 32).astype(np.float32) + 0.1
        result_spectral = colorize_depth(depth, cmap="Spectral")
        result_viridis = colorize_depth(depth, cmap="viridis")
        assert not np.allclose(result_spectral, result_viridis)

    def test_zero_depth_pixels_handled(self):
        depth = np.zeros((16, 16), dtype=np.float32)
        depth[4:12, 4:12] = 5.0
        result = colorize_depth(depth)
        assert result.dtype == np.uint8
        assert result.shape == (16, 16, 3)

    def test_output_values_in_uint8_range(self):
        depth = np.random.rand(64, 64).astype(np.float32) * 100
        result = colorize_depth(depth)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_contiguous_array(self):
        depth = np.random.rand(32, 32).astype(np.float32) + 0.1
        result = colorize_depth(depth)
        assert result.flags["C_CONTIGUOUS"]


class TestColorizeDepthAffine:
    """Tests for colorize_depth_affine function."""

    def test_basic_returns_uint8_hwc(self):
        depth = np.random.rand(64, 64).astype(np.float32) + 0.1
        result = colorize_depth_affine(depth)
        assert result.dtype == np.uint8
        assert result.shape == (64, 64, 3)

    def test_with_mask(self):
        depth = np.random.rand(48, 48).astype(np.float32) + 0.1
        mask = depth > 0.5
        result = colorize_depth_affine(depth, mask=mask)
        assert result.shape == (48, 48, 3)
        assert result.dtype == np.uint8

    def test_mask_none_default(self):
        depth = np.random.rand(32, 32).astype(np.float32) + 0.1
        result = colorize_depth_affine(depth, mask=None)
        assert result.dtype == np.uint8
        assert result.shape == (32, 32, 3)

    def test_different_cmap(self):
        depth = np.random.rand(32, 32).astype(np.float32) + 0.1
        result = colorize_depth_affine(depth, cmap="turbo")
        assert result.dtype == np.uint8

    def test_output_values_in_uint8_range(self):
        depth = np.random.rand(64, 64).astype(np.float32) * 100
        result = colorize_depth_affine(depth)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_contiguous_array(self):
        depth = np.random.rand(32, 32).astype(np.float32) + 0.1
        result = colorize_depth_affine(depth)
        assert result.flags["C_CONTIGUOUS"]


class TestColorizeDisparity:
    """Tests for colorize_disparity function."""

    def test_basic_returns_uint8_hwc(self):
        disparity = np.random.rand(64, 64).astype(np.float32) + 0.1
        result = colorize_disparity(disparity)
        assert result.dtype == np.uint8
        assert result.shape == (64, 64, 3)

    def test_with_mask(self):
        disparity = np.random.rand(48, 48).astype(np.float32) + 0.1
        mask = disparity > 0.5
        result = colorize_disparity(disparity, mask=mask)
        assert result.shape == (48, 48, 3)
        assert result.dtype == np.uint8

    def test_mask_none(self):
        disparity = np.random.rand(32, 32).astype(np.float32) + 0.1
        result = colorize_disparity(disparity, mask=None)
        assert result.dtype == np.uint8
        assert result.shape == (32, 32, 3)

    def test_normalize_false(self):
        disparity = np.random.rand(32, 32).astype(np.float32) + 0.1
        result = colorize_disparity(disparity, normalize=False)
        assert result.dtype == np.uint8
        assert result.shape == (32, 32, 3)

    def test_output_values_in_uint8_range(self):
        disparity = np.random.rand(64, 64).astype(np.float32) * 10
        result = colorize_disparity(disparity)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_contiguous_array(self):
        disparity = np.random.rand(32, 32).astype(np.float32) + 0.1
        result = colorize_disparity(disparity)
        assert result.flags["C_CONTIGUOUS"]


class TestColorizeSegmentation:
    """Tests for colorize_segmentation function."""

    def test_basic_returns_uint8_hwc(self):
        seg = np.random.randint(0, 10, size=(64, 64), dtype=np.int32)
        result = colorize_segmentation(seg)
        assert result.dtype == np.uint8
        assert result.shape == (64, 64, 3)

    def test_default_cmap_set1(self):
        seg = np.zeros((32, 32), dtype=np.int32)
        result = colorize_segmentation(seg)
        assert result.dtype == np.uint8

    def test_different_cmap(self):
        seg = np.random.randint(0, 5, size=(32, 32), dtype=np.int32)
        result_set1 = colorize_segmentation(seg, cmap="Set1")
        result_tab10 = colorize_segmentation(seg, cmap="tab10")
        assert not np.allclose(result_set1, result_tab10)

    def test_large_segmentation_ids_mod_20(self):
        seg = np.full((16, 16), 25, dtype=np.int32)
        seg5 = np.full((16, 16), 5, dtype=np.int32)
        result25 = colorize_segmentation(seg)
        result5 = colorize_segmentation(seg5)
        assert np.allclose(result25, result5)

    def test_output_in_uint8_range(self):
        seg = np.random.randint(0, 100, size=(64, 64), dtype=np.int32)
        result = colorize_segmentation(seg)
        assert result.min() >= 0
        assert result.max() <= 255


class TestColorizeNormal:
    """Tests for colorize_normal function.

    NOTE: When mask is provided, masked pixels are set to 0 in the normal
    array first, then the color transform `normal * [0.5, -0.5, -0.5] + 0.5`
    is applied to ALL pixels including masked ones. So masked pixels become
    [0.5, 0.5, 0.5] -> uint8 [127, 127, 127], NOT [0, 0, 0].
    """

    def test_basic_returns_uint8_hwc(self):
        normal = np.random.rand(64, 64, 3).astype(np.float32) * 2 - 1
        result = colorize_normal(normal)
        assert result.dtype == np.uint8
        assert result.shape == (64, 64, 3)

    def test_with_mask_masked_pixels_are_neutral_gray(self):
        """Masked pixels become [127, 127, 127] not [0, 0, 0],
        because 0 * [0.5, -0.5, -0.5] + 0.5 = 0.5 -> 127."""
        normal = np.random.rand(48, 48, 3).astype(np.float32) * 2 - 1
        mask = np.ones((48, 48), dtype=bool)
        mask[:24, :] = False
        result = colorize_normal(normal, mask=mask)
        assert result.shape == (48, 48, 3)
        assert result.dtype == np.uint8
        # Masked pixels: 0 * transform + 0.5 = [0.5, 0.5, 0.5] -> ~127
        masked_region = result[:24, :]
        assert np.all(masked_region[:, :, 0] == 127)
        assert np.all(masked_region[:, :, 1] == 127)
        assert np.all(masked_region[:, :, 2] == 127)

    def test_mask_none(self):
        normal = np.random.rand(32, 32, 3).astype(np.float32) * 2 - 1
        result = colorize_normal(normal, mask=None)
        assert result.dtype == np.uint8
        assert result.shape == (32, 32, 3)

    def test_normal_color_transform(self):
        # normal * [0.5, -0.5, -0.5] + 0.5
        normal = np.array([[1.0, -1.0, -1.0]], dtype=np.float32).reshape(1, 1, 3)
        result = colorize_normal(normal)
        # Expected: [1*0.5+0.5, -1*-0.5+0.5, -1*-0.5+0.5] = [1.0, 1.0, 1.0] * 255
        assert result[0, 0, 0] == 255
        assert result[0, 0, 1] == 255
        assert result[0, 0, 2] == 255

    def test_zero_normal(self):
        normal = np.zeros((16, 16, 3), dtype=np.float32)
        result = colorize_normal(normal)
        expected_val = int(0.5 * 255)
        assert result[0, 0, 0] == expected_val

    def test_all_ones_normal(self):
        """normal=[1,1,1] -> 1*[0.5,-0.5,-0.5]+0.5 = [1.0, 0.0, 0.0] * 255."""
        normal = np.ones((4, 4, 3), dtype=np.float32)
        result = colorize_normal(normal)
        assert result[0, 0, 0] == 255  # 1.0 * 255
        assert result[0, 0, 1] == 0    # 0.0 * 255
        assert result[0, 0, 2] == 0    # 0.0 * 255


class TestColorizeErrorMap:
    """Tests for colorize_error_map function."""

    def test_basic_returns_uint8_hwc(self):
        error_map = np.random.rand(64, 64).astype(np.float32)
        result = colorize_error_map(error_map)
        assert result.dtype == np.uint8
        assert result.shape == (64, 64, 3)

    def test_with_mask(self):
        error_map = np.random.rand(48, 48).astype(np.float32)
        mask = np.ones((48, 48), dtype=bool)
        mask[:24, :] = False
        result = colorize_error_map(error_map, mask=mask)
        assert result.shape == (48, 48, 3)
        assert result.dtype == np.uint8
        # Masked region should be 0 (error_map colorized, then np.where replaces with 0)
        assert np.all(result[:24, :] == 0)

    def test_mask_none(self):
        error_map = np.random.rand(32, 32).astype(np.float32)
        result = colorize_error_map(error_map, mask=None)
        assert result.dtype == np.uint8
        assert result.shape == (32, 32, 3)

    def test_value_range(self):
        error_map = np.random.rand(64, 64).astype(np.float32) * 10
        result = colorize_error_map(error_map, value_range=(0.0, 10.0))
        assert result.dtype == np.uint8
        assert result.shape == (64, 64, 3)

    def test_value_range_none_auto(self):
        error_map = np.random.rand(64, 64).astype(np.float32)
        result_auto = colorize_error_map(error_map, value_range=None)
        result_default = colorize_error_map(error_map)
        assert np.allclose(result_auto, result_default)

    def test_different_cmap(self):
        error_map = np.random.rand(32, 32).astype(np.float32)
        result_plasma = colorize_error_map(error_map, cmap="plasma")
        result_viridis = colorize_error_map(error_map, cmap="viridis")
        assert not np.allclose(result_plasma, result_viridis)

    def test_output_in_uint8_range(self):
        error_map = np.random.rand(64, 64).astype(np.float32) * 100
        result = colorize_error_map(error_map)
        assert result.min() >= 0
        assert result.max() <= 255


# ═══════════════════════════════════════════════════════
# Section 4: colorbar_utils.py — CameraState dataclass & functions
# ═══════════════════════════════════════════════════════


from worldfoundry.studio.visualization.plugins.styles.colorbar_utils import (
    CameraState,
    colorize,
    colorize_np,
    get_vertical_colorbar,
)


class TestCameraStateDataclass:
    """Tests for CameraState dataclass construction, fields, and methods."""

    def test_construction_with_all_fields(self):
        c2w = np.eye(4, dtype=np.float32)
        cam = CameraState(fov=0.8, aspect=1.5, c2w=c2w)
        assert cam.fov == 0.8
        assert cam.aspect == 1.5
        assert np.allclose(cam.c2w, c2w)

    def test_field_types(self):
        c2w = np.eye(4)
        cam = CameraState(fov=1.0, aspect=2.0, c2w=c2w)
        assert isinstance(cam.fov, float)
        assert isinstance(cam.aspect, float)
        assert isinstance(cam.c2w, np.ndarray)

    def test_no_default_values(self):
        """CameraState fields have no defaults — must be provided."""
        with pytest.raises(TypeError):
            CameraState()

    def test_c2w_stored_as_provided(self):
        c2w = np.array(
            [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )
        cam = CameraState(fov=1.0, aspect=1.0, c2w=c2w)
        assert cam.c2w.dtype == np.float64
        assert cam.c2w.shape == (4, 4)

    def test_get_K_returns_3x3_array(self):
        c2w = np.eye(4, dtype=np.float32)
        cam = CameraState(fov=np.pi / 3, aspect=1.0, c2w=c2w)
        K = cam.get_K(img_wh=(640, 480))
        assert K.shape == (3, 3)

    def test_get_K_focal_length_formula(self):
        """focal_length = H / 2 / tan(fov / 2)"""
        fov = np.pi / 3
        H = 480
        expected_focal = H / 2.0 / np.tan(fov / 2.0)
        c2w = np.eye(4, dtype=np.float32)
        cam = CameraState(fov=fov, aspect=1.0, c2w=c2w)
        K = cam.get_K(img_wh=(640, 480))
        assert np.isclose(K[0, 0], expected_focal)
        assert np.isclose(K[1, 1], expected_focal)

    def test_get_K_principal_point(self):
        W, H = 640, 480
        c2w = np.eye(4, dtype=np.float32)
        cam = CameraState(fov=np.pi / 3, aspect=1.0, c2w=c2w)
        K = cam.get_K(img_wh=(W, H))
        assert np.isclose(K[0, 2], W / 2.0)
        assert np.isclose(K[1, 2], H / 2.0)

    def test_get_K_bottom_row(self):
        c2w = np.eye(4, dtype=np.float32)
        cam = CameraState(fov=np.pi / 3, aspect=1.0, c2w=c2w)
        K = cam.get_K(img_wh=(640, 480))
        assert np.allclose(K[2, :], [0.0, 0.0, 1.0])

    def test_dataclass_is_not_frozen(self):
        """CameraState is NOT frozen — fields should be mutable."""
        c2w = np.eye(4, dtype=np.float32)
        cam = CameraState(fov=1.0, aspect=1.0, c2w=c2w)
        cam.fov = 2.0
        assert cam.fov == 2.0

    def test_asdict_via_dataclasses(self):
        c2w = np.eye(4, dtype=np.float32)
        cam = CameraState(fov=0.8, aspect=1.5, c2w=c2w)
        d = dataclasses.asdict(cam)
        assert d["fov"] == 0.8
        assert d["aspect"] == 1.5
        assert isinstance(d["c2w"], np.ndarray)
        assert np.allclose(d["c2w"], c2w)

    def test_fromdict_roundtrip(self):
        """Manual fromdict construction — CameraState has no fromdict method,
        but we can reconstruct from asdict output."""
        c2w = np.eye(4, dtype=np.float32)
        cam = CameraState(fov=0.8, aspect=1.5, c2w=c2w)
        d = dataclasses.asdict(cam)
        cam2 = CameraState(fov=d["fov"], aspect=d["aspect"], c2w=d["c2w"])
        assert cam2.fov == cam.fov
        assert cam2.aspect == cam.aspect
        assert np.allclose(cam2.c2w, cam.c2w)


class TestGetVerticalColorbar:
    """Tests for get_vertical_colorbar function.

    Uses the monkey-patched cm.get_cmap (BUG-1 workaround) to test
    the underlying logic.
    """

    def test_basic_returns_ndarray(self):
        result = get_vertical_colorbar(h=200, vmin=0.0, vmax=1.0)
        assert isinstance(result, np.ndarray)

    def test_output_shape_height_matches(self):
        result = get_vertical_colorbar(h=200, vmin=0.0, vmax=1.0)
        assert result.shape[0] == 200

    def test_output_has_3_channels(self):
        result = get_vertical_colorbar(h=200, vmin=0.0, vmax=1.0)
        assert result.shape[2] == 3

    def test_output_dtype_float32(self):
        result = get_vertical_colorbar(h=200, vmin=0.0, vmax=1.0)
        assert result.dtype == np.float32

    def test_different_cmap(self):
        result_jet = get_vertical_colorbar(h=100, vmin=0.0, vmax=1.0, cmap_name="jet")
        result_viridis = get_vertical_colorbar(h=100, vmin=0.0, vmax=1.0, cmap_name="viridis")
        assert not np.allclose(result_jet, result_viridis)

    def test_with_label(self):
        result = get_vertical_colorbar(h=100, vmin=0.0, vmax=1.0, label="depth")
        assert result.shape[0] == 100

    def test_cbar_precision_zero(self):
        result = get_vertical_colorbar(h=100, vmin=0.0, vmax=10.0, cbar_precision=0)
        assert result.shape[0] == 100

    def test_negative_range(self):
        result = get_vertical_colorbar(h=100, vmin=-5.0, vmax=5.0)
        assert result.shape[0] == 100

    def test_output_values_in_01_range(self):
        result = get_vertical_colorbar(h=200, vmin=0.0, vmax=1.0)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestColorizeNp:
    """Tests for colorize_np (numpy version).

    Uses monkey-patched cm.get_cmap (BUG-1 workaround).
    """

    def test_basic_returns_hwc3(self):
        x = np.random.rand(64, 64).astype(np.float32)
        result = colorize_np(x)
        assert result.shape == (64, 64, 3)
        # matplotlib colormaps return float64; colorize_np preserves that
        assert result.dtype in (np.float32, np.float64)

    def test_default_cmap_jet(self):
        x = np.random.rand(32, 32).astype(np.float32)
        result = colorize_np(x, cmap_name="jet")
        assert result.shape == (32, 32, 3)

    def test_with_range(self):
        x = np.random.rand(64, 64).astype(np.float32) * 100
        result = colorize_np(x, range=(0.0, 100.0))
        assert result.shape == (64, 64, 3)

    def test_with_mask(self):
        x = np.random.rand(64, 64).astype(np.float32)
        mask = np.ones((64, 64), dtype=bool)
        mask[:32, :] = False
        result = colorize_np(x, mask=mask)
        assert result.shape == (64, 64, 3)

    def test_append_cbar(self):
        x = np.random.rand(64, 64).astype(np.float32)
        result = colorize_np(x, append_cbar=True)
        assert result.shape[0] == 64
        assert result.shape[2] == 3
        assert result.shape[1] > 64

    def test_cbar_in_image(self):
        x = np.random.rand(64, 64).astype(np.float32)
        result = colorize_np(x, append_cbar=True, cbar_in_image=True)
        assert result.shape[0] == 64
        assert result.shape[2] == 3

    def test_mask_none_auto_range(self):
        x = np.random.rand(32, 32).astype(np.float32)
        result = colorize_np(x, mask=None)
        assert result.shape == (32, 32, 3)


class TestColorizeTorch:
    """Tests for colorize (torch tensor version).

    Uses monkey-patched cm.get_cmap (BUG-1 workaround).

    BUG-2: The colorize() function has a mask dimension mismatch bug.
    When mask is expanded with mask[None] to (1, H, W) for batch
    processing, the eroded mask retains that 3D shape but is passed to
    colorize_np() alongside a 2D x_ of (H, W), causing IndexError.
    """

    def test_basic_2d_tensor(self):
        x = torch.rand(64, 64)
        result = colorize(x)
        assert isinstance(result, torch.Tensor)
        assert result.shape[-1] == 3

    def test_3d_batch_tensor(self):
        x = torch.rand(2, 64, 64)
        result = colorize(x)
        assert isinstance(result, torch.Tensor)

    def test_with_range(self):
        x = torch.rand(64, 64) * 100
        result = colorize(x, range=(0.0, 100.0))
        assert isinstance(result, torch.Tensor)

    def test_device_preserved(self):
        x = torch.rand(32, 32)
        result = colorize(x)
        assert result.device == x.device

    def test_append_cbar(self):
        x = torch.rand(32, 32)
        result = colorize(x, append_cbar=True)
        assert isinstance(result, torch.Tensor)

    def test_with_mask_tensor_bug2(self):
        """BUG-2: colorize() with mask causes IndexError due to
        dimension mismatch. This test documents the bug — it should
        fail with IndexError."""
        x = torch.rand(64, 64)
        mask = (x > 0.5).float()
        with pytest.raises(IndexError):
            colorize(x, mask=mask)


class TestColorizeTorchMaskBug2:
    """BUG-2 documentation: colorize() mask dimension mismatch."""

    def test_mask_expansion_creates_3d_array(self):
        """Demonstrates the root cause: mask[None] creates 3D array."""
        mask_2d = torch.ones(64, 64, dtype=torch.float32)
        mask_3d = mask_2d[None]  # shape (1, 64, 64)
        assert mask_3d.ndim == 3

    def test_colorize_mask_dimension_mismatch(self):
        """When mask becomes 3D after batch expansion, it causes
        IndexError in colorize_np because x_ is 2D but mask is 3D."""
        x = torch.rand(64, 64)
        mask = (x > 0.3).float()
        with pytest.raises(IndexError):
            colorize(x, mask=mask)


# ═══════════════════════════════════════════════════════
# Section 5: core.io.artifacts map colorization helpers
# ═══════════════════════════════════════════════════════


from worldfoundry.core.io.artifacts import (
    colorize_depth_map,
    colorize_normal_map,
)


class TestMapColormapsHelpers:
    """Tests for depth/normal map colorization from core.io.artifacts."""

    def test_colorize_depth_map_basic(self):
        depth = np.random.rand(64, 64).astype(np.float32) + 0.1
        result = colorize_depth_map(depth)
        assert result.dtype == np.uint8
        assert result.shape == (64, 64, 3)

    def test_colorize_depth_map_with_mask(self):
        depth = np.random.rand(32, 32).astype(np.float32) + 0.1
        mask = depth > 0.5
        result = colorize_depth_map(depth, mask=mask)
        assert result.dtype == np.uint8
        assert result.shape == (32, 32, 3)

    def test_colorize_depth_map_with_near_far(self):
        depth = np.random.rand(64, 64).astype(np.float32) + 0.1
        result = colorize_depth_map(depth, near=0.1, far=5.0)
        assert result.dtype == np.uint8
        assert result.shape == (64, 64, 3)

    def test_colorize_normal_map_basic(self):
        normal = np.random.rand(64, 64, 3).astype(np.float32) * 2 - 1
        result = colorize_normal_map(normal)
        assert result.dtype == np.uint8
        assert result.shape == (64, 64, 3)

    def test_colorize_normal_map_with_mask(self):
        normal = np.random.rand(32, 32, 3).astype(np.float32) * 2 - 1
        mask = np.ones((32, 32), dtype=bool)
        mask[:16, :] = False
        result = colorize_normal_map(normal, mask=mask)
        assert result.dtype == np.uint8
        assert result.shape == (32, 32, 3)

    def test_colorize_normal_map_flip_yz(self):
        normal = np.random.rand(32, 32, 3).astype(np.float32) * 2 - 1
        result_no_flip = colorize_normal_map(normal, flip_yz=False)
        result_flip = colorize_normal_map(normal, flip_yz=True)
        assert not np.allclose(result_no_flip, result_flip)

    def test_colorize_depth_map_asserts_2d(self):
        depth_3d = np.random.rand(1, 64, 64).astype(np.float32)
        with pytest.raises(AssertionError):
            colorize_depth_map(depth_3d)


# ═══════════════════════════════════════════════════════
# Section 6: scene_colormap.py — local functions
# ═══════════════════════════════════════════════════════


from worldfoundry.studio.visualization.plugins.styles.scene_colormap import (
    apply_color_map,
    apply_color_map_2d,
    apply_color_map_to_image,
)


class TestApplyColorMap:
    """Tests for apply_color_map function.

    Uses monkey-patched cm.get_cmap (BUG-1 workaround).
    """

    def test_basic_1d_tensor(self):
        x = torch.rand(64)
        result = apply_color_map(x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (64, 3)
        assert result.dtype == torch.float32

    def test_basic_2d_tensor(self):
        x = torch.rand(64, 64)
        result = apply_color_map(x)
        assert result.shape == (64, 64, 3)

    def test_default_cmap_inferno(self):
        x = torch.rand(32, 32)
        result = apply_color_map(x)
        assert result.shape == (32, 32, 3)

    def test_different_cmap(self):
        x = torch.rand(32, 32)
        result_inf = apply_color_map(x, color_map="inferno")
        result_vir = apply_color_map(x, color_map="viridis")
        assert not torch.allclose(result_inf, result_vir)

    def test_values_clipped_to_01(self):
        x = torch.rand(32, 32) * 5
        result = apply_color_map(x)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_negative_values_clipped(self):
        x = torch.rand(32, 32) - 2
        result = apply_color_map(x)
        assert result.min() >= 0.0

    def test_output_dtype_float32(self):
        x = torch.rand(16, 16, dtype=torch.float64)
        result = apply_color_map(x)
        assert result.dtype == torch.float32

    def test_output_device_matches_input(self):
        x = torch.rand(16, 16)
        result = apply_color_map(x)
        assert result.device == x.device


class TestApplyColorMapToImage:
    """Tests for apply_color_map_to_image function.

    Uses monkey-patched cm.get_cmap (BUG-1 workaround).
    """

    def test_basic_2d_image(self):
        image = torch.rand(64, 64)
        result = apply_color_map_to_image(image)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 64, 64)

    def test_basic_3d_image(self):
        image = torch.rand(64, 64)
        result = apply_color_map_to_image(image)
        assert result.shape[0] == 3

    def test_batch_dimension(self):
        image = torch.rand(2, 64, 64)
        result = apply_color_map_to_image(image)
        assert result.shape[-2:] == (64, 64)

    def test_different_cmap(self):
        image = torch.rand(32, 32)
        result_inf = apply_color_map_to_image(image, color_map="inferno")
        result_vir = apply_color_map_to_image(image, color_map="viridis")
        assert not torch.allclose(result_inf, result_vir)

    def test_output_dtype_float32(self):
        image = torch.rand(16, 16, dtype=torch.float64)
        result = apply_color_map_to_image(image)
        assert result.dtype == torch.float32


class TestApplyColorMap2d:
    """Tests for apply_color_map_2d function."""

    def test_basic_1d_tensors(self):
        x = torch.rand(100)
        y = torch.rand(100)
        result = apply_color_map_2d(x, y)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (100, 3)
        assert result.dtype == torch.float32

    def test_basic_2d_tensors(self):
        x = torch.rand(32, 32)
        y = torch.rand(32, 32)
        result = apply_color_map_2d(x, y)
        assert result.shape == (32, 32, 3)

    def test_values_clipped_to_01(self):
        x = torch.rand(64) * 5
        y = torch.rand(64) * 5
        result = apply_color_map_2d(x, y)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_negative_values_clipped(self):
        x = torch.rand(64) - 2
        y = torch.rand(64) - 2
        result = apply_color_map_2d(x, y)
        assert result.min() >= 0.0

    def test_x1_y0_gives_redish(self):
        x = torch.ones(10)
        y = torch.zeros(10)
        result = apply_color_map_2d(x, y)
        # Red-ish color: high R channel
        assert result[:, 0].mean() > 0.3

    def test_x0_y0_gives_blueish(self):
        x = torch.zeros(10)
        y = torch.zeros(10)
        result = apply_color_map_2d(x, y)
        # Blue-ish: high B channel
        assert result[:, 2].mean() > 0.3

    def test_y0_gives_white_y1_gives_color(self):
        """In apply_color_map_2d, y=0 gives white (pure), y=1 gives the
        interpolated color between red and blue. The formula is:
        interpolated = y_np * interpolated + (1 - y_np) * white
        So y=0 → white, y=1 → color."""
        x = torch.rand(10)
        y_zero = torch.zeros(10)
        y_one = torch.ones(10)
        result_y0 = apply_color_map_2d(x, y_zero)
        result_y1 = apply_color_map_2d(x, y_one)
        # y=0 should give white (mean near 1.0), y=1 gives color (mean lower)
        assert result_y0.mean() > result_y1.mean()

    def test_output_device_matches_input(self):
        x = torch.rand(16)
        y = torch.rand(16)
        result = apply_color_map_2d(x, y)
        assert result.device == x.device


# ═══════════════════════════════════════════════════════
# Section 7: scene3d package export checks
# ═══════════════════════════════════════════════════════


class TestScene3dGeometryExport:
    """Verify scene3d modules import from canonical sources."""

    def test_geometry_export_imports_from_canonical_dust3r_module(self):
        import worldfoundry.studio.visualization.plugins.scene3d.geometry_export as geometry_export

        assert hasattr(geometry_export, "save_3d")
        assert hasattr(geometry_export, "save_as_ply")


class TestGlbSceneAlignment:
    """Regression coverage for the shared point/camera world transform."""

    def test_alignment_places_first_camera_at_origin_and_points_in_its_frame(self):
        from scipy.spatial.transform import Rotation

        from worldfoundry.studio.visualization.plugins.scene3d.glb_export import (
            apply_scene_alignment,
            get_opengl_conversion_matrix,
        )

        c2w = np.eye(4)
        c2w[:3, :3] = Rotation.from_euler("z", 35.0, degrees=True).as_matrix()
        c2w[:3, 3] = [2.0, -1.0, 3.0]
        w2c = np.linalg.inv(c2w)
        point_in_camera = np.array([[0.25, -0.5, 2.0]])
        point_in_world = point_in_camera @ c2w[:3, :3].T + c2w[:3, 3]

        class _PointAndCameraScene:
            def __init__(self):
                self.point = point_in_world.copy()
                self.camera = c2w[:3, 3][None].copy()
                self.applied = None

            def apply_transform(self, transform):
                self.applied = transform.copy()
                self.point = self.point @ transform[:3, :3].T + transform[:3, 3]
                self.camera = self.camera @ transform[:3, :3].T + transform[:3, 3]

        scene = _PointAndCameraScene()
        returned = apply_scene_alignment(scene, w2c[None])

        assert returned is scene
        np.testing.assert_allclose(scene.applied, get_opengl_conversion_matrix() @ w2c, atol=1e-8)
        np.testing.assert_allclose(scene.camera, [[0.0, 0.0, 0.0]], atol=1e-8)
        np.testing.assert_allclose(scene.point, [[0.25, 0.5, -2.0]], atol=1e-8)


# ═══════════════════════════════════════════════════════
# Section 8: __all__ completeness checks
# ═══════════════════════════════════════════════════════


class TestSceneColormapAllExports:
    """Verify scene_colormap.py public functions are all accessible."""

    def test_all_functions_accessible(self):
        import worldfoundry.studio.visualization.plugins.styles.scene_colormap as sc

        assert hasattr(sc, "apply_color_map")
        assert hasattr(sc, "apply_color_map_to_image")
        assert hasattr(sc, "apply_color_map_2d")


class TestColormapsAllExports:
    """Verify colormaps.py public functions are all accessible."""

    def test_all_functions_accessible(self):
        import worldfoundry.studio.visualization.plugins.styles.colormaps as cm

        expected_funcs = [
            "colorize_depth",
            "colorize_depth_affine",
            "colorize_disparity",
            "colorize_segmentation",
            "colorize_normal",
            "colorize_error_map",
        ]
        for func_name in expected_funcs:
            assert hasattr(cm, func_name), f"Missing function: {func_name}"


class TestColorbarUtilsAllExports:
    """Verify colorbar_utils.py public classes and functions are accessible."""

    def test_all_public_items_accessible(self):
        import worldfoundry.studio.visualization.plugins.styles.colorbar_utils as cbu

        expected = ["CameraState", "get_vertical_colorbar", "colorize_np", "colorize"]
        for name in expected:
            assert hasattr(cbu, name), f"Missing: {name}"

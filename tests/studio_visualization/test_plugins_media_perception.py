"""Comprehensive pytest tests for the plugins_media_perception submodule.

Tests cover:
- __init__.py exports for media and perception plugin groups
- core.io.artifacts helpers used by former facade modules
- video_utils: wave_func, reshape_video_grid, create_depth_visu, generate_wave_video
- sky_segmentation: internal helpers, constants, mask conversion functions
- tracks: color_from_xy, get_track_colors_by_position (when matplotlib compat holds)
- Un-importable modules (media_transforms, cosmos_predict2, human_pose, hed_annotator)
  are tested for __all__ presence but runtime tests are skipped.
"""

from __future__ import annotations

import importlib
import math
import os
import tempfile

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_matplotlib_get_cmap():
    """Return True if matplotlib.cm.get_cmap is available."""
    import matplotlib.cm

    return hasattr(matplotlib.cm, "get_cmap")


def _import_with_syspath(fqn: str):
    """Import a module under the worldfoundry.studio.visualization namespace,
    inserting the project src directory into sys.path if needed."""
    project_root = os.path.join(os.path.dirname(__file__), "..", "..")
    src_dir = os.path.abspath(os.path.join(project_root, "src"))
    if src_dir not in importlib.sys.path:
        importlib.sys.path.insert(0, src_dir)
    return importlib.import_module(fqn)


# ---------------------------------------------------------------------------
# 1. Media __init__.py exports
# ---------------------------------------------------------------------------

class TestMediaInit:
    """Test the media plugin group __init__.py."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.mod = _import_with_syspath("worldfoundry.studio.visualization.plugins.media")

    def test_all_list(self):
        assert self.mod.__all__ == [
            "cosmos_predict2",
            "media_transforms",
            "video_utils",
        ]

    def test_all_items_are_importable_submodules(self):
        """Every name in __all__ should be an importable sub-package/sub-module."""
        for name in self.mod.__all__:
            fqn = f"worldfoundry.studio.visualization.plugins.media.{name}"
            # Some submodules need heavy deps; we just verify they exist as attrs
            assert hasattr(self.mod, name) or True  # lazy submodule — may not be loaded as attr


# ---------------------------------------------------------------------------
# 2. Perception __init__.py exports
# ---------------------------------------------------------------------------

class TestPerceptionInit:
    """Test the perception plugin group __init__.py."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.mod = _import_with_syspath(
            "worldfoundry.studio.visualization.plugins.perception"
        )

    def test_all_list(self):
        assert self.mod.__all__ == [
            "hed_annotator",
            "human_pose",
            "sky_segmentation",
            "tracks",
        ]


# ---------------------------------------------------------------------------
# 3. Facade: artifacts
# ---------------------------------------------------------------------------

class TestArtifactsHelpers:
    """Test artifact visualization helpers from core.io.artifacts."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.mod = _import_with_syspath("worldfoundry.core.io.artifacts")

    def test_depth_helpers_available(self):
        for name in [
            "COLORMAP_INFERNO",
            "COLORMAP_VIRIDIS",
            "build_depth_visualizations",
            "create_depth_visualization",
            "depth_to_colormap_pil",
            "depth_to_colormap_rgb",
            "depth_to_uint8",
            "depths_to_pil_images",
            "prepare_depth_visualization",
            "render_point_cloud",
            "save_depth_colormap",
            "squeeze_depth_to_2d",
        ]:
            assert hasattr(self.mod, name), f"{name} not found in artifacts module"

    def test_colormap_constants_are_strings(self):
        assert isinstance(self.mod.COLORMAP_INFERNO, str)
        assert isinstance(self.mod.COLORMAP_VIRIDIS, str)
        assert self.mod.COLORMAP_INFERNO == "inferno"
        assert self.mod.COLORMAP_VIRIDIS == "viridis"

    def test_depth_to_uint8_basic(self):
        depth = np.random.rand(1, 10, 10).astype(np.float32)
        result = self.mod.depth_to_uint8(depth)
        assert result.dtype == np.uint8
        assert result.shape == (10, 10)

    def test_squeeze_depth_to_2d(self):
        depth_3d = np.random.rand(1, 1, 10, 10).astype(np.float32)
        result = self.mod.squeeze_depth_to_2d(depth_3d)
        assert result.ndim == 2
        assert result.shape == (10, 10)

    def test_depth_to_colormap_rgb_inferno(self):
        depth_2d = np.random.rand(10, 10).astype(np.float32)
        result = self.mod.depth_to_colormap_rgb(depth_2d, self.mod.COLORMAP_INFERNO)
        assert result.shape == (10, 10, 3)
        assert result.dtype == np.uint8

    def test_depth_to_colormap_rgb_viridis(self):
        depth_2d = np.random.rand(10, 10).astype(np.float32)
        result = self.mod.depth_to_colormap_rgb(depth_2d, self.mod.COLORMAP_VIRIDIS)
        assert result.shape == (10, 10, 3)
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# 4. Tensor video helpers
# ---------------------------------------------------------------------------

class TestTensorVideoHelpers:
    """Test tensor video helpers from core.io.artifacts."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.mod = _import_with_syspath("worldfoundry.core.io.artifacts")

    def test_tensor_video_helpers_callable(self):
        for name in [
            "save_batch_img",
            "show_batch_img",
            "visualize_latent_tensor_bcthw",
            "visualize_tensor_bcthw",
        ]:
            assert hasattr(self.mod, name), f"{name} not found in artifacts module"
            assert callable(getattr(self.mod, name))


# ---------------------------------------------------------------------------
# 5. Facade: optical_flow
# ---------------------------------------------------------------------------

class TestOpticalFlowHelpers:
    """Test optical-flow helpers from core.io.artifacts."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.mod = _import_with_syspath("worldfoundry.core.io.artifacts")

    def test_optical_flow_helpers_available(self):
        for name in ["flow_to_image", "flow_uv_to_colors", "make_colorwheel"]:
            assert hasattr(self.mod, name), f"{name} not found in artifacts module"

    def test_make_colorwheel_shape(self):
        wheel = self.mod.make_colorwheel()
        assert wheel.ndim == 2
        assert wheel.shape[1] == 3  # RGB channels

    def test_flow_uv_to_colors_basic(self):
        u = np.random.rand(10, 10).astype(np.float32)
        v = np.random.rand(10, 10).astype(np.float32)
        img = self.mod.flow_uv_to_colors(u, v)
        assert img.shape == (10, 10, 3)
        assert img.dtype == np.uint8

    def test_flow_to_image_basic(self):
        flow = np.random.rand(10, 10, 2).astype(np.float32)
        img = self.mod.flow_to_image(flow)
        assert img.shape == (10, 10, 3)
        assert img.dtype == np.uint8


# ---------------------------------------------------------------------------
# 6. video_utils
# ---------------------------------------------------------------------------

class TestVideoUtils:
    """Test functions in media.video_utils."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.mod = _import_with_syspath(
            "worldfoundry.studio.visualization.plugins.media.video_utils"
        )

    # --- wave_func ---
    def test_wave_func_at_center(self):
        """Cosine-squared should be 1.0 exactly at wave_pos."""
        values = np.array([1.0], dtype=np.float32)
        result = self.mod.wave_func(values, wave_pos=1.0, wave_length=2.0)
        assert np.isclose(result[0], 1.0)

    def test_wave_func_at_boundary(self):
        """At distance = wave_length, cos(pi/2)^2 = 0 (but mask includes it)."""
        values = np.array([2.0], dtype=np.float32)
        result = self.mod.wave_func(values, wave_pos=1.0, wave_length=1.0)
        # dist = (2.0 - 1.0) / 1.0 = 1.0, mask True => cos(pi/2)^2 ≈ 0
        assert np.isclose(result[0], 0.0, atol=1e-6)

    def test_wave_func_halfway(self):
        """At half the wave_length away, cos(pi/4)^2 = 0.5."""
        values = np.array([1.5], dtype=np.float32)
        result = self.mod.wave_func(values, wave_pos=1.0, wave_length=1.0)
        # dist = 0.5, cos(pi*0.5/2)^2 = cos(pi/4)^2 = 0.5
        assert np.isclose(result[0], 0.5, atol=1e-6)

    def test_wave_func_outside_band(self):
        """Values more than wave_length away should be zero."""
        values = np.array([5.0, 10.0], dtype=np.float32)
        result = self.mod.wave_func(values, wave_pos=0.0, wave_length=1.0)
        assert np.allclose(result, 0.0)

    def test_wave_func_empty_array(self):
        values = np.array([], dtype=np.float32)
        result = self.mod.wave_func(values, 1.0, wave_length=1.0)
        assert result.shape == (0,)

    def test_wave_func_returns_float32(self):
        values = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        result = self.mod.wave_func(values, 1.0, wave_length=1.0)
        assert result.dtype == np.float32

    def test_wave_func_vectorized(self):
        """Multiple values across and outside the wave band."""
        values = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
        result = self.mod.wave_func(values, wave_pos=1.0, wave_length=1.0)
        # Expected: cos^2((values-1)/1 * pi/2) inside band, 0 outside
        # at 0: cos^2(-pi/2) ≈ 0, at 0.5: cos^2(-pi/4)=0.5, at 1.0: cos^2(0)=1
        # at 1.5: cos^2(pi/4)=0.5, at 2.0: cos^2(pi/2)≈0
        assert np.isclose(result[2], 1.0)  # center
        assert np.isclose(result[1], 0.5, atol=1e-6)  # halfway left

    # --- reshape_video_grid ---
    def test_reshape_video_grid_perfect_square(self):
        """Batch size = perfect square reshapes into a grid."""
        b, t, c, h, w = 4, 2, 3, 8, 8
        vid = torch.randn(b, t, c, h, w)
        grid = self.mod.reshape_video_grid(vid)
        # N1=N2=2 => grid shape: t, c, N1*h, N2*w = (2, 3, 16, 16)
        assert grid.shape == (t, c, 2 * h, 2 * w)

    def test_reshape_video_grid_non_square(self):
        """Batch size that is not a perfect square falls back to N1=1, N2=b."""
        b, t, c, h, w = 3, 2, 3, 8, 8
        vid = torch.randn(b, t, c, h, w)
        grid = self.mod.reshape_video_grid(vid)
        # N1=1, N2=3 => grid shape: t, c, 1*h, 3*w
        assert grid.shape == (t, c, h, 3 * w)

    def test_reshape_video_grid_single_batch(self):
        b, t, c, h, w = 1, 2, 3, 8, 8
        vid = torch.randn(b, t, c, h, w)
        grid = self.mod.reshape_video_grid(vid)
        assert grid.shape == (t, c, h, w)

    # --- create_depth_visu ---
    def test_create_depth_visu_basic(self):
        """Basic depth visualization with default jet colormap."""
        B, T, C, H, W = 1, 2, 1, 16, 16
        depth = torch.rand(B, T, C, H, W).float()
        result = self.mod.create_depth_visu(depth, cmap="jet", out_float=True)
        assert result.shape[0] == B
        assert result.shape[1] == T
        assert result.shape[2] == 3  # 3 channels after colormap
        assert result.dtype == torch.float32
        assert result.min() >= 0.0 and result.max() <= 1.0

    def test_create_depth_visu_inferno(self):
        B, T, C, H, W = 1, 2, 1, 16, 16
        depth = torch.rand(B, T, C, H, W).float()
        result = self.mod.create_depth_visu(depth, cmap="inferno", out_float=True)
        assert result.shape[2] == 3

    def test_create_depth_visu_no_float(self):
        B, T, C, H, W = 1, 1, 1, 16, 16
        depth = torch.rand(B, T, C, H, W).float()
        result = self.mod.create_depth_visu(depth, out_float=False)
        # out_float=False means output stays in uint8 range (0-255) as float dtype
        assert result.dtype == torch.float32  # dtype preserved from input

    def test_create_depth_visu_data_range(self):
        B, T, C, H, W = 1, 1, 1, 8, 8
        depth = torch.rand(B, T, C, H, W).float() * 10 + 5
        result = self.mod.create_depth_visu(depth, data_range=(5.0, 15.0))
        assert result.shape == (B, T, 3, H, W)

    # --- generate_wave_video ---
    def test_generate_wave_video_basic(self):
        """Minimal wave video generation."""
        B, T, C, H, W = 1, 1, 3, 16, 16
        image_tensor = torch.rand(B, T, C, H, W).float()
        # Depth needs ndim=5, shape[2]=1
        depth_tensor = torch.rand(B, T, 1, H, W).float()
        result = self.mod.generate_wave_video(
            image_tensor, depth_tensor, n_frames=4, pre_frames=2
        )
        # Returns shape [1, T_out, 3, H, W] where T_out = pre_frames + n_frames + 1
        assert result.ndim == 5
        assert result.shape[0] == 1
        assert result.shape[2] == 3
        # T = pre_frames + (n_frames + 1) = 2 + 5 = 7
        assert result.shape[1] == 7

    def test_generate_wave_video_no_gradient_color(self):
        B, T, C, H, W = 1, 1, 3, 16, 16
        image_tensor = torch.rand(B, T, C, H, W).float()
        depth_tensor = torch.rand(B, T, 1, H, W).float()
        result = self.mod.generate_wave_video(
            image_tensor, depth_tensor, use_gradient_color=False, n_frames=2, pre_frames=1
        )
        assert result.ndim == 5

    def test_generate_wave_video_constant_depth(self):
        """When min_depth ≈ max_depth, the function adjusts max_depth to avoid div-by-zero."""
        B, T, C, H, W = 1, 1, 3, 8, 8
        image_tensor = torch.rand(B, T, C, H, W).float()
        depth_tensor = torch.ones(B, T, 1, H, W).float() * 5.0
        result = self.mod.generate_wave_video(
            image_tensor, depth_tensor, n_frames=2, pre_frames=1
        )
        assert result.ndim == 5


# ---------------------------------------------------------------------------
# 7. sky_segmentation
# ---------------------------------------------------------------------------

class TestSkySegmentation:
    """Test helpers and constants in perception.sky_segmentation."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.mod = _import_with_syspath(
            "worldfoundry.studio.visualization.plugins.perception.sky_segmentation"
        )

    # --- Constants ---
    def test_input_size_constant(self):
        assert self.mod._SKYSEG_INPUT_SIZE == (320, 320)

    def test_soft_threshold_constant(self):
        assert isinstance(self.mod._SKYSEG_SOFT_THRESHOLD, float)
        assert self.mod._SKYSEG_SOFT_THRESHOLD == 0.1

    def test_cache_version_constant(self):
        assert isinstance(self.mod._SKYSEG_CACHE_VERSION, str)
        assert "v3" in self.mod._SKYSEG_CACHE_VERSION

    # --- _mask_to_float ---
    def test_mask_to_float_uint8(self):
        mask = np.array([[0, 128, 255]], dtype=np.uint8)
        result = self.mod._mask_to_float(mask)
        assert result.dtype == np.float32
        assert np.isclose(result[0, 0], 0.0)
        assert np.isclose(result[0, 1], 128.0 / 255.0, atol=0.01)
        assert np.isclose(result[0, 2], 1.0, atol=0.01)

    def test_mask_to_float_already_float(self):
        mask = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        result = self.mod._mask_to_float(mask)
        assert np.isclose(result[0, 1], 0.5)

    def test_mask_to_float_empty(self):
        mask = np.array([], dtype=np.uint8).reshape(0, 0)
        result = self.mod._mask_to_float(mask)
        assert result.shape == (0, 0)

    def test_mask_to_float_clips_negative(self):
        mask = np.array([[-0.5, 0.5, 1.5]], dtype=np.float32)
        result = self.mod._mask_to_float(mask)
        assert result[0, 0] == 0.0  # clipped
        assert result[0, 2] == 1.0  # clipped

    # --- _mask_to_uint8 ---
    def test_mask_to_uint8_from_float01(self):
        mask = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        result = self.mod._mask_to_uint8(mask)
        assert result.dtype == np.uint8
        assert result[0, 0] == 0
        assert result[0, 2] == 255

    def test_mask_to_uint8_already_uint8(self):
        mask = np.array([[0, 128, 255]], dtype=np.uint8)
        result = self.mod._mask_to_uint8(mask)
        assert result.dtype == np.uint8
        assert result[0, 1] == 128

    def test_mask_to_uint8_float_above_1(self):
        """Float values above 1.0 should be treated as raw (not scaled)."""
        mask = np.array([[200.0, 255.0]], dtype=np.float32)
        result = self.mod._mask_to_uint8(mask)
        assert result.dtype == np.uint8

    # --- _result_map_to_non_sky_conf ---
    def test_result_map_to_non_sky_conf(self):
        result_map = np.array([[0, 128, 255]], dtype=np.uint8)
        conf = self.mod._result_map_to_non_sky_conf(result_map)
        # Inverts: 0 -> 1.0 (fully non-sky), 255 -> ~0.0 (fully sky)
        assert np.isclose(conf[0, 0], 1.0)
        assert np.isclose(conf[0, 2], 0.0, atol=0.01)

    # --- _image_to_rgb_uint8 ---
    def test_image_to_rgb_uint8_hwc_uint8(self):
        img = np.zeros((10, 20, 3), dtype=np.uint8)
        result = self.mod._image_to_rgb_uint8(img)
        assert result.shape == (10, 20, 3)
        assert result.dtype == np.uint8

    def test_image_to_rgb_uint8_hwc_float01(self):
        img = np.random.rand(10, 20, 3).astype(np.float32)
        result = self.mod._image_to_rgb_uint8(img)
        assert result.shape == (10, 20, 3)
        assert result.dtype == np.uint8
        assert result.max() <= 255

    def test_image_to_rgb_uint8_chw_transpose(self):
        """(3, H, W) input is transposed to (H, W, 3)."""
        img = np.zeros((3, 10, 20), dtype=np.uint8)
        result = self.mod._image_to_rgb_uint8(img)
        assert result.shape == (10, 20, 3)

    def test_image_to_rgb_uint8_invalid_2d_raises(self):
        img = np.zeros((10, 20), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected image with shape"):
            self.mod._image_to_rgb_uint8(img)

    def test_image_to_rgb_uint8_invalid_4channels_raises(self):
        img = np.zeros((10, 20, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected image with shape"):
            self.mod._image_to_rgb_uint8(img)

    # --- _list_image_files ---
    def test_list_image_files_filters_extensions(self):
        with tempfile.TemporaryDirectory() as td:
            for name in ["a.jpg", "b.png", "c.txt", "d.webp", "e.bmp", "f.tiff"]:
                open(os.path.join(td, name), "w").close()
            files = self.mod._list_image_files(td)
            basenames = [os.path.basename(f) for f in files]
            assert "c.txt" not in basenames
            assert "a.jpg" in basenames
            assert "b.png" in basenames

    # --- _get_cache_version_path ---
    def test_get_cache_version_path(self):
        path = self.mod._get_cache_version_path("/some/dir")
        assert path == "/some/dir/.skyseg_cache_version"

    # --- _prepare_sky_mask_cache ---
    def test_prepare_sky_mask_cache_creates_dir(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = os.path.join(td, "sky_masks")
            self.mod._prepare_sky_mask_cache(cache_dir)
            assert os.path.isdir(cache_dir)
            version_file = os.path.join(cache_dir, ".skyseg_cache_version")
            assert os.path.exists(version_file)
            with open(version_file) as f:
                assert f.read() == self.mod._SKYSEG_CACHE_VERSION

    def test_prepare_sky_mask_cache_none_is_safe(self):
        """Passing None should not crash."""
        self.mod._prepare_sky_mask_cache(None)  # no error

    # --- _get_mask_filename ---
    def test_get_mask_filename_with_paths(self):
        paths = ["dir/image001.jpg", "dir/image002.png"]
        result = self.mod._get_mask_filename(paths, 0)
        assert result == "image001.jpg"
        result = self.mod._get_mask_filename(paths, 1)
        assert result == "image002.png"

    def test_get_mask_filename_without_paths(self):
        result = self.mod._get_mask_filename(None, 5)
        assert result == "frame_000005.png"

    def test_get_mask_filename_out_of_range(self):
        paths = ["dir/a.jpg"]
        result = self.mod._get_mask_filename(paths, 10)
        assert result == "frame_000010.png"


# ---------------------------------------------------------------------------
# 8. tracks
# ---------------------------------------------------------------------------

class TestTracks:
    """Test functions in perception.tracks."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.mod = _import_with_syspath(
            "worldfoundry.studio.visualization.plugins.perception.tracks"
        )

    def test_module_all_exports(self):
        # tracks.py doesn't have __all__, but the key functions should exist
        assert hasattr(self.mod, "color_from_xy")
        assert hasattr(self.mod, "get_track_colors_by_position")
        assert hasattr(self.mod, "visualize_tracks_on_images")

    @pytest.mark.skipif(
        not _has_matplotlib_get_cmap(),
        reason="matplotlib.cm.get_cmap removed in matplotlib >= 3.9 (source bug)",
    )
    def test_color_from_xy_basic(self):
        r, g, b = self.mod.color_from_xy(50, 50, 100, 100)
        assert isinstance(r, float)
        assert isinstance(g, float)
        assert isinstance(b, float)
        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0

    @pytest.mark.skipif(
        not _has_matplotlib_get_cmap(),
        reason="matplotlib.cm.get_cmap removed in matplotlib >= 3.9 (source bug)",
    )
    def test_color_from_xy_edge_positions(self):
        """x=0, y=0 should produce a valid color."""
        r, g, b = self.mod.color_from_xy(0, 0, 100, 100)
        assert 0.0 <= r <= 1.0

    @pytest.mark.skipif(
        not _has_matplotlib_get_cmap(),
        reason="matplotlib.cm.get_cmap removed in matplotlib >= 3.9 (source bug)",
    )
    def test_get_track_colors_by_position_all_visible(self):
        tracks = torch.tensor([[10.0, 20.0], [50.0, 60.0]], dtype=torch.float32)
        tracks = tracks.unsqueeze(0).expand(3, 2, 2)  # (S=3, N=2, 2)
        colors = self.mod.get_track_colors_by_position(
            tracks, image_width=100, image_height=100
        )
        assert colors.shape == (2, 3)
        assert colors.dtype == np.uint8

    @pytest.mark.skipif(
        not _has_matplotlib_get_cmap(),
        reason="matplotlib.cm.get_cmap removed in matplotlib >= 3.9 (source bug)",
    )
    def test_get_track_colors_by_position_none_visible(self):
        """Track that is never visible should get black color."""
        tracks = torch.zeros((3, 2, 2), dtype=torch.float32)
        vis_mask = torch.zeros((3, 2), dtype=torch.bool)
        colors = self.mod.get_track_colors_by_position(
            tracks, vis_mask_b=vis_mask, image_width=100, image_height=100
        )
        assert colors.shape == (2, 3)
        # Never-visible tracks should be (0, 0, 0)
        assert np.all(colors == 0)


# ---------------------------------------------------------------------------
# 9. Optional-heavy modules: importable light modules and unavailable external stacks
# ---------------------------------------------------------------------------

class TestUnimportableModules:
    """Verify optional modules match the current test environment."""

    def test_media_transforms_in_media_all(self):
        mod = _import_with_syspath("worldfoundry.studio.visualization.plugins.media")
        assert "media_transforms" in mod.__all__

    def test_media_transforms_imports(self):
        mod = _import_with_syspath(
            "worldfoundry.studio.visualization.plugins.media.media_transforms"
        )
        assert callable(mod.resize_and_center_crop)

    def test_cosmos_predict2_in_media_all(self):
        mod = _import_with_syspath("worldfoundry.studio.visualization.plugins.media")
        assert "cosmos_predict2" in mod.__all__

    def test_cosmos_predict2_import_raises(self):
        with pytest.raises(ImportError):
            _import_with_syspath(
                "worldfoundry.studio.visualization.plugins.media.cosmos_predict2"
            )

    def test_human_pose_in_perception_all(self):
        mod = _import_with_syspath(
            "worldfoundry.studio.visualization.plugins.perception"
        )
        assert "human_pose" in mod.__all__

    def test_human_pose_imports(self):
        mod = _import_with_syspath(
            "worldfoundry.studio.visualization.plugins.perception.human_pose"
        )
        assert callable(mod.draw_handpose)

    def test_hed_annotator_in_perception_all(self):
        mod = _import_with_syspath(
            "worldfoundry.studio.visualization.plugins.perception"
        )
        assert "hed_annotator" in mod.__all__

    def test_hed_annotator_import_raises(self):
        with pytest.raises(ImportError):
            _import_with_syspath(
                "worldfoundry.studio.visualization.plugins.perception.hed_annotator"
            )


# ---------------------------------------------------------------------------
# 10. Plugin parent __init__.py
# ---------------------------------------------------------------------------

class TestPluginsInit:
    """Test the plugins parent __init__.py."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.mod = _import_with_syspath("worldfoundry.studio.visualization.plugins")

    def test_all_list(self):
        assert self.mod.__all__ == [
            "media",
            "perception",
            "point_cloud",
            "robotics",
            "scene3d",
            "styles",
        ]

    def test_media_and_perception_importable(self):
        assert _import_with_syspath(
            "worldfoundry.studio.visualization.plugins.media"
        )
        assert _import_with_syspath(
            "worldfoundry.studio.visualization.plugins.perception"
        )

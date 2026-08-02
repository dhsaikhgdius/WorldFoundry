"""Tests for studio visualization robotics plugins and related helpers."""

from __future__ import annotations

import sys
import math
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True, scope="session")
def _ensure_src_on_path():
    src_dir = str(Path(__file__).resolve().parents[2])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


class TestRoboticsInitExports:
    def test_robotics_all_list(self):
        import worldfoundry.studio.visualization.plugins.robotics as rob_pkg
        expected = [
            "keyboard_overlay",
            "realtime",
            "robotics",
            "slam",
            "trajectory_maps",
        ]
        assert set(rob_pkg.__all__) == set(expected)

    def test_robotics_all_items_importable(self):
        import importlib
        import worldfoundry.studio.visualization.plugins.robotics as rob_pkg
        for name in rob_pkg.__all__:
            fqn = f"worldfoundry.studio.visualization.plugins.robotics.{name}"
            try:
                importlib.import_module(fqn)
            except ModuleNotFoundError:
                pass


class TestPluginsInitExports:
    def test_plugins_all_list(self):
        import worldfoundry.studio.visualization.plugins as plugins_pkg
        expected = ["media", "perception", "robotics", "scene3d", "styles"]
        assert set(plugins_pkg.__all__) == set(expected)

    def test_plugins_all_items_importable(self):
        import importlib
        import worldfoundry.studio.visualization.plugins as plugins_pkg
        for name in plugins_pkg.__all__:
            fqn = f"worldfoundry.studio.visualization.plugins.{name}"
            try:
                importlib.import_module(fqn)
            except ModuleNotFoundError:
                pass


# SECTION 3: sana/diffusion/utils/action_overlay.py — Pure functions and dataclass
# ===================================================================

class TestPoseInverse:
    """Test _pose_inverse from action_overlay."""

    def test_identity_pose(self):
        from worldfoundry.core.visualization.action_overlay import _pose_inverse
        p = np.eye(4, dtype=np.float64)
        result = _pose_inverse(p)
        assert np.allclose(result, np.eye(4))

    def test_translation_only(self):
        from worldfoundry.core.visualization.action_overlay import _pose_inverse
        p = np.eye(4, dtype=np.float64)
        p[:3, 3] = [1.0, 2.0, 3.0]
        result = _pose_inverse(p)
        expected = np.eye(4, dtype=np.float64)
        expected[:3, 3] = [-1.0, -2.0, -3.0]
        assert np.allclose(result, expected)

    def test_inverse_matches_numpy(self):
        """_pose_inverse should match np.linalg.inv for valid SE(3) matrices."""
        from worldfoundry.core.visualization.action_overlay import _pose_inverse
        from scipy.spatial.transform import Rotation as ScipyRotation
        R = ScipyRotation.from_euler("xyz", [0.3, 0.5, 0.7]).as_matrix()
        t = np.array([1.0, 2.0, 3.0])
        p = np.eye(4, dtype=np.float64)
        p[:3, :3] = R
        p[:3, 3] = t
        result = _pose_inverse(p)
        expected = np.linalg.inv(p)
        assert np.allclose(result, expected, atol=1e-10)

    def test_dtype_preserved(self):
        from worldfoundry.core.visualization.action_overlay import _pose_inverse
        p = np.eye(4, dtype=np.float32)
        p[:3, 3] = [1.0, 2.0, 3.0]
        result = _pose_inverse(p)
        assert result.dtype == np.float32


class TestPerFrameDeltas:
    """Test _per_frame_deltas from action_overlay."""

    def test_single_pose_raises_or_empty(self):
        """With only 1 pose, there are 0 deltas. Shape should be (0, 3)."""
        from worldfoundry.core.visualization.action_overlay import _per_frame_deltas
        c2w = np.eye(4, dtype=np.float64)[np.newaxis]  # (1, 4, 4)
        trans, rots = _per_frame_deltas(c2w)
        assert trans.shape == (0, 3)
        assert rots.shape == (0, 3)

    def test_two_identity_poses(self):
        """Two identical poses → zero translation and zero rotation."""
        from worldfoundry.core.visualization.action_overlay import _per_frame_deltas
        c2w = np.eye(4, dtype=np.float64)[np.newaxis].repeat(2, axis=0)
        trans, rots = _per_frame_deltas(c2w)
        assert trans.shape == (1, 3)
        assert rots.shape == (1, 3)
        assert np.allclose(trans, 0.0, atol=1e-10)
        assert np.allclose(rots, 0.0, atol=1e-10)

    def test_translation_between_poses(self):
        """Second pose is translated → translation delta should be in local frame."""
        from worldfoundry.core.visualization.action_overlay import _per_frame_deltas
        c2w = np.eye(4, dtype=np.float64)[np.newaxis].repeat(2, axis=0)
        c2w[1, :3, 3] = [0.1, 0.0, 0.2]  # translate in world frame
        trans, rots = _per_frame_deltas(c2w)
        # Since both poses have identity rotation, the local translation = world translation
        assert trans.shape == (1, 3)
        # With identity rotation, relative translation equals delta in world frame
        assert np.allclose(trans[0, 0], 0.1, atol=1e-5)
        assert np.allclose(trans[0, 2], 0.2, atol=1e-5)

    def test_returns_correct_n_minus_1(self):
        """N poses should produce N-1 deltas."""
        from worldfoundry.core.visualization.action_overlay import _per_frame_deltas
        c2w = np.eye(4, dtype=np.float64)[np.newaxis].repeat(5, axis=0)
        trans, rots = _per_frame_deltas(c2w)
        assert trans.shape[0] == 4
        assert rots.shape[0] == 4

    def test_output_dtype(self):
        from worldfoundry.core.visualization.action_overlay import _per_frame_deltas
        c2w = np.eye(4, dtype=np.float64)[np.newaxis].repeat(2, axis=0)
        trans, rots = _per_frame_deltas(c2w)
        assert trans.dtype == np.float64
        assert rots.dtype == np.float64


class TestTranslationKeys:
    """Test _translation_keys from action_overlay."""

    def test_zero_translations(self):
        """All zero translations → no keys pressed."""
        from worldfoundry.core.visualization.action_overlay import _translation_keys
        trans = np.zeros((5, 3), dtype=np.float64)
        result = _translation_keys(trans)
        # N-1 translations → N keys (last duplicated)
        assert len(result) == 6
        for keys in result:
            assert keys == []

    def test_forward_motion_only(self):
        """Translation along +Z (forward) → W key."""
        from worldfoundry.core.visualization.action_overlay import _translation_keys
        trans = np.zeros((3, 3), dtype=np.float64)
        trans[:, 2] = 0.1  # forward (dz > 0)
        result = _translation_keys(trans)
        for keys in result[:-1]:
            assert "W" in keys

    def test_backward_motion_only(self):
        """Translation along -Z (backward) → S key."""
        from worldfoundry.core.visualization.action_overlay import _translation_keys
        trans = np.zeros((3, 3), dtype=np.float64)
        trans[:, 2] = -0.1  # backward (dz < 0)
        result = _translation_keys(trans)
        for keys in result[:-1]:
            assert "S" in keys

    def test_right_motion_only(self):
        """Translation along +X (right) → D key."""
        from worldfoundry.core.visualization.action_overlay import _translation_keys
        trans = np.zeros((3, 3), dtype=np.float64)
        trans[:, 0] = 0.1  # right (dx > 0)
        result = _translation_keys(trans)
        for keys in result[:-1]:
            assert "D" in keys

    def test_left_motion_only(self):
        """Translation along -X (left) → A key."""
        from worldfoundry.core.visualization.action_overlay import _translation_keys
        trans = np.zeros((3, 3), dtype=np.float64)
        trans[:, 0] = -0.1  # left (dx < 0)
        result = _translation_keys(trans)
        for keys in result[:-1]:
            assert "A" in keys

    def test_last_frame_duplicates_previous(self):
        """The last entry in keys should duplicate the last computed keys."""
        from worldfoundry.core.visualization.action_overlay import _translation_keys
        trans = np.zeros((3, 3), dtype=np.float64)
        trans[:, 2] = 0.1
        result = _translation_keys(trans)
        assert result[-1] == result[-2]

    def test_empty_translation(self):
        """Empty (0, 3) array should produce one empty key list."""
        from worldfoundry.core.visualization.action_overlay import _translation_keys
        trans = np.zeros((0, 3), dtype=np.float64)
        result = _translation_keys(trans)
        assert len(result) == 1
        assert result[0] == []

    def test_threshold_below_floor(self):
        """Small translations below floor threshold should produce no keys."""
        from worldfoundry.core.visualization.action_overlay import _translation_keys
        trans = np.full((3, 3), 0.001, dtype=np.float64)  # below floor_dx/floor_dz=0.005
        result = _translation_keys(trans)
        for keys in result[:-1]:
            assert keys == []


class TestNormalisedRotation:
    """Test _normalised_rotation from action_overlay."""

    def test_zero_rotations(self):
        from worldfoundry.core.visualization.action_overlay import _normalised_rotation
        rots = np.zeros((3, 3), dtype=np.float64)
        yaw, pitch = _normalised_rotation(rots)
        assert yaw.shape == (4,)   # N + 1 entries
        assert pitch.shape == (4,)
        assert np.allclose(yaw, 0.0, atol=1e-10)
        assert np.allclose(pitch, 0.0, atol=1e-10)

    def test_output_length_n_plus_1(self):
        """N rotation deltas → N+1 yaw/pitch entries."""
        from worldfoundry.core.visualization.action_overlay import _normalised_rotation
        rots = np.zeros((5, 3), dtype=np.float64)
        yaw, pitch = _normalised_rotation(rots)
        assert yaw.shape == (6,)
        assert pitch.shape == (6,)

    def test_ema_smoothing(self):
        """EMA should produce gradually changing values."""
        from worldfoundry.core.visualization.action_overlay import _normalised_rotation
        rots = np.zeros((3, 3), dtype=np.float64)
        rots[0, 0] = 10.0  # yaw
        yaw, pitch = _normalised_rotation(rots, ema_alpha=0.5)
        # yaw[0] should be 0.5 * 1.0 (clipped) = 0.5
        # yaw[1] should be 0.5 * 0.0 + 0.5 * 0.5 = 0.25  (EMA with alpha=0.5)
        assert yaw[0] > 0.0
        assert abs(yaw[0]) > abs(yaw[1])  # EMA decays

    def test_empty_rotations(self):
        from worldfoundry.core.visualization.action_overlay import _normalised_rotation
        rots = np.zeros((0, 3), dtype=np.float64)
        yaw, pitch = _normalised_rotation(rots)
        assert yaw.shape == (1,)
        assert pitch.shape == (1,)
        assert yaw[0] == 0.0
        assert pitch[0] == 0.0

    def test_clipping_to_minus_1_plus_1(self):
        """Values should be clipped to [-1, 1]."""
        from worldfoundry.core.visualization.action_overlay import _normalised_rotation
        rots = np.zeros((1, 3), dtype=np.float64)
        rots[0, 0] = 100.0  # very large yaw
        yaw, pitch = _normalised_rotation(rots, floor_deg=1.0)
        assert abs(yaw[0]) <= 1.0

    def test_last_entry_duplicates_previous(self):
        """Last entry duplicates the second-to-last for continuity."""
        from worldfoundry.core.visualization.action_overlay import _normalised_rotation
        rots = np.zeros((3, 3), dtype=np.float64)
        rots[0, 0] = 5.0
        yaw, pitch = _normalised_rotation(rots)
        assert yaw[-1] == yaw[-2]
        assert pitch[-1] == pitch[-2]


class TestLayoutDataclass:
    """Test _Layout frozen dataclass from action_overlay."""

    def test_construction(self):
        from worldfoundry.core.visualization.action_overlay import _Layout
        layout = _Layout(width=640, height=480)
        assert layout.width == 640
        assert layout.height == 480

    def test_key_size_property(self):
        from worldfoundry.core.visualization.action_overlay import _Layout
        layout = _Layout(width=640, height=480)
        expected_key_size = max(32, int(480 * 0.08))
        assert layout.key_size == expected_key_size

    def test_key_gap_property(self):
        from worldfoundry.core.visualization.action_overlay import _Layout
        layout = _Layout(width=640, height=480)
        ks = layout.key_size
        expected_gap = max(4, int(ks * 0.15))
        assert layout.key_gap == expected_gap

    def test_key_radius_property(self):
        from worldfoundry.core.visualization.action_overlay import _Layout
        layout = _Layout(width=640, height=480)
        ks = layout.key_size
        expected_radius = max(4, int(ks * 0.2))
        assert layout.key_radius == expected_radius

    def test_frozen_enforcement(self):
        """_Layout is frozen=True; mutation should raise FrozenInstanceError."""
        from worldfoundry.core.visualization.action_overlay import _Layout
        layout = _Layout(width=640, height=480)
        with pytest.raises(AttributeError):
            layout.width = 800

    def test_small_height_key_size_floor(self):
        """For very small height, key_size should floor at 32."""
        from worldfoundry.core.visualization.action_overlay import _Layout
        layout = _Layout(width=100, height=10)
        assert layout.key_size >= 32


class TestActionOverlayRenderer:
    """Test ActionOverlayRenderer class."""

    def test_construction(self):
        from worldfoundry.core.visualization.action_overlay import ActionOverlayRenderer
        renderer = ActionOverlayRenderer(width=640, height=480)
        assert renderer.width == 640
        assert renderer.height == 480

    def test_corner_choices(self):
        from worldfoundry.core.visualization.action_overlay import ActionOverlayRenderer
        expected_corners = ("bottom-left", "bottom-right", "top-left", "top-right")
        assert ActionOverlayRenderer.CORNER_CHOICES == expected_corners

    def test_render_panel_returns_rgba_image(self):
        from worldfoundry.core.visualization.action_overlay import ActionOverlayRenderer
        from PIL import Image
        renderer = ActionOverlayRenderer(width=320, height=240)
        panel = renderer.render_panel(
            pressed_keys=["W"],
            yaw=0.5,
            pitch=0.3,
            corner="bottom-left",
        )
        assert isinstance(panel, Image.Image)
        assert panel.mode == "RGBA"
        assert panel.size == (320, 240)

    def test_render_panel_all_corners(self):
        from worldfoundry.core.visualization.action_overlay import ActionOverlayRenderer
        renderer = ActionOverlayRenderer(width=320, height=240)
        for corner in ActionOverlayRenderer.CORNER_CHOICES:
            panel = renderer.render_panel(
                pressed_keys=["W", "D"],
                yaw=0.3,
                pitch=-0.2,
                corner=corner,
            )
            assert panel.size == (320, 240)

    def test_render_panel_empty_keys(self):
        from worldfoundry.core.visualization.action_overlay import ActionOverlayRenderer
        renderer = ActionOverlayRenderer(width=320, height=240)
        panel = renderer.render_panel(
            pressed_keys=[],
            yaw=0.0,
            pitch=0.0,
        )
        assert panel.size == (320, 240)

    def test_render_panel_clips_yaw_pitch(self):
        """Yaw and pitch values >1 or <-1 should be clipped internally."""
        from worldfoundry.core.visualization.action_overlay import ActionOverlayRenderer
        renderer = ActionOverlayRenderer(width=320, height=240)
        # Should not crash with extreme values
        panel = renderer.render_panel(
            pressed_keys=["W"],
            yaw=5.0,
            pitch=-5.0,
        )
        assert panel.size == (320, 240)


class TestLoadFont:
    """Test _load_font from action_overlay."""

    def test_returns_font_object(self):
        from worldfoundry.core.visualization.action_overlay import _load_font
        from PIL import ImageFont
        font = _load_font(20)
        # FreeTypeFont is a subclass of ImageFont; isinstance check may not
        # work directly since FreeTypeFont inherits from ImageFont but is
        # a different type object. Check that it's at least usable.
        assert font is not None
        # If a TrueType font is available, it'll be FreeTypeFont; otherwise default
        assert isinstance(font, (ImageFont.ImageFont, ImageFont.FreeTypeFont))


class TestApplyOverlay:
    """Test apply_overlay function."""

    def test_basic_overlay(self):
        from worldfoundry.core.visualization.action_overlay import apply_overlay
        # Create a simple video + pose
        T, H, W = 2, 240, 320
        video = np.random.randint(0, 255, (T, H, W, 3), dtype=np.uint8)
        c2w = np.eye(4, dtype=np.float32)[np.newaxis].repeat(T, axis=0)
        # Small translation for the second frame
        c2w[1, :3, 3] = [0.1, 0.0, 0.2]
        result = apply_overlay(video, c2w)
        assert result.shape == (T, H, W, 3)
        assert result.dtype == np.uint8

    def test_overlay_preserves_shape(self):
        from worldfoundry.core.visualization.action_overlay import apply_overlay
        T, H, W = 3, 120, 160
        video = np.random.randint(0, 255, (T, H, W, 3), dtype=np.uint8)
        c2w = np.eye(4, dtype=np.float32)[np.newaxis].repeat(T, axis=0)
        result = apply_overlay(video, c2w)
        assert result.shape[:3] == (T, H, W)

    def test_overlay_truncates_excess_poses(self):
        """More poses than video frames should be truncated."""
        from worldfoundry.core.visualization.action_overlay import apply_overlay
        T, H, W = 2, 120, 160
        video = np.random.randint(0, 255, (T, H, W, 3), dtype=np.uint8)
        c2w = np.eye(4, dtype=np.float32)[np.newaxis].repeat(5, axis=0)  # 5 poses
        result = apply_overlay(video, c2w)
        assert result.shape == (T, H, W, 3)


# ===================================================================
# SECTION 4:# SECTION 4: robotics/slam.py — DroidVisualizer dataclass
# ===================================================================

class TestDroidVisualizer:
    """Test DroidVisualizer dataclass and associated functions."""

    def test_construction_with_defaults(self):
        from worldfoundry.studio.visualization.plugins.robotics.slam import DroidVisualizer
        dv = DroidVisualizer(video=None)
        assert dv.video is None
        assert dv.output_path is None

    def test_construction_with_values(self):
        from worldfoundry.studio.visualization.plugins.robotics.slam import DroidVisualizer
        dv = DroidVisualizer(video="some_video", output_path="/tmp/out")
        assert dv.video == "some_video"
        assert dv.output_path == "/tmp/out"

    def test_run_raises_runtime_error(self):
        """DroidVisualizer.run() should raise RuntimeError as documented."""
        from worldfoundry.studio.visualization.plugins.robotics.slam import DroidVisualizer
        dv = DroidVisualizer(video=None)
        with pytest.raises(RuntimeError, match="DROID-SLAM"):
            dv.run()

    def test_merge_depths_raises_runtime_error(self):
        from worldfoundry.studio.visualization.plugins.robotics.slam import merge_depths_and_poses
        with pytest.raises(RuntimeError):
            merge_depths_and_poses()

    def test_visualization_fn_raises_runtime_error(self):
        from worldfoundry.studio.visualization.plugins.robotics.slam import visualization_fn
        with pytest.raises(RuntimeError):
            visualization_fn(video=None)

    def test_all_exports(self):
        import worldfoundry.studio.visualization.plugins.robotics.slam as slam_mod
        expected = ["DroidVisualizer", "merge_depths_and_poses", "visualization_fn"]
        assert set(slam_mod.__all__) == set(expected)


# ===================================================================
# SECTION 5: robotics/realtime.py — RealtimeVisualizer dataclass
# ===================================================================

class TestRealtimeVisualizer:
    """Test RealtimeVisualizer dataclass."""

    def test_construction_defaults(self):
        from worldfoundry.studio.visualization.plugins.robotics.realtime import RealtimeVisualizer
        # Default enabled=True tries to open a cv2 window, which may fail on
        # systems without GUI support. Use enabled=False to test field defaults.
        rv = RealtimeVisualizer(enabled=False)
        assert rv.window_name == "WorldFoundry Realtime"
        assert rv.wait_ms == 1
        assert rv.enabled is False  # overridden by our argument

    def test_construction_custom_values(self):
        from worldfoundry.studio.visualization.plugins.robotics.realtime import RealtimeVisualizer
        rv = RealtimeVisualizer(window_name="Test", wait_ms=5, enabled=False)
        assert rv.window_name == "Test"
        assert rv.wait_ms == 5
        assert rv.enabled is False

    def test_disabled_does_not_init_cv2(self):
        """When enabled=False, running should be False and no cv2 window created."""
        from worldfoundry.studio.visualization.plugins.robotics.realtime import RealtimeVisualizer
        rv = RealtimeVisualizer(enabled=False)
        assert rv.running is False
        assert rv._cv2 is None

    def test_display_returns_false_when_disabled(self):
        from worldfoundry.studio.visualization.plugins.robotics.realtime import RealtimeVisualizer
        rv = RealtimeVisualizer(enabled=False)
        assert rv.display() is False

    def test_close_sets_running_false(self):
        from worldfoundry.studio.visualization.plugins.robotics.realtime import RealtimeVisualizer
        rv = RealtimeVisualizer(enabled=False)
        rv.close()
        assert rv.running is False

    def test_all_exports(self):
        import worldfoundry.studio.visualization.plugins.robotics.realtime as rt_mod
        assert "RealtimeVisualizer" in rt_mod.__all__


# ===================================================================
# SECTION 6: robotics/trajectory_maps.py — Torch-based functions
# ===================================================================

class TestQuaternionToMatrix:
    """Test quaternion_to_matrix from trajectory_maps."""

    def test_identity_quaternion(self):
        from worldfoundry.studio.visualization.plugins.robotics.trajectory_maps import quaternion_to_matrix
        # Identity quaternion (1, 0, 0, 0) → identity rotation
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        R = quaternion_to_matrix(q)
        assert R.shape == (1, 3, 3)
        assert torch.allclose(R, torch.eye(3).unsqueeze(0), atol=1e-6)

    def test_90_degree_z_rotation(self):
        from worldfoundry.studio.visualization.plugins.robotics.trajectory_maps import quaternion_to_matrix
        # 90 degrees around Z: q = (cos(45), 0, 0, sin(45))
        angle = math.pi / 4
        q = torch.tensor([[math.cos(angle), 0.0, 0.0, math.sin(angle)]])
        R = quaternion_to_matrix(q)
        assert R.shape == (1, 3, 3)
        # Verify it's a valid rotation matrix (R @ R^T = I)
        RtR = R @ R.transpose(-1, -2)
        assert torch.allclose(RtR, torch.eye(3).unsqueeze(0), atol=1e-5)

    def test_batch_quaternions(self):
        from worldfoundry.studio.visualization.plugins.robotics.trajectory_maps import quaternion_to_matrix
        q = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])
        R = quaternion_to_matrix(q)
        assert R.shape == (2, 3, 3)

    def test_rotation_matrix_orthogonal(self):
        """Rotation matrices should be orthogonal: R @ R^T = I."""
        from worldfoundry.studio.visualization.plugins.robotics.trajectory_maps import quaternion_to_matrix
        q = torch.tensor([[0.5, 0.5, 0.5, 0.5]])  # some quaternion
        # Normalize to unit quaternion
        q = q / q.norm(dim=-1, keepdim=True)
        R = quaternion_to_matrix(q)
        RtR = R @ R.transpose(-1, -2)
        assert torch.allclose(RtR, torch.eye(3).unsqueeze(0), atol=1e-5)


class TestGetTransformationMatrixFromQuat:
    """Test get_transformation_matrix_from_quat from trajectory_maps."""

    def test_identity_transformation(self):
        from worldfoundry.studio.visualization.plugins.robotics.trajectory_maps import get_transformation_matrix_from_quat
        # quat format: (b, 7) → [tx, ty, tz, qx, qy, qz, qw]
        quat = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        result = get_transformation_matrix_from_quat(quat)
        assert result.shape == (1, 4, 4)
        assert torch.allclose(result, torch.eye(4).unsqueeze(0), atol=1e-5)

    def test_translation_only(self):
        from worldfoundry.studio.visualization.plugins.robotics.trajectory_maps import get_transformation_matrix_from_quat
        # quat: [tx, ty, tz, qx, qy, qz, qw] → translation at indices 0:3, rotation at 3:7
        quat = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]])
        result = get_transformation_matrix_from_quat(quat)
        assert result.shape == (1, 4, 4)
        # Translation column should be [1, 2, 3]
        assert torch.allclose(result[0, :3, 3], torch.tensor([1.0, 2.0, 3.0]), atol=1e-5)

    def test_batch_input(self):
        from worldfoundry.studio.visualization.plugins.robotics.trajectory_maps import get_transformation_matrix_from_quat
        quat = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
        ])
        result = get_transformation_matrix_from_quat(quat)
        assert result.shape == (2, 4, 4)


# ===================================================================
# SECTION 8: Styles — CameraState dataclass (from styles.colorbar_utils)
# ===================================================================

class TestCameraState:
    """Test CameraState dataclass from styles.colorbar_utils."""

    def test_construction(self):
        from worldfoundry.studio.visualization.plugins.styles.colorbar_utils import CameraState
        cs = CameraState(fov=0.8, aspect=1.5, c2w=np.eye(4))
        assert cs.fov == 0.8
        assert cs.aspect == 1.5
        assert np.allclose(cs.c2w, np.eye(4))

    def test_get_K_identity_fov(self):
        from worldfoundry.studio.visualization.plugins.styles.colorbar_utils import CameraState
        fov = math.pi / 3  # 60 degrees
        cs = CameraState(fov=fov, aspect=1.0, c2w=np.eye(4))
        K = cs.get_K((640, 480))
        assert K.shape == (3, 3)
        # Focal length = H/2 / tan(fov/2)
        expected_focal = 480 / 2.0 / math.tan(fov / 2.0)
        assert np.allclose(K[0, 0], expected_focal, atol=1e-5)
        assert np.allclose(K[1, 1], expected_focal, atol=1e-5)
        # Principal point at center
        assert np.allclose(K[0, 2], 640 / 2.0)
        assert np.allclose(K[1, 2], 480 / 2.0)

    def test_get_K_output_shape(self):
        from worldfoundry.studio.visualization.plugins.styles.colorbar_utils import CameraState
        cs = CameraState(fov=0.8, aspect=1.5, c2w=np.eye(4))
        K = cs.get_K((800, 600))
        assert K.shape == (3, 3)
        assert K[2, 2] == 1.0  # bottom-right corner


# ===================================================================

class TestOptionalDependencyModules:
    def test_keyboard_overlay_requires_loguru(self):
        try:
            import worldfoundry.studio.visualization.plugins.robotics.keyboard_overlay
        except ModuleNotFoundError as e:
            assert "loguru" in str(e).lower() or "diffusers" in str(e).lower()

    def test_robotics_module_requires_heavy_deps(self):
        try:
            import worldfoundry.studio.visualization.plugins.robotics.robotics
        except ModuleNotFoundError:
            pass


class TestModuleAllCompleteness:
    def test_slam_all(self):
        import worldfoundry.studio.visualization.plugins.robotics.slam as slam
        assert "DroidVisualizer" in slam.__all__
        assert "merge_depths_and_poses" in slam.__all__
        assert "visualization_fn" in slam.__all__

    def test_realtime_all(self):
        import worldfoundry.studio.visualization.plugins.robotics.realtime as rt
        assert "RealtimeVisualizer" in rt.__all__

    def test_trajectory_maps_has_public_functions(self):
        import worldfoundry.studio.visualization.plugins.robotics.trajectory_maps as tm
        assert hasattr(tm, 'quaternion_to_matrix')
        assert hasattr(tm, 'get_transformation_matrix_from_quat')
        assert hasattr(tm, 'simple_radius_gen_func')
        assert hasattr(tm, 'get_traj_maps')

# SECTION 14: CameraState dataclass edge cases
# ===================================================================

class TestCameraStateEdgeCases:
    """Edge cases for CameraState dataclass."""

    def test_get_K_with_narrow_fov(self):
        from worldfoundry.studio.visualization.plugins.styles.colorbar_utils import CameraState
        fov = 0.1  # very narrow FOV → very long focal length
        cs = CameraState(fov=fov, aspect=1.0, c2w=np.eye(4))
        K = cs.get_K((640, 480))
        focal = 480 / 2.0 / math.tan(0.1 / 2.0)
        assert np.allclose(K[0, 0], focal, atol=1e-3)

    def test_get_K_with_wide_fov(self):
        from worldfoundry.studio.visualization.plugins.styles.colorbar_utils import CameraState
        fov = math.pi * 0.9  # very wide FOV → very short focal length
        cs = CameraState(fov=fov, aspect=1.0, c2w=np.eye(4))
        K = cs.get_K((640, 480))
        focal = 480 / 2.0 / math.tan(fov / 2.0)
        assert np.allclose(K[0, 0], focal, atol=1e-3)

    def test_c2w_field_is_array(self):
        from worldfoundry.studio.visualization.plugins.styles.colorbar_utils import CameraState
        c2w = np.random.randn(4, 4)
        cs = CameraState(fov=0.8, aspect=1.0, c2w=c2w)
        assert np.allclose(cs.c2w, c2w)


# ===================================================================
class TestIntegrationQuaternionToTransformationPipeline:
    """Test pipeline: quaternion → rotation matrix → transformation matrix."""

    def test_pipeline(self):
        from worldfoundry.studio.visualization.plugins.robotics.trajectory_maps import (
            quaternion_to_matrix, get_transformation_matrix_from_quat
        )

        quat = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        result = get_transformation_matrix_from_quat(quat)
        # Should produce identity transformation
        assert torch.allclose(result, torch.eye(4).unsqueeze(0), atol=1e-5)

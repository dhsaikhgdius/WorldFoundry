"""Regression tests for studio human_pose drawing helpers (SA-3 / SA-6).

SA-3: ``draw_mask`` used to call the undefined ``alphaMerge`` helper (NameError
on first use) and discarded the resized background because of a variable-name
typo (``backgournd``).

SA-6: ``draw_aapose_new`` used a bare ``raise`` outside any exception handler
for unknown ``stickwidth_type`` values, producing ``RuntimeError: No active
exception to re-raise`` instead of a meaningful error.  ``draw_handpose_new``
had the same bug in the shape of an ``UnboundLocalError`` (no else branch).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")

from worldfoundry.studio.visualization.plugins.perception import human_pose


class TestDrawMask:
    def test_composites_foreground_over_black_background(self) -> None:
        img = np.full((8, 6, 3), 200, dtype=np.uint8)
        mask = np.zeros((8, 6), dtype=np.uint8)
        mask[:4] = 255
        out = human_pose.draw_mask(img, mask, background=0)
        assert out.shape == (8, 6, 3)
        assert out.dtype == np.uint8
        assert (out[:4] == 200).all()
        assert (out[4:] == 0).all()

    def test_white_background_shows_through_zero_mask(self) -> None:
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=np.uint8)
        out = human_pose.draw_mask(img, mask, background=1)
        assert (out == 255).all()

    def test_partial_alpha_blends_linearly(self) -> None:
        img = np.full((2, 2, 3), 100, dtype=np.uint8)
        mask = np.full((2, 2), 128, dtype=np.uint8)
        out = human_pose.draw_mask(img, mask, background=0)
        assert np.allclose(out, 100 * 128 / 255, atol=1)

    def test_background_image_is_resized_to_foreground(self) -> None:
        """Regression: the resized background used to be assigned to a typo variable."""
        img = np.zeros((8, 6, 3), dtype=np.uint8)
        mask = np.zeros((8, 6), dtype=np.uint8)  # fully background
        background = np.full((3, 2, 3), 77, dtype=np.uint8)  # deliberately mismatched size
        out = human_pose.draw_mask(img, mask, background=background)
        assert out.shape == (8, 6, 3)
        assert (out == 77).all()

    def test_mask_with_channel_dim_and_bool_mask(self) -> None:
        img = np.full((4, 4, 3), 10, dtype=np.uint8)
        mask3 = np.full((4, 4, 1), 255, dtype=np.uint8)
        out3 = human_pose.draw_mask(img, mask3, background=0)
        assert (out3 == 10).all()
        maskb = np.ones((4, 4), dtype=bool)
        outb = human_pose.draw_mask(img, maskb, background=0)
        assert (outb == 10).all()

    def test_return_rgba_appends_mask_as_alpha_channel(self) -> None:
        img = np.full((4, 4, 3), 10, dtype=np.uint8)
        mask = np.full((4, 4), 255, dtype=np.uint8)
        out = human_pose.draw_mask(img, mask, background=0, return_rgba=True)
        assert out.shape == (4, 4, 4)
        assert (out[..., :3] == 10).all()
        assert (out[..., 3] == 255).all()


def _zero_confidence_kp2ds(rows: int = 20) -> np.ndarray:
    return np.zeros((rows, 3), dtype=np.float32)


class TestStickwidthTypeValidation:
    def test_draw_aapose_new_rejects_unknown_stickwidth_type(self) -> None:
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="stickwidth_type"):
            human_pose.draw_aapose_new(img, _zero_confidence_kp2ds(), stickwidth_type="v3")

    @pytest.mark.parametrize("stickwidth_type", ["v1", "v2"])
    def test_draw_aapose_new_accepts_known_types(self, stickwidth_type: str) -> None:
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        out = human_pose.draw_aapose_new(img, _zero_confidence_kp2ds(), stickwidth_type=stickwidth_type)
        assert out.shape == (32, 32, 3)

    def test_draw_handpose_new_rejects_unknown_stickwidth_type(self) -> None:
        canvas = np.zeros((32, 32, 3), dtype=np.uint8)
        keypoints = np.zeros((21, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="stickwidth_type"):
            human_pose.draw_handpose_new(canvas, keypoints, stickwidth_type="bogus")

    @pytest.mark.parametrize("stickwidth_type", ["v1", "v2"])
    def test_draw_handpose_new_accepts_known_types(self, stickwidth_type: str) -> None:
        canvas = np.zeros((32, 32, 3), dtype=np.uint8)
        keypoints = np.zeros((21, 3), dtype=np.float32)
        out = human_pose.draw_handpose_new(canvas, keypoints, stickwidth_type=stickwidth_type)
        assert out.shape == (32, 32, 3)

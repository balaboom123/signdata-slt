"""Tests for normalization math (processors/normalize.py helper functions).

This is the most important test file — covers core numerical logic.
"""

import numpy as np
import pytest

from sign_prep.processors.common.normalize import (
    _apply_keypoint_reduction,
    _apply_visibility_mask,
    _get_default_keypoint_indices,
    _normalize_clip_xyz,
)


# ── _get_default_keypoint_indices ───────────────────────────────────────────

class TestGetDefaultKeypointIndices:
    def test_85_returns_none(self):
        assert _get_default_keypoint_indices(85) is None

    def test_133_returns_85_indices(self):
        indices = _get_default_keypoint_indices(133)
        assert indices is not None
        assert len(indices) == 85
        assert all(0 <= i < 133 for i in indices)

    def test_553_returns_85_indices(self):
        indices = _get_default_keypoint_indices(553)
        assert indices is not None
        assert len(indices) == 85
        assert all(0 <= i < 553 for i in indices)

    def test_543_returns_83_indices(self):
        indices = _get_default_keypoint_indices(543)
        assert indices is not None
        assert len(indices) == 83
        assert all(0 <= i < 543 for i in indices)

    def test_unknown_returns_none(self):
        assert _get_default_keypoint_indices(200) is None
        assert _get_default_keypoint_indices(0) is None


# ── _apply_keypoint_reduction ───────────────────────────────────────────────

class TestApplyKeypointReduction:
    def test_select_indices(self):
        arr = np.arange(5 * 10 * 4, dtype=np.float32).reshape(5, 10, 4)
        indices = [0, 3, 7]
        result = _apply_keypoint_reduction(arr, indices)
        assert result.shape == (5, 3, 4)
        np.testing.assert_array_equal(result[:, 0, :], arr[:, 0, :])
        np.testing.assert_array_equal(result[:, 1, :], arr[:, 3, :])
        np.testing.assert_array_equal(result[:, 2, :], arr[:, 7, :])

    def test_output_shape(self):
        arr = np.zeros((10, 133, 4), dtype=np.float32)
        indices = list(range(85))
        result = _apply_keypoint_reduction(arr, indices)
        assert result.shape == (10, 85, 4)

    def test_out_of_bounds_raises(self):
        arr = np.zeros((5, 10, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="out of range"):
            _apply_keypoint_reduction(arr, [0, 5, 15])

    def test_wrong_ndim_raises(self):
        arr = np.zeros((5, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="3D array"):
            _apply_keypoint_reduction(arr, [0, 1])


# ── _apply_visibility_mask ──────────────────────────────────────────────────

SENTINEL = -999.0


class TestApplyVisibilityMask:
    def test_frame_level_all_zero(self):
        """All-zero frame → sentinel -999.0."""
        arr = np.zeros((3, 4, 4), dtype=np.float32)
        # Set frame 1 to have some data
        arr[1, :, :3] = 1.0
        arr[1, :, 3] = 0.9

        result = _apply_visibility_mask(
            arr,
            mask_frame_level=True,
            mask_landmark_level=False,
            visibility_threshold=0.3,
            unvisible_value=SENTINEL,
        )
        assert result.shape == (3, 4, 3)
        # Frame 0 and 2 are all-zero → sentinel
        np.testing.assert_array_equal(result[0], SENTINEL)
        np.testing.assert_array_equal(result[2], SENTINEL)
        # Frame 1 has valid data
        np.testing.assert_array_equal(result[1], arr[1, :, :3])

    def test_landmark_level_low_visibility(self):
        """Low visibility → sentinel -999.0."""
        arr = np.ones((2, 3, 4), dtype=np.float32)
        arr[:, :, 3] = 0.8  # All visible
        arr[0, 1, 3] = 0.1  # Landmark 1 in frame 0 → low visibility

        result = _apply_visibility_mask(
            arr,
            mask_frame_level=False,
            mask_landmark_level=True,
            visibility_threshold=0.3,
            unvisible_value=SENTINEL,
        )
        assert result.shape == (2, 3, 3)
        # Landmark with low vis → sentinel
        np.testing.assert_array_equal(result[0, 1], SENTINEL)
        # Other landmarks intact
        np.testing.assert_array_equal(result[0, 0], arr[0, 0, :3])

    def test_both_disabled_no_masking(self):
        """No masking when both flags are False."""
        arr = np.zeros((2, 3, 4), dtype=np.float32)
        result = _apply_visibility_mask(
            arr,
            mask_frame_level=False,
            mask_landmark_level=False,
            visibility_threshold=0.3,
            unvisible_value=SENTINEL,
        )
        assert result.shape == (2, 3, 3)
        np.testing.assert_array_equal(result, arr[:, :, :3])

    def test_output_strips_visibility_channel(self):
        """Output shape is (T, K, 3) — visibility channel stripped."""
        arr = np.ones((5, 10, 4), dtype=np.float32)
        result = _apply_visibility_mask(
            arr,
            mask_frame_level=True,
            mask_landmark_level=True,
            visibility_threshold=0.3,
            unvisible_value=SENTINEL,
        )
        assert result.shape == (5, 10, 3)


# ── _normalize_clip_xyz ─────────────────────────────────────────────────────

class TestNormalizeClipXYZ:
    def test_isotropic_3d_known_input(self):
        """isotropic_3d: hand-computed expected output."""
        # 2 frames, 2 keypoints
        xyz = np.array([
            [[0.0, 0.0, 0.0],
             [1.0, 0.5, 0.25]],
            [[0.5, 0.25, 0.125],
             [1.0, 1.0, 0.5]],
        ], dtype=np.float32)

        result = _normalize_clip_xyz(xyz, "isotropic_3d", SENTINEL)

        # min = [0, 0, 0], max = [1, 1, 0.5], range = [1, 1, 0.5]
        # max_range = 1.0
        # scaled = (xyz - min) / 1.0 = xyz
        expected = xyz.copy()
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_xy_isotropic_z_minmax_known_input(self):
        """xy_isotropic_z_minmax: hand-computed expected output."""
        xyz = np.array([
            [[0.0, 0.0, 0.0],
             [2.0, 1.0, 10.0]],
            [[1.0, 0.5, 5.0],
             [2.0, 2.0, 10.0]],
        ], dtype=np.float32)

        result = _normalize_clip_xyz(xyz, "xy_isotropic_z_minmax", SENTINEL)

        # x: min=0, max=2 → range=2
        # y: min=0, max=2 → range=2
        # max_xy = 2
        # x_s = (x - 0) / 2, y_s = (y - 0) / 2
        # z: min=0, max=10 → z_s = (z - 0) / 10
        expected = np.array([
            [[0.0, 0.0, 0.0],
             [1.0, 0.5, 1.0]],
            [[0.5, 0.25, 0.5],
             [1.0, 1.0, 1.0]],
        ], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_all_invalid_returns_unchanged(self):
        """All-invalid clip → returns unchanged."""
        xyz = np.full((3, 4, 3), SENTINEL, dtype=np.float32)
        result = _normalize_clip_xyz(xyz, "isotropic_3d", SENTINEL)
        np.testing.assert_array_equal(result, xyz)

    def test_zero_range_no_division_error(self):
        """Zero-range edge case → all zeros (no division error)."""
        # All valid points at the same location
        xyz = np.full((3, 2, 3), 0.5, dtype=np.float32)
        result = _normalize_clip_xyz(xyz, "isotropic_3d", SENTINEL)
        # zero range → scaled to zeros
        np.testing.assert_array_equal(result, np.zeros_like(xyz))

    def test_zero_range_xy_isotropic(self):
        """Zero range in xy_isotropic_z_minmax → all zeros."""
        xyz = np.full((3, 2, 3), 0.5, dtype=np.float32)
        result = _normalize_clip_xyz(xyz, "xy_isotropic_z_minmax", SENTINEL)
        np.testing.assert_array_equal(result, np.zeros_like(xyz))

    def test_sentinel_values_preserved(self):
        """Sentinel values preserved after normalization."""
        xyz = np.array([
            [[0.0, 0.0, 0.0],
             [SENTINEL, SENTINEL, SENTINEL]],
            [[1.0, 1.0, 1.0],
             [0.5, 0.5, 0.5]],
        ], dtype=np.float32)

        result = _normalize_clip_xyz(xyz, "isotropic_3d", SENTINEL)

        # Sentinel point should remain sentinel
        np.testing.assert_array_equal(result[0, 1], [SENTINEL, SENTINEL, SENTINEL])
        # Valid points should be normalized (not sentinel)
        assert not np.any(result[1, 0] == SENTINEL)

    def test_unknown_mode_raises(self):
        xyz = np.zeros((2, 3, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown mode"):
            _normalize_clip_xyz(xyz, "unknown_mode", SENTINEL)

    def test_wrong_shape_raises(self):
        xyz = np.zeros((2, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected shape"):
            _normalize_clip_xyz(xyz, "isotropic_3d", SENTINEL)

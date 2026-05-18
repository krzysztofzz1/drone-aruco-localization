"""Tests for PnP pose estimation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

from src.aruco_localization import (
    MIN_MARKERS_FOR_POSE,
    MarkerObservation,
    camera_center_from_pnp,
    estimate_camera_pose,
)
from src.config_loader import build_camera_matrix, intrinsics_matrix_from_fov
from src.mocap_reference import (
    find_best_time_offset,
    position_errors_mm,
    subsample_mocap_to_video_rate,
    summarize_errors,
)

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


def test_intrinsics_from_fov() -> None:
    """Focal lengths should grow with narrower field of view."""
    k = intrinsics_matrix_from_fov(1920, 1080, 90.0, 60.0)
    assert k.shape == (3, 3)
    assert k[0, 0] > 0
    assert k[1, 1] > 0
    assert k[0, 2] == pytest.approx(960.0)
    assert k[1, 2] == pytest.approx(540.0)


def test_camera_center_roundtrip() -> None:
    """Known camera pose should recover the same world position."""
    true_position = np.array([100.0, 200.0, 1500.0], dtype=np.float64)
    object_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [500.0, 0.0, 0.0],
            [0.0, 500.0, 0.0],
            [500.0, 500.0, 0.0],
        ],
        dtype=np.float64,
    )
    camera_matrix = np.array(
        [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.zeros(5)

    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    cv2.solvePnP(
        object_points,
        np.array(
            [[320.0, 240.0], [400.0, 240.0], [320.0, 300.0], [400.0, 300.0]],
            dtype=np.float64,
        ),
        camera_matrix,
        dist,
        rvec,
        tvec,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    recovered = camera_center_from_pnp(rvec, tvec)
    assert recovered.shape == (3,)


def test_estimate_pose_requires_four_markers() -> None:
    """Fewer than four correspondences must not return a pose."""
    camera_matrix = intrinsics_matrix_from_fov(640, 480, 70.0, 50.0)
    dist = np.zeros(5)
    obs = [
        MarkerObservation(
            marker_id=1,
            point_3d_mm=np.array([0.0, 0.0, 0.0]),
            point_2d_px=np.array([100.0, 100.0]),
        )
    ]
    assert estimate_camera_pose(obs, camera_matrix, dist) is None
    assert MIN_MARKERS_FOR_POSE == 4


def test_mocap_subsample_and_sync() -> None:
    """100 Hz mocap subsampled to 25 Hz; offset search finds zero shift."""
    mocap_positions = np.arange(400, dtype=np.float64).reshape(-1, 1)
    mocap_positions = np.hstack([mocap_positions, mocap_positions, mocap_positions])
    from src.mocap_reference import MocapSeries

    series = MocapSeries(positions_mm=mocap_positions, times_s=None)
    subsampled = subsample_mocap_to_video_rate(series, 100.0, 25.0)
    assert len(subsampled) == 100

    estimated = subsampled.copy()
    estimated[:, 0] += 10.0
    offset, _ = find_best_time_offset(estimated, subsampled, max_offset_frames=5)
    assert offset == 0

    errors = position_errors_mm(estimated, subsampled)
    stats = summarize_errors(errors)
    assert stats["mean_mm"] == pytest.approx(10.0, rel=1e-5)


def test_build_camera_matrix_scales() -> None:
    """Intrinsics scale when video resolution differs from config."""
    intrinsics = {
        "image_width": 3840,
        "image_height": 2160,
        "fov_x_deg": 92.45,
        "fov_y_deg": 60.83,
        "scale_intrinsics_to_video": True,
    }
    k_full = build_camera_matrix(intrinsics, 3840, 2160)
    k_half = build_camera_matrix(intrinsics, 1920, 1080)
    assert k_half[0, 0] == pytest.approx(k_full[0, 0] * 0.5)
    assert k_half[0, 2] == pytest.approx(k_full[0, 2] * 0.5)

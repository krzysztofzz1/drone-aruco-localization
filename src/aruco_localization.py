"""ArUco marker detection and camera pose estimation from 2D–3D correspondences."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

# OpenCV corner order: top-left, top-right, bottom-right, bottom-left.
UPPER_RIGHT_CORNER_INDEX = 1
MIN_MARKERS_FOR_POSE = 4


@dataclass(frozen=True)
class MarkerObservation:
    """Single correspondence: known 3D point and its 2D projection."""

    marker_id: int
    point_3d_mm: npt.NDArray[np.float64]
    point_2d_px: npt.NDArray[np.float64]


@dataclass(frozen=True)
class CameraPose:
    """Camera pose in the world (mocap) coordinate system."""

    position_mm: npt.NDArray[np.float64]
    rotation_matrix: npt.NDArray[np.float64]
    reprojection_error_px: float
    num_markers: int


def create_aruco_detector(
    dictionary_id: int = cv2.aruco.DICT_4X4_1000,
) -> cv2.aruco.ArucoDetector:
    """Create an OpenCV ArUco detector (DPJAIT uses DICT_4X4_1000)."""
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    parameters = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def detect_marker_observations(
    frame_bgr: npt.NDArray[np.uint8],
    detector: cv2.aruco.ArucoDetector,
    markers_3d_mm: dict[int, npt.NDArray[np.float64]],
) -> list[MarkerObservation]:
    """
    Detect ArUco markers and pair each upper-right corner with its known 3D position.

    Only markers present in ``markers_3d_mm`` are used.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return []

    observations: list[MarkerObservation] = []
    flat_ids = ids.flatten()
    for corner_set, marker_id in zip(corners, flat_ids, strict=True):
        marker_id = int(marker_id)
        if marker_id not in markers_3d_mm:
            continue
        corner_2d = corner_set[0][UPPER_RIGHT_CORNER_INDEX].astype(np.float64)
        observations.append(
            MarkerObservation(
                marker_id=marker_id,
                point_3d_mm=markers_3d_mm[marker_id],
                point_2d_px=corner_2d,
            ),
        )
    return observations


def camera_center_from_pnp(
    rvec: npt.NDArray[np.float64],
    tvec: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Compute camera position in world coordinates from solvePnP output.

    OpenCV uses X_cam = R @ X_world + t; camera center C = -R^T @ t.
    """
    rotation, _ = cv2.Rodrigues(rvec)
    return (-rotation.T @ tvec.reshape(3, 1)).reshape(3)


def estimate_camera_pose(
    observations: list[MarkerObservation],
    camera_matrix: npt.NDArray[np.float64],
    dist_coeffs: npt.NDArray[np.float64],
) -> CameraPose | None:
    """
    Estimate camera position in the world frame using PnP on all correspondences.

    Returns None when fewer than ``MIN_MARKERS_FOR_POSE`` markers are available.
    """
    if len(observations) < MIN_MARKERS_FOR_POSE:
        return None

    object_points = np.array(
        [obs.point_3d_mm for obs in observations],
        dtype=np.float64,
    )
    image_points = np.array(
        [obs.point_2d_px for obs in observations],
        dtype=np.float64,
    )

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    projected = projected.reshape(-1, 2)
    reproj_err = float(np.mean(np.linalg.norm(projected - image_points, axis=1)))

    rotation, _ = cv2.Rodrigues(rvec)
    position = camera_center_from_pnp(rvec, tvec)

    return CameraPose(
        position_mm=position,
        rotation_matrix=rotation,
        reprojection_error_px=reproj_err,
        num_markers=len(observations),
    )


def draw_observations(
    frame_bgr: npt.NDArray[np.uint8],
    observations: list[MarkerObservation],
    pose: CameraPose | None,
) -> npt.NDArray[np.uint8]:
    """Annotate frame with detected markers and optional pose text."""
    output = frame_bgr.copy()
    if observations:
        corners = [
            np.array([[obs.point_2d_px]], dtype=np.float32) for obs in observations
        ]
        ids = np.array([[obs.marker_id] for obs in observations], dtype=np.int32)
        cv2.aruco.drawDetectedMarkers(output, corners, ids)

    if pose is not None:
        x, y, z = pose.position_mm
        text = f"cam: ({x:.0f}, {y:.0f}, {z:.0f}) mm  err={pose.reprojection_error_px:.2f}px"
        cv2.putText(
            output,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    return output

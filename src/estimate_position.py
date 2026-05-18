"""
Estimate drone (camera) position from ArUco markers in video and compare to mocap reference.

For each video frame (25 Hz):
  1) Detect and identify ArUco markers
  2) Match upper-right corners to known 3D coordinates
  3) Estimate camera position via PnP (>= 4 markers)
  4) Compare with mocap reference (100 Hz, subsampled to 25 Hz) and report error

Usage:
  python src/estimate_position.py --video data/raw/GX010280.MP4 \\
      --markers data/raw/markers_3d.csv --mocap data/raw/mocap_reference.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from aruco_localization import (
    create_aruco_detector,
    detect_marker_observations,
    draw_observations,
    estimate_camera_pose,
)
from config_loader import (
    build_camera_matrix,
    distortion_coefficients,
    load_intrinsics_yaml,
    load_markers_3d,
)
from mocap_reference import (
    aligned_reference_positions,
    find_best_time_offset,
    load_mocap_csv,
    position_errors_mm,
    subsample_mocap_to_video_rate,
    summarize_errors,
)
from VideoPlayer import VideoPlayer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTRINSICS = PROJECT_ROOT / "config" / "camera_intrinsics.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def _resolve_path(path: str | Path) -> Path:
    """Resolve path relative to project root when not absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def process_video(
    video_path: Path,
    markers_3d_mm: dict[int, npt.NDArray[np.float64]],
    camera_matrix: npt.NDArray[np.float64],
    dist_coeffs: npt.NDArray[np.float64],
    *,
    show_preview: bool = False,
    max_frames: int | None = None,
) -> tuple[npt.NDArray[np.float64], list[int], list[float]]:
    """
    Run localization on all frames.

    Returns:
        positions_mm: (N, 3) array with NaN rows where pose failed
        frame_indices: frame numbers
        reproj_errors: per-frame reprojection error (NaN if failed)
    """
    player = VideoPlayer(str(video_path))
    detector = create_aruco_detector()

    positions: list[npt.NDArray[np.float64]] = []
    frame_indices: list[int] = []
    reproj_errors: list[float] = []

    for frame_idx, frame in player.iteruj_klatki():
        if max_frames is not None and frame_idx >= max_frames:
            break

        observations = detect_marker_observations(frame, detector, markers_3d_mm)
        pose = estimate_camera_pose(observations, camera_matrix, dist_coeffs)

        if pose is not None:
            positions.append(pose.position_mm)
            reproj_errors.append(pose.reprojection_error_px)
        else:
            positions.append(np.full(3, np.nan))
            reproj_errors.append(float("nan"))

        frame_indices.append(frame_idx)

        if show_preview:
            vis = draw_observations(frame, observations, pose)
            vis = cv2.resize(vis, None, fx=0.5, fy=0.5)
            cv2.imshow("ArUco localization", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if show_preview:
        cv2.destroyAllWindows()

    if not positions:
        return np.empty((0, 3)), [], []

    return (
        np.stack(positions, axis=0),
        frame_indices,
        reproj_errors,
    )


def save_trajectory_csv(
    output_path: Path,
    frame_indices: list[int],
    estimated_mm: npt.NDArray[np.float64],
    reference_mm: npt.NDArray[np.float64] | None,
    errors_mm: npt.NDArray[np.float64] | None,
    reproj_errors: list[float],
) -> None:
    """Write per-frame results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "frame",
            "est_x_mm",
            "est_y_mm",
            "est_z_mm",
            "reproj_error_px",
        ]
        if reference_mm is not None:
            fieldnames.extend(["ref_x_mm", "ref_y_mm", "ref_z_mm", "error_mm"])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        n = len(frame_indices)
        for i in range(n):
            row: dict[str, float | int] = {
                "frame": frame_indices[i],
                "est_x_mm": float(estimated_mm[i, 0]),
                "est_y_mm": float(estimated_mm[i, 1]),
                "est_z_mm": float(estimated_mm[i, 2]),
                "reproj_error_px": reproj_errors[i],
            }
            if reference_mm is not None and i < len(reference_mm):
                row["ref_x_mm"] = float(reference_mm[i, 0])
                row["ref_y_mm"] = float(reference_mm[i, 1])
                row["ref_z_mm"] = float(reference_mm[i, 2])
            if errors_mm is not None and i < len(errors_mm):
                row["error_mm"] = float(errors_mm[i])
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    """Configure CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Estimate drone camera position from ArUco markers in video.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/raw/GX010280.MP4",
        help="Path to drone camera MP4 (25 Hz)",
    )
    parser.add_argument(
        "--markers",
        type=str,
        default="data/raw/ArUco_markers_3D.xlsx",
        help="Wall marker 3D coords (xlsx/csv): Marker_ID, X, Y, Z",
    )
    parser.add_argument(
        "--mocap",
        type=str,
        default="data/raw/GX010280_U.csv",
        help="Mocap reference (100 Hz), e.g. GX010280_U.csv; optional",
    )
    parser.add_argument(
        "--intrinsics",
        type=str,
        default=str(DEFAULT_INTRINSICS),
        help="Camera intrinsics YAML",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output trajectory CSV (default: data/processed/<video_stem>_trajectory.csv)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show live preview window",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process only first N frames (debug)",
    )
    parser.add_argument(
        "--sync-offset",
        type=int,
        default=None,
        help="Manual mocap frame offset (skip auto sync)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for position estimation pipeline."""
    args = build_arg_parser().parse_args(argv)

    video_path = _resolve_path(args.video)
    markers_path = _resolve_path(args.markers)
    intrinsics_path = _resolve_path(args.intrinsics)

    if not video_path.exists():
        logger.error("Video not found: %s", video_path)
        return 1
    if not markers_path.exists():
        logger.error(
            "Markers file not found: %s — copy config/markers_3d.csv.example "
            "and fill with your sequence data.",
            markers_path,
        )
        return 1

    markers_3d = load_markers_3d(markers_path)
    intrinsics = load_intrinsics_yaml(intrinsics_path)

    cap = cv2.VideoCapture(str(video_path))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or float(intrinsics.get("video_fps", 25.0))
    cap.release()

    camera_matrix = build_camera_matrix(intrinsics, video_w, video_h)
    dist_coeffs = distortion_coefficients(intrinsics)

    logger.info("Video: %s (%dx%d @ %.2f Hz)", video_path.name, video_w, video_h, video_fps)
    logger.info("Loaded %d marker 3D points", len(markers_3d))

    estimated_mm, frame_indices, reproj_errors = process_video(
        video_path,
        markers_3d,
        camera_matrix,
        dist_coeffs,
        show_preview=args.preview,
        max_frames=args.max_frames,
    )

    valid_frames = int(np.sum(np.all(np.isfinite(estimated_mm), axis=1)))
    logger.info(
        "Pose estimated for %d / %d frames",
        valid_frames,
        len(frame_indices),
    )

    reference_aligned: npt.NDArray[np.float64] | None = None
    errors_mm: npt.NDArray[np.float64] | None = None
    offset = 0

    if args.mocap:
        mocap_path = _resolve_path(args.mocap)
        if not mocap_path.exists():
            logger.error("Mocap file not found: %s", mocap_path)
            return 1

        mocap = load_mocap_csv(mocap_path)
        mocap_fps = float(intrinsics.get("mocap_fps", 100.0))
        ref_subsampled = subsample_mocap_to_video_rate(mocap, mocap_fps, video_fps)

        if args.sync_offset is not None:
            offset = args.sync_offset
        else:
            offset, sync_err = find_best_time_offset(estimated_mm, ref_subsampled)
            logger.info(
                "Auto sync: mocap offset=%d frames (mean error %.2f mm on search)",
                offset,
                sync_err,
            )

        reference_aligned = aligned_reference_positions(
            estimated_mm,
            ref_subsampled,
            offset,
        )
        n_compare = len(reference_aligned)
        errors_mm = position_errors_mm(
            estimated_mm[:n_compare],
            reference_aligned,
        )
        stats = summarize_errors(errors_mm)
        logger.info(
            "Tracking error vs mocap (mm): mean=%.2f median=%.2f std=%.2f rmse=%.2f (n=%d)",
            stats["mean_mm"],
            stats["median_mm"],
            stats["std_mm"],
            stats["rmse_mm"],
            int(stats["count"]),
        )

    output_path = (
        _resolve_path(args.output)
        if args.output
        else DEFAULT_OUTPUT_DIR / f"{video_path.stem}_trajectory.csv"
    )
    save_trajectory_csv(
        output_path,
        frame_indices,
        estimated_mm,
        reference_aligned,
        errors_mm,
        reproj_errors,
    )
    logger.info("Wrote results to %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Load motion-capture reference trajectories and align them with video frames."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class MocapSeries:
    """Reference 3D positions sampled at mocap rate (typically 100 Hz)."""

    positions_mm: npt.NDArray[np.float64]
    times_s: npt.NDArray[np.float64] | None


def _is_dpjait_mocap_row(cells: list[str]) -> bool:
    """Detect DPJAIT headerless mocap CSV (sample_index, 0, 12+ floats)."""
    if len(cells) < 14:
        return False
    try:
        int(float(cells[0]))
        int(float(cells[1]))
        values = [float(c) for c in cells[2:]]
    except ValueError:
        return False
    return len(values) >= 12 and len(values) % 3 == 0


def _find_column(fieldnames: list[str], *candidates: str) -> str | None:
    lowered = {name.strip().lower(): name for name in fieldnames}
    for key in candidates:
        if key in lowered:
            return lowered[key]
    return None


def load_dpjait_mocap_csv(path: Path, num_body_markers: int = 4) -> MocapSeries:
    """
    Load DPJAIT ``*_U.csv`` mocap files (headerless).

    Each row: ``sample_index, 0,`` then ``num_body_markers`` triplets (x,y,z in mm)
    for drone-mounted markers. The centroid is used as the reference position.
    """
    positions: list[npt.NDArray[np.float64]] = []
    times: list[float] = []

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(
            (line for line in handle if not line.lstrip().startswith("#")),
        )
        for row in reader:
            if len(row) < 2 + num_body_markers * 3:
                continue
            sample_idx = int(float(row[0]))
            values = [float(v) for v in row[2 : 2 + num_body_markers * 3]]
            markers = np.array(values, dtype=np.float64).reshape(num_body_markers, 3)
            positions.append(np.mean(markers, axis=0))
            times.append(sample_idx / 100.0)

    if not positions:
        raise ValueError(f"No mocap rows loaded from {path}")

    return MocapSeries(
        positions_mm=np.stack(positions, axis=0),
        times_s=np.asarray(times, dtype=np.float64),
    )


def load_mocap_csv(path: Path) -> MocapSeries:
    """
    Load reference positions from CSV.

    Supports:
    - Direct position columns: x_mm/y_mm/z_mm (or x, y, z)
    - Four drone markers A–D; centroid is used as reference position
    - Optional time column: time, time_s, t
    """
    with path.open(encoding="utf-8", newline="") as handle:
        lines = [line for line in handle if not line.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"Empty mocap file: {path}")

    peek_reader = csv.reader(lines)
    first_row = next(peek_reader)
    if _is_dpjait_mocap_row(first_row):
        return load_dpjait_mocap_csv(path)

    rows: list[dict[str, str]] = []
    reader = csv.DictReader(lines)
    if reader.fieldnames is None:
        raise ValueError(f"No header in mocap file: {path}")
    fieldnames = list(reader.fieldnames)
    rows = list(reader)

    if not rows:
        raise ValueError(f"No data rows in mocap file: {path}")

    time_col = _find_column(fieldnames, "time_s", "time", "t", "timestamp")
    x_col = _find_column(fieldnames, "x_mm", "x", "pos_x", "camera_x")
    y_col = _find_column(fieldnames, "y_mm", "y", "pos_y", "camera_y")
    z_col = _find_column(fieldnames, "z_mm", "z", "pos_z", "camera_z")

    times: list[float] | None = [] if time_col else None
    positions: list[npt.NDArray[np.float64]] = []

    if x_col and y_col and z_col:
        for row in rows:
            positions.append(
                np.array(
                    [float(row[x_col]), float(row[y_col]), float(row[z_col])],
                    dtype=np.float64,
                ),
            )
            if times is not None and time_col is not None:
                times.append(float(row[time_col]))
    else:
        marker_groups = [
            (["a_x", "ax", "marker_a_x"], ["a_y", "ay", "marker_a_y"], ["a_z", "az", "marker_a_z"]),
            (["b_x", "bx"], ["b_y", "by"], ["b_z", "bz"]),
            (["c_x", "cx"], ["c_y", "cy"], ["c_z", "cz"]),
            (["d_x", "dx"], ["d_y", "dy"], ["d_z", "dz"]),
        ]
        resolved: list[tuple[str, str, str]] = []
        for xs, ys, zs in marker_groups:
            x_name = _find_column(fieldnames, *xs)
            y_name = _find_column(fieldnames, *ys)
            z_name = _find_column(fieldnames, *zs)
            if x_name and y_name and z_name:
                resolved.append((x_name, y_name, z_name))
        if len(resolved) < 1:
            raise ValueError(
                f"Could not find position columns in {path}. "
                "Provide x/y/z or A–D marker columns.",
            )
        for row in rows:
            pts = [
                np.array([float(row[x]), float(row[y]), float(row[z])], dtype=np.float64)
                for x, y, z in resolved
            ]
            positions.append(np.mean(np.stack(pts, axis=0), axis=0))
            if times is not None and time_col is not None:
                times.append(float(row[time_col]))

    positions_arr = np.stack(positions, axis=0)
    times_arr = np.asarray(times, dtype=np.float64) if times is not None else None
    return MocapSeries(positions_mm=positions_arr, times_s=times_arr)


def subsample_mocap_to_video_rate(
    mocap: MocapSeries,
    mocap_fps: float,
    video_fps: float,
) -> npt.NDArray[np.float64]:
    """Take every N-th mocap sample so it matches video frame rate (e.g. 100 Hz → 25 Hz)."""
    step = max(1, int(round(mocap_fps / video_fps)))
    return mocap.positions_mm[::step]


def find_best_time_offset(
    estimated_mm: npt.NDArray[np.float64],
    reference_mm: npt.NDArray[np.float64],
    max_offset_frames: int | None = None,
) -> tuple[int, float]:
    """
    Find index offset that minimizes mean 3D distance (DPJAIT sync procedure).

    Slides the longer reference trajectory over the shorter estimated one.
    """
    n_est = len(estimated_mm)
    n_ref = len(reference_mm)
    if n_est == 0 or n_ref == 0:
        return 0, float("inf")

    max_offset = max_offset_frames
    if max_offset is None:
        max_offset = max(0, n_ref - n_est)
    max_offset = min(max_offset, max(0, n_ref - n_est))

    best_offset = 0
    best_error = float("inf")
    for offset in range(max_offset + 1):
        n_compare = min(n_est, n_ref - offset)
        if n_compare <= 0:
            continue
        est = estimated_mm[:n_compare]
        ref = reference_mm[offset : offset + n_compare]
        valid = np.all(np.isfinite(est), axis=1) & np.all(np.isfinite(ref), axis=1)
        if not np.any(valid):
            continue
        dist = np.linalg.norm(est[valid] - ref[valid], axis=1)
        mean_err = float(np.mean(dist))
        if mean_err < best_error:
            best_error = mean_err
            best_offset = offset

    return best_offset, best_error


def aligned_reference_positions(
    estimated_mm: npt.NDArray[np.float64],
    reference_mm: npt.NDArray[np.float64],
    offset_frames: int,
) -> npt.NDArray[np.float64]:
    """Return reference positions aligned to estimated frames using ``offset_frames``."""
    n_compare = min(len(estimated_mm), len(reference_mm) - offset_frames)
    if n_compare <= 0:
        return np.empty((0, 3), dtype=np.float64)
    return reference_mm[offset_frames : offset_frames + n_compare]


def position_errors_mm(
    estimated_mm: npt.NDArray[np.float64],
    reference_mm: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Per-frame Euclidean distance between estimated and reference positions (mm)."""
    n = min(len(estimated_mm), len(reference_mm))
    if n == 0:
        return np.array([], dtype=np.float64)
    est = estimated_mm[:n]
    ref = reference_mm[:n]
    valid = np.all(np.isfinite(est), axis=1) & np.all(np.isfinite(ref), axis=1)
    errors = np.full(n, np.nan, dtype=np.float64)
    errors[valid] = np.linalg.norm(est[valid] - ref[valid], axis=1)
    return errors


def summarize_errors(errors_mm: npt.NDArray[np.float64]) -> dict[str, float]:
    """Compute mean, median, std, and RMSE over valid error samples."""
    valid = errors_mm[np.isfinite(errors_mm)]
    if valid.size == 0:
        return {
            "count": 0.0,
            "mean_mm": float("nan"),
            "median_mm": float("nan"),
            "std_mm": float("nan"),
            "rmse_mm": float("nan"),
        }
    return {
        "count": float(valid.size),
        "mean_mm": float(np.mean(valid)),
        "median_mm": float(np.median(valid)),
        "std_mm": float(np.std(valid)),
        "rmse_mm": float(np.sqrt(np.mean(valid**2))),
    }

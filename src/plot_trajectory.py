"""
Plot estimated vs reference camera positions from a trajectory CSV.

Usage:
  python src/plot_trajectory.py --input data/processed/GX010280_trajectory.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def _resolve_path(path: str | Path) -> Path:
    """Resolve path relative to project root when not absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def valid_pose_mask(
    est: npt.NDArray[np.float64],
    error_mm: npt.NDArray[np.float64] | None,
    *,
    max_error_mm: float = 500.0,
) -> npt.NDArray[np.bool_]:
    """
    Mask frames with finite estimates and acceptable tracking error.

    Failed PnP or mis-detections often produce huge coordinates or errors;
    excluding them keeps comparison plots readable.
    """
    ok = np.isfinite(est).all(axis=1)
    if error_mm is not None:
        ok &= np.isfinite(error_mm) & (error_mm <= max_error_mm)
    return ok


def load_trajectory_csv(path: Path) -> dict[str, npt.NDArray[np.float64]]:
    """
    Load trajectory CSV produced by estimate_position.py.

    Returns:
        Dict with arrays: frame, est (N,3), ref (N,3) or empty, error_mm, reproj_error_px.
    """
    frames: list[float] = []
    est: list[list[float]] = []
    ref: list[list[float]] = []
    errors: list[float] = []
    reproj: list[float] = []
    has_ref = False

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {path}")
        has_ref = "ref_x_mm" in reader.fieldnames

        for row in reader:
            frames.append(float(row["frame"]))
            est.append(
                [float(row["est_x_mm"]), float(row["est_y_mm"]), float(row["est_z_mm"])]
            )
            reproj.append(float(row["reproj_error_px"]))
            if has_ref:
                ref.append(
                    [
                        float(row["ref_x_mm"]),
                        float(row["ref_y_mm"]),
                        float(row["ref_z_mm"]),
                    ]
                )
                errors.append(float(row["error_mm"]))

    result: dict[str, npt.NDArray[np.float64]] = {
        "frame": np.asarray(frames, dtype=np.float64),
        "est": np.asarray(est, dtype=np.float64),
        "reproj_error_px": np.asarray(reproj, dtype=np.float64),
    }
    if has_ref:
        result["ref"] = np.asarray(ref, dtype=np.float64)
        result["error_mm"] = np.asarray(errors, dtype=np.float64)
    return result


def _masked_series(
    values: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Return values with invalid entries replaced by NaN (creates plot gaps)."""
    out = values.astype(np.float64, copy=True)
    out[~mask] = np.nan
    return out


def plot_position_comparison(
    data: dict[str, npt.NDArray[np.float64]],
    *,
    title: str,
    output_path: Path | None = None,
    max_error_mm: float = 500.0,
) -> None:
    """
    Draw time-series (X/Y/Z vs frame), 3D trajectories, and error vs frame.
    """
    frames = data["frame"]
    est = data["est"]
    has_ref = "ref" in data
    error_mm = data.get("error_mm")
    mask = valid_pose_mask(est, error_mm, max_error_mm=max_error_mm)
    n_valid = int(np.sum(mask))
    n_total = len(frames)

    est_plot = est.copy()
    for i in range(3):
        est_plot[:, i] = _masked_series(est[:, i], mask)
    ref_plot = data["ref"].copy() if has_ref else None

    n_rows = 4 if has_ref else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3.2 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    axis_labels = ("X", "Y", "Z")
    colors_est = ("#1f77b4", "#2ca02c", "#9467bd")
    colors_ref = ("#ff7f0e", "#d62728", "#8c564b")

    for ax_idx, (label, c_est, c_ref) in enumerate(zip(axis_labels, colors_est, colors_ref)):
        ax = axes[ax_idx]
        ax.plot(frames, est_plot[:, ax_idx], color=c_est, linewidth=1.2, label=f"ArUco {label}")
        if has_ref and ref_plot is not None:
            ax.plot(
                frames,
                ref_plot[:, ax_idx],
                color=c_ref,
                linewidth=1.0,
                linestyle="--",
                alpha=0.9,
                label=f"Mocap {label}",
            )
        ax.set_ylabel(f"{label} (mm)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    if has_ref:
        ax_err = axes[3]
        err_plot = _masked_series(error_mm, mask) if error_mm is not None else error_mm
        ax_err.plot(frames, err_plot, color="#e377c2", linewidth=1.0, label="3D error")
        ax_err.set_ylabel("Error (mm)")
        ax_err.set_xlabel("Frame")
        ax_err.legend(loc="upper right", fontsize=8)
        ax_err.grid(True, alpha=0.3)
        mean_err = float(np.nanmean(err_plot)) if error_mm is not None else float("nan")
        ax_err.set_title(
            f"Position error (mean {mean_err:.1f} mm, {n_valid}/{n_total} frames)",
            fontsize=10,
        )
    else:
        axes[-1].set_xlabel("Frame")

    subtitle = f"{n_valid}/{n_total} frames (error ≤ {max_error_mm:.0f} mm)"
    fig.suptitle(f"{title}\n{subtitle}", fontsize=12, y=0.995)
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    fig2 = plt.figure(figsize=(9, 7))
    ax3d = fig2.add_subplot(111, projection="3d")
    ax3d.plot(
        est_plot[mask, 0],
        est_plot[mask, 1],
        est_plot[mask, 2],
        color="#1f77b4",
        linewidth=1.2,
        label="ArUco",
    )
    if has_ref and ref_plot is not None:
        ax3d.plot(
            ref_plot[mask, 0],
            ref_plot[mask, 1],
            ref_plot[mask, 2],
            color="#ff7f0e",
            linewidth=1.0,
            linestyle="--",
            alpha=0.9,
            label="Mocap",
        )
    ax3d.set_xlabel("X (mm)")
    ax3d.set_ylabel("Y (mm)")
    ax3d.set_zlabel("Z (mm)")
    ax3d.legend()
    ax3d.set_title(f"{title} — 3D trajectory")

    if output_path is not None:
        path_3d = output_path.with_name(output_path.stem + "_3d" + output_path.suffix)
        fig2.savefig(path_3d, dpi=150, bbox_inches="tight")
        print(f"Saved: {path_3d}")

    if output_path is None:
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig2)


def build_arg_parser() -> argparse.ArgumentParser:
    """Configure CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Plot estimated vs reference positions from trajectory CSV.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/GX010280_trajectory.csv",
        help="Trajectory CSV from estimate_position.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG (default: data/processed/<stem>_comparison.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display interactive plots (in addition to saving if --output set)",
    )
    parser.add_argument(
        "--max-error-mm",
        type=float,
        default=500.0,
        help="Hide ArUco estimates with 3D error above this threshold (mm)",
    )
    return parser


def main() -> int:
    """Entry point."""
    args = build_arg_parser().parse_args()
    input_path = _resolve_path(args.input)
    if not input_path.exists():
        print(f"File not found: {input_path}", file=sys.stderr)
        return 1

    data = load_trajectory_csv(input_path)
    title = f"Position comparison — {input_path.stem}"

    output_path: Path | None
    if args.output:
        output_path = _resolve_path(args.output)
    elif not args.show:
        output_path = DEFAULT_OUTPUT_DIR / f"{input_path.stem}_comparison.png"
    else:
        output_path = None

    plot_position_comparison(
        data,
        title=title,
        output_path=output_path,
        max_error_mm=args.max_error_mm,
    )

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())

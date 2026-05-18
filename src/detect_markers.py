"""Batch ArUco marker detection on a video file (no 3D pose / mocap required)."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter
from pathlib import Path

import cv2

from aruco_localization import create_aruco_detector
from VideoPlayer import VideoPlayer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed"


def run_detection(
    video_path: Path,
    *,
    max_frames: int | None = None,
    save_sample_every: int = 0,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, int | float]:
    """Detect ArUco markers in every frame and return summary statistics."""
    player = VideoPlayer(str(video_path))
    info = player.info
    logger.info(
        "Video: %s  %dx%d  fps=%.2f  frames=%d",
        video_path.name,
        info["szerokosc"],
        info["wysokosc"],
        info["fps"],
        info["liczba_klatek"],
    )

    detector = create_aruco_detector()
    id_hits: Counter[int] = Counter()
    frames_with_markers = 0
    total = 0
    per_frame_rows: list[dict[str, int | str]] = []

    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, frame in player.iteruj_klatki():
        if max_frames is not None and frame_idx >= max_frames:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _rejected = detector.detectMarkers(gray)

        detected_ids: list[int] = []
        if ids is not None:
            frames_with_markers += 1
            detected_ids = [int(x) for x in ids.flatten()]
            for marker_id in detected_ids:
                id_hits[marker_id] += 1
            annotated = frame.copy()
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        else:
            annotated = frame

        per_frame_rows.append(
            {
                "frame": frame_idx,
                "num_markers": len(detected_ids),
                "marker_ids": " ".join(map(str, detected_ids)) if detected_ids else "",
            },
        )

        if save_sample_every > 0 and detected_ids and frame_idx % save_sample_every == 0:
            out_img = output_dir / f"{video_path.stem}_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_img), annotated)

        total += 1
        if total % 500 == 0:
            logger.info("Processed %d / %d frames...", total, info["liczba_klatek"])

    detections_csv = output_dir / f"{video_path.stem}_detections.csv"
    with detections_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame", "num_markers", "marker_ids"],
        )
        writer.writeheader()
        writer.writerows(per_frame_rows)

    logger.info("Wrote per-frame detections to %s", detections_csv)
    logger.info("Frames processed: %d", total)
    logger.info(
        "Frames with markers: %d (%.1f%%)",
        frames_with_markers,
        100.0 * frames_with_markers / max(total, 1),
    )
    if id_hits:
        logger.info("Unique marker IDs: %d", len(id_hits))
        top = id_hits.most_common(20)
        logger.info("Most frequent IDs (frame hits): %s", top)

    return {
        "total_frames": total,
        "frames_with_markers": frames_with_markers,
        "unique_ids": len(id_hits),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for batch marker detection."""
    parser = argparse.ArgumentParser(description="Detect ArUco markers in drone video.")
    parser.add_argument(
        "--video",
        default="data/GX010280.MP4",
        help="Path to MP4 file",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--save-sample-every",
        type=int,
        default=0,
        help="Save annotated JPEG every N frames when markers found (0=off)",
    )
    args = parser.parse_args(argv)

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path
    if not video_path.exists():
        logger.error("Video not found: %s", video_path)
        return 1

    run_detection(
        video_path,
        max_frames=args.max_frames,
        save_sample_every=args.save_sample_every,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

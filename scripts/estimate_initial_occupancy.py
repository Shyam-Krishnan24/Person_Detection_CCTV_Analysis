
"""Estimate initial occupancy from the opening CCTV frames."""

import cv2
import csv
import yaml
from pathlib import Path

from src.onnx_detector import ONNXPersonDetector
from src.byte_tracker import ByteTracker

ROOT = Path(__file__).resolve().parents[1]


def gate_side(point, gate_a, gate_b):
    """Calculate the signed side of a point relative to the gate."""
    ax, ay = gate_a
    bx, by = gate_b
    px, py = point

    return (
        (bx - ax) * (py - ay)
        - (by - ay) * (px - ax)
    )


def main():
    video_path = ROOT / "input" / "store_cctv.mp4"
    config_path = ROOT / "config" / "entrance_band.yaml"

    # Load calibrated gate geometry.
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    gate_a, gate_b = config["gate_line"]
    inside_reference = config["inside_reference_point"]

    inside_sign = (
        1 if gate_side(inside_reference, gate_a, gate_b) > 0 else -1
    )

    # Initialize detector.
    detector = ONNXPersonDetector(
        model_path=ROOT / "models" / "yolov8n.onnx",
        confidence=0.25,
    )

    # Initialize tracker with the same settings used in tracking tests.
    tracker = ByteTracker(
        high_threshold=0.35,
        low_threshold=0.10,
        match_threshold=0.30,
        max_age=30,
        min_hits=1,
    )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        cap.release()
        raise RuntimeError("Invalid video FPS")

    # Analyse the first 2 seconds.
    analysis_frames = max(1, int(fps * 2))

    # A candidate must first appear within the first 0.5 seconds.
    start_window = max(1, int(fps * 0.5))

    # Require repeated observations to reduce temporary detections.
    minimum_observations = 5

    first_seen = {}
    first_point = {}
    observations = {}

    frame_index = 0

    print("Estimating initial occupancy...")
    print(f"Video FPS: {fps:.3f}")
    print(f"Analysis duration: {analysis_frames / fps:.2f} seconds")
    print("-" * 50)

    # Process opening frames.
    while frame_index < analysis_frames:
        ok, frame = cap.read()

        if not ok:
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections)

        for track in tracks:
            tid = track.track_id

            x1, y1, x2, y2 = track.bbox_xyxy

            # Bottom-centre of the person bounding box.
            footpoint = (
                (x1 + x2) / 2.0,
                y2,
            )

            # Preserve the first observed position.
            if tid not in first_seen:
                first_seen[tid] = frame_index
                first_point[tid] = footpoint

            observations[tid] = observations.get(tid, 0) + 1

        frame_index += 1

    cap.release()

    if frame_index == 0:
        raise RuntimeError("No video frames were processed")

    # Select stable tracks already present at the beginning.
    candidates = []

    for tid in sorted(first_seen):
        point = first_point[tid]

        side = gate_side(point, gate_a, gate_b)
        is_inside = side * inside_sign > 0

        appeared_at_start = first_seen[tid] <= start_window
        is_stable = observations[tid] >= minimum_observations

        if appeared_at_start and is_stable and is_inside:
            candidates.append(tid)

    # Prepare detailed report.
    output_dir = ROOT / "reports" / "initial_occupancy"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "initial_occupancy_candidates.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "track_id",
            "first_seen_frame",
            "observations",
            "first_footpoint_x",
            "first_footpoint_y",
            "inside_gate",
            "appeared_at_start",
            "stable_track",
            "selected_for_initial_occupancy",
        ])

        for tid in sorted(first_seen):
            point = first_point[tid]

            side = gate_side(point, gate_a, gate_b)
            is_inside = side * inside_sign > 0

            appeared_at_start = first_seen[tid] <= start_window
            is_stable = observations[tid] >= minimum_observations
            selected = tid in candidates

            writer.writerow([
                tid,
                first_seen[tid],
                observations[tid],
                round(point[0], 2),
                round(point[1], 2),
                is_inside,
                appeared_at_start,
                is_stable,
                selected,
            ])

    # Display results.
    print("\nINITIAL OCCUPANCY ESTIMATE")
    print("-" * 50)

    print("Frames processed:", frame_index)
    print("Unique tracks observed:", len(first_seen))
    print("Stable candidate tracks:", len(candidates))
    print("Estimated initial occupancy:", len(candidates))

    print("\nCandidate track IDs:", candidates)
    print("\nDetailed report:", csv_path)

    print(
        "\nNOTE: This is a provisional estimate. "
        "Visual validation is required before accepting it."
    )


if __name__ == "__main__":
    main()

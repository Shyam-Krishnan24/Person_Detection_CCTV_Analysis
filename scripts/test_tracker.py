
"""Test ByteTrack-style tracking on consecutive CCTV frames."""

import cv2
from pathlib import Path

from src.onnx_detector import ONNXPersonDetector
from src.byte_tracker import ByteTracker

ROOT = Path(__file__).resolve().parents[1]


def main():
    video_path = ROOT / "input" / "store_cctv.mp4"

    detector = ONNXPersonDetector(
        model_path=ROOT / "models" / "yolov8n.onnx",
        confidence=0.25,
    )

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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Process the complete video.
    frames_to_process = total_frames

    print(f"Processing {frames_to_process} frames...")
    print("-" * 55)

    frame_index = 0
    observed_ids = set()
    id_first_seen = {}
    id_last_seen = {}
    id_observations = {}

    while frame_index < frames_to_process:
        ok, frame = cap.read()

        if not ok:
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections)

        for track in tracks:
            tid = track.track_id

            observed_ids.add(tid)
            id_first_seen.setdefault(tid, frame_index)
            id_last_seen[tid] = frame_index
            id_observations[tid] = id_observations.get(tid, 0) + 1

            x1, y1, x2, y2 = map(int, track.bbox_xyxy)

            cv2.rectangle(
                frame, (x1, y1), (x2, y2), (0, 220, 0), 2
            )

            cv2.putText(
                frame,
                f"ID {track.track_id}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 255),
                2,
            )

        cv2.putText(
            frame,
            f"Frame: {frame_index} | Active tracks: {len(tracks)}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        if frame_index % 100 == 0:
            print(
                f"Frame {frame_index:4d} | "
                f"Detections: {len(detections):2d} | "
                f"Active tracks: {len(tracks):2d} | "
                f"Unique IDs: {len(observed_ids):2d}"
            )

        if frame_index % 10 == 0:
            cv2.imwrite(
                str(
                    ROOT / "reports" / "onnx_detector_test"
                    / f"tracking_{frame_index:04d}.jpg"
                ),
                frame,
            )

        frame_index += 1

    cap.release()

    print("-" * 55)
    print("TRACKING TEST COMPLETE")
    print("Frames processed:", frame_index)
    print("Unique IDs observed:", len(observed_ids))
    print("IDs:", sorted(observed_ids))
    print("\nTRACK LIFESPAN REPORT")
    print("-" * 55)

    for tid in sorted(id_first_seen):
        duration = id_last_seen[tid] - id_first_seen[tid] + 1

        print(
            f"ID {tid:3d} | "
            f"First: {id_first_seen[tid]:3d} | "
            f"Last: {id_last_seen[tid]:3d} | "
            f"Span: {duration:3d} frames | "
            f"Observed: {id_observations[tid]:3d}"
        )
    print("Preview images saved in reports/onnx_detector_test/")


if __name__ == "__main__":
    main()

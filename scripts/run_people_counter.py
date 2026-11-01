"""Run the complete offline CCTV people-counting pipeline."""

import csv
import math
from pathlib import Path

import cv2
import yaml

from src.onnx_detector import ONNXPersonDetector
from src.byte_tracker import ByteTracker
from src.people_counter import PeopleCounter


ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = ROOT / "input" / "store_cctv.mp4"
MODEL_PATH = ROOT / "models" / "yolov8n.onnx"
CONFIG_PATH = ROOT / "config" / "entrance_band.yaml"

OUTPUT_DIR = ROOT / "reports" / "people_counter"
INITIAL_DIR = ROOT / "reports" / "initial_occupancy"

ANNOTATED_VIDEO = OUTPUT_DIR / "annotated_people_counter.mp4"
EVENTS_CSV = OUTPUT_DIR / "people_counting_events.csv"
TIMELINE_CSV = OUTPUT_DIR / "occupancy_timeline.csv"
TRACKS_CSV = OUTPUT_DIR / "track_summary.csv"
CANDIDATES_CSV = INITIAL_DIR / "initial_occupancy_candidates.csv"


def create_tracker():
    return ByteTracker(
        high_threshold=0.35,
        low_threshold=0.10,
        match_threshold=0.30,
        max_age=30,
        min_hits=1,
    )


def get_track_value(track, name):
    if isinstance(track, dict):
        return track[name]
    return getattr(track, name)


def footpoint_from_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return ((float(x1) + float(x2)) / 2.0, float(y2))


def signed_distance(point, gate_line):
    (ax, ay), (bx, by) = gate_line
    length = math.hypot(bx - ax, by - ay)
    if length == 0:
        raise ValueError("Gate line has zero length.")

    cross = (
        (bx - ax) * (point[1] - ay)
        - (by - ay) * (point[0] - ax)
    )
    return cross / length


def estimate_initial_occupancy(detector, gate_line, inside_reference):
    """Estimate people already inside during the opening video frames."""

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 13.093

    tracker = create_tracker()
    observations = {}
    frame_index = 0
    max_frames = max(1, int(round(fps * 2.0)))
    first_seen_limit = int(round(fps * 0.5))

    inside_sign = 1 if signed_distance(
        inside_reference, gate_line
    ) > 0 else -1

    while frame_index < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections)

        for track in tracks:
            track_id = int(get_track_value(track, "track_id"))
            bbox = get_track_value(track, "bbox_xyxy")
            point = footpoint_from_bbox(bbox)

            if track_id not in observations:
                observations[track_id] = {
                    "first_seen_frame": frame_index,
                    "observations": 0,
                    "first_x": point[0],
                    "first_y": point[1],
                }

            observations[track_id]["observations"] += 1

        frame_index += 1

    cap.release()

    candidates = []
    for track_id, item in observations.items():
        first_point = (item["first_x"], item["first_y"])
        is_inside = (
            signed_distance(first_point, gate_line) * inside_sign > 0
        )
        appeared_at_start = (
            item["first_seen_frame"] <= first_seen_limit
        )
        stable = item["observations"] >= 5

        selected = is_inside and appeared_at_start and stable

        candidates.append({
            "track_id": track_id,
            "first_seen_frame": item["first_seen_frame"],
            "observations": item["observations"],
            "first_footpoint_x": round(item["first_x"], 2),
            "first_footpoint_y": round(item["first_y"], 2),
            "inside_gate": is_inside,
            "appeared_at_start": appeared_at_start,
            "stable_track": stable,
            "selected_for_initial_occupancy": selected,
        })

    INITIAL_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "track_id",
        "first_seen_frame",
        "observations",
        "first_footpoint_x",
        "first_footpoint_y",
        "inside_gate",
        "appeared_at_start",
        "stable_track",
        "selected_for_initial_occupancy",
    ]

    with CANDIDATES_CSV.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)

    selected_ids = [
        row["track_id"]
        for row in candidates
        if row["selected_for_initial_occupancy"]
    ]

    return len(selected_ids), selected_ids, fps


def main():
    for required in (VIDEO_PATH, MODEL_PATH, CONFIG_PATH):
        if not required.exists():
            raise FileNotFoundError(f"Required file not found: {required}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INITIAL_DIR.mkdir(parents=True, exist_ok=True)

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    gate_line = tuple(
        tuple(map(float, point))
        for point in config["gate_line"]
    )
    inside_reference = tuple(
        map(float, config["inside_reference_point"])
    )

    print("\nLoading ONNX person detector...")
    detector = ONNXPersonDetector(
        model_path=MODEL_PATH,
        confidence=0.25,
    )

    print("\nSTEP 1: ESTIMATING INITIAL OCCUPANCY")
    initial_occupancy, candidate_ids, fps = estimate_initial_occupancy(
        detector,
        gate_line,
        inside_reference,
    )

    print(f"Initial occupancy estimate: {initial_occupancy}")
    print(f"Candidate track IDs: {candidate_ids}")
    print(f"Candidate report: {CANDIDATES_CSV}")

    print("\nSTEP 2: PROCESSING COMPLETE VIDEO")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps and actual_fps > 0:
        fps = actual_fps

    writer = cv2.VideoWriter(
        str(ANNOTATED_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create annotated video.")

    tracker = create_tracker()
    counter = PeopleCounter(
        initial_occupancy=initial_occupancy,
        gate_line=gate_line,
        inside_reference_point=inside_reference,
    )

    track_stats = {}
    timeline_rows = []
    frame_index = 0
    previous_event_count = 0

    event_fields = [
        "event_id",
        "frame_index",
        "video_time_seconds",
        "track_id",
        "event",
        "footpoint_x",
        "footpoint_y",
        "occupancy_after_event",
    ]

    try:
        with EVENTS_CSV.open(
            "w", newline="", encoding="utf-8-sig"
        ) as event_file:

            event_writer = csv.DictWriter(
                event_file, fieldnames=event_fields
            )
            event_writer.writeheader()

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                detections = detector.detect(frame)
                tracks = tracker.update(detections)

                for track in tracks:
                    track_id = int(get_track_value(track, "track_id"))
                    bbox = get_track_value(track, "bbox_xyxy")
                    confidence = float(
                        get_track_value(track, "confidence")
                    )

                    x1, y1, x2, y2 = map(int, bbox)

                    if track_id not in track_stats:
                        track_stats[track_id] = {
                            "first_seen_frame": frame_index,
                            "last_seen_frame": frame_index,
                            "observations": 0,
                        }

                    stats = track_stats[track_id]
                    stats["last_seen_frame"] = frame_index
                    stats["observations"] += 1

                    counter.update(
                        track_id,
                        bbox,
                        frame_index,
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"ID {track_id} {confidence:.2f}",
                        (x1, max(20, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                # Draw the calibrated entrance line.
                p1 = tuple(map(int, gate_line[0]))
                p2 = tuple(map(int, gate_line[1]))
                cv2.line(frame, p1, p2, (255, 0, 255), 3)

                # Write any newly detected crossing events.
                current_events = counter.events

                while previous_event_count < len(current_events):
                    event = current_events[previous_event_count]
                    previous_event_count += 1

                    event_writer.writerow({
                        "event_id": previous_event_count,
                        "frame_index": event.frame_index,
                        "video_time_seconds": round(
                            event.frame_index / fps, 3
                        ),
                        "track_id": event.track_id,
                        "event": event.event,
                        "footpoint_x": round(event.footpoint[0], 2),
                        "footpoint_y": round(event.footpoint[1], 2),
                        "occupancy_after_event": counter.get_occupancy(),
                    })

                    event_text = (
                        f"{event.event} ID {event.track_id}"
                    )
                    cv2.putText(
                        frame,
                        event_text,
                        (30, 110 + (previous_event_count % 5) * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

                occupancy = counter.get_occupancy()
                counts = counter.get_event_counts()

                cv2.rectangle(
                    frame, (10, 10), (390, 90), (0, 0, 0), -1
                )
                cv2.putText(
                    frame,
                    f"Occupancy: {occupancy}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"IN: {counts['IN']}  OUT: {counts['OUT']}",
                    (20, 72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                writer.write(frame)

                timeline_rows.append({
                    "frame_index": frame_index,
                    "video_time_seconds": round(frame_index / fps, 3),
                    "occupancy": occupancy,
                    "in_count": counts["IN"],
                    "out_count": counts["OUT"],
                    "active_tracks": len(tracks),
                })

                if frame_index % 100 == 0:
                    print(
                        f"Frame {frame_index}/{total_frames} | "
                        f"Active tracks: {len(tracks)} | "
                        f"Occupancy: {occupancy} | "
                        f"IN: {counts['IN']} | OUT: {counts['OUT']}"
                    )

                frame_index += 1

    finally:
        cap.release()
        writer.release()

    with TIMELINE_CSV.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        fields = [
            "frame_index",
            "video_time_seconds",
            "occupancy",
            "in_count",
            "out_count",
            "active_tracks",
        ]
        output = csv.DictWriter(file, fieldnames=fields)
        output.writeheader()
        output.writerows(timeline_rows)

    with TRACKS_CSV.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        fields = [
            "track_id",
            "first_seen_frame",
            "last_seen_frame",
            "observations",
        ]
        output = csv.DictWriter(file, fieldnames=fields)
        output.writeheader()

        for track_id, stats in sorted(track_stats.items()):
            output.writerow({
                "track_id": track_id,
                **stats,
            })

    counts = counter.get_event_counts()

    print("\nPEOPLE COUNTING COMPLETE")
    print(f"Frames processed: {frame_index}")
    print(f"Initial occupancy: {initial_occupancy}")
    print(f"IN events: {counts['IN']}")
    print(f"OUT events: {counts['OUT']}")
    print(f"Final occupancy: {counter.get_occupancy()}")

    print("\nOUTPUT FILES")
    print(ANNOTATED_VIDEO)
    print(EVENTS_CSV)
    print(TIMELINE_CSV)
    print(TRACKS_CSV)
    print(CANDIDATES_CSV)


if __name__ == "__main__":
    main()

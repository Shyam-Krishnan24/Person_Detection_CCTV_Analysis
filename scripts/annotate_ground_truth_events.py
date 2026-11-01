"""Manually label true entrance/exit crossings for evaluation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.annotation_io import load_entrance_config, load_events_csv, project_path, save_events_csv

WINDOW = "Section 2 - Ground-truth event annotation"


def next_event_number(events: list[dict]) -> int:
    numbers = []
    for row in events:
        event_id = str(row.get("event_id", ""))
        if event_id.startswith("EVT") and event_id[3:].isdigit():
            numbers.append(int(event_id[3:]))
    return max(numbers, default=0) + 1


def draw_geometry(frame, config):
    overlay = frame.copy()
    polygon = np.asarray(config["approach_polygon"], dtype=np.int32)
    cv2.polylines(overlay, [polygon], True, (0, 220, 255), 2)
    gate = config["gate_line"]
    p1 = tuple(map(int, gate[0]))
    p2 = tuple(map(int, gate[1]))
    cv2.line(overlay, p1, p2, (0, 0, 255), 3)
    inside = tuple(map(int, config["inside_reference_point"]))
    cv2.circle(overlay, inside, 7, (0, 255, 0), -1)
    cv2.putText(overlay, "STORE / INSIDE", (inside[0] + 8, max(20, inside[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return overlay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="input/store_cctv.mp4")
    parser.add_argument("--geometry", default="config/entrance_band.yaml")
    parser.add_argument("--output", default="annotations/ground_truth_events.csv")
    args = parser.parse_args()

    video_path = project_path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")
    config = load_entrance_config(args.geometry)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 13.093
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1452
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width != int(config["frame_width"]) or height != int(config["frame_height"]):
        cap.release()
        raise SystemExit("Video dimensions do not match entrance geometry. Re-run calibrate_entrance.py.")

    events = load_events_csv(args.output)
    event_number = next_event_number(events)
    frame_index = 0
    playing = False
    speed = 1.0
    status = f"Loaded {len(events)} event labels. Pause near a crossing, step to it, then press I or O."
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print("Controls are displayed in the video window.")
    print("Enter a stable manual person ID (for example GT001) when prompted; leave blank if uncertain.")
    print("For a group crossing, create one event per person. Do not label disappearance away from the gate as OUT.")

    try:
        while 0 <= frame_index < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                status = "End of decoded video. Saving current labels."
                save_events_csv(args.output, events)
                print(f"Saved {len(events)} event labels to {project_path(args.output)}")
                break

            display = draw_geometry(frame, config)
            video_seconds = frame_index / fps
            recent = events[-3:]
            summary = " | ".join(
                f"{row['event_id']} {row['event']} @{float(row['video_time_seconds']):.2f}s"
                for row in recent
            ) or "No events labelled yet"
            lines = [
                f"Frame {frame_index}/{total_frames - 1} | video time {video_seconds:.3f}s | FPS {fps:.3f} | speed {speed:.2f}x | {'PLAY' if playing else 'PAUSED'}",
                "SPACE play/pause | , / . frame step | [ / ] seek 5s | +/- speed | I=IN | O=OUT",
                "U undo last event | S save CSV | Q save and quit | IN=outside to inside; OUT=inside to outside",
                summary,
                status,
            ]
            y = 22
            for line in lines:
                cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                            (255, 255, 255), 2, cv2.LINE_AA)
                y += 23
            scale = min(1.0, 1400 / width, 820 / height)
            if scale < 1:
                display = cv2.resize(display, (int(width * scale), int(height * scale)))
            cv2.imshow(WINDOW, display)

            delay_ms = max(1, int(round(1000 / fps / speed))) if playing else 30
            key = cv2.waitKey(delay_ms) & 0xFF
            if playing and key == 255:
                frame_index += 1
                continue

            if key in (ord(" "),):
                playing = not playing
            elif key == ord(","):
                playing = False
                frame_index = max(0, frame_index - 1)
            elif key == ord("."):
                playing = False
                frame_index = min(total_frames - 1, frame_index + 1)
            elif key == ord("["):
                playing = False
                frame_index = max(0, frame_index - int(round(5 * fps)))
            elif key == ord("]"):
                playing = False
                frame_index = min(total_frames - 1, frame_index + int(round(5 * fps)))
            elif key in (ord("+"), ord("=")):
                speed = min(4.0, speed * 1.25)
                status = f"Playback speed: {speed:.2f}x"
            elif key in (ord("-"), ord("_")):
                speed = max(0.1, speed / 1.25)
                status = f"Playback speed: {speed:.2f}x"
            elif key in (ord("i"), ord("I"), ord("o"), ord("O")):
                playing = False
                event_type = "IN" if key in (ord("i"), ord("I")) else "OUT"
                print(f"\n{event_type} at frame {frame_index}, video time {video_seconds:.3f}s")
                person_id = input("Person ID (e.g. GT001; blank if uncertain): ").strip()
                notes = input("Notes (optional; press Enter to skip): ").strip()
                event_id = f"EVT{event_number:03d}"
                events.append({
                    "event_id": event_id,
                    "frame_index": frame_index,
                    "video_time_seconds": round(video_seconds, 3),
                    "event": event_type,
                    "person_id": person_id,
                    "notes": notes,
                })
                event_number += 1
                status = f"Recorded {event_id}: {event_type} at {video_seconds:.3f}s"
            elif key in (ord("u"), ord("U")):
                if events:
                    removed = events.pop()
                    event_number = next_event_number(events)
                    status = f"Undid {removed['event_id']} ({removed['event']})"
                else:
                    status = "No event to undo."
            elif key in (ord("s"), ord("S")):
                save_events_csv(args.output, events)
                status = f"Saved {len(events)} event labels to {args.output}"
                print(status)
            elif key in (ord("q"), ord("Q"), 27):
                save_events_csv(args.output, events)
                print(f"Saved {len(events)} event labels to {project_path(args.output)}")
                break
            elif not playing and key == 255:
                continue

            if playing and key != 255:
                # If a key is pressed during playback, process it but do not skip a frame.
                continue
        else:
            save_events_csv(args.output, events)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

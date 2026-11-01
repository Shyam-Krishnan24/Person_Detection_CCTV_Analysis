"""Shared read/write helpers for Section 2 ground-truth annotations."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALID_ROLES = {"STAFF", "CUSTOMER", "UNKNOWN"}
VALID_EVENTS = {"IN", "OUT"}
EVENT_COLUMNS = [
    "event_id",
    "frame_index",
    "video_time_seconds",
    "event",
    "person_id",
    "notes",
]


def project_path(relative_path: str | Path) -> Path:
    """Resolve a project-relative path against the project root."""
    path = Path(relative_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_box_xyxy(box: Iterable[int], width: int, height: int) -> list[int]:
    """Validate and clamp an [x1, y1, x2, y2] box to the image bounds."""
    values = list(box)
    if len(values) != 4:
        raise ValueError("A bounding box must contain exactly four values: x1,y1,x2,y2")
    x1, y1, x2, y2 = (int(round(float(v))) for v in values)
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Bounding box has zero or negative area")
    return [x1, y1, x2, y2]


def build_initial_occupancy_payload(
    people: list[dict[str, Any]], width: int, height: int, video_file: str = "input/store_cctv.mp4"
) -> dict[str, Any]:
    """Build a validated frame-zero annotation record."""
    cleaned: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for person in people:
        person_id = str(person["person_id"]).strip()
        role = str(person.get("role", "UNKNOWN")).upper()
        if not person_id or person_id in seen_ids:
            raise ValueError(f"Missing or duplicate person_id: {person_id!r}")
        if role not in VALID_ROLES:
            raise ValueError(f"Invalid role {role!r}; expected one of {sorted(VALID_ROLES)}")
        seen_ids.add(person_id)
        cleaned.append(
            {
                "person_id": person_id,
                "bbox_xyxy": validate_box_xyxy(person["bbox_xyxy"], width, height),
                "role": role,
                "notes": str(person.get("notes", "")).strip(),
            }
        )

    return {
        "schema_version": 1,
        "video_file": video_file,
        "frame_index": 0,
        "video_time_seconds": 0.0,
        "frame_width": int(width),
        "frame_height": int(height),
        "annotation_status": "manually_annotated",
        "visible_person_count": len(cleaned),
        "people": cleaned,
        "counting_note": (
            "Count distinct visible people in frame zero, including staff and children. "
            "A child being carried is a separate person when visibly distinguishable."
        ),
    }


def save_initial_occupancy(
    output_json: str | Path,
    people: list[dict[str, Any]],
    width: int,
    height: int,
    video_file: str = "input/store_cctv.mp4",
) -> dict[str, Any]:
    payload = build_initial_occupancy_payload(people, width, height, video_file)
    output_path = project_path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_initial_occupancy(path: str | Path) -> dict[str, Any] | None:
    file_path = project_path(path)
    if not file_path.exists():
        return None
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if payload.get("frame_index") != 0 or not isinstance(payload.get("people"), list):
        raise ValueError(f"Unsupported initial occupancy file: {file_path}")
    return payload


def save_events_csv(path: str | Path, events: list[dict[str, Any]]) -> None:
    """Write event labels in chronological order with a stable CSV schema."""
    cleaned: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    for row in events:
        event_id = str(row.get("event_id", "")).strip()
        event = str(row.get("event", "")).upper()
        if not event_id or event_id in seen_event_ids:
            raise ValueError(f"Missing or duplicate event_id: {event_id!r}")
        if event not in VALID_EVENTS:
            raise ValueError(f"Invalid event {event!r}; expected IN or OUT")
        frame_index = int(row["frame_index"])
        video_time = float(row["video_time_seconds"])
        if frame_index < 0 or video_time < 0:
            raise ValueError("Frame index and video time must be non-negative")
        seen_event_ids.add(event_id)
        cleaned.append(
            {
                "event_id": event_id,
                "frame_index": frame_index,
                "video_time_seconds": f"{video_time:.3f}",
                "event": event,
                "person_id": str(row.get("person_id", "")).strip(),
                "notes": str(row.get("notes", "")).strip(),
            }
        )
    cleaned.sort(key=lambda item: (float(item["video_time_seconds"]), item["event_id"]))

    output_path = project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_COLUMNS)
        writer.writeheader()
        writer.writerows(cleaned)


def load_events_csv(path: str | Path) -> list[dict[str, Any]]:
    input_path = project_path(path)
    if not input_path.exists():
        return []
    with input_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EVENT_COLUMNS:
            raise ValueError(
                f"Unexpected event CSV columns in {input_path}. Expected: {EVENT_COLUMNS}"
            )
        rows = list(reader)
    return [
        {
            "event_id": row["event_id"],
            "frame_index": int(row["frame_index"]),
            "video_time_seconds": float(row["video_time_seconds"]),
            "event": row["event"].upper(),
            "person_id": row.get("person_id", ""),
            "notes": row.get("notes", ""),
        }
        for row in rows
    ]


def save_entrance_config(
    path: str | Path,
    width: int,
    height: int,
    gate_line: list[list[int]],
    approach_polygon: list[list[int]],
    inside_reference_point: list[int],
) -> dict[str, Any]:
    if len(gate_line) != 2:
        raise ValueError("Gate line must have exactly two endpoints")
    if len(approach_polygon) < 3:
        raise ValueError("Approach polygon must have at least three vertices")
    if len(inside_reference_point) != 2:
        raise ValueError("Inside reference point must contain x and y")

    payload = {
        "schema_version": 1,
        "frame_width": int(width),
        "frame_height": int(height),
        "coordinate_system": "pixel coordinates; origin at top-left",
        "gate_line": [[int(x), int(y)] for x, y in gate_line],
        "approach_polygon": [[int(x), int(y)] for x, y in approach_polygon],
        "inside_reference_point": [int(inside_reference_point[0]), int(inside_reference_point[1])],
        "event_direction": {
            "IN": "cross from the side opposite inside_reference_point to the inside side",
            "OUT": "cross from the inside side to the opposite side",
        },
        "annotation_note": (
            "Geometry is a manually calibrated candidate gate. Recheck against visible crossings. "
            "Use the tracked person's bottom-centre/footpoint for later crossing tests."
        ),
    }
    output_path = project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return payload


def load_entrance_config(path: str | Path) -> dict[str, Any]:
    input_path = project_path(path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Entrance geometry not found: {input_path}. Run scripts/calibrate_entrance.py first."
        )
    payload = yaml.safe_load(input_path.read_text(encoding="utf-8")) or {}
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported entrance config schema in {input_path}")
    return payload

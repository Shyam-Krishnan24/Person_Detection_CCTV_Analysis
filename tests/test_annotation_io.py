"""Unit tests for Section 2 annotation validation and persistence."""
import csv

import pytest

from src.annotation_io import (
    EVENT_COLUMNS,
    build_initial_occupancy_payload,
    save_events_csv,
    load_events_csv,
)


def test_initial_occupancy_counts_distinct_annotated_people():
    payload = build_initial_occupancy_payload(
        [
            {"person_id": "B001", "bbox_xyxy": [10, 20, 40, 80], "role": "CUSTOMER"},
            {"person_id": "B002", "bbox_xyxy": [50, 20, 75, 90], "role": "UNKNOWN"},
        ],
        width=1270,
        height=720,
    )
    assert payload["frame_index"] == 0
    assert payload["visible_person_count"] == 2


def test_duplicate_person_ids_are_rejected():
    people = [
        {"person_id": "B001", "bbox_xyxy": [10, 20, 40, 80], "role": "CUSTOMER"},
        {"person_id": "B001", "bbox_xyxy": [50, 20, 75, 90], "role": "UNKNOWN"},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        build_initial_occupancy_payload(people, 1270, 720)


def test_invalid_bounding_box_is_rejected():
    with pytest.raises(ValueError, match="zero or negative area"):
        build_initial_occupancy_payload(
            [{"person_id": "B001", "bbox_xyxy": [20, 20, 20, 60]}], 1270, 720
        )


def test_event_csv_is_sorted_and_round_trips(tmp_path, monkeypatch):
    # Save directly to an absolute temporary path; project paths preserve absolute paths.
    output = tmp_path / "ground_truth_events.csv"
    events = [
        {"event_id": "EVT002", "frame_index": 52, "video_time_seconds": 4.0, "event": "OUT", "person_id": "GT002"},
        {"event_id": "EVT001", "frame_index": 13, "video_time_seconds": 1.0, "event": "IN", "person_id": "GT001"},
    ]
    save_events_csv(output, events)
    with output.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == EVENT_COLUMNS
        rows = list(reader)
    assert [row["event_id"] for row in rows] == ["EVT001", "EVT002"]
    loaded = load_events_csv(output)
    assert [row["event"] for row in loaded] == ["IN", "OUT"]

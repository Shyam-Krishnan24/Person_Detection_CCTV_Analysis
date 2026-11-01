
"""Export people-counting events in the required submission format."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCE_EVENTS = ROOT / "reports" / "people_counter" / "people_counting_events.csv"
CANDIDATES = ROOT / "reports" / "initial_occupancy" / "initial_occupancy_candidates.csv"
OUTPUT = ROOT / "reports" / "people_counter" / "events.csv"


def truthy(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def format_timestamp(seconds):
    total_seconds = max(0, int(float(seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main():
    if not SOURCE_EVENTS.exists():
        raise FileNotFoundError(SOURCE_EVENTS)

    if not CANDIDATES.exists():
        raise FileNotFoundError(CANDIDATES)

    with CANDIDATES.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        candidates = list(csv.DictReader(file))

    initial_occupancy = sum(
        truthy(row["selected_for_initial_occupancy"])
        for row in candidates
    )

    with SOURCE_EVENTS.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        events = list(csv.DictReader(file))

    events.sort(
        key=lambda row: (
            int(row["frame_index"]),
            int(row["event_id"]),
        )
    )

    headcount = initial_occupancy
    output_rows = []

    for event in events:
        direction = event["event"].strip().upper()

        if direction == "IN":
            headcount += 1
        elif direction == "OUT":
            headcount = max(0, headcount - 1)
        else:
            continue

        output_rows.append({
            "timestamp": format_timestamp(
                event["video_time_seconds"]
            ),
            "person_id": event["track_id"],
            "event": direction,
            "headcount": headcount,
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        fields = ["timestamp", "person_id", "event", "headcount"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    print("SUBMISSION CSV CREATED")
    print(f"Initial occupancy: {initial_occupancy}")
    print(f"Events exported: {len(output_rows)}")
    print(f"Final headcount: {headcount}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()


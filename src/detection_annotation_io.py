"""Read/write helpers for manually labelled detector-evaluation frames."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def validate_box_xyxy(box: Iterable[int], width: int, height: int) -> list[int]:
    values = list(box)
    if len(values) != 4:
        raise ValueError("A box must contain x1, y1, x2, y2")
    x1, y1, x2, y2 = (int(round(float(v))) for v in values)
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Box must have positive width and height")
    return [x1, y1, x2, y2]


def build_detection_payload(
    video_file: str,
    width: int,
    height: int,
    fps: float,
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    clean_frames: list[dict[str, Any]] = []
    seen_frames: set[int] = set()
    for frame in sorted(frames, key=lambda item: int(item["frame_index"])):
        frame_index = int(frame["frame_index"])
        if frame_index < 0 or frame_index in seen_frames:
            raise ValueError(f"Negative or duplicate frame index: {frame_index}")
        seen_frames.add(frame_index)
        annotations = []
        for idx, ann in enumerate(frame.get("annotations", []), start=1):
            annotations.append(
                {
                    "annotation_id": str(ann.get("annotation_id") or f"F{frame_index:06d}_A{idx:03d}"),
                    "class_name": "person",
                    "bbox_xyxy": validate_box_xyxy(ann["bbox_xyxy"], width, height),
                }
            )
        clean_frames.append(
            {
                "frame_index": frame_index,
                "video_time_seconds": round(float(frame["video_time_seconds"]), 3),
                "image_width": int(width),
                "image_height": int(height),
                "annotations": annotations,
                "person_count": len(annotations),
            }
        )
    return {
        "schema_version": 1,
        "annotation_purpose": "detector_evaluation_ground_truth_only",
        "video_file": video_file,
        "video": {"width": int(width), "height": int(height), "fps": float(fps)},
        "class_names": ["person"],
        "frames": clean_frames,
        "frame_count": len(clean_frames),
        "total_person_boxes": sum(item["person_count"] for item in clean_frames),
    }


def save_detection_payload(path: str | Path, payload: dict[str, Any]) -> Path:
    output = project_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def load_detection_payload(path: str | Path) -> dict[str, Any]:
    source = project_path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("annotation_purpose") != "detector_evaluation_ground_truth_only":
        raise ValueError(f"Unsupported detection annotation file: {source}")
    return payload

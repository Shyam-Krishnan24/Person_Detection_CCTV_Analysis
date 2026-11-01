"""Draw person boxes on selected video frames for detector evaluation only.

Mouse: drag a tight box around one visible person.
Keys: N next sample, P previous sample, U undo last box on this frame,
      D delete selected box, click inside a box to select it, S save, Q save/quit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection_annotation_io import build_detection_payload, project_path, save_detection_payload

WINDOW = "Section 2 - Detector ground-truth frames"


class FrameAnnotator:
    def __init__(self, frames: list[dict], width: int, height: int, fps: float, video_file: str, output: str):
        self.frames = frames
        self.width = width
        self.height = height
        self.fps = fps
        self.video_file = video_file
        self.output = output
        self.index = 0
        self.selected: int | None = None
        self.drag_start: tuple[int, int] | None = None
        self.drag_end: tuple[int, int] | None = None
        self.dragging = False
        self.status = "Draw one box per distinct visible person. Include clearly visible partial people."
        self.dirty = False

    def scale(self) -> float:
        return min(1.0, 1400 / self.width, 820 / self.height)

    def to_source(self, x: int, y: int) -> tuple[int, int]:
        scale = self.scale()
        return (max(0, min(self.width - 1, int(round(x / scale)))),
                max(0, min(self.height - 1, int(round(y / scale)))))

    @property
    def current(self) -> dict:
        return self.frames[self.index]

    def mouse(self, event, x, y, _flags, _param):
        point = self.to_source(x, y)
        annotations = self.current["annotations"]
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = point
            self.drag_end = point
            self.dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.drag_end = point
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.drag_end = point
            if self.drag_start is None:
                return
            x1, y1 = self.drag_start
            x2, y2 = self.drag_end
            if abs(x2 - x1) < 6 or abs(y2 - y1) < 6:
                candidates = []
                for idx, ann in enumerate(annotations):
                    bx1, by1, bx2, by2 = ann["bbox_xyxy"]
                    if bx1 <= point[0] <= bx2 and by1 <= point[1] <= by2:
                        candidates.append(((bx2 - bx1) * (by2 - by1), idx))
                self.selected = min(candidates)[1] if candidates else None
                self.status = "Selected box." if self.selected is not None else "No box selected; drag to add one."
            else:
                left, right = sorted((x1, x2))
                top, bottom = sorted((y1, y2))
                if right > left and bottom > top:
                    annotations.append({"bbox_xyxy": [left, top, right, bottom]})
                    self.selected = len(annotations) - 1
                    self.dirty = True
                    self.status = f"Added person box. This frame: {len(annotations)} people."
            self.drag_start = None
            self.drag_end = None

    def render(self):
        canvas = self.current["image"].copy()
        for idx, ann in enumerate(self.current["annotations"]):
            x1, y1, x2, y2 = ann["bbox_xyxy"]
            color = (0, 165, 255) if idx == self.selected else (0, 220, 0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, f"P{idx + 1:02d}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if self.dragging and self.drag_start and self.drag_end:
            cv2.rectangle(canvas, self.drag_start, self.drag_end, (255, 180, 0), 2)
        scale = self.scale()
        canvas = cv2.resize(canvas, (int(self.width * scale), int(self.height * scale)))
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 100), (20, 20, 20), -1)
        canvas = cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0)
        frame = self.current
        lines = [
            f"Sample {self.index + 1}/{len(self.frames)} | frame {frame['frame_index']} | t={frame['video_time_seconds']:.2f}s | boxes={len(frame['annotations'])}",
            "Drag=add box | Click box=select | U=undo | D=delete | N/P=next/previous sample",
            "S=save JSON | Q=save and quit | Label each distinct visible person once",
            self.status + (" [unsaved]" if self.dirty else " [saved]"),
        ]
        y = 20
        for line in lines:
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1, cv2.LINE_AA)
            y += 23
        return canvas

    def save(self):
        serial_frames = []
        for frame in self.frames:
            serial_frames.append({
                "frame_index": frame["frame_index"],
                "video_time_seconds": frame["video_time_seconds"],
                "annotations": frame["annotations"],
            })
        payload = build_detection_payload(self.video_file, self.width, self.height, self.fps, serial_frames)
        saved_path = save_detection_payload(self.output, payload)
        self.dirty = False
        self.status = f"Saved {payload['total_person_boxes']} boxes across {payload['frame_count']} frames."
        print(f"Saved detector ground truth: {saved_path}")
        print(f"Frames: {payload['frame_count']} | person boxes: {payload['total_person_boxes']}")


def parse_times(raw: str) -> list[float]:
    try:
        values = sorted({float(part.strip()) for part in raw.split(",") if part.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Times must be comma-separated seconds, e.g. 0,10,20") from exc
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("Provide at least one non-negative sample time")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="input/store_cctv.mp4")
    parser.add_argument("--times", type=parse_times, default=parse_times("0,10,20,30,40,50,60,70,80,90,100"),
                        help="Comma-separated sample times in seconds")
    parser.add_argument("--output", default="annotations/detection_labels.json")
    args = parser.parse_args()

    video_path = project_path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        cap.release()
        raise SystemExit("Video FPS is invalid")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1452

    frames = []
    seen_indices = set()
    for seconds in args.times:
        frame_index = min(total - 1, int(round(seconds * fps)))
        if frame_index in seen_indices:
            continue
        seen_indices.add(frame_index)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, image = cap.read()
        if ok and image is not None:
            frames.append({
                "frame_index": frame_index,
                "video_time_seconds": frame_index / fps,
                "image": image,
                "annotations": [],
            })
    cap.release()
    if not frames:
        raise SystemExit("None of the requested sample frames could be decoded")

    # Resume existing work when the JSON file exists and the selected frame list matches.
    output_path = project_path(args.output)
    if output_path.exists():
        import json
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        old_by_idx = {int(item["frame_index"]): item for item in existing.get("frames", [])}
        for frame in frames:
            old = old_by_idx.get(frame["frame_index"])
            if old:
                frame["annotations"] = old.get("annotations", [])

    app = FrameAnnotator(frames, width, height, fps, args.video, args.output)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, app.mouse)
    print("Annotate a small evaluation set. This data is NOT used as the system's initial headcount.")
    print("For a carried child, draw a separate box only when the child is visually distinguishable.")
    try:
        while True:
            cv2.imshow(WINDOW, app.render())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("n"), ord("N")):
                app.index = min(len(frames) - 1, app.index + 1)
                app.selected = None
                app.status = "Review this sample and annotate each visible person."
            elif key in (ord("p"), ord("P")):
                app.index = max(0, app.index - 1)
                app.selected = None
                app.status = "Review this sample and annotate each visible person."
            elif key in (ord("u"), ord("U")):
                if app.current["annotations"]:
                    app.current["annotations"].pop()
                    app.selected = None
                    app.dirty = True
                    app.status = "Removed the last box on this frame."
            elif key in (ord("d"), ord("D")):
                if app.selected is not None:
                    app.current["annotations"].pop(app.selected)
                    app.selected = None
                    app.dirty = True
                    app.status = "Deleted selected box."
            elif key in (ord("s"), ord("S")):
                app.save()
            elif key in (ord("q"), ord("Q"), 27):
                if app.dirty or not output_path.exists():
                    app.save()
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

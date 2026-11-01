"""Draw the entrance gate, approach polygon and inside reference point."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.annotation_io import project_path, save_entrance_config

WINDOW = "Section 2 - Entrance calibration"


class EntranceCalibrator:
    def __init__(self, frame):
        self.frame = frame
        self.height, self.width = frame.shape[:2]
        self.mode = "idle"
        self.gate_line: list[tuple[int, int]] = []
        self.polygon: list[tuple[int, int]] = []
        self.inside_point: tuple[int, int] | None = None
        self.status = "Press L to draw the gate line, P for approach polygon, I for inside point."

    def scale(self):
        return min(1.0, 1400 / self.width, 820 / self.height)

    def to_source(self, x, y):
        scale = self.scale()
        return (max(0, min(self.width - 1, int(round(x / scale)))),
                max(0, min(self.height - 1, int(round(y / scale)))))

    def mouse(self, event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        point = self.to_source(x, y)
        if self.mode == "line":
            self.gate_line.append(point)
            if len(self.gate_line) == 2:
                self.mode = "idle"
                self.status = "Gate line set. Draw the entrance approach polygon with P."
            else:
                self.status = "First gate endpoint set. Click the second endpoint."
        elif self.mode == "polygon":
            self.polygon.append(point)
            self.status = f"Approach polygon: {len(self.polygon)} vertices. Press Enter when complete."
        elif self.mode == "inside":
            self.inside_point = point
            self.mode = "idle"
            self.status = "Inside reference point set. Press S to save after checking geometry."

    def render(self):
        canvas = self.frame.copy()
        if self.polygon:
            pts = [self.to_display(p) for p in self.polygon]
            for p in pts:
                cv2.circle(canvas, self.to_display(self.to_source(*p)), 4, (0, 220, 255), -1)
            if len(pts) > 1:
                cv2.polylines(canvas, [__import__("numpy").array(pts, dtype="int32")], False, (0, 220, 255), 2)
            if len(self.polygon) >= 3 and self.mode != "polygon":
                cv2.polylines(canvas, [__import__("numpy").array(pts, dtype="int32")], True, (0, 220, 255), 2)
        if self.gate_line:
            for point in self.gate_line:
                cv2.circle(canvas, point, 6, (0, 0, 255), -1)
            if len(self.gate_line) == 2:
                cv2.line(canvas, self.gate_line[0], self.gate_line[1], (0, 0, 255), 3)
        if self.inside_point:
            cv2.circle(canvas, self.inside_point, 8, (0, 255, 0), -1)
            cv2.putText(canvas, "INSIDE", (self.inside_point[0] + 8, self.inside_point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        scale = self.scale()
        canvas = cv2.resize(canvas, (int(self.width * scale), int(self.height * scale)))
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 106), (15, 15, 15), -1)
        canvas = cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0)
        lines = [
            "L: gate line (2 clicks) | P: approach polygon (click vertices, Enter to finish) | I: inside point",
            "R: reset all | U: undo last polygon vertex | S: save | Q: quit",
            f"Line endpoints: {len(self.gate_line)}/2 | Polygon vertices: {len(self.polygon)} | Inside point: {self.inside_point}",
            self.status,
        ]
        y = 22
        for line in lines:
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            y += 24
        return canvas

    def to_display(self, point):
        scale = self.scale()
        return int(round(point[0] * scale)), int(round(point[1] * scale))

    def save(self, output):
        if len(self.gate_line) != 2:
            raise ValueError("Set exactly two gate-line endpoints with L before saving.")
        if len(self.polygon) < 3:
            raise ValueError("Draw at least three approach-polygon vertices with P before saving.")
        if self.inside_point is None:
            raise ValueError("Set an inside reference point with I before saving.")
        save_entrance_config(
            output,
            self.width,
            self.height,
            [list(p) for p in self.gate_line],
            [list(p) for p in self.polygon],
            list(self.inside_point),
        )
        print(f"Entrance geometry saved: {project_path(output)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="input/store_cctv.mp4")
    parser.add_argument("--output", default="config/entrance_band.yaml")
    args = parser.parse_args()
    video_path = project_path(args.video)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Could not decode frame zero")

    app = EntranceCalibrator(frame)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, app.mouse)
    print("Calibrate with the actual entrance movement in the bottom-edge sequence.")
    print("Draw only the main doorway path, not the whole lower edge or checkout area.")
    try:
        while True:
            cv2.imshow(WINDOW, app.render())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("l"), ord("L")):
                app.gate_line = []
                app.mode = "line"
                app.status = "Click the two endpoints of the crossing gate line."
            elif key in (ord("p"), ord("P")):
                app.polygon = []
                app.mode = "polygon"
                app.status = "Click at least three vertices around the entrance approach. Press Enter to finish."
            elif key in (ord("i"), ord("I")):
                app.mode = "inside"
                app.status = "Click a point clearly inside the shop side of the gate."
            elif key in (13, 10) and app.mode == "polygon":
                if len(app.polygon) >= 3:
                    app.mode = "idle"
                    app.status = "Approach polygon completed. Set the inside point with I."
                else:
                    app.status = "Polygon needs at least three vertices."
            elif key in (ord("u"), ord("U")) and app.polygon:
                app.polygon.pop()
                app.status = "Removed the last polygon vertex."
            elif key in (ord("r"), ord("R")):
                app.mode = "idle"
                app.gate_line = []
                app.polygon = []
                app.inside_point = None
                app.status = "Geometry cleared. Start again with L, P and I."
            elif key in (ord("s"), ord("S")):
                try:
                    app.save(args.output)
                    app.status = f"Saved {args.output}. Review it before event annotation."
                except (ValueError, OSError) as exc:
                    app.status = str(exc)
                    print(f"Not saved: {exc}")
            elif key in (ord("q"), ord("Q"), 27):
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

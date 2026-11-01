
"""Lightweight ByteTrack-style multi-person tracker.

Uses ONNX detections, a constant-velocity Kalman filter,
IoU association, and two-stage confidence matching.
Does not depend on PyTorch or Ultralytics.
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    confidence: float
    hits: int = 1
    age: int = 1
    missed: int = 0

    @property
    def bbox_xyxy(self):
        cx, cy, width, height = self.state[:4]
        width = max(1.0, width)
        height = max(1.0, height)

        return [
            float(cx - width / 2),
            float(cy - height / 2),
            float(cx + width / 2),
            float(cy + height / 2),
        ]


class ByteTracker:
    def __init__(
        self,
        high_threshold=0.50,
        low_threshold=0.10,
        match_threshold=0.30,
        max_age=30,
        min_hits=1,
    ):
        if not 0 <= low_threshold <= high_threshold <= 1:
            raise ValueError(
                "Require 0 <= low_threshold <= high_threshold <= 1"
            )

        if not 0 <= match_threshold <= 1:
            raise ValueError("match_threshold must be between 0 and 1")

        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.match_threshold = match_threshold
        self.max_age = max_age
        self.min_hits = min_hits

        self.tracks = []
        self.next_id = 1

        # State: [cx, cy, width, height, vx, vy, vw, vh]
        self.F = np.eye(8, dtype=np.float64)

        for i in range(4):
            self.F[i, i + 4] = 1.0

        self.H = np.zeros((4, 8), dtype=np.float64)
        self.H[:4, :4] = np.eye(4)

        self.Q = np.eye(8, dtype=np.float64) * 0.05
        self.R = np.eye(4, dtype=np.float64) * 4.0
        self.I = np.eye(8, dtype=np.float64)

    @staticmethod
    def _measurement(box):
        x1, y1, x2, y2 = map(float, box)

        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)

        return np.array(
            [
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                width,
                height,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        intersection = (
            max(0.0, ix2 - ix1) *
            max(0.0, iy2 - iy1)
        )

        area_a = (
            max(0.0, ax2 - ax1) *
            max(0.0, ay2 - ay1)
        )

        area_b = (
            max(0.0, bx2 - bx1) *
            max(0.0, by2 - by1)
        )

        union = area_a + area_b - intersection

        return intersection / union if union > 0 else 0.0

    def _predict(self, track):
        track.state = self.F @ track.state

        track.covariance = (
            self.F @ track.covariance @ self.F.T + self.Q
        )

        track.age += 1
        track.missed += 1

    def _correct(self, track, detection):
        z = self._measurement(detection.bbox_xyxy)

        innovation = z - self.H @ track.state

        S = (
            self.H @ track.covariance @ self.H.T
            + self.R
        )

        K = (
            track.covariance @ self.H.T
            @ np.linalg.inv(S)
        )

        track.state = track.state + K @ innovation

        track.covariance = (
            self.I - K @ self.H
        ) @ track.covariance

        track.state[2:4] = np.maximum(
            track.state[2:4], 1.0
        )

        track.confidence = float(detection.confidence)
        track.hits += 1
        track.missed = 0

    def _associate(self, tracks, detections):
        if not tracks or not detections:
            return (
                [],
                list(range(len(tracks))),
                list(range(len(detections))),
            )

        cost = np.full(
            (len(tracks), len(detections)),
            1e6,
            dtype=np.float64,
        )

        similarities = np.zeros(
            (len(tracks), len(detections)),
            dtype=np.float64,
        )

        for i, track in enumerate(tracks):
            tb = track.bbox_xyxy

            tx = (tb[0] + tb[2]) / 2
            ty = (tb[1] + tb[3]) / 2

            tw = max(1.0, tb[2] - tb[0])
            th = max(1.0, tb[3] - tb[1])

            track_diag = max(1.0, np.hypot(tw, th))

            for j, detection in enumerate(detections):
                db = detection.bbox_xyxy

                dx = (db[0] + db[2]) / 2
                dy = (db[1] + db[3]) / 2

                dw = max(1.0, db[2] - db[0])
                dh = max(1.0, db[3] - db[1])

                iou = self._iou(tb, db)

                center_distance = float(
                    np.hypot(tx - dx, ty - dy)
                )

                distance_ratio = center_distance / max(
                    track_diag,
                    np.hypot(dw, dh),
                    1.0,
                )

                width_ratio = max(tw, dw) / min(tw, dw)
                height_ratio = max(th, dh) / min(th, dh)

                if width_ratio > 2.5 or height_ratio > 2.5:
                    continue

                if (
                    iou < self.match_threshold
                    and distance_ratio > 0.25
                ):
                    continue

                similarity = (
                    0.7 * iou
                    + 0.3 * max(0.0, 1.0 - distance_ratio)
                )

                similarities[i, j] = similarity
                cost[i, j] = 1.0 - similarity

        rows, cols = linear_sum_assignment(cost)

        matches = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_detections = set(range(len(detections)))

        for row, col in zip(rows, cols):
            if cost[row, col] >= 1e5:
                continue

            if similarities[row, col] >= self.match_threshold:
                matches.append((row, col))
                unmatched_tracks.discard(row)
                unmatched_detections.discard(col)

        return (
            matches,
            sorted(unmatched_tracks),
            sorted(unmatched_detections),
        )

    def _start_track(self, detection):
        z = self._measurement(detection.bbox_xyxy)

        state = np.zeros(8, dtype=np.float64)
        state[:4] = z

        track = Track(
            track_id=self.next_id,
            state=state,
            covariance=np.eye(8, dtype=np.float64) * 10.0,
            confidence=float(detection.confidence),
        )

        self.next_id += 1
        self.tracks.append(track)

    def update(self, detections):
        """Update tracker with a list of PersonDetection objects."""

        for track in self.tracks:
            self._predict(track)

        high = [
            d for d in detections
            if d.confidence >= self.high_threshold
        ]

        low = [
            d for d in detections
            if self.low_threshold <= d.confidence < self.high_threshold
        ]

        # Stage 1: high-confidence association.
        matches, unmatched_tracks, unmatched_high = (
            self._associate(self.tracks, high)
        )

        for track_idx, detection_idx in matches:
            self._correct(
                self.tracks[track_idx],
                high[detection_idx],
            )

        # Stage 2: low-confidence recovery.
        remaining_tracks = [
            self.tracks[i] for i in unmatched_tracks
        ]

        low_matches, _, _ = self._associate(
            remaining_tracks,
            low,
        )

        for local_track_idx, detection_idx in low_matches:
            track = remaining_tracks[local_track_idx]
            self._correct(track, low[detection_idx])

        # Create tracks only from unmatched high-confidence detections.
        for detection_idx in unmatched_high:
            detection = high[detection_idx]

            duplicate = any(
                self._iou(
                    track.bbox_xyxy,
                    detection.bbox_xyxy,
                ) >= 0.65
                for track in self.tracks
            )

            if not duplicate:
                self._start_track(detection)

        # Remove expired tracks.
        self.tracks = [
            track for track in self.tracks
            if track.missed <= self.max_age
        ]

        return [
            track for track in self.tracks
            if track.missed == 0
            and track.hits >= self.min_hits
        ]


"""Finite entrance-line crossing counter with corrected direction logic."""

from dataclasses import dataclass
import math


@dataclass
class CrossingEvent:
    track_id: int
    event: str
    frame_index: int
    footpoint: tuple


class PeopleCounter:
    def __init__(
        self,
        initial_occupancy=0,
        gate_line=((957, 715), (1100, 553)),
        inside_reference_point=(705, 505),
        deadband=5.0,
        cooldown_frames=5,
    ):
        self.initial_occupancy = max(0, int(initial_occupancy))
        self.occupancy = self.initial_occupancy

        self.gate_line = gate_line
        self.inside_reference_point = inside_reference_point
        self.deadband = float(deadband)
        self.cooldown_frames = int(cooldown_frames)

        a, b = gate_line
        self.line_length = math.hypot(b[0] - a[0], b[1] - a[1])

        if self.line_length == 0:
            raise ValueError("Gate line endpoints must be different")

        reference_distance = self._signed_distance(
            inside_reference_point
        )

        self.inside_sign = (
            1 if reference_distance > 0 else -1
        )

        self.previous = {}
        self.last_event_frame = {}

        self.in_count = 0
        self.out_count = 0
        self.events = []

    def _signed_distance(self, point):
        a, b = self.gate_line

        cross = (
            (b[0] - a[0]) * (point[1] - a[1])
            - (b[1] - a[1]) * (point[0] - a[0])
        )

        return cross / self.line_length

    @staticmethod
    def _footpoint(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, y2)

    def _crosses_finite_gate(self, p1, p2):
        a, b = self.gate_line

        r = (p2[0] - p1[0], p2[1] - p1[1])
        s = (b[0] - a[0], b[1] - a[1])

        denominator = r[0] * s[1] - r[1] * s[0]

        if abs(denominator) < 1e-9:
            return False

        q = (a[0] - p1[0], a[1] - p1[1])

        t = (
            q[0] * s[1] - q[1] * s[0]
        ) / denominator

        u = (
            q[0] * r[1] - q[1] * r[0]
        ) / denominator

        return 0 <= t <= 1 and 0 <= u <= 1

    def update(self, track_id, bbox_xyxy, frame_index):
        point = self._footpoint(bbox_xyxy)
        distance = self._signed_distance(point)

        if abs(distance) <= self.deadband:
            return None

        current_sign = 1 if distance > 0 else -1

        previous = self.previous.get(track_id)

        self.previous[track_id] = {
            "point": point,
            "sign": current_sign,
        }

        if previous is None:
            return None

        if previous["sign"] == current_sign:
            return None

        if not self._crosses_finite_gate(
            previous["point"], point
        ):
            return None

        last_frame = self.last_event_frame.get(
            track_id, -10**9
        )

        if frame_index - last_frame < self.cooldown_frames:
            return None

        if previous["sign"] != self.inside_sign:
            event_type = "IN"
            self.in_count += 1
            self.occupancy += 1
        else:
            event_type = "OUT"
            self.out_count += 1
            self.occupancy = max(0, self.occupancy - 1)

        self.last_event_frame[track_id] = frame_index

        event = CrossingEvent(
            track_id=track_id,
            event=event_type,
            frame_index=frame_index,
            footpoint=point,
        )

        self.events.append(event)
        return event

    def get_occupancy(self):
        return self.occupancy

    def get_event_counts(self):
        return {
            "IN": self.in_count,
            "OUT": self.out_count,
        }

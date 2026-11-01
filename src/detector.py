"""Reusable Ultralytics person detector for the Store People Counter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ultralytics import YOLO


@dataclass
class PersonDetection:
    """One person detection in pixel coordinates (x1, y1, x2, y2)."""
    bbox_xyxy: list[float]
    confidence: float
    class_id: int = 0


class PersonDetector:
    """Loads a YOLO model and returns only COCO person detections."""

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        confidence: float = 0.25,
        iou: float = 0.45,
        image_size: int = 640,
        device: str | int = 0,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not 0.0 <= iou <= 1.0:
            raise ValueError("iou must be between 0 and 1")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self.weights = str(weights)
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.image_size = int(image_size)
        self.device = device
        self.model = YOLO(self.weights)

    def detect(self, frame: Any) -> list[PersonDetection]:
        """Detect people in an image array."""
        if frame is None or getattr(frame, "size", 0) == 0:
            raise ValueError("frame is empty")
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            classes=[0],
            device=self.device,
            verbose=False,
        )
        detections: list[PersonDetection] = []
        if not results or results[0].boxes is None:
            return detections
        for box in results[0].boxes:
            coords = box.xyxy[0].detach().cpu().tolist()
            score = float(box.conf[0].detach().cpu().item())
            class_id = int(box.cls[0].detach().cpu().item())
            if class_id == 0:
                detections.append(
                    PersonDetection(
                        bbox_xyxy=[float(value) for value in coords],
                        confidence=score,
                        class_id=class_id,
                    )
                )
        return detections

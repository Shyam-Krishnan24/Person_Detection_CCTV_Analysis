"""YOLOv8 ONNX person detector using ONNX Runtime CPU only.

Uses a slightly stricter person NMS threshold to reduce duplicate boxes.
"""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class PersonDetection:
    bbox_xyxy: list[float]
    confidence: float
    class_id: int = 0


class ONNXPersonDetector:
    def __init__(self, model_path="models/yolov8n.onnx", confidence=0.25, iou=0.40, input_size=640):
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if not 0 <= iou <= 1:
            raise ValueError("iou must be between 0 and 1")
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        self.confidence, self.iou, self.input_size = float(confidence), float(iou), int(input_size)
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, frame):
        if frame is None or frame.size == 0:
            raise ValueError("frame is empty")
        h, w = frame.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        px, py = (self.input_size - nw) // 2, (self.input_size - nh) // 2
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[py:py+nh, px:px+nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None]
        tensor = np.ascontiguousarray(tensor)

        output = self.session.run(None, {self.input_name: tensor})[0]
        # YOLOv8: [1, 84, 8400] = xywh + 80 class scores; no objectness column.
        pred = np.squeeze(output, axis=0).T
        scores_all = pred[:, 4:]
        scores = scores_all[:, 0]  # COCO person class
        ids = np.where(scores >= self.confidence)[0]
        if not len(ids):
            return []

        xywh = pred[ids, :4]
        boxes = np.empty_like(xywh)
        boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)

        nms_boxes = [[float(x1), float(y1), float(max(0, x2-x1)), float(max(0, y2-y1))]
                     for x1, y1, x2, y2 in boxes]
        keep = cv2.dnn.NMSBoxes(nms_boxes, scores[ids].astype(float).tolist(),
                                self.confidence, self.iou)
        detections = []
        for i in np.asarray(keep).reshape(-1) if len(keep) else []:
            x1, y1, x2, y2 = boxes[int(i)]
            if x2 > x1 and y2 > y1:
                detections.append(PersonDetection(
                    [round(float(x1), 2), round(float(y1), 2),
                     round(float(x2), 2), round(float(y2), 2)],
                    float(scores[ids[int(i)]]), 0))
        return detections

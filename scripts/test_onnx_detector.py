"""Test ONNX person detection on the 11 annotated sample timestamps."""
import argparse
import csv
from pathlib import Path
import cv2
from src.onnx_detector import ONNXPersonDetector

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="input/store_cctv.mp4")
    parser.add_argument("--model", default="models/yolov8n.onnx")
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()

    video = ROOT / args.video
    model = ROOT / args.model
    out_dir = ROOT / "reports" / "onnx_detector_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    detector = ONNXPersonDetector(model, confidence=args.confidence)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise SystemExit("Invalid video FPS")

    rows, total = [], 0
    for seconds in range(0, 101, 10):
        frame_index = round(seconds * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            print(f"Skipping {seconds}s: frame unavailable")
            continue
        detections = detector.detect(frame)
        total += len(detections)
        for d in detections:
            x1, y1, x2, y2 = map(int, d.bbox_xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(frame, f"person {d.confidence:.2f}", (x1, max(18, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
            rows.append({"timestamp_seconds": seconds, "frame_index": frame_index,
                         "confidence": round(d.confidence, 4),
                         "x1": d.bbox_xyxy[0], "y1": d.bbox_xyxy[1],
                         "x2": d.bbox_xyxy[2], "y2": d.bbox_xyxy[3]})
        cv2.putText(frame, f"{seconds}s | people: {len(detections)}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
        cv2.imwrite(str(out_dir / f"frame_{seconds:03d}s.jpg"), frame)
        print(f"{seconds:3d}s | frame {frame_index:4d} | people: {len(detections)}")
    cap.release()

    with (out_dir / "detections.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp_seconds", "frame_index",
            "confidence", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Total detections across sampled frames: {total}")
    print(f"CSV: {out_dir / 'detections.csv'}")
    print(f"Annotated frames: {out_dir}")


if __name__ == "__main__":
    main()

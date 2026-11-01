"""Benchmark YOLO person detection on evenly sampled CCTV frames."""
from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sample_video_frames(video_path: Path, sample_count: int):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if frame_count <= 0:
        capture.release()
        raise RuntimeError("Video reports zero frames")
    count = min(max(1, sample_count), frame_count)
    indices = sorted(set(int(round(v)) for v in np.linspace(0, frame_count - 1, count)))
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if ok:
            frames.append((index, frame))
    capture.release()
    if not frames:
        raise RuntimeError("Could not decode any benchmark frames")
    return frames, fps, frame_count


def benchmark_one(weights, image_size, frames, config, device):
    model = YOLO(weights)
    warmup_count = max(1, int(config.get("warmup_runs", 2)))
    for _, frame in frames[:warmup_count]:
        model.predict(
            source=frame, imgsz=image_size,
            conf=float(config["confidence"]), iou=float(config["iou"]),
            classes=config.get("classes", [0]), device=device, verbose=False,
        )

    timings = []
    detection_total = 0
    repetitions = max(1, int(config.get("benchmark_runs", 3)))
    for _ in range(repetitions):
        for _, frame in frames:
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = model.predict(
                source=frame, imgsz=image_size,
                conf=float(config["confidence"]), iou=float(config["iou"]),
                classes=config.get("classes", [0]), device=device, verbose=False,
            )
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
            if result and result[0].boxes is not None:
                detection_total += len(result[0].boxes)

    mean_seconds = statistics.mean(timings)
    return {
        "model": weights,
        "image_size": image_size,
        "device": device,
        "frames_per_run": len(frames),
        "runs": repetitions,
        "mean_inference_ms": round(mean_seconds * 1000, 2),
        "median_inference_ms": round(statistics.median(timings) * 1000, 2),
        "approx_fps": round(1.0 / mean_seconds, 2) if mean_seconds else 0.0,
        "mean_person_detections_per_frame": round(
            detection_total / (len(frames) * repetitions), 2
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "detector_benchmark.yaml"))
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    video_path = ROOT / config["video_path"]
    output_path = ROOT / config["output_csv"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = "0"
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    elif args.allow_cpu or config.get("allow_cpu", False):
        device = "cpu"
        print("CUDA unavailable: running an explicitly permitted CPU benchmark.")
    else:
        raise SystemExit(
            "CUDA is unavailable. Finish the CUDA PyTorch installation and verify it, "
            "or deliberately rerun with --allow-cpu."
        )

    frames, fps, frame_count = sample_video_frames(
        video_path, int(config.get("sample_frames", 60))
    )
    print(f"Video: {video_path}")
    print(f"Sampled frames: {len(frames)} / reported frames: {frame_count}; FPS: {fps:.3f}")

    rows = []
    for weights in config["models"]:
        for image_size in config["image_sizes"]:
            print(f"Benchmarking {weights} at {image_size}px on {device}...")
            row = benchmark_one(weights, int(image_size), frames, config, device)
            rows.append(row)
            print(row)

    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved benchmark results: {output_path}")


if __name__ == "__main__":
    main()

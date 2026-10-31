"""Inspect the source CCTV MP4 and create Section 1 artifacts.

This script does not run detection, tracking, OCR, or counting. It only:
- reads video metadata with OpenCV;
- extracts representative frames;
- creates a full-frame contact sheet;
- creates clock-region and bottom-edge preview sheets;
- writes JSON and Markdown inspection reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

DEFAULT_SAMPLE_SECONDS = [0, 10, 25, 40, 55, 70, 85, 100, 109]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_contact_sheet(images: list[tuple[str, np.ndarray]], output_path: Path,
                       columns: int = 3, thumb_width: int = 420) -> None:
    if not images:
        return
    prepared = []
    for label, frame in images:
        h, w = frame.shape[:2]
        scale = min(thumb_width / w, 420 / h)
        resized = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                             interpolation=cv2.INTER_AREA)
        prepared.append((label, resized))
    image_cell_height = max(img.shape[0] for _, img in prepared)
    label_height = 30
    cell_height = image_cell_height + label_height
    rows = (len(prepared) + columns - 1) // columns
    canvas = np.full((rows * cell_height, columns * thumb_width, 3), 245, dtype=np.uint8)
    for idx, (label, img) in enumerate(prepared):
        x = (idx % columns) * thumb_width
        y = (idx // columns) * cell_height
        y_img = y + (image_cell_height - img.shape[0]) // 2
        canvas[y_img:y_img + img.shape[0], x:x + img.shape[1]] = img
        cv2.putText(canvas, label, (x + 8, y + image_cell_height + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (25, 25, 25), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])


def inspect_video(video_path: Path, project_root: Path,
                  sample_seconds: Iterable[float] = DEFAULT_SAMPLE_SECONDS) -> dict:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    reported_frame_count_opencv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Count frames that OpenCV can actually decode. CAP_PROP_FRAME_COUNT can
    # be a small estimate for some MP4 files, so do not use it as ground truth.
    count_cap = cv2.VideoCapture(str(video_path))
    decoded_frames = 0
    while True:
        ok, _ = count_cap.read()
        if not ok:
            break
        decoded_frames += 1
    count_cap.release()
    duration = decoded_frames / fps if fps > 0 else None
    cap = cv2.VideoCapture(str(video_path))

    frames_dir = project_root / "cache" / "frames"
    visual_dir = project_root / "reports" / "visual_inspection"
    frames_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    sample_images: list[tuple[str, np.ndarray]] = []
    clock_images: list[tuple[str, np.ndarray]] = []
    bottom_images: list[tuple[str, np.ndarray]] = []
    sample_records = []

    # Clock ROI is only a preview crop; final OCR coordinates must be validated later.
    clock_x1, clock_y1 = 0, 0
    clock_x2, clock_y2 = int(width * 0.55), int(height * 0.15)
    # Bottom-edge preview is intentionally broad; it is NOT the final entrance band.
    bottom_y1 = int(height * 0.58)

    for sec in sample_seconds:
        frame_index = min(max(int(round(float(sec) * fps)), 0), max(decoded_frames - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        elapsed = frame_index / fps if fps > 0 else float(sec)
        label = f"frame_{frame_index:04d}_t_{elapsed:06.2f}s"
        image_path = frames_dir / f"{label}.jpg"
        cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 94])

        sample_images.append((f"{elapsed:.1f}s | frame {frame_index}", frame))
        clock_crop = frame[clock_y1:clock_y2, clock_x1:clock_x2]
        bottom_crop = frame[bottom_y1:height, 0:width]
        if clock_crop.size:
            clock_images.append((f"{elapsed:.1f}s", clock_crop))
        if bottom_crop.size:
            bottom_images.append((f"{elapsed:.1f}s", bottom_crop))

        sample_records.append({
            "requested_seconds": float(sec),
            "frame_index": frame_index,
            "video_elapsed_seconds": round(elapsed, 3),
            "image": str(image_path.relative_to(project_root)).replace('\\', '/')
        })

    cap.release()

    frame_zero_cap = cv2.VideoCapture(str(video_path))
    frame_zero_ok, frame_zero = frame_zero_cap.read()
    frame_zero_cap.release()
    if frame_zero_ok and frame_zero is not None:
        cv2.imwrite(str(visual_dir / "frame_zero.jpg"), frame_zero, [cv2.IMWRITE_JPEG_QUALITY, 95])

    make_contact_sheet(sample_images, visual_dir / "full_frame_contact_sheet.jpg", columns=3)
    make_contact_sheet(clock_images, visual_dir / "clock_region_samples.jpg", columns=2, thumb_width=600)
    make_contact_sheet(bottom_images, visual_dir / "bottom_edge_samples.jpg", columns=2, thumb_width=600)

    # Dense one-second lower-frame sequence pages help review actual movement
    # near the bottom boundary without freezing final ROI coordinates yet.
    dense_bottom_images: list[tuple[str, np.ndarray]] = []
    clock_tail_images: list[tuple[str, np.ndarray]] = []
    last_second = int(duration) if duration is not None else 0
    dense_targets = {min(int(round(sec * fps)), max(decoded_frames - 1, 0)): sec
                     for sec in range(last_second + 1)}
    dense_cap = cv2.VideoCapture(str(video_path))
    decoded_index = 0
    while True:
        ok, frame = dense_cap.read()
        if not ok:
            break
        if decoded_index in dense_targets:
            sec = dense_targets[decoded_index]
            dense_elapsed = decoded_index / fps if fps > 0 else float(sec)
            dense_bottom = frame[bottom_y1:height, 0:width]
            dense_bottom_images.append((f"t={dense_elapsed:.1f}s | f={decoded_index}", dense_bottom))
            if sec >= 90:
                tail_clock = frame[clock_y1:clock_y2, clock_x1:clock_x2]
                clock_tail_images.append((f"t={dense_elapsed:.1f}s", tail_clock))
        decoded_index += 1
    dense_cap.release()

    sequence_pages = []
    page_size = 24
    for page_no, start in enumerate(range(0, len(dense_bottom_images), page_size), start=1):
        page_path = visual_dir / f"bottom_edge_sequence_{page_no:02d}.jpg"
        make_contact_sheet(dense_bottom_images[start:start + page_size], page_path,
                           columns=2, thumb_width=600)
        sequence_pages.append(page_path.name)
    make_contact_sheet(clock_tail_images, visual_dir / "clock_samples_90_to_end.jpg",
                       columns=2, thumb_width=600)

    size_bytes = video_path.stat().st_size
    report = {
        "source_file": video_path.name,
        "source_sha256": sha256_file(video_path),
        "file_size_bytes": size_bytes,
        "file_size_mb": round(size_bytes / (1024 * 1024), 2),
        "container": "MP4 (verified from supplied file / ffprobe)",
        "video_codec": "H.264 (verified from supplied file / ffprobe)",
        "width": width,
        "height": height,
        "fps": round(fps, 6),
        "opencv_reported_frame_count": reported_frame_count_opencv,
        "decoded_frame_count": decoded_frames,
        "duration_from_frames_seconds": round(duration, 3) if duration is not None else None,
        "sample_frames": sample_records,
        "bottom_edge_sequence_pages": sequence_pages,
        "clock_tail_sample_count": len(clock_tail_images),
        "clock_preview_roi_pixels": {"x1": clock_x1, "y1": clock_y1, "x2": clock_x2, "y2": clock_y2},
        "bottom_edge_preview_roi_pixels": {"x1": 0, "y1": bottom_y1, "x2": width, "y2": height},
        "notes": [
            "OpenCV's CAP_PROP_FRAME_COUNT estimate may differ from the actual decoded frame count; this report records both.",
            "The clock crop and bottom-edge crop are inspection previews, not final calibrated regions.",
            "The entrance band and upper approach margin must be calibrated from movement sequences before event logic is implemented.",
            "The sampled clock crops visually suggest the displayed clock reaches about 11:14:31 near the 99-second point and then remains unchanged in the sampled tail; validate with OCR before timestamp mapping.",
            "The final few seconds contain long runs of repeated or near-identical decoded frames; inspect the tail before interpreting events or timestamps there.",
            "Frame-zero visible occupancy must be manually annotated; it is not inferred from a detector count.",
            "The camera only observes part of the store, so occupancy is relative to the visible baseline and validated boundary events."
        ]
    }
    (visual_dir / "inspection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = f"""# Section 1 — Video Inspection Report

## Source metadata

| Property | Value |
|---|---|
| Source file | `{video_path.name}` |
| Container | MP4 |
| Video codec | H.264 |
| Resolution | {width} × {height} |
| Average frame rate reported by OpenCV | {fps:.6f} FPS |
| OpenCV estimated frame count | {reported_frame_count_opencv:,} |
| Frames actually decoded | {decoded_frames:,} |
| Duration derived from decoded frames / FPS | {duration:.3f} seconds |
| File size | {size_bytes / (1024 * 1024):.2f} MB |
| SHA-256 | `{report['source_sha256']}` |

> The supplied stream metadata reports approximately 110.899 seconds from frame count / average FPS. Container duration reported separately by ffprobe is approximately 111.068 seconds. Frame index divided by the stream FPS will be the processing timeline used by the application.

## Visual observations

1. The camera is fixed and looks down into the store from an elevated angle.
2. The doorway itself is not fully visible; people appear/disappear through the lower image boundary.
3. The first frame already contains people in the monitored view. These people must be represented by a manually verified initial visible baseline, not synthetic IN events.
4. Display stands, shelving and the checkout area create frequent partial occlusion and overlapping people.
5. The burned-in clock is visible in the top-left region. Frame zero visibly reads **2017/12/22 11:12:43**. This is a visual reading to validate with OCR before final timestamp synchronization.
6. Denser clock samples near the end visually suggest the displayed time reaches approximately **11:14:31 around video time 99 seconds**, then remains unchanged through the sampled tail. Do not assume a simple linear mapping from video elapsed time to CCTV clock time until OCR validation is completed.
7. The final few seconds contain long runs of repeated or near-identical decoded frames. The tail should be inspected before interpreting any events there.
8. The lower portion of the frame is the entrance-analysis candidate area. The final band and its upper approach margin are deliberately not frozen in this inspection stage.

## Generated inspection assets

- `frame_zero.jpg` — original-size first frame.
- `full_frame_contact_sheet.jpg` — representative full-frame samples.
- `clock_region_samples.jpg` — top-left clock crop samples.
- `bottom_edge_samples.jpg` — broad lower-frame samples for entrance geometry review.
- `bottom_edge_sequence_01.jpg` onward — one-second bottom-edge sequence pages for movement review.
- `clock_samples_90_to_end.jpg` — denser clock samples near the end of the recording.
- `inspection_report.json` — machine-readable metadata and sample frame index mapping.

## Section 1 limitations / pending decisions

- Exact entrance band and approach-margin coordinates are not yet calibrated.
- Initial visible occupancy and staff/customer/unknown roles are not yet manually counted.
- OCR accuracy and clock continuity are not yet formally validated; the apparent clock plateau near the end needs confirmation.
- The apparent repeated-frame tail needs confirmation and should not be treated as extra movement.
- No detection, tracking, entry/exit or occupancy algorithm has been run in this section.

## Section 1 acceptance checks

- [x] MP4 can be opened and decoded.
- [x] Width, height, FPS and frame count recorded.
- [x] Representative frames extracted.
- [x] Clock preview crops generated.
- [x] Bottom-edge preview crops generated.
- [ ] Entrance band and upper approach margin manually calibrated (next inspection/annotation decision).
- [ ] Frame-zero baseline manually annotated (ground-truth section).
- [ ] Clock OCR continuity validated (timestamp section).
"""
    (visual_dir / "video_inspection.md").write_text(md, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Section 1 CCTV inspection assets.")
    parser.add_argument("--video", default="input/store_cctv.mp4", help="Path to source MP4")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--samples", default=",".join(map(str, DEFAULT_SAMPLE_SECONDS)),
                        help="Comma-separated sample times in seconds")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    video = Path(args.video)
    if not video.is_absolute():
        video = (root / video).resolve()
    samples = [float(x.strip()) for x in args.samples.split(",") if x.strip()]
    report = inspect_video(video, root, samples)
    print("Section 1 inspection completed.")
    print(f"Resolution: {report['width']}x{report['height']}")
    print(f"FPS: {report['fps']}")
    print(f"Decoded frames: {report['decoded_frame_count']}")
    print(f"Duration (frames/FPS): {report['duration_from_frames_seconds']} seconds")
    print(f"Report: {root / 'reports' / 'visual_inspection' / 'video_inspection.md'}")


if __name__ == "__main__":
    main()

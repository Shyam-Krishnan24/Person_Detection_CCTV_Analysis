
@'
# Store CCTV People Counter

An offline computer-vision system that detects and tracks people in a recorded CCTV video, identifies candidate entrance and exit crossings, and estimates store occupancy.

## 1. Features

- Person detection using YOLOv8n in ONNX format.
- ONNX Runtime CPU inference.
- Bounding boxes and tracking IDs.
- Entrance/exit line visualization.
- IN and OUT event detection using tracked footpoints.
- Automatic initial occupancy estimation.
- Live occupancy overlay in the processed video.
- CSV event log and occupancy timeline.
- Offline processing; no cloud service or live CCTV connection required.

## 2. Technology

| Component | Technology |
|---|---|
| Person detection | YOLOv8n ONNX |
| Model runtime | ONNX Runtime |
| Video processing | OpenCV |
| Object tracking | Custom ByteTrack-style tracker |
| Association | IoU, motion prediction and Hungarian assignment |
| Data processing | Python, NumPy, CSV |
| Entrance calibration | YAML configuration |

The tracker is a lightweight custom implementation inspired by ByteTrack-style two-stage association. It is not the official Ultralytics ByteTrack implementation.

## 3. Project structure

```text
Store_People_Counter/
|
|-- input/
|   |-- store_cctv.mp4
|
|-- models/
|   |-- yolov8n.onnx
|
|-- config/
|   |-- entrance_band.yaml
|
|-- src/
|   |-- onnx_detector.py
|   |-- byte_tracker.py
|   |-- people_counter.py
|
|-- scripts/
|   |-- run_people_counter.py
|   |-- export_submission_events.py
|
|-- annotations/
|   |-- detection_labels.json
|   |-- detection_labels_fullbody.json
|   |-- ground_truth_events.csv
|
|-- reports/
    |-- initial_occupancy/
    |-- people_counter/
        |-- annotated_people_counter.mp4
        |-- people_counting_events.csv
        |-- events.csv
        |-- occupancy_timeline.csv
        |-- track_summary.csv
```

## 4. Requirements

- Windows 10/11
- Python 3.12
- OpenCV
- NumPy
- ONNX Runtime
- PyYAML
- SciPy

Install the required packages in your Python environment:

```powershell
python -m pip install numpy opencv-python onnxruntime PyYAML scipy
```

The project uses ONNX Runtime CPU execution. PyTorch and a CUDA-enabled GPU are not required.

## 5. Input files

Place the CCTV video here:

```text
input/store_cctv.mp4
```

Place the YOLOv8n ONNX model here:

```text
models/yolov8n.onnx
```

The entrance line and inside reference point are configured in:

```text
config/entrance_band.yaml
```

The line must be calibrated to the entrance visible in the CCTV footage.

## 6. Run the system

Activate the virtual environment if it is not already active:

```powershell
.venv\Scripts\Activate.ps1
```

From the project root, run:

```powershell
python -m scripts.run_people_counter
```

The program performs two passes:

1. Analyses the opening two seconds to estimate initial occupancy.
2. Processes the complete video, detects and tracks people, checks entrance-line crossings, and renders the output.

Export the required submission CSV:

```powershell
python -m scripts.export_submission_events
```

## 7. Output files

All main outputs are saved under:

```text
reports/people_counter/
```

| Output | Description |
|---|---|
| annotated_people_counter.mp4 | Annotated video with bounding boxes, IDs, entrance line and occupancy overlay |
| people_counting_events.csv | Detailed event log, including frame number and footpoint coordinates |
| events.csv | Submission CSV with timestamp, person_id, event and headcount |
| occupancy_timeline.csv | Frame-by-frame occupancy and event totals |
| track_summary.csv | Track start, end and observation statistics |

The initial occupancy candidate report is saved at:

```text
reports/initial_occupancy/initial_occupancy_candidates.csv
```

## 8. Counting logic

The estimated occupancy is calculated as:

```text
Current Occupancy = Initial Occupancy + IN Events - OUT Events
```

The system uses the bottom-centre point of each person's bounding box as the footpoint. It checks whether a tracked person's movement crosses the calibrated entrance segment.

The initial occupancy is estimated from stable tracks observed near the beginning of the video and classified on the inside of the entrance line.

## 9. Known limitations

- Partial occlusion can cause missed detections or fragmented tracking IDs.
- People entering close together may overlap and be detected as one person.
- A person who disappears behind an obstruction may receive a new tracking ID after reappearing.
- Incorrect footpoint positions or an imperfect entrance calibration can cause missed or incorrect crossing events.
- Initial occupancy is an estimate, not a manually confirmed count.
- Staff and customers are not separately classified. All detected people are treated equally.
- The system has been evaluated on a limited set of manually annotated frames and events. Its event counts should not be treated as fully accurate.

## 10. Future improvements

- Improve tracking through longer occlusions.
- Use appearance-based re-identification to reduce ID changes.
- Improve group-entry separation.
- Validate entrance geometry and crossing direction against more annotated events.
- Add staff/customer classification using a suitable labeled dataset.
- Evaluate precision, recall, F1-score and occupancy error on a larger ground-truth sample.

## 11. Execution mode

This project processes a recorded CCTV video offline. It does not connect to a live camera, transmit footage to a cloud service, or identify people by their real-world identity.
'@ | Set-Content -Encoding UTF8 README.md

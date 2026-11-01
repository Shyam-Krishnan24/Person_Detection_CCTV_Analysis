
## Project Deliverables

| Deliverable | Description | Link |
|---|---|---|
| Annotated Output Video | Detection boxes, tracking IDs, entrance/exit line and live occupancy count | [View Output Video](reports/people_counter/annotated_people_counter.mp4) |
| Event CSV | Timestamp, person ID, IN/OUT event and headcount | [View Events CSV](reports/people_counter/events.csv) |
| Approach and Limitations | One-page summary of the methodology, evaluation, failures and future improvements | [Read Approach Note](reports/approach_and_limitations.md) |
| Scenario Handling | Explanation of occlusion, group entry, re-entry and staff/customer classification | [Read Scenario Handling](reports/scenario_handling.md) |

> **Note:** The annotated video is generated locally and excluded from Git because of its file size. Include it separately when submitting the project.

## How to Run

### 1. Clone or open the project

Open the project directory in VS Code.

### 2. Create and activate the virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install numpy opencv-python onnxruntime PyYAML scipy
```

### 4. Run the people-counting system

Ensure the input video is available at:

`input/store_cctv.mp4`

Ensure the ONNX model is available at:

`models/yolov8n.onnx`

Run:

```powershell
python -m scripts.run_people_counter
```

### 5. Export the submission CSV

```powershell
python -m scripts.export_submission_events
```

### 6. View the generated results

The output files are saved under:

`reports/people_counter/`

- `annotated_people_counter.mp4` — annotated video.
- `events.csv` — final submission event CSV.
- `people_counting_events.csv` — detailed counting events.
- `occupancy_timeline.csv` — occupancy changes over time.
- `track_summary.csv` — tracking information.

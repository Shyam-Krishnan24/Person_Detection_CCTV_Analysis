# Section 2 — Ground-Truth Annotation (Revised)

## Purpose

Create a small manually labelled reference set for evaluating the detector and event counter.

These labels are evaluation ground truth only. They do not initialise or override the system's automatic headcount.

## Keep and use these files

| Relative path | Action | Purpose |
|---|---|---|
| `src/detection_annotation_io.py` | NEW | Validate and save selected-frame person boxes. |
| `scripts/annotate_detection_frames.py` | NEW | Draw person boxes on selected sample frames for detector evaluation. |
| `scripts/calibrate_entrance.py` | KEEP | Mark candidate gate line, approach polygon and inside point. |
| `scripts/annotate_ground_truth_events.py` | KEEP | Label genuine IN/OUT crossings for event evaluation. |
| `src/annotation_io.py` | KEEP | Shared helpers for entrance geometry and event CSV. |
| `tests/test_detection_annotation_io.py` | NEW | Test the revised detection-label format. |

## Do not use these old baseline-only files

- `scripts/annotate_initial_occupancy.py` — do not run; it manually labels frame-zero occupancy and is obsolete under the revised requirement.
- `annotations/initial_occupancy.json` — do not populate; it is not the runtime initial count.
- `annotations/ground_truth_frames/frame_0000_initial_occupancy.jpg` — old baseline preview; not needed.

You may leave these files in the project; they do not affect the revised pipeline as long as the old script is not run.

## Task 2.1 — Selected-frame person-box labels

From PowerShell:

    cd F:\Store_People_Counter
    .\.venv\Scripts\Activate.ps1
    python scripts\annotate_detection_frames.py

Default sample times are 0, 10, 20, ... 100 seconds.

To choose different times:

    python scripts\annotate_detection_frames.py --times 5,15,25,35,45,55,65,75,85,95,105

### Controls

- Drag: draw a tight box around one distinct visible person.
- Click inside a box: select it.
- U: undo last box on current frame.
- D: delete selected box.
- N / P: next / previous sample frame.
- S: save JSON.
- Q: save and quit.

Count every distinct visible person in each selected frame, including staff.

Draw a separate box around a carried child only if the child is visibly distinguishable.

Do not invent boxes for people fully hidden by occlusion.

Output:

    annotations/detection_labels.json

## Task 2.2 — Entrance geometry

Run after reviewing the lower-edge sequence:

    python scripts\calibrate_entrance.py

Set the crossing gate line, narrow entrance approach polygon and inside reference point.

Save to:

    config/entrance_band.yaml

This is geometry configuration, not a manually entered occupancy count.

## Task 2.3 — Ground-truth IN/OUT event labels

Run only after the entrance geometry is saved:

    python scripts\annotate_ground_truth_events.py

Mark actual crossings only.

- Use video frame/time, not the CCTV clock.
- Label one event per person in a group.
- If identity is uncertain, leave the manual person ID blank.
- Do not label disappearance away from the entrance as OUT.

Output:

    annotations/ground_truth_events.csv

## Verify

    python -m pytest -q tests\test_detection_annotation_io.py tests\test_annotation_io.py

The expected total includes the revised detection-label tests and existing shared annotation tests.

## Critical runtime rule

The production pipeline will estimate initial headcount automatically from detector/tracker output during initialisation.

The manually labelled JSON and event CSV are only a reference set used for measuring accuracy.

They must never be read as the production occupancy baseline.


@'
# Store CCTV People Counting — Approach and Limitations

## Objective
Develop an offline computer-vision system to detect and track people in recorded store CCTV footage, identify entry and exit movements, and estimate occupancy.

## Approach
The system uses the open-source YOLOv8n model exported to ONNX format, executed through ONNX Runtime on the CPU. OpenCV reads and processes the video. A custom ByteTrack-style tracker associates detections across frames using motion prediction, IoU matching and two-stage confidence association. Each tracked person's bottom-centre bounding-box point is used to test crossings of a manually calibrated entrance line.

Initial occupancy is estimated automatically from stable tracks observed during the opening two seconds. Occupancy is updated using:

**Occupancy = Initial Occupancy + IN events - OUT events**

The processed video displays bounding boxes, tracking IDs, the entrance line, and the current estimated headcount. CSV outputs record crossing events and occupancy over time.

## What failed and limitations
The first detector evaluation used upper-body annotations against full-person predictions, producing an invalid comparison. Full-body annotations were created for a more appropriate comparison. On the 11 sampled frames, the detector achieved 72.00% precision, 67.92% recall and 69.90% F1 at IoU 0.50. These are limited-sample results, not full-video performance.

Tracking IDs sometimes fragment when people overlap or disappear behind shelves and displays. The entrance is partly outside the camera view, and the calibrated line and footpoint crossing method do not capture every manually annotated event. In the current run, the system estimated 7 people initially, detected 5 IN and 2 OUT events, and estimated a final occupancy of 10. The manual event annotations recorded 6 IN and 4 OUT events, showing that counting accuracy needs improvement.

## Handling specific scenarios
- **Occlusion:** The tracker uses motion prediction and high-/low-confidence detection association to maintain tracks where possible. Long or complete occlusions can still break a track and create a new ID.
- **Groups entering together:** Each visible person is detected and tracked separately when the detector can distinguish them. Overlapping people may be merged into one detection or missed, so group counts are not guaranteed.
- **People re-entering:** A person crossing out and later crossing in can generate another event if the tracker and crossing logic maintain a usable track. Re-identification after a lost track is not implemented, so the same person may receive a different ID.
- **Staff versus customers:** No reliable staff/customer classifier is implemented. All detected people contribute equally to occupancy. A future version could use a labeled uniform/appearance dataset and a validated classification model.

## What I would improve with more time
I would improve tracking through occlusion with appearance-based re-identification, refine the entrance region using more annotated crossings, separate overlapping people, and validate the system against a larger ground-truth set. I would also add staff/customer classification only after collecting suitable labeled examples. Finally, I would report event precision/recall and occupancy error across the complete video.

## Reproducibility
Run `python -m scripts.run_people_counter` to generate the annotated video and detailed reports. Run `python -m scripts.export_submission_events` to create the required four-column `events.csv`.
'@ | Set-Content -Encoding UTF8 reports\approach_and_limitations.md


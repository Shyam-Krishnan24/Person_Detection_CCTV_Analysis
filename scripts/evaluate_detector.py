
"""Evaluate ONNX person detections against manually annotated boxes."""

import csv
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]

GROUND_TRUTH_FILE = ROOT / "annotations" / "detection_labels_fullbody.json"
PREDICTIONS_FILE = ROOT / "reports" / "onnx_detector_test" / "detections.csv"

IOU_THRESHOLD = 0.50


def calculate_iou(box_a, box_b):
    """Calculate Intersection over Union for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def match_detections(ground_truth, predictions):
    """Greedily match each ground-truth box to at most one prediction."""
    candidates = []

    for gt_index, gt_box in enumerate(ground_truth):
        for pred_index, pred_box in enumerate(predictions):
            iou = calculate_iou(gt_box, pred_box)
            if iou >= IOU_THRESHOLD:
                candidates.append((iou, gt_index, pred_index))

    candidates.sort(reverse=True)

    matched_gt = set()
    matched_pred = set()

    for _, gt_index, pred_index in candidates:
        if gt_index not in matched_gt and pred_index not in matched_pred:
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)

    tp = len(matched_gt)
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp

    return tp, fp, fn


def main():
    with GROUND_TRUTH_FILE.open(encoding="utf-8") as f:
        ground_truth_data = json.load(f)

    ground_truth = {
        frame["frame_index"]: [
            annotation["bbox_xyxy"]
            for annotation in frame["annotations"]
            if annotation["class_name"] == "person"
        ]
        for frame in ground_truth_data["frames"]
    }

    predictions = defaultdict(list)

    with PREDICTIONS_FILE.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            frame_index = int(row["frame_index"])
            box = [
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
            ]
            predictions[frame_index].append(box)

    total_tp = total_fp = total_fn = 0

    print("\nPER-FRAME EVALUATION")
    print("-" * 65)
    print(f"{'Frame':>8} {'GT':>6} {'Pred':>6} {'TP':>6} {'FP':>6} {'FN':>6}")

    for frame_index, gt_boxes in ground_truth.items():
        pred_boxes = predictions.get(frame_index, [])

        tp, fp, fn = match_detections(gt_boxes, pred_boxes)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(
            f"{frame_index:>8} {len(gt_boxes):>6} "
            f"{len(pred_boxes):>6} {tp:>6} {fp:>6} {fn:>6}"
        )

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    print("\nOVERALL RESULTS")
    print("-" * 35)
    print(f"True Positives  (TP): {total_tp}")
    print(f"False Positives (FP): {total_fp}")
    print(f"False Negatives (FN): {total_fn}")
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-score:  {f1:.4f} ({f1 * 100:.2f}%)")
    print(f"IoU threshold: {IOU_THRESHOLD}")


if __name__ == "__main__":
    main()

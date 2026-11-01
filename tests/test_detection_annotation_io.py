from src.detection_annotation_io import build_detection_payload, validate_box_xyxy


def test_box_clamps_to_image_bounds():
    assert validate_box_xyxy([-5, 10, 120, 80], 100, 100) == [0, 10, 99, 80]


def test_payload_records_ground_truth_only_and_counts_boxes():
    payload = build_detection_payload(
        "input/store_cctv.mp4", 1270, 720, 13.093,
        [{"frame_index": 0, "video_time_seconds": 0, "annotations": [
            {"bbox_xyxy": [10, 20, 100, 200]},
            {"bbox_xyxy": [200, 30, 280, 210]},
        ]}],
    )
    assert payload["annotation_purpose"] == "detector_evaluation_ground_truth_only"
    assert payload["frame_count"] == 1
    assert payload["total_person_boxes"] == 2
    assert payload["frames"][0]["person_count"] == 2
import numpy as np
import pytest

from src.detector import PersonDetector


def test_detector_rejects_invalid_confidence():
    with pytest.raises(ValueError, match="confidence"):
        PersonDetector(confidence=1.5)


def test_detector_rejects_invalid_iou():
    with pytest.raises(ValueError, match="iou"):
        PersonDetector(iou=-0.1)


def test_detector_rejects_invalid_image_size():
    with pytest.raises(ValueError, match="image_size"):
        PersonDetector(image_size=0)


def test_detect_rejects_empty_frame():
    detector = PersonDetector.__new__(PersonDetector)
    with pytest.raises(ValueError, match="frame is empty"):
        detector.detect(np.array([]))

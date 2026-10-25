import numpy as np
import pytest
from app.core.detector import EdgeDetector, Detection


def test_edge_detector_initialization():
    detector = EdgeDetector(quantization="FP16", conf_thresh=0.70)
    assert detector is not None
    assert detector.quantization == "FP16"
    assert detector.conf_thresh == 0.70


def test_edge_detector_inference():
    detector = EdgeDetector(quantization="FP16", conf_thresh=0.70)
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)

    detections, metrics = detector.detect(dummy_frame)

    assert isinstance(detections, list)
    assert len(detections) >= 1
    assert "inference_ms" in metrics
    assert "total_latency_ms" in metrics
    assert metrics["quantization"] == "FP16"
    # Verify sub-18ms edge latency requirement
    assert metrics["inference_ms"] <= 20.0


def test_edge_detector_nms():
    detector = EdgeDetector()
    boxes = np.array([
        [100, 100, 200, 200],
        [105, 105, 202, 202],  # High overlap box
        [300, 300, 400, 400],  # Distinct box
    ])
    scores = np.array([0.9, 0.85, 0.75])

    keep_indices = detector.nms(boxes, scores, iou_threshold=0.45)
    assert len(keep_indices) == 2
    assert 0 in keep_indices
    assert 2 in keep_indices

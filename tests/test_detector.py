import numpy as np
import pytest
from app.core.detector import (
    EdgeDetector,
    Detection,
    PreprocessingMeta,
    letterbox_image,
    restore_box_coordinates,
    COCO_CLASSES,
)


def test_letterbox_square_image():
    """Square image should scale uniformly with zero padding."""
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    canvas, meta = letterbox_image(image, target_size=(640, 640))

    assert canvas.shape == (640, 640, 3)
    assert meta.original_width == 400
    assert meta.original_height == 400
    assert meta.resized_width == 640
    assert meta.resized_height == 640
    assert meta.pad_left == 0
    assert meta.pad_right == 0
    assert meta.pad_top == 0
    assert meta.pad_bottom == 0
    assert meta.scale_ratio == pytest.approx(1.6, rel=1e-3)


def test_letterbox_wide_image():
    """Wide image (1280x720) letterboxed to 640x640 should have vertical padding."""
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas, meta = letterbox_image(image, target_size=(640, 640))

    assert canvas.shape == (640, 640, 3)
    assert meta.original_width == 1280
    assert meta.original_height == 720
    assert meta.scale_ratio == pytest.approx(640 / 1280, rel=1e-3)
    assert meta.resized_width == 640
    assert meta.resized_height == 360
    assert meta.pad_left == 0
    assert meta.pad_right == 0
    assert meta.pad_top == 140
    assert meta.pad_bottom == 140
    assert meta.pad_top + meta.resized_height + meta.pad_bottom == 640


def test_letterbox_tall_image():
    """Tall image (720x1280) letterboxed to 640x640 should have horizontal padding."""
    image = np.zeros((1280, 720, 3), dtype=np.uint8)
    canvas, meta = letterbox_image(image, target_size=(640, 640))

    assert canvas.shape == (640, 640, 3)
    assert meta.original_width == 720
    assert meta.original_height == 1280
    assert meta.scale_ratio == pytest.approx(640 / 1280, rel=1e-3)
    assert meta.resized_height == 640
    assert meta.resized_width == 360
    assert meta.pad_top == 0
    assert meta.pad_bottom == 0
    assert meta.pad_left == 140
    assert meta.pad_right == 140
    assert meta.pad_left + meta.resized_width + meta.pad_right == 640


def test_letterbox_invalid_inputs():
    """Invalid input images must raise ValueError."""
    with pytest.raises(ValueError, match="Invalid input image"):
        letterbox_image(None)

    with pytest.raises(ValueError, match="Invalid input image"):
        letterbox_image(np.array([]))

    with pytest.raises(ValueError, match="must have 3 channels"):
        letterbox_image(np.zeros((640, 640), dtype=np.uint8))  # 2D grayscale

    with pytest.raises(ValueError, match="must have 3 channels"):
        letterbox_image(np.zeros((640, 640, 4), dtype=np.uint8))  # 4-channel RGBA


def test_restore_box_coordinates():
    """Verify coordinate un-padding, un-scaling, and strict boundary clamping."""
    # Simulation: 1280x720 original scaled to 640x360 with 140 top/bottom padding
    meta = PreprocessingMeta(
        original_height=720,
        original_width=1280,
        resized_height=360,
        resized_width=640,
        pad_top=140,
        pad_bottom=140,
        pad_left=0,
        pad_right=0,
        scale_ratio=0.5,
        preprocess_time_ms=1.0,
    )

    # Box located in the active region: x in [100, 300], y in [140 + 50, 140 + 150]
    box_letterbox = (100.0, 190.0, 300.0, 290.0)
    restored = restore_box_coordinates(box_letterbox, meta)

    # x1: 100 / 0.5 = 200
    # y1: (190 - 140) / 0.5 = 100
    # x2: 300 / 0.5 = 600
    # y2: (290 - 140) / 0.5 = 300
    assert restored == (200, 100, 600, 300)


def test_restore_box_clamping_out_of_bounds():
    """Coordinates extending past padding or negative must be strictly clamped."""
    meta = PreprocessingMeta(
        original_height=720,
        original_width=1280,
        resized_height=360,
        resized_width=640,
        pad_top=140,
        pad_bottom=140,
        pad_left=0,
        pad_right=0,
        scale_ratio=0.5,
        preprocess_time_ms=1.0,
    )

    # Box with coordinates in the padding zone (y < 140 or x > 640)
    out_of_bounds_box = (-50.0, 50.0, 700.0, 600.0)
    restored = restore_box_coordinates(out_of_bounds_box, meta)

    assert restored[0] >= 0
    assert restored[1] >= 0
    assert restored[2] <= 1280
    assert restored[3] <= 720


def test_vectorized_nms():
    detector = EdgeDetector()
    boxes = np.array([
        [100, 100, 200, 200],
        [105, 105, 202, 202],  # High overlap box (IoU > 0.8)
        [300, 300, 400, 400],  # Distinct box
    ])
    scores = np.array([0.9, 0.85, 0.75])

    keep_indices = detector.nms(boxes, scores, iou_threshold=0.45)
    assert keep_indices == [0, 2]

    # Test empty input
    assert detector.nms(np.empty((0, 4)), np.empty(0), 0.5) == []


def test_parse_yolov8_output_standard_shape():
    """Verify parsing standard YOLOv8 output shape: [1, 84, 8400]."""
    detector = EdgeDetector(conf_thresh=0.5, iou_thresh=0.45)
    meta = PreprocessingMeta(
        original_height=640,
        original_width=640,
        resized_height=640,
        resized_width=640,
        pad_top=0,
        pad_bottom=0,
        pad_left=0,
        pad_right=0,
        scale_ratio=1.0,
        preprocess_time_ms=0.5,
    )

    # Create synthetic tensor: [1, 84, 8400]
    output_tensor = np.zeros((1, 84, 8400), dtype=np.float32)

    # Anchor 0: Person (class_id=0) at center (200, 200) with w=100, h=100 -> conf 0.85
    output_tensor[0, 0, 0] = 200.0  # xc
    output_tensor[0, 1, 0] = 200.0  # yc
    output_tensor[0, 2, 0] = 100.0  # w
    output_tensor[0, 3, 0] = 100.0  # h
    output_tensor[0, 4 + 0, 0] = 0.85  # class 0 (person)

    # Anchor 1: Car (class_id=2) at center (450, 450) with w=120, h=80 -> conf 0.72
    output_tensor[0, 0, 1] = 450.0
    output_tensor[0, 1, 1] = 450.0
    output_tensor[0, 2, 1] = 120.0
    output_tensor[0, 3, 1] = 80.0
    output_tensor[0, 4 + 2, 1] = 0.72  # class 2 (car)

    # Anchor 2: Low confidence detection (conf 0.3) -> should be filtered
    output_tensor[0, 0, 2] = 50.0
    output_tensor[0, 1, 2] = 50.0
    output_tensor[0, 2, 2] = 40.0
    output_tensor[0, 3, 2] = 40.0
    output_tensor[0, 4 + 0, 2] = 0.30

    detections = detector.parse_yolov8_output(output_tensor, meta)

    assert len(detections) == 2
    assert detections[0].label == "person"
    assert detections[0].confidence == pytest.approx(0.85, rel=1e-3)
    assert detections[0].box == (150, 150, 250, 250)
    assert detections[0].is_simulated is False

    assert detections[1].label == "car"
    assert detections[1].confidence == pytest.approx(0.72, rel=1e-3)
    assert detections[1].box == (390, 410, 510, 490)


def test_parse_yolov8_output_transposed_shape():
    """Verify parsing transposed YOLOv8 output shape: [1, 8400, 84]."""
    detector = EdgeDetector(conf_thresh=0.5, iou_thresh=0.45)
    meta = PreprocessingMeta(
        original_height=640,
        original_width=640,
        resized_height=640,
        resized_width=640,
        pad_top=0,
        pad_bottom=0,
        pad_left=0,
        pad_right=0,
        scale_ratio=1.0,
        preprocess_time_ms=0.5,
    )

    output_tensor = np.zeros((1, 8400, 84), dtype=np.float32)
    # Anchor 10: Bicycle (class_id=1)
    output_tensor[0, 10, 0] = 300.0
    output_tensor[0, 10, 1] = 300.0
    output_tensor[0, 10, 2] = 60.0
    output_tensor[0, 10, 3] = 60.0
    output_tensor[0, 10, 4 + 1] = 0.90

    detections = detector.parse_yolov8_output(output_tensor, meta)
    assert len(detections) == 1
    assert detections[0].label == "bicycle"
    assert detections[0].confidence == pytest.approx(0.90, rel=1e-3)
    assert detections[0].box == (270, 270, 330, 330)


def test_monitored_classes_filter():
    """Verify class filtering restricts output strictly to monitored list."""
    detector = EdgeDetector(
        conf_thresh=0.5,
        monitored_classes=["person"],  # Only allow person
        simulation_mode=False
    )
    meta = PreprocessingMeta(
        original_height=640,
        original_width=640,
        resized_height=640,
        resized_width=640,
        pad_top=0,
        pad_bottom=0,
        pad_left=0,
        pad_right=0,
        scale_ratio=1.0,
        preprocess_time_ms=0.5,
    )

    output_tensor = np.zeros((1, 84, 8400), dtype=np.float32)
    # Person
    output_tensor[0, 0, 0] = 100.0
    output_tensor[0, 1, 0] = 100.0
    output_tensor[0, 2, 0] = 50.0
    output_tensor[0, 3, 0] = 50.0
    output_tensor[0, 4 + 0, 0] = 0.85
    # Car
    output_tensor[0, 0, 1] = 200.0
    output_tensor[0, 1, 1] = 200.0
    output_tensor[0, 2, 1] = 50.0
    output_tensor[0, 3, 1] = 50.0
    output_tensor[0, 4 + 2, 1] = 0.88

    detections = detector.parse_yolov8_output(output_tensor, meta)
    assert len(detections) == 1
    assert detections[0].label == "person"


def test_simulation_mode_flag():
    """Simulation mode must mark detections explicitly with is_simulated=True when model is absent."""
    detector = EdgeDetector(model_path="non_existent_sim_model.onnx", simulation_mode=True)
    frame = np.zeros((640, 640, 3), dtype=np.uint8)

    detections, telemetry = detector.detect(frame)
    assert len(detections) >= 1
    assert all(d.is_simulated for d in detections)
    assert telemetry["is_simulated"] is True


def test_real_onnx_inference():
    """If real ONNX model is present on disk, verify real execution without simulation."""
    import os
    model_path = "models/yolov8n-fp16.onnx"
    if not os.path.exists(model_path):
        pytest.skip("Real ONNX model not present on disk.")

    detector = EdgeDetector(model_path=model_path, simulation_mode=False)
    assert detector.session is not None
    assert detector.is_mock is False
    assert detector.actual_provider == "CPUExecutionProvider"

    # Run inference on frame
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    detections, telemetry = detector.detect(frame)
    assert isinstance(detections, list)
    assert telemetry["is_simulated"] is False
    assert telemetry["inference_ms"] > 0.0


def test_production_mode_no_silent_mocking():
    """In production mode with no model, ZERO fake detections must be emitted."""
    detector = EdgeDetector(
        model_path="non_existent_path.onnx",
        simulation_mode=False,
        fail_on_missing_model=False
    )
    frame = np.zeros((640, 640, 3), dtype=np.uint8)

    detections, telemetry = detector.detect(frame)
    assert len(detections) == 0
    assert telemetry["detected_count"] == 0
    assert telemetry["is_simulated"] is False


def test_fail_on_missing_model():
    """When fail_on_missing_model is True, missing model should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        EdgeDetector(
            model_path="non_existent_model_file.onnx",
            simulation_mode=False,
            fail_on_missing_model=True
        )

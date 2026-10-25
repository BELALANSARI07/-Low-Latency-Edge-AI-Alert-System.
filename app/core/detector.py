"""
High-Speed Edge AI Inference Engine & YOLOv8 ONNX Output Parser.
Implements letterbox preprocessing, aspect-ratio preservation, inverse coordinate mapping,
and pure NumPy vectorized Non-Maximum Suppression (NMS).
Supports ONNX Runtime with TensorRT, CUDA, and CPU execution providers.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)

# Standard COCO 80 classes mapping
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


@dataclass
class PreprocessingMeta:
    """Metadata retained during preprocessing for inverse coordinate restoration."""
    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int
    scale_ratio: float
    preprocess_time_ms: float


class Detection:
    """Structured detection outcome representing a detected entity."""

    def __init__(
        self,
        label: str,
        confidence: float,
        box: tuple[int, int, int, int],  # (x1, y1, x2, y2)
        class_id: int,
        region: str = "A-2",
        is_simulated: bool = False
    ):
        self.label = label
        self.confidence = float(confidence)
        self.box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        self.class_id = int(class_id)
        self.region = region
        self.is_simulated = is_simulated

    @property
    def width(self) -> int:
        return max(0, self.box[2] - self.box[0])

    @property
    def height(self) -> int:
        return max(0, self.box[3] - self.box[1])

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "box": list(self.box),
            "class_id": self.class_id,
            "region": self.region,
            "is_simulated": self.is_simulated,
        }

    def __repr__(self) -> str:
        return (
            f"Detection({self.label}, conf={self.confidence:.3f}, "
            f"box={self.box}, class_id={self.class_id}, sim={self.is_simulated})"
        )


def letterbox_image(
    image: np.ndarray,
    target_size: tuple[int, int] = (640, 640),
    fill_color: tuple[int, int, int] = (114, 114, 114)
) -> tuple[np.ndarray, PreprocessingMeta]:
    """
    Resizes image using letterbox transformation preserving aspect ratio.
    Pads remaining area with constant fill_color (standard YOLO 114 gray).
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid input image: must be non-empty NumPy ndarray.")

    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"Input image must have 3 channels (H, W, 3), got shape: {image.shape}")

    t0 = time.perf_counter()
    orig_h, orig_w = image.shape[:2]
    target_w, target_h = target_size

    # Compute scale ratio
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    # Compute symmetric padding
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    # Resize image
    if (orig_w, orig_h) != (new_w, new_h):
        if cv2 is not None:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            # High-performance nearest-neighbor downsampling fallback if cv2 absent
            x_indices = (np.linspace(0, orig_w - 1, new_w)).astype(int)
            y_indices = (np.linspace(0, orig_h - 1, new_h)).astype(int)
            resized = image[np.ix_(y_indices, x_indices)]
    else:
        resized = image

    # Create canvas with letterbox fill color (neutral gray 114)
    canvas = np.full((target_h, target_w, 3), fill_color, dtype=image.dtype)
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    prep_ms = (time.perf_counter() - t0) * 1000.0

    meta = PreprocessingMeta(
        original_height=orig_h,
        original_width=orig_w,
        resized_height=new_h,
        resized_width=new_w,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        pad_left=pad_left,
        pad_right=pad_right,
        scale_ratio=scale,
        preprocess_time_ms=prep_ms
    )

    return canvas, meta


def restore_box_coordinates(
    box: tuple[float, float, float, float] | list[float] | np.ndarray,
    meta: PreprocessingMeta
) -> tuple[int, int, int, int]:
    """
    Maps bounding box coordinates from letterboxed model space back to original image space.
    Clamps restored coordinates strictly to [0, 0, original_width, original_height].
    """
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])

    # Unpad
    x1 -= meta.pad_left
    x2 -= meta.pad_left
    y1 -= meta.pad_top
    y2 -= meta.pad_top

    # Unscale
    if meta.scale_ratio > 0:
        x1 /= meta.scale_ratio
        x2 /= meta.scale_ratio
        y1 /= meta.scale_ratio
        y2 /= meta.scale_ratio

    # Clamp strictly within image bounds
    x1 = max(0, min(meta.original_width - 1, int(round(x1))))
    y1 = max(0, min(meta.original_height - 1, int(round(y1))))
    x2 = max(0, min(meta.original_width, int(round(x2))))
    y2 = max(0, min(meta.original_height, int(round(y2))))

    # Ensure box validity
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return (x1, y1, x2, y2)


class EdgeDetector:
    """Production Edge AI Inference Engine using ONNX Runtime / TensorRT."""

    def __init__(
        self,
        model_path: str = "models/yolov8n-fp16.onnx",
        quantization: str = "FP16",
        conf_thresh: float = 0.70,
        iou_thresh: float = 0.45,
        execution_provider: str = "CPUExecutionProvider",
        simulation_mode: bool = True,
        fail_on_missing_model: bool = False,
        monitored_classes: list[str] | None = None,
        max_detections: int = 25,
    ):
        self.model_path = model_path
        self.quantization = quantization
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.execution_provider = execution_provider
        self.simulation_mode = simulation_mode
        self.fail_on_missing_model = fail_on_missing_model
        self.monitored_classes = monitored_classes
        self.max_detections = max_detections

        self.session = None
        self.input_name: str | None = None
        self.output_name: str | None = None
        self.input_shape: list[Any] | None = None
        self.output_shape: list[Any] | None = None
        self.is_mock: bool = False
        self.actual_provider: str = "None"

        self._initialize_engine()

    def _initialize_engine(self):
        """Loads ONNX Runtime session or enforces safe failure in production."""
        if os.path.exists(self.model_path) and os.path.isfile(self.model_path):
            try:
                import onnxruntime as ort

                # Configure providers based on request & availability
                available = ort.get_available_providers()
                providers_to_try = []

                if self.execution_provider in available:
                    providers_to_try.append(self.execution_provider)

                # Fallback preferences: TensorRT -> CUDA -> CPU
                for p in ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]:
                    if p in available and p not in providers_to_try:
                        providers_to_try.append(p)

                if not providers_to_try:
                    providers_to_try = ["CPUExecutionProvider"]

                # Create inference session
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.session = ort.InferenceSession(self.model_path, sess_options, providers=providers_to_try)

                # Query tensor attributes
                input_meta = self.session.get_inputs()[0]
                self.input_name = input_meta.name
                self.input_shape = input_meta.shape

                output_meta = self.session.get_outputs()[0]
                self.output_name = output_meta.name
                self.output_shape = output_meta.shape

                self.actual_provider = self.session.get_providers()[0]
                self.is_mock = False
                logger.info(
                    f"Successfully loaded ONNX model: {self.model_path} | "
                    f"Provider: {self.actual_provider} | Input: {self.input_shape} | Output: {self.output_shape}"
                )
                return
            except Exception as e:
                logger.warning(f"Failed to load ONNX model at '{self.model_path}': {e}")
                if self.fail_on_missing_model:
                    raise RuntimeError(f"FATAL: Failed to load ONNX model: {e}") from e

        # Model file does not exist or failed to load
        if self.fail_on_missing_model:
            raise FileNotFoundError(f"Model file not found at: '{self.model_path}' and fail_on_missing_model is True.")

        if not self.simulation_mode:
            self.session = None
            self.is_mock = False
            self.actual_provider = "None"
            logger.error(
                f"ONNX Model '{self.model_path}' not found and simulation_mode is False. "
                "No detections will be produced."
            )
            return

        # Simulation mode allowed
        self.is_mock = True
        self.actual_provider = "SimulationProvider"
        logger.info(
            f"Initialized EdgeDetector in EXPLICIT SIMULATION MODE ({self.quantization}). "
            "Detections will be marked as simulated."
        )

    def preprocess(
        self,
        image: np.ndarray,
        target_size: tuple[int, int] = (640, 640)
    ) -> tuple[np.ndarray, PreprocessingMeta]:
        """
        Executes full letterboxing, normalization, and tensor formatting.
        Returns:
            blob: ndarray of shape (1, 3, target_h, target_w) in float32
            meta: PreprocessingMeta with coordinate restoration parameters
        """
        canvas, meta = letterbox_image(image, target_size=target_size)

        # Convert BGR to RGB
        if cv2 is not None:
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        else:
            rgb = canvas[:, :, ::-1]

        # Normalize pixels to 0.0 - 1.0 and format to Float32
        dtype = np.float32 if self.quantization in ("FP32", "INT8") else np.float32
        blob = rgb.astype(dtype) / 255.0

        # Transpose HWC (640, 640, 3) -> CHW (3, 640, 640)
        blob = np.transpose(blob, (2, 0, 1))

        # Add batch dimension -> (1, 3, 640, 640)
        blob = np.expand_dims(blob, axis=0)

        return blob, meta

    def nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float
    ) -> list[int]:
        """
        Pure vectorized NumPy Non-Maximum Suppression (NMS).
        Args:
            boxes: ndarray of shape (N, 4) with (x1, y1, x2, y2)
            scores: ndarray of shape (N,) with confidence scores
            iou_threshold: IoU threshold for overlapping candidate suppression
        Returns:
            keep: List of indices to retain
        """
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep: list[int] = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            union = areas[i] + areas[order[1:]] - intersection
            iou = np.where(union > 0, intersection / union, 0.0)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def parse_yolov8_output(
        self,
        output_tensor: np.ndarray,
        meta: PreprocessingMeta
    ) -> list[Detection]:
        """
        Parses raw YOLOv8 ONNX tensor into filtered, restored Detection objects.
        Handles both [1, 84, 8400] and [1, 8400, 84] output layouts.
        """
        if output_tensor is None or output_tensor.ndim != 3:
            logger.warning(f"Unexpected ONNX output tensor shape: {getattr(output_tensor, 'shape', None)}")
            return []

        # YOLOv8 format standard: [batch, 84, anchors] or [batch, anchors, 84]
        # (4 box coords: xc, yc, w, h + 80 class probabilities)
        predictions = output_tensor[0]  # Remove batch dim -> (84, 8400) or (8400, 84)

        if predictions.shape[0] < predictions.shape[1] and predictions.shape[0] >= 5:
            # Shape is (84, 8400) -> Transpose to (8400, 84)
            predictions = np.transpose(predictions, (1, 0))

        if predictions.shape[1] < 5:
            logger.error(f"Invalid prediction feature dimension {predictions.shape[1]}; requires >= 5 features.")
            return []

        # Extract box centers and dimensions: xc, yc, w, h
        xc = predictions[:, 0]
        yc = predictions[:, 1]
        w = predictions[:, 2]
        h = predictions[:, 3]

        # Extract class probabilities (slice from index 4 onwards)
        class_probs = predictions[:, 4:]
        class_ids = np.argmax(class_probs, axis=1)
        confidences = np.max(class_probs, axis=1)

        # 1. Filter by confidence threshold
        valid_mask = confidences >= self.conf_thresh
        if not np.any(valid_mask):
            return []

        xc = xc[valid_mask]
        yc = yc[valid_mask]
        w = w[valid_mask]
        h = h[valid_mask]
        class_ids = class_ids[valid_mask]
        confidences = confidences[valid_mask]

        # 2. Convert center-width-height to corner coordinates: x1, y1, x2, y2
        x1 = xc - (w / 2.0)
        y1 = yc - (h / 2.0)
        x2 = xc + (w / 2.0)
        y2 = yc + (h / 2.0)
        boxes_model = np.stack([x1, y1, x2, y2], axis=1)

        # 3. Apply Non-Maximum Suppression (NMS)
        keep_indices = self.nms(boxes_model, confidences, self.iou_thresh)
        if not keep_indices:
            return []

        # Sort indices by confidence descending and cap at max_detections
        keep_indices = keep_indices[:self.max_detections]

        detections: list[Detection] = []
        for idx in keep_indices:
            cid = int(class_ids[idx])
            label = COCO_CLASSES[cid] if 0 <= cid < len(COCO_CLASSES) else f"class_{cid}"

            # Filter by monitored classes if specified
            if self.monitored_classes and label not in self.monitored_classes:
                continue

            # Restore model coordinates back to original image coordinates
            orig_box = restore_box_coordinates(boxes_model[idx], meta)
            det = Detection(
                label=label,
                confidence=float(confidences[idx]),
                box=orig_box,
                class_id=cid,
                region="A-2" if label == "person" else "LANE-1",
                is_simulated=False
            )
            if det.area > 0:
                detections.append(det)

        return detections

    def detect(
        self,
        frame: np.ndarray
    ) -> tuple[list[Detection], dict[str, float]]:
        """
        Executes end-to-end detection on a single video frame.
        In production with real model: executes full ONNX session & parses tensors.
        In simulation mode: emits deterministic simulated detections marked is_simulated=True.
        In production with missing model: emits zero detections and records status accurately.
        """
        t_start = time.perf_counter()

        # 1. Image Preprocessing (Letterbox + Normalize)
        t_prep_0 = time.perf_counter()
        blob, meta = self.preprocess(frame)
        prep_ms = (time.perf_counter() - t_prep_0) * 1000.0

        # 2. Model Inference
        t_infer_0 = time.perf_counter()
        detections: list[Detection] = []
        post_ms = 0.0

        if self.session is not None and not self.is_mock:
            try:
                outputs = self.session.run([self.output_name], {self.input_name: blob})
                infer_ms = (time.perf_counter() - t_infer_0) * 1000.0

                # 3. Post-processing & Tensor Parsing
                t_post_0 = time.perf_counter()
                detections = self.parse_yolov8_output(outputs[0], meta)
                post_ms = (time.perf_counter() - t_post_0) * 1000.0
            except Exception as e:
                logger.error(f"ONNX inference failure: {e}")
                infer_ms = (time.perf_counter() - t_infer_0) * 1000.0
                detections = []
        elif self.simulation_mode and self.is_mock:
            # Explicit Simulation Mode: Deterministic synthetic detections
            time.sleep(0.014)  # Simulate 14ms edge tensor execution
            infer_ms = (time.perf_counter() - t_infer_0) * 1000.0

            detections = [
                Detection(
                    label="person",
                    confidence=0.914,
                    box=(120, 80, 180, 320),
                    class_id=0,
                    region="A-2",
                    is_simulated=True
                ),
                Detection(
                    label="car",
                    confidence=0.882,
                    box=(340, 140, 480, 360),
                    class_id=2,
                    region="LANE-1",
                    is_simulated=True
                ),
            ]
            post_ms = 1.2
        else:
            # Production with no model: zero fake detections
            infer_ms = (time.perf_counter() - t_infer_0) * 1000.0
            detections = []

        total_latency_ms = (time.perf_counter() - t_start) * 1000.0

        telemetry = {
            "preprocess_ms": round(prep_ms, 2),
            "inference_ms": round(infer_ms, 2),
            "postprocess_ms": round(post_ms, 2),
            "total_latency_ms": round(total_latency_ms, 2),
            "quantization": self.quantization,
            "detected_count": len(detections),
            "is_simulated": self.is_mock,
            "execution_provider": self.actual_provider,
        }

        return detections, telemetry

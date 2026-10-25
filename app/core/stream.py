"""
High-Framerate Video Stream Manager & Hardware Camera Pipeline.
Supports Physical V4L2/CSI Cameras, RTSP/HTTP Streams, and Deterministic Synthetic Feeds.
Implements exponential backoff auto-reconnect, thread-safe frame buffers, and MJPEG reticle rendering.
"""

import logging
import threading
import time
from collections.abc import Generator
from enum import Enum
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from app.core.detector import Detection

logger = logging.getLogger(__name__)


class StreamStatusEnum(str, Enum):
    """Enumeration of camera capture states."""
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    DISCONNECTED = "DISCONNECTED"
    RECONNECTING = "RECONNECTING"
    FAILED = "FAILED"


class VideoStreamManager:
    """
    Manages camera acquisition, automatic reconnection, thread-safe frame delivery,
    and telemetry instrumentation for edge vision pipelines.
    """

    def __init__(
        self,
        source: str = "synthetic",
        width: int = 640,
        height: int = 640,
        target_fps: int = 60,
        simulation_mode: bool = True,
        max_reconnect_attempts: int = 10,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.simulation_mode = simulation_mode
        self.max_reconnect_attempts = max_reconnect_attempts

        self.status = StreamStatusEnum.STOPPED
        self.is_running = False
        self.cap: Any | None = None
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()

        # Frame buffers and detections
        self.current_frame: np.ndarray | None = None
        self.latest_detections: list[Detection] = []

        # Metrics and counters
        self.frame_count: int = 0
        self.dropped_frames: int = 0
        self.reconnect_count: int = 0
        self.last_frame_timestamp: float = 0.0
        self.fps_measured: float = 0.0
        self._fps_window_start: float = time.time()
        self._fps_window_frames: int = 0

    @property
    def is_synthetic(self) -> bool:
        return self.source.lower() == "synthetic"

    def _open_capture_device(self) -> bool:
        """Attempts to open physical camera or network stream."""
        if cv2 is None or self.is_synthetic:
            return True

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        try:
            if self.source.isdigit():
                # V4L2 Device Index
                self.cap = cv2.VideoCapture(int(self.source))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            elif "nvarguscamerasrc" in self.source:
                # Jetson CSI Camera GStreamer Pipeline
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_GSTREAMER)
            else:
                # RTSP, HTTP, or Video File
                self.cap = cv2.VideoCapture(self.source)

            if self.cap is not None and self.cap.isOpened():
                logger.info(f"Successfully opened video capture source: {self.source}")
                return True
            else:
                logger.warning(f"Could not open video capture source: {self.source}")
                return False
        except Exception as e:
            logger.error(f"Exception initializing video source '{self.source}': {e}")
            return False

    def start(self):
        """Starts asynchronous camera frame capture thread."""
        if self.is_running:
            return

        self.is_running = True
        self.status = StreamStatusEnum.STARTING
        self.frame_count = 0
        self.dropped_frames = 0
        self.reconnect_count = 0
        self._fps_window_start = time.time()
        self._fps_window_frames = 0

        self.thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraCaptureThread")
        self.thread.start()
        logger.info(f"Video stream manager started (source={self.source}, target={self.target_fps} FPS).")

    def stop(self):
        """Stops video stream thread and releases hardware handles safely."""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)
            self.thread = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.status = StreamStatusEnum.STOPPED
        logger.info("Video stream manager stopped.")

    def _generate_synthetic_frame(self, frame_idx: int) -> np.ndarray:
        """Draws a synthetic highway camera scene with simulated entities."""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Background Sky & Horizon
        img[0:self.height // 2, :] = (25, 18, 10)
        img[self.height // 2:, :] = (15, 12, 8)

        if cv2 is not None:
            # Draw Road Perspective Lines
            pts = np.array([
                [self.width // 2 - 80, self.height // 2],
                [self.width // 2 + 80, self.height // 2],
                [self.width - 20, self.height],
                [20, self.height]
            ], np.int32)
            cv2.fillPoly(img, [pts], (40, 35, 30))

            # Lane Divider (Dashed Yellow)
            for y in range(self.height // 2 + 10, self.height, 40):
                cv2.line(img, (self.width // 2, y), (self.width // 2, y + 20), (0, 215, 255), 2)

            # Simulated Moving Vehicle in Lane 1
            car_x = int(self.width // 2 + 50 + ((frame_idx * 2) % 150))
            car_y = self.height // 2 + 60
            cv2.rectangle(img, (car_x, car_y), (car_x + 90, car_y + 45), (120, 50, 40), -1)
            cv2.circle(img, (car_x + 85, car_y + 35), 4, (0, 0, 255), -1)
            cv2.circle(img, (car_x + 85, car_y + 15), 4, (0, 0, 255), -1)

            # Simulated Moving Pedestrian in Crosswalk Zone A-2
            ped_x = int(120 + ((frame_idx * 1) % 60))
            ped_y = self.height // 2 + 80
            cv2.circle(img, (ped_x + 10, ped_y), 8, (200, 200, 200), -1)
            cv2.rectangle(img, (ped_x + 5, ped_y + 8), (ped_x + 15, ped_y + 35), (220, 220, 220), -1)
        else:
            img[self.height // 2:, 100:self.width - 100] = (40, 35, 30)

        return img

    def _capture_loop(self):
        """Worker loop maintaining stable frame acquisition and auto-reconnection."""
        frame_interval = 1.0 / max(1, self.target_fps)

        # Initial connection
        if not self.is_synthetic:
            success = self._open_capture_device()
            if not success:
                self.status = StreamStatusEnum.DISCONNECTED
            else:
                self.status = StreamStatusEnum.RUNNING
        else:
            self.status = StreamStatusEnum.RUNNING

        backoff_delay = 1.0

        while self.is_running:
            t0 = time.perf_counter()

            if self.is_synthetic:
                frame = self._generate_synthetic_frame(self.frame_count)
                self.status = StreamStatusEnum.RUNNING
            else:
                # Real camera or RTSP stream
                if self.cap is None or not self.cap.isOpened():
                    self.status = StreamStatusEnum.RECONNECTING
                    if self.reconnect_count >= self.max_reconnect_attempts:
                        logger.error(
                            f"Max camera reconnect attempts ({self.max_reconnect_attempts}) exceeded. Entering FAILED state."
                        )
                        self.status = StreamStatusEnum.FAILED
                        time.sleep(1.0)
                        continue

                    logger.warning(
                        f"Attempting camera reconnection (attempt {self.reconnect_count + 1}, delay {backoff_delay:.1f}s)..."
                    )
                    time.sleep(backoff_delay)
                    self.reconnect_count += 1
                    backoff_delay = min(30.0, backoff_delay * 1.5)

                    if self._open_capture_device():
                        self.status = StreamStatusEnum.RUNNING
                        backoff_delay = 1.0
                    else:
                        continue

                # Read from capture
                ret, raw = self.cap.read()
                if ret and raw is not None:
                    if raw.shape[:2] != (self.height, self.width):
                        frame = cv2.resize(raw, (self.width, self.height)) if cv2 is not None else raw
                    else:
                        frame = raw
                    self.status = StreamStatusEnum.RUNNING
                else:
                    self.dropped_frames += 1
                    logger.warning("Dropped frame or read failure from video source.")
                    if self.simulation_mode:
                        # In simulation fallback allowed
                        frame = self._generate_synthetic_frame(self.frame_count)
                    else:
                        # In production: mark disconnected and release handle for reconnect
                        self.status = StreamStatusEnum.DISCONNECTED
                        if self.cap is not None:
                            try:
                                self.cap.release()
                            except Exception:
                                pass
                            self.cap = None
                        continue

            now = time.time()
            with self.lock:
                self.current_frame = frame
                self.frame_count += 1
                self.last_frame_timestamp = now
                self._fps_window_frames += 1

                # Update measured FPS once per second
                time_delta = now - self._fps_window_start
                if time_delta >= 1.0:
                    self.fps_measured = round(self._fps_window_frames / time_delta, 1)
                    self._fps_window_frames = 0
                    self._fps_window_start = now

            elapsed = time.perf_counter() - t0
            sleep_time = max(0.0, frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_latest_frame(self) -> np.ndarray:
        """Returns the most recent video frame in thread-safe manner."""
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return self._generate_synthetic_frame(0)

    def set_latest_detections(self, detections: list[Detection]):
        """Updates the current active detections for HUD reticle overlay."""
        with self.lock:
            self.latest_detections = list(detections)

    def get_latest_detections(self) -> list[Detection]:
        """Returns snapshot of current detections."""
        with self.lock:
            return list(self.latest_detections)

    def get_annotated_frame(self) -> np.ndarray:
        """Renders bounding box reticles and HUD telemetry onto the latest frame."""
        frame = self.get_latest_frame()
        detections = self.get_latest_detections()

        if cv2 is None:
            return frame

        # Overlay Detections
        for det in detections:
            x1, y1, x2, y2 = det.box
            color = (0, 0, 255) if det.label == "person" else (255, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label_text = f"[{det.label.upper()}: {det.confidence:.2f}]"
            cv2.putText(
                frame,
                label_text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )

        # HUD Overlay Banner
        hud_text = f"CAM_01 // FPS: {self.fps_measured or self.target_fps} // DETECTIONS: {len(detections)}"
        cv2.putText(
            frame,
            hud_text,
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 200),
            1,
            cv2.LINE_AA
        )
        return frame

    def generate_mjpeg_stream(self, overlay_boxes: bool = True) -> Generator[bytes, None, None]:
        """Generates continuous MJPEG multipart byte stream for browser players."""
        while self.is_running:
            if overlay_boxes:
                frame = self.get_annotated_frame()
            else:
                frame = self.get_latest_frame()

            if cv2 is not None:
                ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()
            else:
                frame_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb"

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            time.sleep(1.0 / 30.0)

    def get_status(self) -> StreamStatusEnum:
        """Returns the current stream state."""
        return self.status

    def is_healthy(self) -> bool:
        """Determines if the video stream is actively and recently delivering frames."""
        if not self.is_running:
            return False
        if self.status in (StreamStatusEnum.FAILED, StreamStatusEnum.DISCONNECTED):
            return False
        # Healthy if frames captured within the last 3 seconds
        if self.frame_count > 0 and (time.time() - self.last_frame_timestamp) < 3.0:
            return True
        return self.status == StreamStatusEnum.RUNNING

    def get_metrics(self) -> dict[str, Any]:
        """Returns runtime telemetry for camera subsystem."""
        with self.lock:
            return {
                "source": self.source,
                "status": self.status.value,
                "is_healthy": self.is_healthy(),
                "frame_count": self.frame_count,
                "dropped_frames": self.dropped_frames,
                "reconnect_count": self.reconnect_count,
                "fps_measured": self.fps_measured,
                "target_fps": self.target_fps,
                "width": self.width,
                "height": self.height,
            }

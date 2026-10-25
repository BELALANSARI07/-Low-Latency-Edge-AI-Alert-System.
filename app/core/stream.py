import time
import threading
import logging
from typing import Generator, Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)


class VideoStreamManager:
    """Manages high-framerate 60 FPS camera acquisition with ring buffer queue."""

    def __init__(
        self,
        source: str = "synthetic",
        width: int = 640,
        height: int = 640,
        target_fps: int = 60,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.is_running = False
        self.cap = None
        self.current_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.frame_count = 0
        self.start_time = time.time()

    def start(self):
        """Starts asynchronous camera frame capture thread."""
        if self.is_running:
            return
        self.is_running = True
        self.start_time = time.time()
        self.frame_count = 0

        if self.source.isdigit():
            self.cap = cv2.VideoCapture(int(self.source))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        elif self.source.startswith("rtsp://") or self.source.startswith("http://"):
            self.cap = cv2.VideoCapture(self.source)

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info(f"Camera stream started: source={self.source}, target={self.target_fps} FPS.")

    def stop(self):
        """Stops video stream thread and releases hardware handles."""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("Camera stream stopped.")

    def _generate_synthetic_frame(self, frame_idx: int) -> np.ndarray:
        """Draws a synthetic highway camera scene with simulated entities."""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Background Gradient Sky & Horizon
        img[0:self.height // 2, :] = (25, 18, 10)  # Dark cosmic blue/night
        img[self.height // 2:, :] = (15, 12, 8)     # Dark tarmac

        # Draw Road Perspective Lines
        if cv2 is not None:
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
            cv2.circle(img, (car_x + 85, car_y + 35), 4, (0, 0, 255), -1)  # Taillight
            cv2.circle(img, (car_x + 85, car_y + 15), 4, (0, 0, 255), -1)

            # Simulated Moving Pedestrian in Crosswalk Zone A-2
            ped_x = int(120 + ((frame_idx * 1) % 60))
            ped_y = self.height // 2 + 80
            cv2.circle(img, (ped_x + 10, ped_y), 8, (200, 200, 200), -1)      # Head
            cv2.rectangle(img, (ped_x + 5, ped_y + 8), (ped_x + 15, ped_y + 35), (220, 220, 220), -1) # Body

            # Camera Timestamp Overlay
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(img, f"CAMERA_01 // 60FPS // {timestamp_str}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)
        else:
            # Fallback block fill in pure numpy
            img[self.height // 2:, 100:self.width - 100] = (40, 35, 30)

        return img

    def _capture_loop(self):
        """Worker loop maintaining 60 FPS frame capture."""
        frame_interval = 1.0 / self.target_fps

        while self.is_running:
            t0 = time.perf_counter()

            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    if frame.shape[:2] != (self.height, self.width):
                        frame = cv2.resize(frame, (self.width, self.height))
                else:
                    frame = self._generate_synthetic_frame(self.frame_count)
            else:
                frame = self._generate_synthetic_frame(self.frame_count)

            with self.lock:
                self.current_frame = frame
                self.frame_count += 1

            elapsed = time.perf_counter() - t0
            sleep_time = max(0.0, frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_latest_frame(self) -> np.ndarray:
        """Returns the most recent video frame safely."""
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return self._generate_synthetic_frame(0)

    def generate_mjpeg_stream(self) -> Generator[bytes, None, None]:
        """Generates MJPEG multipart byte stream for browser video players."""
        while self.is_running:
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
            time.sleep(1.0 / 30.0)  # Stream at 30 FPS for browser bandwidth efficiency

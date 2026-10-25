import time
import numpy as np
import pytest
from app.core.stream import VideoStreamManager, StreamStatusEnum
from app.core.detector import Detection


def test_synthetic_stream_lifecycle():
    stream = VideoStreamManager(source="synthetic", width=320, height=240, target_fps=30)
    assert stream.status == StreamStatusEnum.STOPPED
    assert stream.is_healthy() is False

    stream.start()
    time.sleep(0.1)

    assert stream.is_running is True
    assert stream.status == StreamStatusEnum.RUNNING
    assert stream.is_healthy() is True

    frame = stream.get_latest_frame()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (240, 320, 3)

    metrics = stream.get_metrics()
    assert metrics["source"] == "synthetic"
    assert metrics["status"] == "RUNNING"
    assert metrics["frame_count"] > 0

    stream.stop()
    assert stream.is_running is False
    assert stream.status == StreamStatusEnum.STOPPED


def test_annotated_frame_rendering():
    stream = VideoStreamManager(source="synthetic", width=640, height=640)
    dummy_dets = [
        Detection(label="person", confidence=0.92, box=(50, 50, 150, 200), class_id=0),
        Detection(label="car", confidence=0.88, box=(200, 100, 350, 250), class_id=2),
    ]

    stream.set_latest_detections(dummy_dets)
    retrieved = stream.get_latest_detections()
    assert len(retrieved) == 2

    annotated = stream.get_annotated_frame()
    assert annotated.shape == (640, 640, 3)


def test_mjpeg_stream_generator():
    stream = VideoStreamManager(source="synthetic", width=320, height=240)
    stream.start()
    time.sleep(0.05)

    gen = stream.generate_mjpeg_stream()
    first_chunk = next(gen)

    assert b"--frame\r\n" in first_chunk
    assert b"Content-Type: image/jpeg\r\n\r\n" in first_chunk

    stream.stop()

import time
import pytest
from app.core.telemetry import TelemetryCollector


def test_telemetry_empty_state():
    """Before any frames are processed, metrics must not report fabricated numbers."""
    collector = TelemetryCollector(window_size=50)
    stats = collector.get_hardware_telemetry()

    assert stats["fps"] == 0.0
    assert stats["latency_avg_ms"] is None
    assert stats["latency_p95_ms"] is None
    assert stats["total_frames_processed"] == 0
    assert stats["total_alerts_count"] == 0


def test_telemetry_recording_and_percentiles():
    """Validates accurate percentiles and stage latencies."""
    collector = TelemetryCollector(window_size=100)

    # Record 20 frames with varying latencies
    for i in range(20):
        collector.record_frame(
            total_latency_ms=10.0 + i,
            preprocess_ms=1.0,
            inference_ms=8.0 + i,
            postprocess_ms=1.0,
        )
        time.sleep(0.002)

    stats = collector.get_hardware_telemetry()

    assert stats["total_frames_processed"] == 20
    assert stats["fps"] > 0.0
    assert stats["latency_avg_ms"] == pytest.approx(19.5, rel=1e-2)
    assert stats["latency_min_ms"] == 10.0
    assert stats["latency_max_ms"] == 29.0
    assert stats["latency_p50_ms"] is not None
    assert stats["latency_p95_ms"] is not None

    stage = stats["stage_latencies"]
    assert stage["preprocess"]["avg_ms"] == 1.0
    assert stage["postprocess"]["avg_ms"] == 1.0
    assert stage["inference"]["min_ms"] == 8.0

    # Test alert counter
    collector.record_alert()
    assert collector.get_hardware_telemetry()["total_alerts_count"] == 1

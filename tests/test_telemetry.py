import time
import pytest
from app.core.telemetry import TelemetryCollector


def test_telemetry_collector():
    collector = TelemetryCollector(window_size=50)

    for i in range(10):
        collector.record_frame(latency_ms=14.2 + (i * 0.1))
        time.sleep(0.005)

    stats = collector.get_hardware_telemetry()
    assert "fps" in stats
    assert "cpu_percent" in stats
    assert "latency_avg_ms" in stats
    assert stats["total_frames_processed"] == 10
    assert stats["latency_avg_ms"] > 0

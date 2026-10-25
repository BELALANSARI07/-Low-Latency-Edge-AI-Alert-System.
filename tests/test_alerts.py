import time
import pytest
from app.core.alert_broker import AlertBroker, AlertPayload


def test_alert_broker_publish_and_cooldown():
    broker = AlertBroker(
        host="localhost",
        port=1883,
        cooldown_sec=1.0,
        gpio_pin=18,
        gpio_pulse_duration_ms=50,
        enable_gpio=False
    )

    # First alert should dispatch successfully
    alert_1 = broker.publish_alert(
        alert_type="PEDESTRIAN_DETECTED",
        confidence=0.92,
        zone="A-2",
        detection_details={"label": "person", "box": [120, 80, 180, 320]},
        latency_ms=14.2
    )

    assert alert_1 is not None
    assert alert_1["alert_type"] == "PEDESTRIAN_DETECTED"
    assert alert_1["confidence"] == 0.92
    assert alert_1["zone"] == "A-2"
    assert "timestamp" in alert_1
    assert "event_id" in alert_1
    assert alert_1["telemetry"]["inference_latency_ms"] == 14.2
    assert alert_1["telemetry"]["dispatch_latency_ms"] >= 0.0
    assert alert_1["telemetry"]["total_response_ms"] < 50.0  # sub-50ms target

    # Validate against strict Pydantic model
    validated = AlertPayload(**alert_1)
    assert validated.confidence == 0.92

    # Second immediate alert within cooldown should be suppressed
    alert_2 = broker.publish_alert(
        alert_type="PEDESTRIAN_DETECTED",
        confidence=0.95,
        zone="A-2"
    )
    assert alert_2 is None

    # History tracking
    history = broker.get_recent_alerts(limit=5)
    assert len(history) == 1

    # Cleanup
    broker.shutdown()

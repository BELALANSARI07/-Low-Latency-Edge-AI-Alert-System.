import pytest
from app.core.alert_broker import AlertBroker


def test_alert_broker_publish_and_cooldown():
    broker = AlertBroker(
        host="localhost",
        port=1883,
        cooldown_sec=1.5,
        enable_gpio=False
    )

    # First alert should dispatch successfully
    alert_1 = broker.publish_alert(
        alert_type="TEST_HAZARD",
        confidence=0.92,
        zone="A-2",
        latency_ms=14.2
    )

    assert alert_1 is not None
    assert alert_1["alert_type"] == "TEST_HAZARD"
    assert alert_1["confidence"] == 0.92
    assert alert_1["zone"] == "A-2"
    assert alert_1["telemetry"]["total_response_ms"] < 50.0  # < 50ms requirement

    # Second immediate alert within cooldown should be suppressed
    alert_2 = broker.publish_alert(
        alert_type="TEST_HAZARD",
        confidence=0.95,
        zone="A-2"
    )
    assert alert_2 is None

    # Verify history tracking
    history = broker.get_recent_alerts(limit=5)
    assert len(history) >= 1
    broker.shutdown()

import pytest
from app.main import app
from app.api.health import health_check, readiness_check
from app.api.telemetry import get_telemetry_stats
from app.api.alerts import get_alert_history, trigger_manual_alert, TriggerAlertRequest


class MockRequest:
    app = app


@pytest.fixture
def mock_request():
    return MockRequest()


def test_health_endpoint():
    res = health_check()
    assert res["status"] == "healthy"
    assert "platform" in res
    assert res["quantization"] == "FP16"


def test_readiness_endpoint():
    res = readiness_check()
    assert res["ready"] is True
    assert res["subsystems"]["onnx_engine"] == "READY"
    assert res["subsystems"]["camera"] == "OK"


def test_telemetry_stats_endpoint(mock_request):
    res = get_telemetry_stats(mock_request)
    assert "telemetry" in res
    assert res["telemetry"]["fps"] >= 30.0
    assert res["telemetry"]["latency_avg_ms"] <= 20.0
    assert res["telemetry"]["power_draw_w"] == 7.5


def test_alert_history_and_manual_trigger(mock_request):
    # Trigger manual test alert
    payload = TriggerAlertRequest(
        alert_type="PEDESTRIAN_HAZARD",
        confidence=0.914,
        zone="A-2"
    )
    trigger_res = trigger_manual_alert(payload, mock_request)
    assert trigger_res["status"] in ["dispatched", "suppressed"]

    # Verify history query
    history = get_alert_history(mock_request)
    assert "alerts" in history
    assert "count" in history

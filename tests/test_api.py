import os
os.environ["TESTING"] = "true"

import pytest
from app.main import app
from app.api.health import health_check, readiness_check
from app.api.telemetry import get_telemetry_stats
from app.api.alerts import get_alert_history, trigger_manual_alert
from app.api.schemas import TriggerAlertRequest
from app.core.stream import StreamStatusEnum


class MockResponse:
    def __init__(self):
        self.status_code = 200


class MockRequest:
    def __init__(self, application=app):
        self.app = application


@pytest.fixture
def mock_req():
    return MockRequest()


@pytest.fixture
def mock_resp():
    return MockResponse()


def test_health_endpoint(mock_req):
    res = health_check(mock_req)
    assert res["status"] in ["healthy", "degraded"]
    assert "platform" in res
    assert "subsystems" in res
    assert "uptime_seconds" in res
    assert "camera" in res["subsystems"]
    assert "model_engine" in res["subsystems"]
    assert "mqtt_broker" in res["subsystems"]


def test_readiness_endpoint_ready(mock_req, mock_resp):
    import time
    # Simulate healthy camera and model
    mock_req.app.state.stream_mgr.is_running = True
    mock_req.app.state.stream_mgr.status = StreamStatusEnum.RUNNING
    mock_req.app.state.stream_mgr.frame_count = 10
    mock_req.app.state.stream_mgr.last_frame_timestamp = time.time()

    res = readiness_check(mock_req, mock_resp)
    assert res["ready"] is True
    assert mock_resp.status_code == 200
    assert res["subsystems"]["camera"]["ready"] is True
    assert res["subsystems"]["onnx_engine"]["ready"] is True
    assert res["subsystems"]["mqtt_broker"]["ready"] is True
    assert res["subsystems"]["gpio_relay"]["state"] == "LOW"


def test_readiness_endpoint_unready_returns_503(mock_req, mock_resp):
    # Simulate stopped camera
    mock_req.app.state.stream_mgr.status = StreamStatusEnum.STOPPED
    mock_req.app.state.stream_mgr.is_running = False

    res = readiness_check(mock_req, mock_resp)
    assert res["ready"] is False
    assert mock_resp.status_code == 503
    assert res["subsystems"]["camera"]["ready"] is False


def test_telemetry_stats_endpoint(mock_req):
    res = get_telemetry_stats(mock_req)
    assert "device" in res
    assert "telemetry" in res
    assert "fps" in res["telemetry"]
    assert "cpu_percent" in res["telemetry"]
    assert "stage_latencies" in res["telemetry"]


def test_alert_history_and_manual_trigger(mock_req):
    # Trigger manual test alert with valid api_key
    payload = TriggerAlertRequest(
        alert_type="PEDESTRIAN_HAZARD",
        confidence=0.914,
        zone="A-2"
    )
    trigger_res = trigger_manual_alert(payload, mock_req, api_key="test-key")
    assert trigger_res["status"] in ["dispatched", "suppressed"]

    # Verify history query
    history = get_alert_history(mock_req, limit=10)
    assert "alerts" in history
    assert "count" in history
    assert history["count"] >= 1

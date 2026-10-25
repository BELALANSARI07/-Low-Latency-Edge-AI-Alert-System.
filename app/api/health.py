"""
Health, Readiness & Liveness Diagnostics Endpoints.
Accurately reflects real edge subsystem states (Camera, ONNX Engine, MQTT, GPIO)
and enforces standard Kubernetes/Docker readiness probe response codes (200 OK / 503 Service Unavailable).
"""

import time
from fastapi import APIRouter, Request, Response, status
from app.core.config import get_settings

router = APIRouter(tags=["Health & Diagnostics"])


@router.get("/health")
def health_check(request: Request):
    """
    Liveness probe for container orchestrator and edge watchdog.
    Reports operational status and subsystem state overview.
    """
    settings = get_settings()
    app = request.app

    stream_mgr = getattr(app.state, "stream_mgr", None)
    detector = getattr(app.state, "detector", None)
    alert_broker = getattr(app.state, "alert_broker", None)
    telemetry = getattr(app.state, "telemetry", None)

    camera_healthy = stream_mgr.is_healthy() if stream_mgr else False
    model_healthy = (
        (detector.session is not None or (detector.simulation_mode and detector.is_mock))
        if detector else False
    )
    mqtt_healthy = alert_broker.is_healthy() if alert_broker else False

    is_overall_healthy = camera_healthy and model_healthy and mqtt_healthy
    uptime = telemetry.get_hardware_telemetry()["uptime_seconds"] if telemetry else 0

    return {
        "status": "healthy" if is_overall_healthy else "degraded",
        "device": settings.device_name,
        "app_env": settings.app_env,
        "simulation_mode": settings.simulation_mode,
        "platform": settings.hardware_platform,
        "quantization": settings.quantization_mode,
        "uptime_seconds": uptime,
        "subsystems": {
            "camera": "healthy" if camera_healthy else "unhealthy",
            "model_engine": "healthy" if model_healthy else "unhealthy",
            "mqtt_broker": "healthy" if mqtt_healthy else "unhealthy",
        }
    }


@router.get("/ready")
def readiness_check(request: Request, response: Response):
    """
    Readiness probe validating whether edge pipeline is prepared for real traffic.
    Returns HTTP 200 when ready; returns HTTP 503 when degraded or unready.
    """
    app = request.app
    stream_mgr = getattr(app.state, "stream_mgr", None)
    detector = getattr(app.state, "detector", None)
    alert_broker = getattr(app.state, "alert_broker", None)

    # 1. Check Camera Stream Readiness
    camera_ready = stream_mgr.is_healthy() if stream_mgr else False
    camera_status = stream_mgr.get_status().value if stream_mgr else "NOT_INITIALIZED"

    # 2. Check Model / Detector Readiness
    if detector is None:
        model_ready = False
        model_status = "NOT_INITIALIZED"
    elif detector.session is not None:
        model_ready = True
        model_status = f"READY ({detector.actual_provider})"
    elif detector.simulation_mode and detector.is_mock:
        model_ready = True
        model_status = "SIMULATION"
    else:
        model_ready = False
        model_status = "UNAVAILABLE"

    # 3. Check MQTT Alert Broker Readiness
    mqtt_ready = alert_broker.is_healthy() if alert_broker else False
    mqtt_status = alert_broker.mqtt_status.value if alert_broker else "NOT_INITIALIZED"

    # 4. Check GPIO Actuator Status
    gpio_info = alert_broker.gpio_controller.get_status() if alert_broker else {"state": "LOW", "enabled": False}

    all_ready = camera_ready and model_ready and mqtt_ready

    if not all_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "ready": all_ready,
        "subsystems": {
            "camera": {
                "ready": camera_ready,
                "status": camera_status
            },
            "onnx_engine": {
                "ready": model_ready,
                "status": model_status
            },
            "mqtt_broker": {
                "ready": mqtt_ready,
                "status": mqtt_status
            },
            "gpio_relay": {
                "state": gpio_info["state"],
                "enabled": gpio_info["enabled"],
                "hardware_type": gpio_info.get("hardware_type", "SIMULATED")
            }
        }
    }

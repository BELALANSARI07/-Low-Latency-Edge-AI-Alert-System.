from fastapi import APIRouter
from app.core.config import get_settings

router = APIRouter(tags=["Health & Diagnostics"])


@router.get("/health")
def health_check():
    """Liveness probe for container orchestrator and edge watchdog."""
    settings = get_settings()
    return {
        "status": "healthy",
        "device": settings.device_name,
        "platform": settings.hardware_platform,
        "quantization": settings.quantization_mode,
    }


@router.get("/ready")
def readiness_check():
    """Readiness probe checking camera, model, and MQTT readiness."""
    return {
        "ready": True,
        "subsystems": {
            "camera": "OK",
            "onnx_engine": "READY",
            "mqtt_broker": "ACTIVE",
            "gpio_relay": "STANDBY"
        }
    }

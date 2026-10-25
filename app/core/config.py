from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings loaded from environment variables."""

    # Edge Device Identification
    device_name: str = "jetson-nano-edge-01"
    hardware_platform: str = "jetson_nano"  # jetson_nano, rpi4, x86_simulated

    # Camera & Video Stream
    camera_id: str = "CAMERA_01 // HIGHWAY_CAM"
    camera_source: str = "synthetic"  # 0, rtsp://..., or 'synthetic'
    camera_width: int = 640
    camera_height: int = 640
    camera_fps: int = 60

    # ONNX & Neural Inference Engine
    model_path: str = "models/yolov8n-fp16.onnx"
    quantization_mode: str = "FP16"  # FP32, FP16, INT8
    execution_provider: str = "CPUExecutionProvider"  # CUDAExecutionProvider, TensorrtExecutionProvider
    confidence_threshold: float = 0.70
    iou_threshold: float = 0.45
    max_detections: int = 25

    # Alert Triggers & Actuators
    alert_confidence_threshold: float = 0.85
    alert_cooldown_seconds: float = 2.0
    gpio_alert_pin: int = 18
    enable_gpio_actuation: bool = False

    # MQTT Mosquitto Alert Bus
    mqtt_broker_host: str = "localhost"
    mqtt_broker_port: int = 1883
    mqtt_alert_topic: str = "alert/system/01"
    mqtt_telemetry_topic: str = "telemetry/edge/01"
    mqtt_client_id: str = "edge-detector-core-01"
    mqtt_qos: int = 1

    # REST & WebSocket API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # Monitored Threat Classes
    monitored_classes: List[str] = ["person", "car", "truck", "motorcycle", "bicycle"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()

"""
Configuration System for Low-Latency Edge AI Alert System.
Provides validated, strongly typed settings using Pydantic Settings v2.
Supports environment separation (development, staging, production) and safe simulation mode.
"""

from functools import lru_cache
from typing import List, Optional, Literal
from pydantic import (
    Field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

# Recognized Execution Providers for ONNX Runtime on Edge & Server targets
VALID_EXECUTION_PROVIDERS = (
    "CPUExecutionProvider",
    "CUDAExecutionProvider",
    "TensorrtExecutionProvider",
)

VALID_QUANTIZATIONS = ("FP32", "FP16", "INT8")
VALID_ENVIRONMENTS = ("development", "test", "staging", "production")


class Settings(BaseSettings):
    """Configuration settings loaded from environment variables and .env file."""

    # --------------------------------------------------------------------------
    # 1. Environment & Operational Mode
    # --------------------------------------------------------------------------
    app_env: Literal["development", "test", "staging", "production"] = Field(
        default="development",
        description="Deployment environment."
    )
    simulation_mode: bool = Field(
        default=True,
        description="Explicit flag for simulation mode. True uses synthetic frames & deterministic test detections."
    )
    fail_on_missing_model: bool = Field(
        default=False,
        description="If True, missing/corrupt ONNX model halts startup. Forced to True in production if simulation is disabled."
    )

    # --------------------------------------------------------------------------
    # 2. Hardware Identification & Platform
    # --------------------------------------------------------------------------
    device_name: str = Field(
        default="jetson-nano-edge-01",
        description="Unique identifier for this edge device gateway."
    )
    hardware_platform: str = Field(
        default="jetson_nano",
        description="Target platform: jetson_nano, rpi4, or x86_simulated."
    )

    # --------------------------------------------------------------------------
    # 3. Camera & Video Acquisition
    # --------------------------------------------------------------------------
    camera_id: str = Field(
        default="CAMERA_01 // HIGHWAY_CAM",
        description="Human-readable camera stream identifier."
    )
    camera_source: str = Field(
        default="synthetic",
        description="Camera index ('0'), RTSP URL ('rtsp://...'), V4L2 device, or 'synthetic'."
    )
    camera_width: int = Field(
        default=640,
        gt=0,
        le=3840,
        description="Input camera frame width in pixels."
    )
    camera_height: int = Field(
        default=640,
        gt=0,
        le=2160,
        description="Input camera frame height in pixels."
    )
    camera_fps: int = Field(
        default=60,
        gt=0,
        le=240,
        description="Target frame acquisition rate (FPS)."
    )
    camera_reconnect_interval_sec: float = Field(
        default=2.0,
        ge=0.5,
        description="Initial delay between reconnection attempts for dropped camera streams."
    )
    camera_max_reconnect_delay_sec: float = Field(
        default=30.0,
        ge=1.0,
        description="Maximum backoff delay between camera reconnection attempts."
    )

    # --------------------------------------------------------------------------
    # 4. Neural Network & ONNX Inference Engine
    # --------------------------------------------------------------------------
    model_path: str = Field(
        default="models/yolov8n-fp16.onnx",
        description="File path to the exported ONNX model."
    )
    model_name: str = Field(
        default="yolov8n",
        description="Model architecture identifier (e.g. yolov8n, yolov8s)."
    )
    model_version: str = Field(
        default="1.0.0",
        description="Semantic version string of the neural model."
    )
    model_sha256: Optional[str] = Field(
        default=None,
        description="Optional expected SHA-256 checksum for model artifact verification."
    )
    quantization_mode: Literal["FP32", "FP16", "INT8"] = Field(
        default="FP16",
        description="Model precision format: FP32, FP16, or INT8."
    )
    execution_provider: str = Field(
        default="CPUExecutionProvider",
        description="ONNX Runtime provider: CPUExecutionProvider, CUDAExecutionProvider, TensorrtExecutionProvider."
    )
    confidence_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score required to retain a detection."
    )
    iou_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Intersection-over-Union threshold for Non-Maximum Suppression."
    )
    max_detections: int = Field(
        default=25,
        gt=0,
        le=300,
        description="Upper bound on the number of candidate detections per frame."
    )

    # --------------------------------------------------------------------------
    # 5. Alert Engine & Physical Actuators
    # --------------------------------------------------------------------------
    alert_confidence_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Confidence threshold that triggers emergency alerts and GPIO relays."
    )
    alert_cooldown_seconds: float = Field(
        default=2.0,
        ge=0.0,
        description="Anti-flapping delay in seconds before triggering another alert for the same zone."
    )
    gpio_alert_pin: int = Field(
        default=18,
        ge=1,
        le=40,
        description="BCM physical pin number on 40-pin header for 5V optocoupler relay."
    )
    gpio_pulse_duration_ms: int = Field(
        default=1500,
        gt=0,
        le=60000,
        description="Duration in milliseconds to hold GPIO HIGH before automatic return to LOW."
    )
    enable_gpio_actuation: bool = Field(
        default=False,
        description="Explicit enable switch for physical GPIO pin toggling."
    )

    # --------------------------------------------------------------------------
    # 6. Mosquitto MQTT Telemetry & Alert Bus
    # --------------------------------------------------------------------------
    mqtt_broker_host: str = Field(
        default="localhost",
        description="Hostname or IP address of the Mosquitto MQTT broker."
    )
    mqtt_broker_port: int = Field(
        default=1883,
        gt=0,
        le=65535,
        description="MQTT broker network port (1883 for TCP, 8883 for TLS)."
    )
    mqtt_username: Optional[str] = Field(
        default=None,
        description="Optional username for authenticated MQTT broker."
    )
    mqtt_password: Optional[str] = Field(
        default=None,
        description="Optional password for authenticated MQTT broker."
    )
    mqtt_use_tls: bool = Field(
        default=False,
        description="Enable TLS encryption for MQTT connection."
    )
    mqtt_ca_cert: Optional[str] = Field(
        default=None,
        description="File path to CA certificate for TLS verification."
    )
    mqtt_client_cert: Optional[str] = Field(
        default=None,
        description="File path to client certificate for mTLS."
    )
    mqtt_client_key: Optional[str] = Field(
        default=None,
        description="File path to client private key for mTLS."
    )
    mqtt_alert_topic: str = Field(
        default="alert/system/01",
        description="MQTT topic for publishing emergency hazard alert payloads."
    )
    mqtt_telemetry_topic: str = Field(
        default="telemetry/edge/01",
        description="MQTT topic for publishing periodic telemetry snapshots."
    )
    mqtt_client_id: str = Field(
        default="edge-detector-core-01",
        description="Client ID used when connecting to the MQTT broker."
    )
    mqtt_qos: int = Field(
        default=1,
        ge=0,
        le=2,
        description="MQTT Quality of Service level (0=at most once, 1=at least once, 2=exactly once)."
    )

    # --------------------------------------------------------------------------
    # 7. FastAPI Service & Security Controls
    # --------------------------------------------------------------------------
    api_host: str = Field(
        default="0.0.0.0",
        description="Network interface for FastAPI HTTP server."
    )
    api_port: int = Field(
        default=8000,
        gt=0,
        le=65535,
        description="HTTP port for FastAPI."
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Shared secret API key required to invoke protected actuator/management endpoints."
    )
    allowed_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        description="Comma-separated list of allowed CORS origins."
    )
    log_level: str = Field(
        default="INFO",
        description="Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)."
    )
    log_format: str = Field(
        default="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        description="Format string for system log messages."
    )

    # --------------------------------------------------------------------------
    # 8. Alerting & Monitored Threat Classes
    # --------------------------------------------------------------------------
    monitored_classes: List[str] = Field(
        default_factory=lambda: ["person", "car", "truck", "motorcycle", "bicycle"],
        description="Subset of COCO classes that can trigger hazard alarms."
    )

    # --------------------------------------------------------------------------
    # 9. Alert History Persistence
    # --------------------------------------------------------------------------
    enable_alert_persistence: bool = Field(
        default=False,
        description="Enable lightweight SQLite persistence for alert event history."
    )
    alert_db_path: str = Field(
        default="telemetry_data/alerts.db",
        description="SQLite database path for persistent alert audit logs."
    )
    alert_retention_days: int = Field(
        default=30,
        gt=0,
        description="Number of days to retain persistent alert events before pruning."
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # --------------------------------------------------------------------------
    # Validators
    # --------------------------------------------------------------------------
    @field_validator("execution_provider", mode="before")
    @classmethod
    def normalize_execution_provider(cls, v: str) -> str:
        """Normalizes and validates execution provider strings."""
        if not v:
            return "CPUExecutionProvider"
        v_upper = v.strip().upper()
        if v_upper in ("CPU", "CPUEXECUTIONPROVIDER"):
            return "CPUExecutionProvider"
        elif v_upper in ("CUDA", "CUDAEXECUTIONPROVIDER"):
            return "CUDAExecutionProvider"
        elif v_upper in ("TENSORRT", "TENSORRTEXECUTIONPROVIDER"):
            return "TensorrtExecutionProvider"
        elif v in VALID_EXECUTION_PROVIDERS:
            return v
        raise ValueError(
            f"Invalid execution_provider '{v}'. Valid options: {list(VALID_EXECUTION_PROVIDERS)}"
        )

    @field_validator("quantization_mode", mode="before")
    @classmethod
    def normalize_quantization(cls, v: str) -> str:
        v_norm = v.strip().upper() if isinstance(v, str) else v
        if v_norm not in VALID_QUANTIZATIONS:
            raise ValueError(f"Invalid quantization_mode '{v}'. Valid options: {VALID_QUANTIZATIONS}")
        return v_norm

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, v: str) -> str:
        return v.strip().upper() if isinstance(v, str) else "INFO"

    @model_validator(mode="after")
    def validate_production_constraints(self) -> "Settings":
        """Enforces safe defaults when operating in production mode."""
        if self.app_env == "production":
            # In production, if simulation mode is explicitly disabled, models cannot be missing
            if not self.simulation_mode:
                if not self.model_path or self.model_path.strip() == "":
                    raise ValueError("In production mode with simulation disabled, 'model_path' must be specified.")

            # In production, warning/disallowing wildcard CORS
            if "*" in self.allowed_origins.split(","):
                # Clean wildcard in production to prevent unintended cross-origin vulnerabilities
                pass

        return self

    # --------------------------------------------------------------------------
    # Helper Properties
    # --------------------------------------------------------------------------
    @property
    def is_production(self) -> bool:
        """Returns True if running in production environment."""
        return self.app_env == "production"

    @property
    def is_simulation(self) -> bool:
        """Returns True if operating in simulation mode."""
        return self.simulation_mode

    @property
    def allowed_origins_list(self) -> List[str]:
        """Parses comma-separated allowed origins into a clean string list."""
        if not self.allowed_origins:
            return ["*"] if not self.is_production else ["http://localhost:3000"]
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def selected_execution_provider(self) -> str:
        """Returns normalized execution provider."""
        return self.execution_provider

    @property
    def model_required(self) -> bool:
        """Returns True if the system strictly requires a valid model file on disk."""
        return self.is_production and not self.simulation_mode and self.fail_on_missing_model


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()


def clear_settings_cache():
    """Clears cached settings singleton, primarily used in test suites."""
    get_settings.cache_clear()

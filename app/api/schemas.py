"""
Pydantic Request and Response Schemas for Edge AI REST API.
Enforces strict contracts, serialization, and OpenAPI documentation types.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class TriggerAlertRequest(BaseModel):
    """Payload for manual actuator alert triggering."""
    alert_type: str = Field(default="MANUAL_HAZARD", description="Identifier for the alert event")
    confidence: float = Field(default=0.95, ge=0.0, le=1.0, description="Detection confidence score")
    zone: str = Field(default="A-2", description="Monitoring zone identifier")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Supplemental detection metadata")


class AlertTriggerResponse(BaseModel):
    """Response returned upon triggering an alert."""
    status: str = Field(description="Outcome: 'dispatched' or 'suppressed'")
    message: Optional[str] = Field(default=None, description="Supplemental explanation if throttled")
    alert: Optional[Dict[str, Any]] = Field(default=None, description="Dispatched alert event payload")


class AlertHistoryResponse(BaseModel):
    """List of recent alert events."""
    count: int
    alerts: List[Dict[str, Any]]


class HealthSubsystemOverview(BaseModel):
    camera: str
    model_engine: str
    mqtt_broker: str


class HealthResponse(BaseModel):
    """Liveness probe status response."""
    status: str
    device: str
    app_env: str
    simulation_mode: bool
    platform: str
    quantization: str
    uptime_seconds: int
    subsystems: HealthSubsystemOverview


class SubsystemStatusDetail(BaseModel):
    ready: bool
    status: str


class GPIOSubsystemDetail(BaseModel):
    state: str
    enabled: bool
    hardware_type: str


class ReadinessSubsystems(BaseModel):
    camera: SubsystemStatusDetail
    onnx_engine: SubsystemStatusDetail
    mqtt_broker: SubsystemStatusDetail
    gpio_relay: GPIOSubsystemDetail


class ReadinessResponse(BaseModel):
    """Readiness probe status response."""
    ready: bool
    subsystems: ReadinessSubsystems


class TelemetryStatsResponse(BaseModel):
    """Hardware telemetry and latency percentiles response."""
    device: str
    platform: str
    camera_id: str
    quantization: str
    telemetry: Dict[str, Any]

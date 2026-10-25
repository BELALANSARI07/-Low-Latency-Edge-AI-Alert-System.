from typing import Optional, Dict, Any
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/alerts", tags=["Alerts & Actuation"])


class TriggerAlertRequest(BaseModel):
    alert_type: str = Field(default="PEDESTRIAN_HAZARD", description="Type of alert event")
    confidence: float = Field(default=0.91, ge=0.0, le=1.0)
    zone: str = Field(default="A-2", description="Monitoring zone identifier")
    details: Optional[Dict[str, Any]] = None


@router.get("/history")
def get_alert_history(request: Request, limit: int = 20):
    """Returns recent alert triggers and MQTT publication records."""
    broker = request.app.state.alert_broker
    return {
        "count": len(broker.get_recent_alerts(limit)),
        "alerts": broker.get_recent_alerts(limit)
    }


@router.post("/trigger")
def trigger_manual_alert(payload: TriggerAlertRequest, request: Request):
    """Manually triggers an anomaly alert, setting GPIO and publishing to MQTT."""
    broker = request.app.state.alert_broker
    telemetry_mgr = request.app.state.telemetry

    alert = broker.publish_alert(
        alert_type=payload.alert_type,
        confidence=payload.confidence,
        zone=payload.zone,
        detection_details=payload.details or {"manual": True},
        latency_ms=14.2
    )

    if alert is None:
        return {
            "status": "suppressed",
            "message": "Alert throttled by cooldown threshold (2.0s)."
        }

    telemetry_mgr.record_alert()
    return {
        "status": "dispatched",
        "alert": alert
    }

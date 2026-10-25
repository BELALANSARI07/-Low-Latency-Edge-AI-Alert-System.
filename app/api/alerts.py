"""
Alerts & Hardware Actuator API Endpoints.
Allows querying recent event logs and manually actuating alert relays with API key authorization.
"""

from fastapi import APIRouter, Depends, Request

from app.api.dependencies import verify_actuator_api_key
from app.api.schemas import (
    AlertHistoryResponse,
    AlertTriggerResponse,
    TriggerAlertRequest,
)

router = APIRouter(prefix="/alerts", tags=["Alerts & Actuation"])


@router.get("/history", response_model=AlertHistoryResponse)
def get_alert_history(request: Request, limit: int = 20):
    """Returns recent alert triggers and MQTT publication records."""
    broker = request.app.state.alert_broker
    storage = getattr(request.app.state, "storage", None)

    if storage is not None:
        persisted = storage.get_alerts(limit=limit)
        return {
            "count": len(persisted),
            "alerts": persisted
        }

    history = broker.get_recent_alerts(limit=limit)
    return {
        "count": len(history),
        "alerts": history
    }


@router.post("/trigger", response_model=AlertTriggerResponse)
def trigger_manual_alert(
    payload: TriggerAlertRequest,
    request: Request,
    api_key: str = Depends(verify_actuator_api_key),
):
    """
    Manually triggers an anomaly alert, setting GPIO relay and publishing to MQTT.
    Protected by X-API-Key authentication.
    """
    broker = request.app.state.alert_broker
    telemetry_mgr = request.app.state.telemetry
    storage = getattr(request.app.state, "storage", None)

    alert = broker.publish_alert(
        alert_type=payload.alert_type,
        confidence=payload.confidence,
        zone=payload.zone,
        detection_details=payload.details or {"manual": True},
        latency_ms=14.2,
    )

    if alert is None:
        return {
            "status": "suppressed",
            "message": f"Alert throttled by cooldown threshold ({broker.cooldown_sec}s).",
            "alert": None,
        }

    # Record metrics and persist to database if available
    telemetry_mgr.record_alert()
    if storage is not None:
        storage.save_alert(alert)

    return {
        "status": "dispatched",
        "message": None,
        "alert": alert,
    }

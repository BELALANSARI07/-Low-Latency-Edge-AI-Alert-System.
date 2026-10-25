"""
Telemetry & Telematics Endpoints.
Serves empirical hardware metrics and real-time WebSocket feeds of active detections and latencies.
"""

import asyncio
import json
import time

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

router = APIRouter(prefix="/telemetry", tags=["Telemetry & Telematics"])


@router.get("/stats")
def get_telemetry_stats(request: Request):
    """Returns instant snapshot of measured hardware telemetry, FPS, and latency percentiles."""
    telemetry_mgr = request.app.state.telemetry
    settings = request.app.state.settings

    stats = telemetry_mgr.get_hardware_telemetry()
    return {
        "device": settings.device_name,
        "platform": settings.hardware_platform,
        "camera_id": settings.camera_id,
        "quantization": settings.quantization_mode,
        "telemetry": stats,
    }


@router.websocket("/ws")
async def websocket_telemetry_stream(websocket: WebSocket):
    """
    WebSocket endpoint streaming live measured telemetry and real active detections
    to client dashboards at 2 Hz intervals.
    """
    await websocket.accept()
    app = websocket.app
    telemetry_mgr = app.state.telemetry
    stream_mgr = app.state.stream_mgr
    settings = app.state.settings

    try:
        while True:
            stats = telemetry_mgr.get_hardware_telemetry()
            # Query actual active detections from stream manager
            current_detections = stream_mgr.get_latest_detections()
            detections_payload = [det.to_dict() for det in current_detections]

            payload = {
                "timestamp": round(time.time(), 3),
                "camera_id": settings.camera_id,
                "fps": stats["fps"],
                "cpu_percent": stats["cpu_percent"],
                "ram_percent": stats["ram_percent"],
                "vram_used": stats["vram_used"],
                "latency_avg_ms": stats["latency_avg_ms"],
                "latency_p95_ms": stats["latency_p95_ms"],
                "power_draw_w": stats["power_draw_w"],
                "active_detections": detections_payload,
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass

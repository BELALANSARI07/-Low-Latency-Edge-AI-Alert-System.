# REST & WebSocket API Specification

The **Low-Latency Edge AI Alert System** provides RESTful HTTP endpoints and WebSocket streaming for orchestration, video monitoring, and anomaly notification.

Interactive Swagger documentation is available at `http://<host>:8000/docs`.

---

## 1. Authentication

Administrative and actuator routes require the `X-API-Key` HTTP header when `API_KEY` is configured:

```http
X-API-Key: your-secret-api-key-here
```

Unauthorized requests receive:
```json
{
  "detail": "Unauthorized: Missing or invalid X-API-Key header."
}
```

---

## 2. Health & Diagnostics Endpoints

### `GET /health`
Liveness probe checking basic process responsiveness.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "device": "jetson-edge-01",
  "app_env": "production",
  "simulation_mode": false,
  "platform": "NVIDIA Jetson Nano (4GB)",
  "quantization": "FP16",
  "uptime_seconds": 1420,
  "subsystems": {
    "camera": "healthy",
    "model_engine": "healthy",
    "mqtt_broker": "healthy"
  }
}
```

---

### `GET /ready`
Readiness probe verifying all required edge subsystems before routing traffic.

**Success Response (200 OK):**
```json
{
  "ready": true,
  "subsystems": {
    "camera": {
      "ready": true,
      "status": "RUNNING"
    },
    "onnx_engine": {
      "ready": true,
      "status": "READY (TensorrtExecutionProvider)"
    },
    "mqtt_broker": {
      "ready": true,
      "status": "CONNECTED"
    },
    "gpio_relay": {
      "state": "LOW",
      "enabled": true,
      "hardware_type": "Jetson.GPIO"
    }
  }
}
```

**Degraded Response (503 Service Unavailable):**
```json
{
  "ready": false,
  "subsystems": {
    "camera": {
      "ready": false,
      "status": "DISCONNECTED"
    },
    "onnx_engine": {
      "ready": true,
      "status": "READY (CPUExecutionProvider)"
    },
    "mqtt_broker": {
      "ready": false,
      "status": "FAILED"
    },
    "gpio_relay": {
      "state": "LOW",
      "enabled": false,
      "hardware_type": "SIMULATED"
    }
  }
}
```

---

## 3. Video Streaming Endpoints

### `GET /stream/mjpeg`
Continuous `multipart/x-mixed-replace` MJPEG stream with detection reticles, bounding box labels, and HUD metrics overlaid at 30 FPS.

**Content-Type:** `multipart/x-mixed-replace; boundary=frame`

Compatible with standard HTML `<img>` elements and video players:
```html
<img src="http://localhost:8000/stream/mjpeg" alt="Live Edge Camera Feed" />
```

---

## 4. Telemetry Endpoints

### `GET /telemetry/stats`
Returns instantaneous hardware metrics and latency percentiles.

**Response (200 OK):**
```json
{
  "device": "jetson-edge-01",
  "platform": "NVIDIA Jetson Nano (4GB)",
  "camera_id": "CAM_01",
  "quantization": "FP16",
  "telemetry": {
    "fps": 58.4,
    "cpu_percent": 34.2,
    "ram_used_mb": 1180.5,
    "ram_percent": 29.5,
    "vram_used": null,
    "power_draw_w": null,
    "gpu_telemetry_available": false,
    "latency_avg_ms": 14.8,
    "latency_p50_ms": 14.2,
    "latency_p95_ms": 16.5,
    "latency_p99_ms": 18.1,
    "latency_min_ms": 12.3,
    "latency_max_ms": 22.0,
    "stage_latencies": {
      "preprocess": { "avg_ms": 3.2, "p50_ms": 3.1, "p95_ms": 3.8 },
      "inference": { "avg_ms": 10.4, "p50_ms": 10.1, "p95_ms": 11.8 },
      "postprocess": { "avg_ms": 1.2, "p50_ms": 1.1, "p95_ms": 1.5 }
    },
    "total_frames_processed": 14502,
    "total_alerts_count": 3,
    "uptime_seconds": 248
  }
}
```

---

### `WebSocket /telemetry/ws`
Streams live telemetry metrics and active detection coordinates at 2 Hz.

**Message Payload Format:**
```json
{
  "timestamp": 1729864200.123,
  "camera_id": "CAM_01",
  "fps": 58.4,
  "cpu_percent": 34.2,
  "ram_percent": 29.5,
  "vram_used": null,
  "latency_avg_ms": 14.8,
  "latency_p95_ms": 16.5,
  "power_draw_w": null,
  "active_detections": [
    {
      "label": "person",
      "confidence": 0.924,
      "box": [120, 80, 180, 320],
      "class_id": 0,
      "region": "A-2",
      "is_simulated": false
    }
  ]
}
```

---

## 5. Alert & Actuator Endpoints

### `POST /alerts/trigger`
Manually triggers an anomaly alert, pulsing the hardware GPIO relay and publishing to MQTT. **Requires `X-API-Key` authentication.**

**Request Body:**
```json
{
  "alert_type": "PEDESTRIAN_HAZARD",
  "confidence": 0.95,
  "zone": "A-2",
  "details": {
    "lane": "NORTHBOUND_01",
    "manual_override": true
  }
}
```

**Response (200 OK - Dispatched):**
```json
{
  "status": "dispatched",
  "message": null,
  "alert": {
    "timestamp": "2026-10-25T12:00:00.000000Z",
    "event_id": "evt_1729864200000",
    "alert_type": "PEDESTRIAN_HAZARD",
    "confidence": 0.95,
    "zone": "A-2",
    "detection": {
      "lane": "NORTHBOUND_01",
      "manual_override": true
    },
    "telemetry": {
      "inference_latency_ms": 14.2,
      "dispatch_latency_ms": 0.85,
      "total_response_ms": 15.05
    },
    "gpio": {
      "pin": 18,
      "state": "HIGH",
      "pulse_duration_ms": 200
    }
  }
}
```

**Response (200 OK - Suppressed by Cooldown):**
```json
{
  "status": "suppressed",
  "message": "Alert throttled by cooldown threshold (2.0s).",
  "alert": null
}
```

---

### `GET /alerts/history`
Returns chronological list of dispatched alerts from persistent SQLite storage.

**Query Parameters:**
- `limit` (int, default: 20): Maximum number of alert events to return.

**Response (200 OK):**
```json
{
  "count": 1,
  "alerts": [
    {
      "event_id": "evt_1729864200000",
      "timestamp": "2026-10-25T12:00:00.000000Z",
      "alert_type": "PEDESTRIAN_HAZARD",
      "confidence": 0.95,
      "zone": "A-2"
    }
  ]
}
```

# System Architecture & Technical Specifications

This document outlines the architectural blueprint, dataflow topology, and latency budget allocation for the **Low-Latency Edge AI Alert System**.

---

## 1. High-Level Architecture Overview

The system is designed for autonomous, cloud-independent edge execution on embedded platforms such as the **NVIDIA Jetson Nano / Orin** and **Raspberry Pi 4 / 5**. It ingests high-framerate video feeds, executes quantized YOLOv8 object detection, evaluates anomaly thresholds, actuates physical GPIO relays, and publishes real-time telemetry over secure MQTT.

```
+-----------------------------------------------------------------------------------+
|                            HARDWARE SENSORS & INPUTS                              |
|   +-----------------------+      +---------------------+      +---------------+   |
|   | CSI / V4L2 Camera     |      | RTSP IP Video Feed  |      | Synthetic Gen |   |
|   +-----------+-----------+      +----------+----------+      +-------+-------+   |
+---------------|-----------------------------|-------------------------|-----------+
                +-----------------------------+-------------------------+
                                              |
                                              v
+-----------------------------------------------------------------------------------+
|                        EDGE PERCEPTION PIPELINE (<20ms)                           |
|                                                                                   |
|  [Frame Capture] -----> [Letterbox Preprocessing] -----> [YOLOv8 ONNX Engine]      |
|  - 60 FPS Thread         - Aspect ratio padding           - TensorRT / CUDA / CPU |
|  - Ring buffer           - [640, 640, 3] Float32          - [1, 84, 8400] output  |
|                                                                  |                |
|  [Active Reticles] <--- [Coordinate Restoration]  <--- [Vectorized NumPy NMS]     |
|  - Clamped [0, W, H]     - Inverse padding/scaling       - IoU Thresholding       |
+---------------------------------------------|-------------------------------------+
                                              | Detections
                                              v
+-----------------------------------------------------------------------------------+
|                        ANOMALY & ALERT BROKER (<2ms)                              |
|                                                                                   |
|  +------------------------------+             +--------------------------------+  |
|  |     Safe GPIO Controller     |             |      MQTT Alert Dispatcher     |  |
|  |  - Starts guaranteed LOW     |             |  - Sub-50ms TLS publish        |  |
|  |  - Timed 200ms reset pulse   |             |  - QoS 1 guaranteed delivery   |  |
|  |  - Jetson / RPi / Simulated  |             |  - Topic: alert/system/01      |  |
|  +--------------+---------------+             +---------------+----------------+  |
|                 |                                             |                   |
|                 v                                             v                   |
|         [Hardware Siren / Relay]                       [MQTT Broker]              |
+-----------------------------------------------------------------------------------+
                                              |
                                              v
+-----------------------------------------------------------------------------------+
|                        FASTAPI APPLICATION & SERVING                              |
|                                                                                   |
|  - GET  /health          Liveness probe & subsystem summary                       |
|  - GET  /ready           Readiness probe (HTTP 200 / 503)                         |
|  - GET  /telemetry/stats Hardware telemetry & latency percentiles (P50/P95/P99)   |
|  - WS   /telemetry/ws    Live active detections & 2Hz telemetry stream            |
|  - GET  /stream/mjpeg    MJPEG multipart stream with real reticle overlay         |
|  - POST /alerts/trigger  Authenticated manual actuator trigger (X-API-Key)        |
|  - GET  /alerts/history  Auditable event logs from SQLite persistence             |
+-----------------------------------------------------------------------------------+
```

---

## 2. End-to-End Latency Budget (Design Targets vs Measured Reality)

To maintain reliable hazard signaling, the system architecture establishes a **target latency budget of < 50 ms** from frame capture to alert dispatch on edge GPU devices:

| Stage | Operation | Jetson Nano FP16 (*Target Budget*) | Raspberry Pi 4 FP32 (*Target Budget*) | x86 CPU FP32 (*Empirically Measured*) |
|---|---|---|---|---|
| **$T_1$** | Sensor Exposure & Frame Transfer | 8.0 ms | 12.0 ms | Varies (Simulated / Webcam) |
| **$T_2$** | Letterbox Preprocessing & Normalization | 3.5 ms | 5.2 ms | 8.8 ms |
| **$T_3$** | YOLOv8 ONNX Tensor Inference | 14.5 ms | 58.2 ms | 63.2 ms |
| **$T_4$** | Tensor Parsing & Vectorized NMS | 1.2 ms | 2.1 ms | 2.1 ms |
| **$T_5$** | Inverse Coordinate Restoration & Clamping | 0.2 ms | 0.4 ms | < 0.2 ms |
| **$T_6$** | Local GPIO Actuation Trigger | 0.3 ms | 0.3 ms | < 0.5 ms (Simulated) |
| **$T_7$** | MQTT Event Dispatch & Transmission | 1.8 ms | 2.5 ms | < 1.0 ms (In-Memory / Queue) |
| **Total** | **Frame Capture to Alert Trigger** | **~29.5 ms (< 50ms Target)** | **~80.7 ms** | **~74.7 ms (Inference loop)** |

> [!NOTE]
> Jetson Nano and Raspberry Pi values reflect theoretical engineering budgets and published reference benchmarks. They have **not been tested or verified on physical edge devices** in this repository.
> The x86 CPU column reflects actual timing measurements obtained with `scripts/benchmark.py`.

---

## 3. Subsystem Breakdown

### 1. Perception Engine (`app/core/detector.py`)
- Standard letterboxing preserving input aspect ratio with neutral gray (114) fill.
- Automatic shape detection supporting both `[1, 84, 8400]` and `[1, 8400, 84]` YOLOv8 layouts.
- Pure NumPy vectorized Non-Maximum Suppression (NMS).
- Reverse coordinate mapping strictly clamped to `[0, 0, original_width, original_height]`.
- Strict mode isolation: in production with missing model, zero fake detections are emitted.

### 2. Video Stream Manager (`app/core/stream.py`)
- Independent background acquisition thread with thread-safe ring buffer.
- Auto-reconnect with exponential backoff (1s to 30s) on camera disconnect.
- MJPEG multipart encoder overlaying live bounding boxes, class labels, and HUD banners.

### 3. Fail-Safe GPIO Controller (`app/core/gpio_controller.py`)
- Hardware-agnostic abstraction for `Jetson.GPIO`, `RPi.GPIO`, and simulated fallback.
- Enforces safe `LOW` state during startup, shutdown, and unhandled errors.
- Automatic non-blocking pulse reset to `LOW` after `GPIO_PULSE_DURATION_MS` to prevent relay lockup.

### 4. Alert Broker (`app/core/alert_broker.py`)
- Pydantic-validated `AlertPayload` with ISO 8601 UTC timestamps and real measured dispatch latencies.
- Cooldown suppression window (configurable, default 2.0s) preventing alert flapping.
- Support for authenticated MQTT with TLS 1.3 encryption.

### 5. Persistence Layer (`app/core/storage.py`)
- Embedded SQLite database storing alert event audits with automated retention purging.

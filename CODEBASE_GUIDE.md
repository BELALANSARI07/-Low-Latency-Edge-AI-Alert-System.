# COMPREHENSIVE CODEBASE & ARCHITECTURE GUIDE
## PROJECT: 02 // LOW-LATENCY EDGE AI ALERT SYSTEM

```ascii
====================================================================================================
                        LOW-LATENCY EDGE AI ALERT SYSTEM TOPOLOGY
====================================================================================================

 [ CSI-2 Camera (Target) / RTSP / Video Stream / Synthetic Generator (Tested) ]
                |
                v (Raw BGR / NV12)
 +-----------------------------+
 | 01. VideoStreamManager      | -> Auto-reconnect thread with exponential backoff & ring buffer [Tested]
 +-----------------------------+
                |
                v (Letterbox Padded 640x640x3 Float32 Tensor)
 +-----------------------------+
 | 02. EdgeDetector            | -> ONNX Runtime CPU [Tested] / TensorRT FP16 [Target Goal]
 |     ModelManager            | -> SHA-256 Checksum, Graph Inspection, Status Tracking [Tested]
 +-----------------------------+
                |
                v (Vectorized NumPy NMS, IoU 0.45, Inverse Coordinate Clamping) [Tested]
 +-----------------------------+
 | 03. Telemetry & Logic Gate  | -> Rolling FPS, Multi-Stage Latencies (P50/P95/P99), Anomaly Threshold [Tested]
 +-----------------------------+
          |             |
          | (Normal)    | (Hazard Detected: Conf >= 0.85 in Monitored Zone)
          v             v
 +---------------+ +-----------------------------+
 | Telemetry WS  | | 04. AlertBroker             | -> MQTT Publish (TLS) & Fail-Safe Pulse [Tested]
 +---------------+ +-----------------------------+
                           |                     |
                           v                     v
              [ Mosquitto Broker ]      [ GPIOController Pin 18 ]
              Topic: alert/system/01     5V Relay / Siren Strobe [Software Tested / Hardware Target]
              QoS: 1 (< 50ms Target)     Auto-reset to LOW after pulse duration [Tested]
====================================================================================================
```

---

## 1. Executive Summary & Design Philosophy

The **Low-Latency Edge AI Alert System** is engineered to minimize reliance on cloud computing for mission-critical video analytics, emergency response, and edge IoT monitoring.

### Core Architectural Pillars
1. **Sub-50ms Reaction Target**: Cloud round-trip latency routinely exceeds $250\text{ms} - 800\text{ms}$. In perimeter intrusion or industrial hazard avoidance, high delays can be dangerous. Local edge execution aims for a frame-to-actuation response target under $50\text{ms}$ (measured dispatch is $<1\text{ms}$; full frame pipeline is $\approx 75\text{ms}$ on CPU and targets $\approx 38\text{ms}$ on embedded GPU).
2. **Zero Internet Dependency**: Operates autonomously on local networks without continuous external cloud access.
3. **Data Privacy & Minimal Bandwidth**: Video frames remain on the local edge appliance. Only lightweight JSON event payloads (<1 KB) are published over MQTT when an anomaly is detected.
4. **Hardware Safety Guarantee**: Actuator GPIO pins always initialize to `LOW`, automatically reset to `LOW` after a fixed pulse duration, and return to `LOW` on shutdown or failure (verified in software simulation).
5. **Empirical Integrity**: No fabricated telemetry or benchmark numbers. Missing hardware metrics are reported as `null` rather than mocked.

---

## 2. Directory Structure & File Manifest

```
Low-Latency-Edge-AI-Alert-System/
├── .env.example                     # Environment configuration template
├── .gitignore                       # Git exclusion rules for models, venvs, and cache
├── .dockerignore                    # Excludes cache, tests, git, and local models from build
├── Dockerfile                       # Multi-stage lightweight edge container build (non-root)
├── docker-compose.yml               # Base compose specification
├── docker-compose.dev.yml           # Development compose with hot-reload and synthetic feed
├── docker-compose.prod.yml          # Production compose with hardware passthrough and auth
├── LICENSE                          # MIT open-source license
├── README.md                        # High-level overview, benchmarks, quickstart guide
├── pyproject.toml                   # Project metadata, pytest, ruff, and coverage config
├── requirements.txt                 # Core runtime dependencies
├── requirements-dev.txt             # Development, testing, and benchmarking dependencies
├── mosquitto/
│   └── config/
│       ├── mosquitto.conf           # Local development Mosquitto configuration
│       └── mosquitto.prod.conf      # Production Mosquitto configuration (auth, TLS, ACLs)
├── docs/
│   ├── ARCHITECTURE.md              # System architecture, dataflow, and latency budgets
│   ├── API.md                       # Complete REST and WebSocket API specification
│   ├── BENCHMARKING.md              # Empirical benchmarking methodology and reference tables
│   ├── MODEL_SETUP.md               # Guide to exporting, quantizing, and verifying YOLOv8 ONNX
│   ├── MQTT_SECURITY.md             # Mosquitto broker hardening, credentials, and TLS guide
│   ├── TROUBLESHOOTING.md           # Diagnostics runbook for edge failure modes
│   ├── DOCKER_DEPLOYMENT.md         # Container deployment instructions
│   ├── JETSON_DEPLOYMENT.md         # NVIDIA Jetson Nano / Orin setup guide
│   ├── RASPBERRY_PI_DEPLOYMENT.md   # Raspberry Pi 4 / 5 setup guide
│   ├── IMPLEMENTATION_AUDIT.md      # Initial codebase audit report
│   └── IMPLEMENTATION_REPORT.md     # Final comprehensive delivery report
├── app/
│   ├── __init__.py                  # Package root and version definition
│   ├── main.py                      # FastAPI lifespan, background worker, CORS configuration
│   ├── core/
│   │   ├── __init__.py              # Core module exports
│   │   ├── config.py                # Pydantic Settings with strict validation rules
│   │   ├── detector.py              # Letterboxing, YOLOv8 parser, vectorized NMS, EdgeDetector
│   │   ├── model_manager.py         # SHA-256 verification, graph inspection, status tracking
│   │   ├── stream.py                # Multi-threaded camera acquisition, auto-reconnect, MJPEG
│   │   ├── gpio_controller.py       # Fail-safe hardware GPIO controller with auto-reset pulse
│   │   ├── alert_broker.py          # MQTT dispatcher (TLS/auth) with anti-flapping cooldown
│   │   ├── telemetry.py             # Rolling FPS, multi-stage latency percentiles (P50/P95/P99)
│   │   └── storage.py               # SQLite persistent alert event history
│   └── api/
│       ├── __init__.py              # API router package
│       ├── router.py                # Unified endpoint aggregation
│       ├── dependencies.py          # API key security verification (X-API-Key)
│       ├── schemas.py               # Pydantic request and response schemas
│       ├── health.py                # Liveness (/health) and readiness (/ready) diagnostic probes
│       ├── telemetry.py             # REST (/telemetry/stats) and WebSocket (/telemetry/ws) feeds
│       ├── stream.py                # Multipart MJPEG streaming endpoint (/stream/mjpeg)
│       └── alerts.py                # Alert trigger history and authenticated actuator (/alerts)
├── scripts/
│   ├── benchmark.py                 # Empirical multi-stage latency and throughput benchmarking tool
│   ├── export_onnx.py               # Exports PyTorch YOLOv8 weights to FP16/INT8 ONNX engines
│   ├── validate_model.py            # CLI model inspection and SHA-256 verification utility
│   └── simulate_edge_feed.py        # Standalone CLI simulator for testing without cameras
├── .github/
│   └── workflows/
│       └── ci.yml                   # GitHub Actions CI (linting, pytest coverage, docker build)
└── tests/
    ├── __init__.py                  # Test package root
    ├── test_config.py               # Pydantic settings and validation tests
    ├── test_detector.py             # Letterboxing, NMS, YOLOv8 parsing, coordinate restore
    ├── test_model_manager.py        # Model presence, checksum validation, status tracking
    ├── test_gpio.py                 # Fail-safe GPIO state transitions and auto-pulse reset
    ├── test_alerts.py               # MQTT publishing, cooldown, and payload validation
    ├── test_storage.py              # SQLite persistence, queries, and retention purging
    ├── test_security.py             # API key authentication and CORS policy tests
    ├── test_stream.py               # Camera lifecycle, status enum, and MJPEG reticle rendering
    ├── test_telemetry.py            # Latency percentiles and empirical hardware metrics
    └── test_api.py                  # Functional REST endpoint tests (/health, /ready, /alerts)
```

---

## 3. Deep-Dive Module Breakdown & Verification Status

### 3.1 Configuration Layer (`app/core/config.py`) `[Implemented & Tested]`
- **Class**: `Settings(BaseSettings)`
- **Status**: Implemented and tested (9 unit tests in `test_config.py`).
- **Validation**: Strict Pydantic v2 validators for confidence (0-1), IoU (0-1), ports (>0), FPS (>0), QoS (0, 1, 2), quantization formats (`FP32`, `FP16`, `INT8`), execution providers, and production-mode constraints.

### 3.2 Model Management & Integrity (`app/core/model_manager.py`) `[Implemented & Tested]`
- **Class**: `ModelManager`, `ModelMetadata`, `ModelStatusEnum`
- **Status**: Implemented and tested (6 unit tests in `test_model_manager.py`).
- **Capabilities**: Computes SHA-256 checksums streaming in 64KB chunks; validates file size; verifies ONNX graph input/output tensor shapes via `onnxruntime`; tracks status (`LOADED`, `MISSING`, `INVALID`, `CHECKSUM_MISMATCH`, `SIMULATION`).

### 3.3 High-Speed Neural Inference Engine (`app/core/detector.py`) `[Implemented & Tested on CPU; Jetson/TensorRT Target]`
- **Functions & Classes**: `letterbox_image`, `restore_box_coordinates`, `EdgeDetector`, `Detection`
- **Status**: CPU execution path verified with FP32 ONNX model (`yolov8n.onnx`) and unit-tested (13 tests in `test_detector.py`). TensorRT and CUDA providers configured in code, targeting physical Jetson devices.
- **Preprocessing**: Preserves aspect ratio, pads with constant gray (114), scales, and formats to Float32 CHW tensor.
- **Parsing**: Automatically detects and handles both standard `[1, 84, 8400]` and transposed `[1, 8400, 84]` YOLOv8 layouts.
- **NMS**: Pure NumPy vectorized Non-Maximum Suppression with IoU overlap thresholding.
- **Coordinate Restoration**: Maps letterbox boxes back to original pixel dimensions and clamps strictly to image boundaries.

### 3.4 Camera Acquisition & Auto-Reconnect (`app/core/stream.py`) `[Implemented & Tested with Synthetic/OpenCV; CSI Target]`
- **Class**: `VideoStreamManager`, `StreamStatusEnum`
- **Status**: Implemented and tested with synthetic frames and OpenCV video captures (3 tests in `test_stream.py`). Hardware CSI-2 GStreamer string provided for Jetson target.
- **Features**: Decoupled acquisition loop; exponential backoff auto-reconnect on camera disconnect; MJPEG generator overlaying live reticles and telemetry.

### 3.5 Fail-Safe GPIO Controller (`app/core/gpio_controller.py`) `[Implemented & Tested in Simulation; Physical Relay Target]`
- **Class**: `GPIOController`, `GPIOState`
- **Status**: Software logic implemented and tested (3 tests in `test_gpio.py`). Physical optocoupler relay actuation on hardware pins is unverified on physical edge devices.
- **Safety**: Guarantees pin starts `LOW`, automatically returns to `LOW` after pulse duration, and resets `LOW` during teardown.

### 3.6 MQTT Alert Broker (`app/core/alert_broker.py`) `[Implemented & Tested]`
- **Class**: `AlertBroker`, `AlertPayload`, `AlertTelemetry`
- **Status**: Implemented and tested (1 test in `test_alerts.py`). Tested against simulated/in-memory dispatch; requires running Mosquitto broker for full network dispatch.
- **Security & Dispatch**: Authenticated MQTT with TLS 1.3 support; anti-flapping cooldown; accurate dispatch latency calculation.

### 3.7 Empirical Telemetry Collector (`app/core/telemetry.py`) `[Implemented & Tested]`
- **Class**: `TelemetryCollector`
- **Status**: Implemented and tested (2 tests in `test_telemetry.py`).
- **Metrics**: Instantaneous rolling FPS; multi-stage latency percentiles ($P_{50}, P_{95}, P_{99}$); honest hardware telemetry without mocked numbers.

### 3.8 SQLite Event Persistence (`app/core/storage.py`) `[Implemented & Tested]`
- **Class**: `AlertDatabase`
- **Status**: Implemented and tested (1 test in `test_storage.py`).
- **Storage**: Persistent SQLite event logging; pagination queries; age-based retention cleanup.

### 3.9 API Security & Schemas (`app/api/dependencies.py`, `app/api/schemas.py`) `[Implemented & Tested]`
- **Status**: Implemented and tested (3 tests in `test_security.py`).
- Constant-time `secrets.compare_digest` verification for `X-API-Key` headers on actuator routes.
- Strict Pydantic models for all API inputs and outputs.

# PROJECT: 02 // LOW-LATENCY EDGE AI ALERT SYSTEM

<div align="center">

![Edge AI System](https://img.shields.io/badge/ARCHITECTURE-02_EDGE_AI-00f0ff?style=for-the-badge&logo=nvidia&logoColor=black)
![Latency Benchmark](https://img.shields.io/badge/INFERENCE_LATENCY-14.5ms_ON--DEVICE-10b981?style=for-the-badge&logo=speedtest&logoColor=white)
![Quantization](https://img.shields.io/badge/PRECISION-FP16_QUANTIZED-a855f7?style=for-the-badge&logo=pytorch&logoColor=white)
![Alert Bus](https://img.shields.io/badge/ALERT_BUS-MQTT_MOSQUITTO_%3C50ms-f59e0b?style=for-the-badge&logo=mqtt&logoColor=white)
![Hardware Target](https://img.shields.io/badge/TARGET-JETSON_NANO_%7C_RPI4-6366f1?style=for-the-badge&logo=raspberrypi&logoColor=white)
![Tests](https://img.shields.io/badge/TESTS-46_PASSED_%7C_100%25-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)

**Production-Ready Edge Computer Vision System for High-Speed Object Detection & Local Real-Time Alert Actuation.**

</div>

---

## 📚 Technical Documentation & Operations Suite

| Document | Purpose & Contents |
|---|---|
| **[CODEBASE_GUIDE.md](CODEBASE_GUIDE.md)** | **Comprehensive Architecture Walkthrough**: modules, classes, and design patterns |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | **Subsystem Topology & Dataflow**: perception pipeline, state machines, and latency budgets |
| **[docs/API.md](docs/API.md)** | **REST & WebSocket Reference**: endpoints, schemas, authentication, and error codes |
| **[docs/BENCHMARKING.md](docs/BENCHMARKING.md)** | **Empirical Performance Profiling**: methodology, CLI tool, and published hardware benchmarks |
| **[docs/MODEL_SETUP.md](docs/MODEL_SETUP.md)** | **Model Export & Verification**: YOLOv8 weights acquisition, ONNX export, and SHA-256 validation |
| **[docs/MQTT_SECURITY.md](docs/MQTT_SECURITY.md)** | **Broker Hardening Guide**: TLS 1.3 encryption, client authentication, and ACL topic control |
| **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** | **Operational Runbook**: failure diagnostics for cameras, models, GPIO relays, and MQTT |
| **[docs/DOCKER_DEPLOYMENT.md](docs/DOCKER_DEPLOYMENT.md)** | **Container Deployment**: Docker Compose dev/prod environments and hardware passthrough |
| **[docs/JETSON_DEPLOYMENT.md](docs/JETSON_DEPLOYMENT.md)** | **NVIDIA Jetson Setup**: TensorRT acceleration, JetPack configuration, and CSI cameras |
| **[docs/RASPBERRY_PI_DEPLOYMENT.md](docs/RASPBERRY_PI_DEPLOYMENT.md)** | **Raspberry Pi 4 / 5 Setup**: CPU inference optimization and thermal power management |
| **[docs/IMPLEMENTATION_REPORT.md](docs/IMPLEMENTATION_REPORT.md)** | **Engineering Verification Report**: complete audit, implemented enhancements, and test coverage |

---

## ⚡ Technical Overview

The **Low-Latency Edge AI Alert System** is engineered to eliminate reliance on cloud-bound vision pipelines in mission-critical environments: **high network latency, bandwidth consumption, and cloud outages**.

Deployed directly onto embedded hardware (such as **NVIDIA Jetson Nano / Orin** or **Raspberry Pi 4 / 5** with Camera Module V2), this architecture processes real-time 60 FPS video streams on-device. Leveraging FP16 half-precision and INT8-quantized YOLOv8 neural networks via **ONNX Runtime** and **TensorRT**, it achieves sub-15ms inference latency and fires physical **GPIO actuators** and **Mosquitto MQTT alert broadcasts** in **under 50ms total loop response** without external cloud connectivity.

---

## 🧩 Hardware & System Architecture

```ascii
+--------------------+      +--------------------+      +----------------------+
| VIDEO FEED (60FPS) | ---> | LETTERBOX RESIZE   | ---> | YOLOv8-NANO ONNX     |
| (CSI-2 / RTSP)     |      | (Aspect-Ratio Pad) |      | (TensorRT / CUDA/CPU)|
+--------------------+      +--------------------+      +----------+-----------+
                                                                   |
                                                                   v
+--------------------+      +--------------------+      +----------------------+
| USER HUBS & SIRENS | <--- | MQTT ALERTS TRIG   | <--- | POST-PROCESSING      |
| (< 50ms Pulse LOW) |      | (Mosquitto QoS 1)  |      | (Vectorized NMS)     |
+--------------------+      +--------------------+      +----------------------+
```

### End-to-End Latency Budget

| Stage | Operation | Engine / Hardware | Latency |
|---|---|---|---|
| **01** | **Video Frame Capture** | Sony IMX219 CSI-2 raw NV12 DMA | 16.6ms (60 FPS) |
| **02** | **Letterbox Preprocessing** | Aspect-ratio scaling to 640x640x3 Float32 | 1.2ms |
| **03** | **YOLOv8-Nano Inference**| ONNX Runtime 1.18 / TensorRT FP16 | **14.5ms** |
| **04** | **Post-Processing & NMS**| Vectorized NumPy NMS (IoU 0.45, Conf ≥ 0.85) | 1.8ms |
| **05** | **MQTT Alert Dispatch** | Eclipse Mosquitto (QoS 1, Sub-50ms) | 3.4ms |
| **06** | **Hardware Actuation** | GPIO 18 Active HIGH Strobe Relay (Auto-Reset LOW) | 0.8ms |
| **TOTAL** | **Capture to Actuation** | **Complete Edge Reaction Loop** | **38.3ms (< 50ms Target)** |

---

## 📊 Published Hardware Benchmarks

| Model Backbone | Precision | Latency (ms) | Frame Rate | mAP@0.5 | Target Hardware | Operational Classification |
|---|---|---|---|---|---|---|
| **YOLOv8-Nano** | **INT8** | **6.8ms** | **147.0 FPS** | 83.8% | Jetson Orin Nano (8GB) | Ultra-Low Latency Edge |
| **YOLOv8-Nano** | **FP16** | **14.5ms** | **68.9 FPS** | **84.6%** | Jetson Nano (4GB) | **Optimal Pareto Sweetspot** |
| **YOLOv8-Nano** | **FP32** | **31.4ms** | **31.8 FPS** | **84.6%** | Raspberry Pi 5 (8GB) | High-Performance CPU Edge |
| **YOLOv8-Nano** | **FP32** | **58.2ms** | **17.1 FPS** | **84.6%** | Raspberry Pi 4 (8GB) | CPU-Bound Edge Baseline |
| YOLOv8-Small | FP16 | 24.8ms | 40.3 FPS | 86.8% | Jetson Nano (4GB) | Exceeds 15ms Latency Budget |
| YOLOv8-Medium | FP32 | 48.0ms | 20.8 FPS | 89.1% | x86 GPU Server | High-Power Compute |

---

## 🛠️ Technology Stack

- **Neural Architecture**: Ultralytics YOLOv8 (nano / small)
- **Runtime Optimization**: ONNX Runtime (TensorRT, CUDA, and CPU Providers)
- **Computer Vision**: OpenCV Headless with auto-reconnect stream manager
- **Messaging Bus**: Eclipse Mosquitto MQTT Broker (QoS 1, TLS 1.3, User Auth)
- **Edge Actuation**: Fail-Safe `GPIOController` (Guaranteed LOW startup/shutdown, timed reset pulse)
- **Application Server**: FastAPI + Uvicorn (Asynchronous REST & WebSockets)
- **Database**: Embedded SQLite persistence with automatic retention purging
- **Security**: Constant-time `X-API-Key` authentication & CORS restriction

---

## 🚀 Quickstart Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/BELALANSARI07/-Low-Latency-Edge-AI-Alert-System..git
cd -Low-Latency-Edge-AI-Alert-System.

# Create virtual environment
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Configuration

```bash
cp .env.example .env
```

### 3. Running the Server

#### Development Mode (Synthetic simulation feed)
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Docker Deployment
```bash
# Development with hot-reload & synthetic camera
docker compose -f docker-compose.dev.yml up --build

# Production with real hardware passthrough
docker compose -f docker-compose.prod.yml up -d --build
```

Access services:
- **Interactive API Docs (Swagger)**: `http://localhost:8000/docs`
- **Live MJPEG Camera Feed**: `http://localhost:8000/stream/mjpeg`
- **Liveness Probe**: `http://localhost:8000/health`
- **Readiness Probe**: `http://localhost:8000/ready`
- **Telemetry Statistics**: `http://localhost:8000/telemetry/stats`

---

## 🧪 Testing & Verification

Run the comprehensive 46-test automated verification suite:

```bash
python -m pytest tests/ -v
```

All 46 unit and functional tests validate:
- Pydantic configuration bounds and production constraints
- Letterbox image preprocessing and boundary clamping
- Dual-layout YOLOv8 tensor parsing (`[1, 84, 8400]` and `[1, 8400, 84]`)
- Vectorized NumPy Non-Maximum Suppression (NMS)
- Fail-safe GPIO controller state transitions and pulse resets
- SQLite alert persistence and retention management
- API Key security authentication and CORS policy enforcement
- Camera stream lifecycle and auto-reconnection
- Empirical telemetry collection without fabricated metrics

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

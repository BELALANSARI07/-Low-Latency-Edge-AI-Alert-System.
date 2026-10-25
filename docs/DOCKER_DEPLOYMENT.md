# Docker Deployment Guide

This guide details how to build, run, and orchestrate the **Low-Latency Edge AI Alert System** container stack using Docker and Docker Compose.

---

## 1. Quick Start (Development Simulation)

In development mode, the container stack operates with synthetic video generation and local simulated MQTT alerting:

```bash
# Build and run with hot-reload development compose
docker compose -f docker-compose.dev.yml up --build
```

The system will be available at:
- API Server: `http://localhost:8000`
- API Documentation (Swagger): `http://localhost:8000/docs`
- Live MJPEG Stream: `http://localhost:8000/stream/mjpeg`
- Health Probe: `http://localhost:8000/health`
- Readiness Probe: `http://localhost:8000/ready`

---

## 2. Production Deployment

In production, real camera devices, authenticated MQTT, and persistent databases are mounted:

### Step 1: Configure Environment

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Ensure the following variables are configured:
```ini
APP_ENV=production
SIMULATION_MODE=false
FAIL_ON_MISSING_MODEL=true
MODEL_PATH=models/yolov8n-fp16.onnx
API_KEY=GenerateAStrongRandomSecretKeyHere!
CAMERA_SOURCE=0
```

### Step 2: Place Validated Model File

Place your exported ONNX model in the `models/` directory:
```bash
python scripts/validate_model.py --model-path models/yolov8n-fp16.onnx
```

### Step 3: Launch Production Compose Stack

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

---

## 3. Container Hardware Passthrough

### Camera Device Passthrough (`/dev/video*`)
To access physical USB/CSI cameras inside the container, pass the device in `docker-compose.prod.yml`:
```yaml
devices:
  - "/dev/video0:/dev/video0"
```

### GPU Passthrough (NVIDIA Jetson / Container Toolkit)
For NVIDIA GPU / TensorRT acceleration on NVIDIA Linux hosts:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

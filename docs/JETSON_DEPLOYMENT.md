# NVIDIA Jetson Nano & Orin Deployment Guide

This guide details physical deployment, TensorRT acceleration, CSI camera interfacing, and GPIO relay actuation on **NVIDIA Jetson Nano (4GB)** and **Jetson Orin Nano**.

---

## 1. Hardware Prerequisites

- **Host Device**: NVIDIA Jetson Nano B01 (4GB) or Jetson Orin Nano Developer Kit
- **Camera**: Raspberry Pi Camera Module V2 (Sony IMX219) via CSI ribbon cable or USB V4L2 WebCam
- **OS**: NVIDIA JetPack 4.6.1 (L4T 32.7.1) for Nano / JetPack 5.1+ for Orin
- **Power**: 5V/4A DC Barrel Jack (micro-USB cannot deliver peak 10W inference loads reliably)

---

## 2. Jetson System Configuration

### Maximum Power Mode
Set Jetson Nano to 10W high-performance power profile:
```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

### CSI Camera Hardware Pipeline
Test the IMX219 CSI camera via GStreamer:
```bash
gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! nvvidconv ! xvimagesink
```

Configure `.env` for CSI camera capture:
```ini
CAMERA_SOURCE="nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=60/1 ! nvvidconv ! video/x-raw, width=640, height=640, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
CAMERA_FPS=60
```

---

## 3. ONNX Runtime with TensorRT Acceleration

Install the JetPack-compatible ONNX Runtime wheel with GPU / TensorRT execution providers:

```bash
# Verify GPU availability
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Output should contain: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

In `.env`:
```ini
EXECUTION_PROVIDER=TensorrtExecutionProvider
QUANTIZATION_MODE=FP16
```

---

## 4. Jetson GPIO Permissions & Hardware Relays

Jetson Nano requires udev permissions for unprivileged GPIO access:

```bash
# Add user to gpio group
sudo usermod -a -G gpio $USER

# Install Jetson.GPIO
pip install Jetson.GPIO
```

Configure physical pin in `.env`:
```ini
ENABLE_GPIO_ACTUATION=true
GPIO_ALERT_PIN=18
GPIO_PULSE_DURATION_MS=200
```

---

## 5. Systemd Production Service Setup

Create `/etc/systemd/system/edge-detector.service`:

```ini
[Unit]
Description=Low-Latency Edge AI Alert System
After=network.target mosquitto.service

[Service]
Type=simple
User=jetson
WorkingDirectory=/home/jetson/Low-Latency-Edge-AI-Alert-System
ExecStart=/home/jetson/venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3
EnvironmentFile=/home/jetson/Low-Latency-Edge-AI-Alert-System/.env

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable edge-detector
sudo systemctl start edge-detector
```

# Raspberry Pi 4 & 5 Edge Deployment Guide

> [!NOTE]
> **Target Hardware Guide**: The instructions below document the deployment procedure for Raspberry Pi single-board computers.
> Physical execution on ARM64 Raspberry Pi hardware and physical GPIO pin actuation have **not yet been verified on physical hardware** in this repository. Software shims and simulated execution have been tested.

This guide details configuring and optimizing the **Low-Latency Edge AI Alert System** on **Raspberry Pi 4 (8GB)** and **Raspberry Pi 5** using CPU-optimized ONNX Runtime inference.

---

## 1. Hardware Requirements

- **Device**: Raspberry Pi 4 Model B (4GB / 8GB) or Raspberry Pi 5 (8GB)
- **Cooling**: Active cooler / aluminum heatsink case (CPU will throttle without cooling under continuous inference)
- **Camera**: Raspberry Pi Camera Module V2 (8MP IMX219) or Camera Module V3 (12MP IMX708)
- **Power**: Official 5.1V / 3A (Pi 4) or 5V / 5A USB-C Power Supply (Pi 5)
- **OS**: Raspberry Pi OS 64-bit (Debian Bookworm)

---

## 2. Environment Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-pip python3-venv libgl1 libglib2.0-0 mosquitto mosquitto-clients

# Create virtual environment with system site-packages for camera/GPIO access
python3 -m venv --system-site-packages venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

## 3. Camera Setup & Testing

### Using Raspberry Pi Camera Module V2/V3

Enable camera interface:
```bash
# Verify camera detection
rpicam-hello --list-cameras
```

In `.env`:
```ini
CAMERA_SOURCE=0
CAMERA_WIDTH=640
CAMERA_HEIGHT=640
CAMERA_FPS=30
```

---

## 4. Raspberry Pi Latency Optimization

Because Raspberry Pi lacks dedicated Tensor Cores, inference runs on ARM Cortex-A72 / A76 CPU cores.

Recommended configuration for sub-50ms CPU latency:
```ini
QUANTIZATION_MODE=FP32
EXECUTION_PROVIDER=CPUExecutionProvider
CONFIDENCE_THRESHOLD=0.65
ALERT_CONFIDENCE_THRESHOLD=0.80
```

| Resolution | RPi 4 Target Latency | RPi 5 Target Latency | Published mAP@0.5 |
|---|---|---|---|
| **640x640** | 58.2 ms | 31.4 ms | 84.6% (published) |
| **480x480** | 36.5 ms | 19.8 ms | 83.2% (published) |
| **320x320** | 18.2 ms | 9.7 ms | 79.5% (published) |

*(Note: Target latencies and accuracy figures represent published Ultralytics reference numbers on ARM64; not evaluated locally on physical Raspberry Pi boards in this repository).*

---

## 5. Relay Actuator & GPIO Setup

Install RPi.GPIO:
```bash
pip install RPi.GPIO
```

Configure `.env`:
```ini
ENABLE_GPIO_ACTUATION=true
GPIO_ALERT_PIN=18
GPIO_PULSE_DURATION_MS=200
```

The system ensures pin 18 initializes LOW, pulses HIGH for 200ms when a hazard is identified, and immediately resets LOW.

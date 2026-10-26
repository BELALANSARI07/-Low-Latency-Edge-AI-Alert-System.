# YOLOv8 ONNX Model Setup & Optimization Guide

This guide describes how to acquire, export, quantize, verify, and deploy YOLOv8 ONNX models for the **Low-Latency Edge AI Alert System** targeting Jetson Nano, Raspberry Pi 4/5, and x86/ARM edge devices.

---

## 1. Acquiring Weights

Download the official lightweight YOLOv8 weights from Ultralytics or train custom models:

```bash
# YOLOv8 Nano (Default recommended for sub-18ms edge latency)
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt -O models/yolov8n.pt

# YOLOv8 Small (Higher accuracy if compute permits)
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt -O models/yolov8s.pt
```

---

## 2. Exporting to ONNX

Use the provided export script `scripts/export_onnx.py` or the official Ultralytics CLI.

### Option A: Using `scripts/export_onnx.py`

```bash
# Export FP16 model (Recommended for NVIDIA Jetson Nano / TensorRT)
python scripts/export_onnx.py --weights models/yolov8n.pt --output models/yolov8n-fp16.onnx --quant FP16 --imgsz 640

# Export FP32 model (Recommended for Raspberry Pi / CPU execution)
python scripts/export_onnx.py --weights models/yolov8n.pt --output models/yolov8n-fp32.onnx --quant FP32 --imgsz 640
```

### Option B: Using Ultralytics CLI directly

```bash
# Install ultralytics
pip install ultralytics onnx onnxslim

# Export FP16 ONNX
yolo export model=models/yolov8n.pt format=onnx imgsz=640 half=True opset=17 simplify=True

# Export FP32 ONNX
yolo export model=models/yolov8n.pt format=onnx imgsz=640 half=False opset=17 simplify=True
```

---

## 3. Quantization Modes & Hardware Targets

> [!NOTE]
> Latency targets and accuracy numbers are **published reference benchmarks** from Ultralytics and hardware vendors (COCO val2017 mAP@0.5). Accuracy has not been independently evaluated inside this repository.
> Currently, the repository includes an FP32 ONNX model (`yolov8n.onnx`) running on `CPUExecutionProvider`.

| Quantization | Status in Repo | Target Hardware | Execution Provider | Latency Target | Published mAP@0.5 |
|---|---|---|---|---|---|
| **FP32** | **Implemented & Tested** | x86 CPU / Raspberry Pi | `CPUExecutionProvider` | ~60ms - 75ms | 84.6% (published) |
| **FP16** | **Target Optimization** | NVIDIA Jetson Nano | `TensorrtExecutionProvider` | **12ms - 18ms** | 84.4% (published) |
| **INT8** | **Target Optimization** | Jetson Orin / EdgeTPU | `TensorrtExecutionProvider` | **6ms - 10ms** | 83.8% (published) |

---

## 4. Checksum Verification

To prevent corrupted model deployments in production environments, compute and configure the SHA-256 hash:

### Step 1: Compute Checksum

On Linux / macOS / Raspberry Pi:
```bash
sha256sum models/yolov8n-fp16.onnx
```

On Windows PowerShell:
```powershell
Get-FileHash models/yolov8n-fp16.onnx -Algorithm SHA256
```

### Step 2: Configure in `.env`

Add the hash to your `.env` configuration file:
```ini
MODEL_PATH=models/yolov8n-fp16.onnx
MODEL_SHA256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

---

## 5. Validating Model with CLI

Before starting the server, run `scripts/validate_model.py` to inspect the ONNX graph:

```bash
# Verify integrity and check input/output shapes
python scripts/validate_model.py --model-path models/yolov8n-fp16.onnx

# Verify with expected checksum
python scripts/validate_model.py \
  --model-path models/yolov8n-fp16.onnx \
  --expected-sha256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

# Machine-readable JSON output for CI/CD checks
python scripts/validate_model.py --model-path models/yolov8n-fp16.onnx --json
```

---

## 6. Simulation Mode vs Production Mode

- **Development / CI (`SIMULATION_MODE=true`)**: If no model file is present at `MODEL_PATH`, the system initializes in synthetic simulation mode. Simulated detections are clearly marked with `"is_simulated": true` and logs explicitly identify simulation mode.
- **Production (`SIMULATION_MODE=false`)**: If the model is missing, invalid, or fails checksum comparison, the system fails safely:
  - If `FAIL_ON_MISSING_MODEL=true`, the process terminates immediately on boot.
  - If `FAIL_ON_MISSING_MODEL=false`, the health check marks the model subsystem as `unhealthy`, the `/ready` API endpoint returns `503 Service Unavailable`, and zero fake detections are produced.

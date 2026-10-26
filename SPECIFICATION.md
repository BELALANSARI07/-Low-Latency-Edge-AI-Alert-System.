# ARCHITECTURE & DESIGN SPECIFICATION (RFC-002)
## PROJECT: 02 // LOW-LATENCY EDGE AI ALERT SYSTEM

> [!NOTE]
> **Status**: Design & Target Specification (RFC-002).
> Latency budgets, target hardware configurations, and power envelopes represent **design targets and published reference benchmarks**.
> Current verified implementation runs on x86_64 CPU (ONNX Runtime, FP32) and simulated edge hardware. Physical validation on NVIDIA Jetson and Raspberry Pi boards is planned.

```ascii
+--------------------+      +--------------------+      +----------------------+
| VIDEO FEED (60FPS) | ---> | ON-DEVICE RESIZING | ---> | YOLOv8-NANO INFER    |
| (CSI-2 / RTSP)     |      | (V4L2 / OpenCV)    |      | (ONNX FP16 / TensorRT|
+--------------------+      +--------------------+      +----------+-----------+
                                                                   |
                                                                   v
+--------------------+      +--------------------+      +----------------------+
| USER HUBS & SIRENS | <--- | MQTT ALERTS TRIG   | <--- | POST-PROCESSING      |
| (< 50ms Relay)     |      | (Mosquitto QoS 1)  |      | (NMS & Conf Thresh)  |
+--------------------+      +--------------------+      +----------------------+
```

---

### 1. Technical Overview & Mission
- **Project Identifier**: `PROJECT: 02 // LOW-LATENCY EDGE AI ALERT SYSTEM`
- **Headline**: Edge-Deployed Computer Vision System for High-Speed Object Detection & Local Real-Time Alert Triggering.
- **Core Mission**: Developing a hardened, zero-cloud pipeline to process real-time video streams on edge devices (NVIDIA Jetson Nano, Raspberry Pi 4, or local industrial hardware) using lightweight YOLO architectures, running strictly below 15ms per frame to trigger alarms without cloud latency or internet dependency.

---

### 2. Target Hardware & Architecture Specifications

| Parameter | Specification | Engineering Details & Status |
|---|---|---|
| **Target Hardware** | NVIDIA Jetson Nano 4GB / Raspberry Pi 4 (8GB) | Quad-Core ARM Cortex-A57 @ 1.43GHz, 128-core NVIDIA Maxwell GPU / ARM Cortex-A72 (*Target platforms*) |
| **Verified Dev Platform** | x86_64 host (CPU) & Linux Docker container | AMD Ryzen / Intel x86_64 CPU; tested via Pytest and CI/CD (*Verified*) |
| **Camera Sensor** | Raspberry Pi Camera Module V2 (Sony IMX219) | 8-Megapixel, 1080p @ 60 FPS raw NV12 / Bayer CSI-2 (*Target*); OpenCV / RTSP / Synthetic (*Verified*) |
| **Core Stack** | PyTorch, YOLOv8 (nano), OpenCV, ONNX Runtime, Mosquitto, FastAPI | ONNX Runtime (CPU verified; CUDA/TensorRT target), paho-mqtt, FastAPI REST & WS (*Verified*) |
| **Model Precision** | FP32 ONNX (*Implemented*); FP16 / INT8 (*Target*) | FP32 ONNX weights active by default; FP16 / INT8 targeted for embedded GPU/TensorRT |
| **Inference Latency** | **12ms to 18ms** on-device (*Target Goal*) | Published reference target for Jetson Nano TensorRT FP16; local x86 CPU measured at ~63ms |
| **Alert Response Target**| **< 50ms** end-to-end (*Design Budget*) | Frame capture (16.6ms) + Resizing (1.2ms) + Inference (14.5ms) + Post-proc (1.8ms) + MQTT publish (3.4ms) |
| **Power Envelope** | **~7.5W** Peak Edge Load (*Target Budget*) | Theoretical envelope for Jetson Nano 10W power profile; unmeasured in current repo |
| **Operating System** | Linux Ubuntu 20.04+ / JetPack 4.6.1+ / Debian Bookworm | Multi-platform Linux support; validated on Ubuntu in GitHub Actions CI |

---

### 3. Pipeline Flow & Target Latency Budget (Design RFC)

```
[01. Frame Capture]   --> CSI-2 Camera DMA Ring Buffer            (16.6ms target frame interval)
[02. Hardware Resize] --> V4L2 / OpenCV CUDA to 640x640x3 Float   (1.2ms target)
[03. YOLOv8 Inference]--> TensorRT / ONNX Runtime FP16 Engine     (14.5ms target on Jetson Nano)
[04. Post-Processing] --> Vectorized NMS (IoU 0.45, Conf >= 0.85) (1.8ms target)
[05. MQTT Dispatch]   --> Mosquitto Broker QoS 1 Socket           (3.4ms target)
[06. Hardware IO]     --> GPIO 18 Active HIGH Strobe Siren Relay  (0.8ms target)
----------------------------------------------------------------------------------------
THEORETICAL DESIGN BUDGET (Capture to Actuation):                 38.3ms (< 50ms Hard Target)
MEASURED x86 CPU BASELINE (Preprocess + Infer + NMS + Dispatch):  ~75ms (at 640x640 FP32 CPU)
```

---

### 4. Quantization Tradeoff Targets & Published References

> The following metrics reflect published reference benchmarks from Ultralytics and NVIDIA hardware literature, serving as optimization targets for this architecture. They have **not been evaluated on physical edge hardware in this repository**.

| Model Architecture | Quantization | Target Latency | Target FPS | Published mAP@0.5 (COCO) | Estimated Power | Target Hardware Classification |
|---|---|---|---|---|---|---|
| **YOLOv8-Nano** | **INT8** | **10.4ms** | **96.1** | 81.2% | 6.2W | Ultra-low latency target |
| **YOLOv8-Nano** | **FP16** | **14.5ms** | **68.9** | **84.6%** | **7.5W** | **Optimal Jetson Pareto Target** |
| YOLOv5-Tiny | FP16 | 16.2ms | 61.7 | 78.4% | 8.1W | Legacy Reference |
| YOLOv8-Small | FP16 | 24.8ms | 40.3 | 86.8% | 9.8W | Higher precision target |
| YOLOv8-Medium | FP32 | 48.0ms | 20.8 | 89.1% | 14.2W | High compute reference |

---

### 5. Mosquitto MQTT Telemetry Protocol

**Topic**: `alert/system/01`  
**QoS**: `1` (Guaranteed At Least Once Delivery)  
**Encoding**: `application/json`

```json
{
  "timestamp": "2026-10-25T16:45:02.124Z",
  "camera_id": "CAMERA_01 // HIGHWAY_CAM",
  "zone": "REGION_A2_CROSSWALK",
  "alert_type": "PEDESTRIAN_HAZARD",
  "confidence": 0.914,
  "bounding_box": [120, 80, 180, 320],
  "telemetry": {
    "inference_ms": 14.2,
    "fps": 45.2,
    "quantization": "FP16_ONNX",
    "hardware_power_w": 7.42
  },
  "action": {
    "gpio_pin": 18,
    "gpio_state": "HIGH",
    "relay_duration_ms": 1500
  }
}
```

---

### 6. Edge GPIO Hardware Actuator Pinout

- **Platform**: 40-pin Expansion Header (Jetson Nano / Raspberry Pi)
- **Signal Pin**: Physical Pin 12 (`GPIO 18 / BCM 18`)
- **Electrical Characteristic**: 3.3V Logic Level to Optocoupler Relay Board
- **Actuation Pulse**: 1500ms Active HIGH upon Hazard Trigger
- **Failsafe**: Default LOW on boot and watchdog timeout

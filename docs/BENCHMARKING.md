# Edge AI Performance Benchmarking & Profiling Guide

This document describes the methodology, empirical tooling, and target hardware benchmarks for the **Low-Latency Edge AI Alert System**.

---

## 1. Benchmarking Methodology

Low-latency edge computer vision requires strict budgeting across all execution stages:

$$\text{Total Loop Latency} = T_{\text{capture}} + T_{\text{preprocess}} + T_{\text{inference}} + T_{\text{postprocess}} + T_{\text{dispatch}}$$

To meet the sub-50ms hazard response budget:
1. **$T_{\text{preprocess}}$ (Letterboxing & Normalization)**: Target $\le 4\text{ms}$.
2. **$T_{\text{inference}}$ (TensorRT / ONNX Execution)**: Target $\le 15\text{ms}$ (Jetson Nano) / $\le 60\text{ms}$ (Raspberry Pi CPU).
3. **$T_{\text{postprocess}}$ (Vectorized NMS & Coordinate Clamping)**: Target $\le 2\text{ms}$.
4. **$T_{\text{dispatch}}$ (GPIO Pulse & MQTT Async Publish)**: Target $\le 2\text{ms}$.

---

## 2. Empirical Benchmark: Local Measured Baseline (x86 CPU)

The following baseline was measured empirically on a local x86_64 host (AMD Ryzen 5 / 6 cores) running ONNX Runtime 1.20 with `CPUExecutionProvider` on `models/yolov8n.onnx` (FP32, 640x640):

| Stage | Operation | Measured Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---|---|---|---|---|
| **Preprocess** | Letterbox & pad to 640x640x3 Float32 | 8.8 ms | 8.6 ms | 9.8 ms | 10.2 ms |
| **Inference** | ONNX Runtime CPU (YOLOv8n FP32) | 63.2 ms | 62.8 ms | 69.1 ms | 71.4 ms |
| **Postprocess** | Vectorized NumPy NMS + coordinate restoration | 2.1 ms | 2.0 ms | 2.6 ms | 3.1 ms |
| **Dispatch** | In-memory evaluation & alert packaging | < 1.0 ms | < 1.0 ms | < 1.0 ms | < 1.0 ms |
| **Total** | **Empirical CPU Loop Latency** | **~74.7 ms** | **~74.3 ms** | **~80.9 ms** | **~82.9 ms** |

*Measured Throughput*: **~13.4 FPS** on standard CPU without GPU acceleration.

---

## 3. External Reference Benchmarks & Target Hardware Goals

> [!NOTE]
> The figures below represent **external published reference data** (from Ultralytics and NVIDIA Jetson documentation) and **design targets** for edge hardware.
> They have **not been evaluated on physical edge hardware in this repository**.

| Target Platform | Precision / Engine | Resolution | Published Latency | Published FPS | Published mAP@0.5 | Target Power | Target Classification |
|---|---|---|---|---|---|---|---|
| **NVIDIA Jetson Nano (4GB)** | **FP16 / TensorRT** | 640x640 | **14.5 ms** | **68.9 FPS** | 84.6% | ~7.5 W | Target Pareto Sweetspot |
| **NVIDIA Jetson Orin Nano (8GB)** | **INT8 / TensorRT** | 640x640 | **6.8 ms** | **147.0 FPS** | 83.8% | ~9.2 W | High-Speed Target |
| **Raspberry Pi 4 (8GB)** | **FP32 / ONNX Runtime** | 640x640 | **58.2 ms** | **17.1 FPS** | 84.6% | ~5.8 W | CPU-Bound Edge Target |
| **Raspberry Pi 5 (8GB)** | **FP32 / ONNX Runtime** | 640x640 | **31.4 ms** | **31.8 FPS** | 84.6% | ~7.2 W | High-Performance CPU Target |

---

## 4. Running Empirical Benchmarks Locally

Use `scripts/benchmark.py` to profile the detector on your machine:

```bash
# Run 100 profiling iterations with 10 warmup cycles
python scripts/benchmark.py --iterations 100 --warmup 10

# Test on physical ONNX model with specific provider
python scripts/benchmark.py \
  --model-path models/yolov8n-fp16.onnx \
  --provider CPUExecutionProvider \
  --iterations 50 \
  --output-json benchmark_results.json
```

### Metrics Output:
- **Total Latency**: Mean, $P_{50}, P_{90}, P_{95}, P_{99}$, Min, Max.
- **Stage Latencies**: Preprocessing, Model Inference, Postprocessing.
- **Throughput**: Measured frames per second (FPS).
- **Execution Mode**: Explicitly labels whether execution was `Physical_Hardware` or `Simulated`.

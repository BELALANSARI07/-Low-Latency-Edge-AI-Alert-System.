# Edge AI Troubleshooting & Diagnostics Runbook

This guide covers failure modes, error diagnostics, and mitigation procedures for the **Low-Latency Edge AI Alert System**.

---

## 1. Model & Inference Issues

### Error: `FileNotFoundError: Model file not found at: 'models/yolov8n-fp16.onnx'`
- **Cause**: `FAIL_ON_MISSING_MODEL=true` is set, but the model file does not exist at `MODEL_PATH`.
- **Mitigation**:
  1. In development, set `SIMULATION_MODE=true` to enable deterministic synthetic inference.
  2. In production, export or copy your YOLOv8 ONNX model into the `models/` directory following [MODEL_SETUP.md](MODEL_SETUP.md).

### Error: `Checksum mismatch for 'models/yolov8n-fp16.onnx'`
- **Cause**: The SHA-256 hash of the model on disk differs from `MODEL_SHA256` in `.env`.
- **Mitigation**: Recompute the file hash with `python scripts/validate_model.py --model-path models/yolov8n-fp16.onnx` and update `MODEL_SHA256` in `.env`.

---

## 2. Camera & Video Stream Issues

### Camera reports `StreamStatusEnum.DISCONNECTED` or `FAILED`
- **Cause**: Physical USB cable unplugged, CSI ribbon cable loose, or RTSP camera network packet drop.
- **Diagnostic Command**:
  ```bash
  # Check if V4L2 device exists
  v4l2-ctl --list-devices
  ls -la /dev/video*
  ```
- **Mitigation**: The system automatically attempts reconnection with exponential backoff. Check camera power and verify that the user running the process belongs to the `video` group:
  ```bash
  sudo usermod -a -G video $USER
  ```

### High Frame Drop Count (`dropped_frames > 0`)
- **Cause**: Host processor overloaded or USB bandwidth bottleneck when streaming 4K or uncompressed YUYV video.
- **Mitigation**: Lower camera resolution to 640x640 or 1280x720 in `.env` (`CAMERA_WIDTH=640`, `CAMERA_HEIGHT=640`) and ensure MJPEG format is selected.

---

## 3. MQTT Broker Connection Issues

### Broker reports `MQTTStatusEnum.FAILED` or `DISCONNECTED`
- **Cause**: Mosquitto daemon not running, wrong host/port, or firewall blocking port 1883/8883.
- **Diagnostic Command**:
  ```bash
  # Check broker status
  sudo systemctl status mosquitto

  # Test subscription manually
  mosquitto_sub -h localhost -p 1883 -t "alert/system/#" -v
  ```
- **Mitigation**: Verify that `MQTT_BROKER_HOST` points to the correct IP or Docker service name (`mqtt-broker`).

### Connection Return Code 5 (Connection Refused: Not Authorized)
- **Cause**: Broker requires authentication (`allow_anonymous false`), but credentials are missing or incorrect.
- **Mitigation**: Set `MQTT_USERNAME` and `MQTT_PASSWORD` in `.env`. See [MQTT_SECURITY.md](MQTT_SECURITY.md).

---

## 4. GPIO & Hardware Actuator Issues

### Error: `Permission denied: /dev/gpiomem`
- **Cause**: Unprivileged process attempting to access GPIO registers without group membership.
- **Mitigation**:
  ```bash
  sudo usermod -a -G gpio $USER
  # Log out and log back in for group membership to take effect
  ```

### Relay Stays `HIGH` (Actuator Lockup)
- **Safety Guarantee**: The `GPIOController` automatically initiates a background reset timer that drives the pin to `LOW` after `GPIO_PULSE_DURATION_MS`.
- **Mitigation**: If hardware is malfunctioning, send a manual reset request or restart the process (which forces pin LOW on startup and shutdown).

---

## 5. API & Readiness Probe Failures

### `GET /ready` returns HTTP 503 Service Unavailable
- **Cause**: One or more critical subsystems (camera, model engine, or MQTT broker) are not healthy.
- **Diagnosis**: Query `/ready` and examine the `subsystems` breakdown in the response body to isolate the failing component:
  ```bash
  curl -s http://localhost:8000/ready | jq .
  ```

### `POST /alerts/trigger` returns HTTP 401 Unauthorized
- **Cause**: Missing or incorrect `X-API-Key` header.
- **Mitigation**: Include the header `X-API-Key: <your-key>` in the POST request.

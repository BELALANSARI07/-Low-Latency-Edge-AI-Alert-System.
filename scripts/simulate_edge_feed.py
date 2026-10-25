"""
Standalone Edge Feed & MQTT Alert Simulator:
Continuously generates camera frames, runs inference detection, and broadcasts
telemetry and hazard alerts to Eclipse Mosquitto broker at topic 'alert/system/01'.
"""

import logging
import time

from app.core.alert_broker import AlertBroker
from app.core.config import get_settings
from app.core.detector import EdgeDetector
from app.core.stream import VideoStreamManager
from app.core.telemetry import TelemetryCollector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("edge_simulator")


def main():
    settings = get_settings()
    logger.info("Initializing Edge Alert Simulator...")

    detector = EdgeDetector(
        quantization=settings.quantization_mode,
        conf_thresh=settings.confidence_threshold,
        iou_thresh=settings.iou_threshold
    )

    stream_mgr = VideoStreamManager(
        source="synthetic",
        width=settings.camera_width,
        height=settings.camera_height,
        target_fps=settings.camera_fps
    )
    stream_mgr.start()

    broker = AlertBroker(
        host=settings.mqtt_broker_host,
        port=settings.mqtt_broker_port,
        alert_topic=settings.mqtt_alert_topic,
        cooldown_sec=settings.alert_cooldown_seconds,
        gpio_pin=settings.gpio_alert_pin,
        enable_gpio=False
    )

    telemetry = TelemetryCollector()

    logger.info("Simulating active edge video feed. Press Ctrl+C to terminate.")

    try:
        frame_idx = 0
        while True:
            frame = stream_mgr.get_latest_frame()
            detections, metrics = detector.detect(frame)
            telemetry.record_frame(metrics["total_latency_ms"])

            frame_idx += 1
            if frame_idx % 30 == 0:
                stats = telemetry.get_hardware_telemetry()
                logger.info(
                    f"HUD: FPS={stats['fps']} | CPU={stats['cpu_percent']}% | "
                    f"Latency={metrics['total_latency_ms']}ms | Quant={metrics['quantization']}"
                )

            # Trigger alert every ~50 frames for demonstration
            if frame_idx % 50 == 0:
                broker.publish_alert(
                    alert_type="PEDESTRIAN_HAZARD",
                    confidence=0.914,
                    zone="A-2",
                    detection_details={"label": "person", "conf": 0.914, "box": [120, 80, 180, 320]},
                    latency_ms=metrics["total_latency_ms"]
                )

            time.sleep(0.02)  # Simulate 50 FPS loop
    except KeyboardInterrupt:
        logger.info("Simulation halted by user.")
    finally:
        stream_mgr.stop()
        broker.shutdown()


if __name__ == "__main__":
    main()

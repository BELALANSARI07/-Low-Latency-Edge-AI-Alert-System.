import asyncio
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.detector import EdgeDetector
from app.core.stream import VideoStreamManager
from app.core.alert_broker import AlertBroker
from app.core.telemetry import TelemetryCollector
from app.api.router import api_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("edge_system")


async def run_inference_loop(app: FastAPI):
    """Background task executing continuous inference loop on camera frames."""
    detector: EdgeDetector = app.state.detector
    stream_mgr: VideoStreamManager = app.state.stream_mgr
    alert_broker: AlertBroker = app.state.alert_broker
    telemetry: TelemetryCollector = app.state.telemetry
    settings = app.state.settings

    logger.info("Starting background edge AI inference worker loop...")

    try:
        while app.state.is_running:
            frame = stream_mgr.get_latest_frame()
            detections, metrics = detector.detect(frame)

            # Record telemetry
            telemetry.record_frame(metrics["total_latency_ms"])

            # Check for hazard alerts
            for d in detections:
                if d.confidence >= settings.alert_confidence_threshold:
                    alert = alert_broker.publish_alert(
                        alert_type="HAZARD_DETECTED",
                        confidence=d.confidence,
                        zone=d.region,
                        detection_details=d.to_dict(),
                        latency_ms=metrics["total_latency_ms"]
                    )
                    if alert:
                        telemetry.record_alert()

            await asyncio.sleep(0.05)
    except asyncio.CancelledError:
        logger.info("Inference worker task cancelled.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager initializing edge hardware pipelines."""
    settings = get_settings()
    app.state.settings = settings
    app.state.is_running = True

    # 1. Initialize Telemetry Collector
    app.state.telemetry = TelemetryCollector()

    # 2. Initialize Edge Object Detector
    app.state.detector = EdgeDetector(
        model_path=settings.model_path,
        quantization=settings.quantization_mode,
        conf_thresh=settings.confidence_threshold,
        iou_thresh=settings.iou_threshold,
        execution_provider=settings.execution_provider
    )

    # 3. Initialize Camera Stream Manager
    app.state.stream_mgr = VideoStreamManager(
        source=settings.camera_source,
        width=settings.camera_width,
        height=settings.camera_height,
        target_fps=settings.camera_fps
    )
    # 5. Launch camera streaming and background inference worker unless in testing mode
    worker_task = None
    if os.environ.get("TESTING") != "true":
        app.state.stream_mgr.start()
        worker_task = asyncio.create_task(run_inference_loop(app))

    logger.info("Low-Latency Edge AI Alert System fully initialized and active.")

    yield

    # Teardown
    logger.info("Shutting down edge pipelines...")
    app.state.is_running = False
    if worker_task:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
    if os.environ.get("TESTING") != "true":
        app.state.stream_mgr.stop()
    app.state.alert_broker.shutdown()
    logger.info("Shutdown complete.")


def create_application() -> FastAPI:
    """Builds and configures the FastAPI application instance."""
    app = FastAPI(
        title="PROJECT: 02 // LOW-LATENCY EDGE AI ALERT SYSTEM",
        description="Edge-Deployed Computer Vision System for High-Speed Object Detection & Local Real-Time Alert Triggering.",
        version="1.0.0",
        lifespan=lifespan
    )

    settings = get_settings()
    app.state.settings = settings
    app.state.telemetry = TelemetryCollector()
    app.state.detector = EdgeDetector()
    app.state.stream_mgr = VideoStreamManager()
    app.state.alert_broker = AlertBroker(enable_gpio=False)
    app.state.is_running = False

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)
    return app


app = create_application()

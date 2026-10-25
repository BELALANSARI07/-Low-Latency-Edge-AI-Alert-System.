from fastapi import APIRouter
from app.api.health import router as health_router
from app.api.telemetry import router as telemetry_router
from app.api.stream import router as stream_router
from app.api.alerts import router as alerts_router

api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(telemetry_router)
api_router.include_router(stream_router)
api_router.include_router(alerts_router)

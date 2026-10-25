"""
FastAPI Security & Authentication Dependencies.
Protects sensitive hardware actuator endpoints with constant-time API key verification.
"""

import secrets

from fastapi import Header, HTTPException, status

from app.core.config import get_settings


def verify_actuator_api_key(
    x_api_key: str | None = Header(None, alias="X-API-Key", description="Secret API key for actuator access")
) -> str:
    """
    Enforces API key authentication on actuator and privileged routes.
    Uses secrets.compare_digest to prevent timing attack vulnerabilities.
    """
    settings = get_settings()

    # If an API key is configured (always required in production)
    if settings.api_key:
        if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Missing or invalid X-API-Key header.",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        return x_api_key

    # In development environment without API_KEY set, allow with warning
    if settings.app_env == "production":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: API_KEY must be configured in production environment.",
        )

    return x_api_key or "dev-unauthenticated"

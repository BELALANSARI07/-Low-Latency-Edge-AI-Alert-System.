import pytest
from fastapi import HTTPException

from app.api.dependencies import verify_actuator_api_key
from app.core.config import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_api_key_valid(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-test-token-12345")
    get_settings.cache_clear()

    # Verify with matching key
    result = verify_actuator_api_key(x_api_key="secret-test-token-12345")
    assert result == "secret-test-token-12345"


def test_api_key_missing_or_invalid(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-test-token-12345")
    get_settings.cache_clear()

    # Missing key
    with pytest.raises(HTTPException) as exc_info:
        verify_actuator_api_key(x_api_key=None)
    assert exc_info.value.status_code == 401

    # Wrong key
    with pytest.raises(HTTPException) as exc_info:
        verify_actuator_api_key(x_api_key="wrong-key")
    assert exc_info.value.status_code == 401


def test_api_key_unconfigured_in_production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("API_KEY", raising=False)
    get_settings.cache_clear()

    with pytest.raises(HTTPException) as exc_info:
        verify_actuator_api_key(x_api_key=None)
    assert exc_info.value.status_code == 500

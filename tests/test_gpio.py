import time
import pytest
from app.core.gpio_controller import GPIOController, GPIOState


def test_gpio_initial_state_is_low():
    controller = GPIOController(pin=18, pulse_duration_ms=100, enabled=False)
    status = controller.get_status()
    assert status["pin"] == 18
    assert status["state"] == "LOW"
    assert status["is_active"] is False


def test_gpio_timed_pulse_auto_resets_to_low():
    controller = GPIOController(pin=18, pulse_duration_ms=50, enabled=False)

    # Trigger pulse -> goes HIGH
    controller.trigger_pulse()
    assert controller.get_status()["state"] == "HIGH"
    assert controller.get_status()["is_active"] is True
    assert controller.get_status()["pulse_count"] == 1

    # Wait for pulse duration + margin -> auto resets to LOW
    time.sleep(0.08)
    assert controller.get_status()["state"] == "LOW"
    assert controller.get_status()["is_active"] is False


def test_gpio_forced_reset_and_shutdown():
    controller = GPIOController(pin=18, pulse_duration_ms=200, enabled=False)
    controller.trigger_pulse()
    assert controller.get_status()["state"] == "HIGH"

    # Immediate forced reset
    controller.reset_low()
    assert controller.get_status()["state"] == "LOW"

    # Shutdown
    controller.shutdown()
    assert controller.get_status()["state"] == "LOW"

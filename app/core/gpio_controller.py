"""
Fail-Safe Hardware & Simulated GPIO Actuator Controller.
Guarantees GPIO pin state is LOW on startup, shutdown, and error.
Implements non-blocking timed pulse reset to prevent hardware relay lockup.
Supports NVIDIA Jetson Nano (Jetson.GPIO), Raspberry Pi (RPi.GPIO), and Mock simulation.
"""

import logging
import threading
import time
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class GPIOState(str, Enum):
    LOW = "LOW"
    HIGH = "HIGH"


class GPIOController:
    """
    Hardware-agnostic safe GPIO controller.
    Ensures safe LOW state upon initialization and graceful teardown.
    """

    def __init__(
        self,
        pin: int = 18,
        pulse_duration_ms: int = 200,
        enabled: bool = False,
    ):
        self.pin = pin
        self.pulse_duration_ms = pulse_duration_ms
        self.enabled = enabled

        self._state = GPIOState.LOW
        self._lock = threading.Lock()
        self._pulse_timer: threading.Timer | None = None
        self._hardware_type = "SIMULATED"
        self._pulse_count = 0
        self._last_trigger_timestamp: float = 0.0

        self._initialize_hardware()

    def _initialize_hardware(self):
        """Discovers and initializes hardware GPIO library or falls back safely."""
        if not self.enabled:
            self._hardware_type = "SIMULATED"
            logger.info("GPIO Controller running in SIMULATION mode (actuation disabled).")
            return

        # Attempt Jetson.GPIO
        try:
            from Jetson import GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
            self._hardware_type = "Jetson.GPIO"
            self._gpio_module = GPIO
            logger.info(f"Initialized Jetson.GPIO on BCM pin {self.pin} (initial: LOW).")
            return
        except (ImportError, Exception):
            pass

        # Attempt RPi.GPIO
        try:
            from RPi import GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
            self._hardware_type = "RPi.GPIO"
            self._gpio_module = GPIO
            logger.info(f"Initialized RPi.GPIO on BCM pin {self.pin} (initial: LOW).")
            return
        except (ImportError, Exception):
            pass

        # Fallback to simulated if physical hardware unavailable
        self._hardware_type = "SIMULATED_FALLBACK"
        logger.warning(
            f"Physical GPIO libraries (Jetson.GPIO/RPi.GPIO) not available. "
            f"Falling back to simulated actuator for pin {self.pin}."
        )

    def trigger_pulse(self) -> bool:
        """
        Drives the actuator pin HIGH and schedules an automatic reset to LOW
        after pulse_duration_ms. Thread-safe and cancels any pending reset timer.
        """
        with self._lock:
            # Cancel any existing reset timer to prevent race conditions
            if self._pulse_timer is not None:
                self._pulse_timer.cancel()
                self._pulse_timer = None

            self._set_pin_state(GPIOState.HIGH)
            self._state = GPIOState.HIGH
            self._pulse_count += 1
            self._last_trigger_timestamp = time.time()

            # Schedule automatic reset to LOW
            delay_sec = max(0.01, self.pulse_duration_ms / 1000.0)
            self._pulse_timer = threading.Timer(delay_sec, self._auto_reset_low)
            self._pulse_timer.daemon = True
            self._pulse_timer.start()

            logger.info(
                f"GPIO Pin {self.pin} triggered HIGH (pulse duration: {self.pulse_duration_ms}ms)."
            )
            return True

    def _auto_reset_low(self):
        """Callback invoked by timer to reset GPIO pin safely to LOW."""
        with self._lock:
            self._set_pin_state(GPIOState.LOW)
            self._state = GPIOState.LOW
            self._pulse_timer = None
            logger.debug(f"GPIO Pin {self.pin} reset safely to LOW.")

    def _set_pin_state(self, state: GPIOState):
        """Executes physical pin write if hardware module is present."""
        if self._hardware_type in ("Jetson.GPIO", "RPi.GPIO") and hasattr(self, "_gpio_module"):
            try:
                val = self._gpio_module.HIGH if state == GPIOState.HIGH else self._gpio_module.LOW
                self._gpio_module.output(self.pin, val)
            except Exception as e:
                logger.error(f"Hardware write to GPIO pin {self.pin} failed: {e}")

    def reset_low(self):
        """Forces the actuator immediately to safe LOW state."""
        with self._lock:
            if self._pulse_timer is not None:
                self._pulse_timer.cancel()
                self._pulse_timer = None
            self._set_pin_state(GPIOState.LOW)
            self._state = GPIOState.LOW

    def shutdown(self):
        """Ensures pin is LOW and releases hardware handles safely."""
        self.reset_low()
        if self._hardware_type in ("Jetson.GPIO", "RPi.GPIO") and hasattr(self, "_gpio_module"):
            try:
                self._gpio_module.cleanup(self.pin)
            except Exception as e:
                logger.debug(f"GPIO cleanup warning: {e}")
        logger.info(f"GPIO Controller for pin {self.pin} shutdown safely.")

    def get_status(self) -> dict[str, Any]:
        """Returns the current actuator state and metrics."""
        with self._lock:
            return {
                "pin": self.pin,
                "state": self._state.value,
                "is_active": self._state == GPIOState.HIGH,
                "enabled": self.enabled,
                "hardware_type": self._hardware_type,
                "pulse_duration_ms": self.pulse_duration_ms,
                "pulse_count": self._pulse_count,
                "last_trigger_timestamp": self._last_trigger_timestamp,
            }

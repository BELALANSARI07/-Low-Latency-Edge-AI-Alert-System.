import pytest
from pydantic import ValidationError
from app.core.config import Settings, clear_settings_cache


def test_default_development_configuration():
    clear_settings_cache()
    settings = Settings()

    assert settings.app_env == "development"
    assert settings.simulation_mode is True
    assert settings.enable_gpio_actuation is False
    assert settings.execution_provider == "CPUExecutionProvider"
    assert settings.is_production is False
    assert settings.is_simulation is True
    assert settings.quantization_mode == "FP16"
    assert settings.camera_fps == 60
    assert settings.confidence_threshold == 0.70
    assert settings.iou_threshold == 0.45
    assert settings.gpio_pulse_duration_ms == 1500


def test_execution_provider_normalization():
    # Test shorthand "CPU"
    s_cpu = Settings(execution_provider="CPU")
    assert s_cpu.execution_provider == "CPUExecutionProvider"

    # Test shorthand "CUDA"
    s_cuda = Settings(execution_provider="CUDA")
    assert s_cuda.execution_provider == "CUDAExecutionProvider"

    # Test shorthand "TensorRT"
    s_trt = Settings(execution_provider="TensorRT")
    assert s_trt.execution_provider == "TensorrtExecutionProvider"

    # Test explicit full name
    s_full = Settings(execution_provider="CUDAExecutionProvider")
    assert s_full.execution_provider == "CUDAExecutionProvider"

    # Test invalid provider
    with pytest.raises(ValidationError):
        Settings(execution_provider="InvalidUnknownProvider")


def test_quantization_mode_validation():
    for q in ["FP32", "FP16", "INT8", "fp16", "int8"]:
        s = Settings(quantization_mode=q.upper())
        assert s.quantization_mode in ["FP32", "FP16", "INT8"]

    with pytest.raises(ValidationError):
        Settings(quantization_mode="INVALID_QUANT")


def test_confidence_and_iou_bounds():
    # Valid bounds
    Settings(confidence_threshold=0.0, iou_threshold=1.0)
    Settings(confidence_threshold=0.5, iou_threshold=0.5)

    # Invalid confidence > 1.0
    with pytest.raises(ValidationError):
        Settings(confidence_threshold=1.05)

    # Invalid confidence < 0.0
    with pytest.raises(ValidationError):
        Settings(confidence_threshold=-0.1)

    # Invalid IoU > 1.0
    with pytest.raises(ValidationError):
        Settings(iou_threshold=1.5)


def test_camera_dimension_bounds():
    # Negative width
    with pytest.raises(ValidationError):
        Settings(camera_width=-640)

    # Zero FPS
    with pytest.raises(ValidationError):
        Settings(camera_fps=0)


def test_mqtt_validation():
    # Valid QoS
    for qos in [0, 1, 2]:
        s = Settings(mqtt_qos=qos)
        assert s.mqtt_qos == qos

    # Invalid QoS
    with pytest.raises(ValidationError):
        Settings(mqtt_qos=3)

    # Invalid port
    with pytest.raises(ValidationError):
        Settings(mqtt_broker_port=70000)


def test_gpio_and_cooldown_validation():
    with pytest.raises(ValidationError):
        Settings(gpio_pulse_duration_ms=0)

    with pytest.raises(ValidationError):
        Settings(alert_cooldown_seconds=-1.0)


def test_production_constraints():
    # In production, if simulation mode is false, model path must be non-empty
    with pytest.raises(ValidationError):
        Settings(
            app_env="production",
            simulation_mode=False,
            model_path=""
        )

    s_prod = Settings(
        app_env="production",
        simulation_mode=False,
        model_path="models/yolov8n.onnx"
    )
    assert s_prod.is_production is True
    assert s_prod.is_simulation is False


def test_allowed_origins_list():
    s = Settings(allowed_origins="https://my-dashboard.com, https://edge-hub.local")
    assert s.allowed_origins_list == ["https://my-dashboard.com", "https://edge-hub.local"]

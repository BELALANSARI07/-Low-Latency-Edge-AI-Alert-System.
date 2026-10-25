import os
import tempfile
import pytest
from app.core.model_manager import (
    ModelManager,
    ModelStatusEnum,
    compute_file_sha256,
)


def test_model_missing_simulation_mode():
    """In simulation mode, a missing model file reports SIMULATION status and is marked usable."""
    manager = ModelManager(
        model_path="non_existent_model.onnx",
        simulation_mode=True,
        fail_on_missing_model=False,
    )
    meta = manager.validate()
    assert meta.status == ModelStatusEnum.SIMULATION
    assert meta.is_valid is True
    assert meta.active_provider == "SimulationProvider"
    assert manager.is_healthy() is True


def test_model_missing_production_mode():
    """In production mode without simulation, a missing model reports MISSING status and fails."""
    manager = ModelManager(
        model_path="non_existent_model.onnx",
        simulation_mode=False,
        fail_on_missing_model=False,
    )
    meta = manager.validate()
    assert meta.status == ModelStatusEnum.MISSING
    assert meta.is_valid is False
    assert manager.is_healthy() is False


def test_model_missing_fail_fast():
    """When fail_on_missing_model is True, validation raises FileNotFoundError."""
    manager = ModelManager(
        model_path="non_existent_model.onnx",
        simulation_mode=False,
        fail_on_missing_model=True,
    )
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        manager.validate()


def test_model_empty_file():
    """An empty file (0 bytes) must report INVALID status."""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        temp_path = f.name

    try:
        manager = ModelManager(model_path=temp_path, simulation_mode=False)
        meta = manager.validate()
        assert meta.status == ModelStatusEnum.INVALID
        assert meta.is_valid is False
        assert "empty" in meta.error_message.lower()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_checksum_verification_and_mismatch():
    """Validates checksum calculation and mismatch detection."""
    content = b"TEST_MODEL_BINARY_CONTENTS_FOR_SHA256_INTEGRITY_VERIFICATION"
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        f.write(content)
        temp_path = f.name

    try:
        expected_hash = compute_file_sha256(temp_path)
        assert len(expected_hash) == 64

        # Test with wrong checksum
        manager_bad = ModelManager(
            model_path=temp_path,
            expected_sha256="0000000000000000000000000000000000000000000000000000000000000000",
            simulation_mode=False,
        )
        meta_bad = manager_bad.validate()
        assert meta_bad.status == ModelStatusEnum.CHECKSUM_MISMATCH
        assert meta_bad.is_valid is False

        # Test with matching checksum (will fail ONNX session parsing unless real ONNX, but checksum matches)
        manager_good = ModelManager(
            model_path=temp_path,
            expected_sha256=expected_hash,
            simulation_mode=False,
        )
        meta_good = manager_good.validate()
        assert meta_good.sha256 == expected_hash
        # Since temp file is not a valid ONNX graph, ONNX session parser marks it INVALID
        assert meta_good.status == ModelStatusEnum.INVALID
        assert "onnx" in meta_good.error_message.lower()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_model_manager_to_dict():
    """ModelMetadata serialization produces valid dictionary."""
    manager = ModelManager(model_path="dummy.onnx", simulation_mode=True)
    meta = manager.validate()
    d = meta.to_dict()
    assert isinstance(d, dict)
    assert d["status"] == "SIMULATION"
    assert d["model_name"] == "dummy.onnx"

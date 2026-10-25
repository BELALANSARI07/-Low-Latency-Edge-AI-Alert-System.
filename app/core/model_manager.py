"""
Model Lifecycle and Integrity Management Layer.
Validates ONNX model binaries, computes SHA-256 integrity checksums,
inspects input/output tensor shapes, and enforces explicit status tracking.
"""

import hashlib
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ModelStatusEnum(str, Enum):
    """Enumeration of model runtime and validation states."""
    LOADED = "LOADED"
    MISSING = "MISSING"
    INVALID = "INVALID"
    CHECKSUM_MISMATCH = "CHECKSUM_MISMATCH"
    SIMULATION = "SIMULATION"


@dataclass
class ModelMetadata:
    """Detailed structural and integrity metadata for an edge AI model."""
    status: ModelStatusEnum
    is_valid: bool
    model_path: str
    model_name: str
    file_size_bytes: int = 0
    sha256: str | None = None
    expected_sha256: str | None = None
    input_names: list[str] = field(default_factory=list)
    input_shapes: list[list[Any]] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)
    output_shapes: list[list[Any]] = field(default_factory=list)
    available_providers: list[str] = field(default_factory=list)
    active_provider: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "is_valid": self.is_valid,
            "model_path": self.model_path,
            "model_name": self.model_name,
            "file_size_bytes": self.file_size_bytes,
            "sha256": self.sha256,
            "expected_sha256": self.expected_sha256,
            "input_names": self.input_names,
            "input_shapes": self.input_shapes,
            "output_names": self.output_names,
            "output_shapes": self.output_shapes,
            "available_providers": self.available_providers,
            "active_provider": self.active_provider,
            "error_message": self.error_message,
        }


def compute_file_sha256(filepath: str, chunk_size: int = 65536) -> str:
    """Computes SHA-256 hexadecimal checksum of a file streaming in chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest().lower()


class ModelManager:
    """
    Manages model discovery, verification, and metadata extraction.
    Protects edge runtime against corrupt or missing weight files.
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n-fp16.onnx",
        expected_sha256: str | None = None,
        simulation_mode: bool = True,
        fail_on_missing_model: bool = False,
    ):
        self.model_path = model_path
        self.expected_sha256 = expected_sha256.lower() if expected_sha256 else None
        self.simulation_mode = simulation_mode
        self.fail_on_missing_model = fail_on_missing_model
        self._metadata: ModelMetadata | None = None

    @property
    def model_name(self) -> str:
        return os.path.basename(self.model_path)

    def validate(self, force_refresh: bool = False) -> ModelMetadata:
        """
        Validates model presence, checksum, and ONNX session compatibility.
        Caches result unless force_refresh is True.
        """
        if self._metadata is not None and not force_refresh:
            return self._metadata

        # Case 1: File does not exist
        if not os.path.exists(self.model_path) or not os.path.isfile(self.model_path):
            if self.fail_on_missing_model:
                raise FileNotFoundError(
                    f"Model file not found at: '{self.model_path}' and fail_on_missing_model is True."
                )

            if self.simulation_mode:
                self._metadata = ModelMetadata(
                    status=ModelStatusEnum.SIMULATION,
                    is_valid=True,
                    model_path=self.model_path,
                    model_name=self.model_name,
                    file_size_bytes=0,
                    available_providers=["SimulationProvider"],
                    active_provider="SimulationProvider",
                    error_message="Running in simulation mode without physical ONNX model file.",
                )
            else:
                self._metadata = ModelMetadata(
                    status=ModelStatusEnum.MISSING,
                    is_valid=False,
                    model_path=self.model_path,
                    model_name=self.model_name,
                    file_size_bytes=0,
                    error_message=f"Model file not found at: '{self.model_path}'.",
                )
            return self._metadata

        # Case 2: File exists -> Check file size
        file_size = os.path.getsize(self.model_path)
        if file_size == 0:
            self._metadata = ModelMetadata(
                status=ModelStatusEnum.INVALID,
                is_valid=False,
                model_path=self.model_path,
                model_name=self.model_name,
                file_size_bytes=0,
                error_message="Model file exists but is empty (0 bytes).",
            )
            return self._metadata

        # Case 3: Compute SHA-256 Checksum
        try:
            actual_sha256 = compute_file_sha256(self.model_path)
        except Exception as e:
            self._metadata = ModelMetadata(
                status=ModelStatusEnum.INVALID,
                is_valid=False,
                model_path=self.model_path,
                model_name=self.model_name,
                file_size_bytes=file_size,
                error_message=f"Failed to read file for checksum computation: {e}",
            )
            return self._metadata

        # Case 4: Verify expected SHA-256 if configured
        if self.expected_sha256 and actual_sha256 != self.expected_sha256:
            msg = (
                f"Checksum mismatch for '{self.model_path}'. "
                f"Expected: {self.expected_sha256}, Got: {actual_sha256}"
            )
            logger.error(msg)
            self._metadata = ModelMetadata(
                status=ModelStatusEnum.CHECKSUM_MISMATCH,
                is_valid=False,
                model_path=self.model_path,
                model_name=self.model_name,
                file_size_bytes=file_size,
                sha256=actual_sha256,
                expected_sha256=self.expected_sha256,
                error_message=msg,
            )
            return self._metadata

        # Case 5: Inspect ONNX runtime session structure
        input_names: list[str] = []
        input_shapes: list[list[Any]] = []
        output_names: list[str] = []
        output_shapes: list[list[Any]] = []
        available_providers: list[str] = []
        active_provider: str | None = None

        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()

            # Attempt creating minimal session to inspect graph
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            sess = ort.InferenceSession(self.model_path, sess_opts, providers=["CPUExecutionProvider"])

            for inp in sess.get_inputs():
                input_names.append(inp.name)
                input_shapes.append(inp.shape)

            for out in sess.get_outputs():
                output_names.append(out.name)
                output_shapes.append(out.shape)

            active_provider = sess.get_providers()[0] if sess.get_providers() else "CPUExecutionProvider"

        except ImportError:
            # onnxruntime not installed in environment, but file exists and checksum matched
            logger.warning("onnxruntime is not installed; ONNX graph validation skipped.")
        except Exception as e:
            msg = f"Failed to load ONNX model graph: {e}"
            logger.error(msg)
            self._metadata = ModelMetadata(
                status=ModelStatusEnum.INVALID,
                is_valid=False,
                model_path=self.model_path,
                model_name=self.model_name,
                file_size_bytes=file_size,
                sha256=actual_sha256,
                expected_sha256=self.expected_sha256,
                error_message=msg,
            )
            return self._metadata

        # Model is fully valid and loaded
        self._metadata = ModelMetadata(
            status=ModelStatusEnum.LOADED,
            is_valid=True,
            model_path=self.model_path,
            model_name=self.model_name,
            file_size_bytes=file_size,
            sha256=actual_sha256,
            expected_sha256=self.expected_sha256,
            input_names=input_names,
            input_shapes=input_shapes,
            output_names=output_names,
            output_shapes=output_shapes,
            available_providers=available_providers,
            active_provider=active_provider,
        )
        return self._metadata

    def get_status(self) -> ModelStatusEnum:
        """Returns current model status enum."""
        meta = self.validate()
        return meta.status

    def is_healthy(self) -> bool:
        """Returns True if model is usable for inference or valid simulation."""
        meta = self.validate()
        return meta.is_valid

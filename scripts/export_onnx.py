"""
Model Export & Quantization Pipeline:
Exports YOLOv8-nano / small models from PyTorch into ONNX FP16 and INT8 formats
calibrated for NVIDIA Jetson Nano TensorRT / ONNX Runtime EdgeGPU execution.
"""

import os
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_exporter")


def export_yolo_to_onnx(
    weights_path: str = "yolov8n.pt",
    output_path: str = "models/yolov8n-fp16.onnx",
    quantization: str = "FP16",
    img_size: int = 640,
    opset: int = 17
):
    """
    Exports YOLOv8 PyTorch weights to ONNX format with FP16 half precision.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    logger.info(f"Initiating export for {weights_path} -> {output_path} (Quant: {quantization})")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Input weights file not found: {weights_path}")

    try:
        from ultralytics import YOLO
        model = YOLO(weights_path)

        half_precision = (quantization == "FP16")
        int8_quant = (quantization == "INT8")

        exported_file = model.export(
            format="onnx",
            imgsz=img_size,
            half=half_precision,
            int8=int8_quant,
            opset=opset,
            simplify=True,
            dynamic=False,
        )
        logger.info(f"Model successfully exported to: {exported_file}")
        return exported_file
    except ImportError as e:
        logger.error(
            "The 'ultralytics' package is required to export YOLO models to ONNX. "
            "Install it using: pip install ultralytics onnx"
        )
        raise RuntimeError(
            "Cannot export model: 'ultralytics' package is not installed."
        ) from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to Edge-Optimized ONNX")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Path to .pt weights")
    parser.add_argument("--output", type=str, default="models/yolov8n-fp16.onnx", help="Target .onnx path")
    parser.add_argument("--quant", type=str, default="FP16", choices=["FP32", "FP16", "INT8"], help="Quantization")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image dimension")

    args = parser.parse_args()
    export_yolo_to_onnx(
        weights_path=args.weights,
        output_path=args.output,
        quantization=args.quant,
        img_size=args.imgsz
    )

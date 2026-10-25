#!/usr/bin/env python3
"""
CLI Model Verification & Inspection Utility.
Inspects ONNX weights, validates SHA-256 integrity checksums,
and verifies tensor graph input/output signatures.
"""

import argparse
import json
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.model_manager import ModelManager


def main():
    parser = argparse.ArgumentParser(
        description="Verify edge ONNX model file integrity, checksum, and tensor signatures."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/yolov8n-fp16.onnx",
        help="Path to the ONNX model file"
    )
    parser.add_argument(
        "--expected-sha256",
        type=str,
        default=None,
        help="Expected SHA-256 checksum for verification"
    )
    parser.add_argument(
        "--simulation-mode",
        action="store_true",
        help="Allow simulation mode if model file is not found"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format"
    )

    args = parser.parse_args()

    manager = ModelManager(
        model_path=args.model_path,
        expected_sha256=args.expected_sha256,
        simulation_mode=args.simulation_mode,
        fail_on_missing_model=False,
    )

    try:
        meta = manager.validate()
    except Exception as e:
        print(f"ERROR: Model validation raised exception: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(meta.to_dict(), indent=2))
    else:
        print("=" * 60)
        print("EDGE AI MODEL VERIFICATION REPORT")
        print("=" * 60)
        print(f"Model Path:         {meta.model_path}")
        print(f"Model Name:         {meta.model_name}")
        print(f"Validation Status:  {meta.status.value}")
        print(f"Usable / Valid:     {'YES' if meta.is_valid else 'NO'}")
        print(f"File Size:          {meta.file_size_bytes} bytes")
        print(f"Computed SHA-256:   {meta.sha256 or 'N/A'}")
        if meta.expected_sha256:
            print(f"Expected SHA-256:   {meta.expected_sha256}")
            print(f"Checksum Match:     {'YES' if meta.sha256 == meta.expected_sha256 else 'NO'}")
        if meta.input_names:
            print("Model Inputs:")
            for name, shape in zip(meta.input_names, meta.input_shapes):
                print(f"  - {name}: {shape}")
        if meta.output_names:
            print("Model Outputs:")
            for name, shape in zip(meta.output_names, meta.output_shapes):
                print(f"  - {name}: {shape}")
        if meta.active_provider:
            print(f"Active Provider:    {meta.active_provider}")
        if meta.error_message:
            print(f"Diagnostics:        {meta.error_message}")
        print("=" * 60)

    sys.exit(0 if meta.is_valid else 1)


if __name__ == "__main__":
    main()

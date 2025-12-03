#!/usr/bin/env python3
"""Quantize YOLO11 FP32 TFLite model to INT8 with calibration.

This script:
- Loads the FP32 TFLite model exported from best.pt
- Uses a representative dataset of training images
- Produces an INT8 TFLite model suitable for NPU / embedded deployment

Output model:
- models/pytorch/best_saved_model/best_int8_from_fp32.tflite
"""

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
FP32_TFLITE_PATH = ROOT / "models/pytorch/best_saved_model/best_float32.tflite"
INT8_TFLITE_PATH = ROOT / "models/pytorch/best_saved_model/best_int8_from_fp32.tflite"

# Representative images directory: adjust if your 300-image dataset lives elsewhere
REP_DATA_DIRS = [
    ROOT / "data_1-300",  # main folder if exists
]

IMG_SIZE = 1024


def iter_rep_images(max_images: int = 200) -> Iterable[np.ndarray]:
    """Yield preprocessed images for representative dataset.

    We keep it simple: sample up to `max_images` from REP_DATA_DIRS recursively.
    """

    images = []
    for base in REP_DATA_DIRS:
        if not base.exists():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in base.rglob(ext):
                images.append(p)
    if not images:
        raise RuntimeError(
            f"No representative images found under: {[str(d) for d in REP_DATA_DIRS]}"
        )

    images = sorted(images)[:max_images]
    print(f"Using {len(images)} images for representative dataset")

    for img_path in images:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        # FP32 model expects float32 [0,1]
        img = img_rgb.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        yield img


def representative_dataset() -> Iterable[dict]:
    for img in iter_rep_images():
        # Single input tensor; converter will map this to input[0]
        yield [img]


def main():
    print("=" * 80)
    print("  Quantizing FP32 TFLite to INT8 (with calibration)")
    print("=" * 80)

    if not FP32_TFLITE_PATH.exists():
        raise FileNotFoundError(f"FP32 TFLite model not found at {FP32_TFLITE_PATH}")

    print(f"Loading FP32 TFLite model from: {FP32_TFLITE_PATH.relative_to(ROOT)}")

    # Note: There's no direct 'from_tflite' quantization. We load via a TF function
    # and re-convert, keeping the same computation graph.
    # Here, we use the TFLite model as a black box; quantization primarily affects
    # weights and activations inside the TFLite kernel implementations.

    # Load the FP32 model back as a ConcreteFunction using tf.lite.Interpreter
    interpreter = tf.lite.Interpreter(model_path=str(FP32_TFLITE_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_shape = input_details["shape"]

    print(f"Input shape: {input_shape}, dtype: {input_details['dtype']}")
    print(f"Output shape: {output_details['shape']}, dtype: {output_details['dtype']}")

    # Build a simple tf.function wrapper around the TFLite model
    @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)])
    def tflite_model(x):
        # Run through interpreter
        interpreter.set_tensor(input_details["index"], x.numpy())
        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])
        return tf.convert_to_tensor(out)

    concrete_func = tflite_model.get_concrete_function()

    # Convert this function to INT8 TFLite
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    print("Converting to INT8... this may take a while")
    int8_model = converter.convert()

    INT8_TFLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INT8_TFLITE_PATH, "wb") as f:
        f.write(int8_model)

    print(f"\n✅ Saved INT8 TFLite model to: {INT8_TFLITE_PATH.relative_to(ROOT)}")
    print("=" * 80)


if __name__ == "__main__":
    main()

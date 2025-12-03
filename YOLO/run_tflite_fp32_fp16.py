#!/usr/bin/env python3
"""Run YOLO11 TFLite FP32/FP16 models on test images and compare to PyTorch.

This script:
- Uses the original PyTorch model for reference
- Runs the float32 TFLite model
- Runs the float16 TFLite model
- Uses careful preprocessing/postprocessing to match YOLO11 behaviour
"""

import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# Paths
ROOT = Path(__file__).resolve().parent
PT_MODEL_PATH = ROOT / "models/pytorch/best.pt"
# These are the clean TFLite exports produced by Ultralytics
TFLITE_FP32_PATH = ROOT / "models/pytorch/best_saved_model/best_float32.tflite"
TFLITE_FP16_PATH = ROOT / "models/pytorch/best_saved_model/best_float16.tflite"
TEST_IMAGES_DIR = ROOT / "test_images"
OUT_DIR = ROOT / "pipeline_output_tflite_fp"
OUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 1024  # YOLO11 input size
CLASSES = ["worker", "truck", "bike", "bulldozer", "car"]


def load_interpreter(model_path: Path) -> tf.lite.Interpreter:
    print(f"\n🔄 Loading TFLite model: {model_path.relative_to(ROOT)}")
    interpreter = tf.lite.Interpreter(str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print(f"   Input shape: {input_details['shape']}, dtype: {input_details['dtype']}")
    print(f"   Output shape: {output_details['shape']}, dtype: {output_details['dtype']}")
    print(f"   Input quant: {input_details['quantization']}")
    print(f"   Output quant: {output_details['quantization']}")
    return interpreter


def preprocess_image(img_bgr: np.ndarray, input_details) -> np.ndarray:
    """Resize and normalize image for TFLite model.

    For float models: [0,1] float32
    For quantized models: uint8 [0,255] with given scale/zero_point
    """
    h, w = IMG_SIZE, IMG_SIZE
    img_resized = cv2.resize(img_bgr, (w, h))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    dtype = input_details["dtype"]
    scale, zero_point = input_details["quantization"]

    if dtype == np.float32:
        # For FP32/FP16 models exported by Ultralytics, input is float32 [0,1]
        img = img_rgb.astype(np.float32) / 255.0
    elif dtype == np.uint8:
        # For quantized models, use uint8 with proper scaling
        if scale == 0 or scale is None:
            img = img_rgb.astype(np.uint8)
        else:
            img = (img_rgb.astype(np.float32) / 255.0 / scale + zero_point).astype(
                np.uint8
            )
    else:
        raise ValueError(f"Unsupported input dtype: {dtype}")

    img = np.expand_dims(img, axis=0)
    return img


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def postprocess_yolo_output(
    output: np.ndarray,
    output_details,
    img_w: int,
    img_h: int,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
):
    """Convert raw YOLO head output to detection dicts.

    Expected output shape from YOLO11 TFLite: [1, 9, 21504]
    9 = [x, y, w, h, c1, c2, c3, c4, c5]
    """
    # Remove batch dim -> [9, 21504]
    output = output[0]

    # Dequantize if needed
    if output_details["dtype"] == np.uint8:
        scale, zero_point = output_details["quantization"]
        output = scale * (output.astype(np.float32) - zero_point)
    else:
        output = output.astype(np.float32)

    # Transpose to [21504, 9]
    output = output.transpose(1, 0)

    # Apply sigmoid to all channels (as in YOLO head)
    output = sigmoid(output)

    boxes = output[:, :4]  # [x_center, y_center, w, h] in [0,1]
    class_scores = output[:, 4:]  # [21504, num_classes]

    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)

    # Filter by confidence
    mask = confidences > conf_threshold
    boxes = boxes[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    if len(boxes) == 0:
        return []

    detections = []
    for box, cls_id, conf in zip(boxes, class_ids, confidences):
        x_c, y_c, w_box, h_box = box
        # clip
        x_c = float(np.clip(x_c, 0, 1))
        y_c = float(np.clip(y_c, 0, 1))
        w_box = float(np.clip(w_box, 0, 1))
        h_box = float(np.clip(h_box, 0, 1))

        x1 = (x_c - w_box / 2.0) * img_w
        y1 = (y_c - h_box / 2.0) * img_h
        x2 = (x_c + w_box / 2.0) * img_w
        y2 = (y_c + h_box / 2.0) * img_h

        x1 = float(np.clip(x1, 0, img_w))
        y1 = float(np.clip(y1, 0, img_h))
        x2 = float(np.clip(x2, 0, img_w))
        y2 = float(np.clip(y2, 0, img_h))

        detections.append(
            {
                "class": CLASSES[int(cls_id)],
                "class_id": int(cls_id),
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2],
            }
        )

    # Simple NMS
    detections = nms(detections, iou_threshold)
    return detections


def nms(detections, iou_thresh: float):
    if not detections:
        return []
    boxes = np.array([d["bbox"] for d in detections])
    scores = np.array([d["confidence"] for d in detections])
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rem = (
            (boxes[order[1:], 2] - boxes[order[1:], 0])
            * (boxes[order[1:], 3] - boxes[order[1:], 1])
        )
        union = area_i + area_rem - inter
        iou = inter / (union + 1e-6)

        remaining = np.where(iou <= iou_thresh)[0]
        order = order[remaining + 1]

    return [detections[i] for i in keep]


def run_tflite_model(interpreter, image_paths, tag: str):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    results_summary = []
    total = 0

    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"   ⚠️ Could not read {img_path.name}")
            continue
        h, w = img_bgr.shape[:2]

        inp = preprocess_image(img_bgr, input_details)
        interpreter.set_tensor(input_details["index"], inp)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])

        detections = postprocess_yolo_output(out, output_details, w, h)
        print(f"   [{tag}] {img_path.name}: {len(detections)} detections")
        total += len(detections)

        # save JSON only; visualization we already have from PyTorch
        results_summary.append(
            {
                "image": img_path.name,
                "model": tag,
                "detections": detections,
            }
        )

    return results_summary, total


def run_pytorch_reference(image_paths):
    model = YOLO(str(PT_MODEL_PATH))
    results_summary = []
    total = 0

    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        res = model(img_bgr, conf=0.25, iou=0.45, imgsz=IMG_SIZE, verbose=False)[0]

        detections = []
        if res.boxes is not None:
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                detections.append(
                    {
                        "class": CLASSES[cls_id],
                        "class_id": cls_id,
                        "confidence": conf,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    }
                )

        print(f"   [pt] {img_path.name}: {len(detections)} detections")
        total += len(detections)
        results_summary.append(
            {
                "image": img_path.name,
                "model": "pytorch",
                "detections": detections,
            }
        )
    return results_summary, total


def main():
    print("=" * 80)
    print("  YOLO11 TFLite FP32/FP16 Inference Runner")
    print("=" * 80)

    image_paths = sorted(TEST_IMAGES_DIR.glob("*.jpg"))
    print(f"Found {len(image_paths)} test images")

    # 1. PyTorch reference
    print("\n📌 Running PyTorch reference model...")
    pt_results, pt_total = run_pytorch_reference(image_paths)
    print(f"   PyTorch total detections: {pt_total}")

    # 2. FP32 TFLite
    print("\n📌 Running TFLite FP32 model...")
    fp32_interp = load_interpreter(TFLITE_FP32_PATH)
    fp32_results, fp32_total = run_tflite_model(fp32_interp, image_paths, "tflite_fp32")
    print(f"   FP32 total detections: {fp32_total}")

    # 3. FP16 TFLite
    print("\n📌 Running TFLite FP16 model...")
    fp16_interp = load_interpreter(TFLITE_FP16_PATH)
    fp16_results, fp16_total = run_tflite_model(fp16_interp, image_paths, "tflite_fp16")
    print(f"   FP16 total detections: {fp16_total}")

    summary = {
        "images": [p.name for p in image_paths],
        "pytorch_total": pt_total,
        "fp32_total": fp32_total,
        "fp16_total": fp16_total,
        "pytorch": pt_results,
        "fp32": fp32_results,
        "fp16": fp16_results,
    }

    out_path = OUT_DIR / "tflite_fp_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("✅ Finished TFLite FP32/FP16 inference")
    print(f"   Summary saved to: {out_path.relative_to(ROOT)}")
    print("=" * 80)


if __name__ == "__main__":
    main()

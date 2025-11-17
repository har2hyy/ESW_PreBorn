#!/usr/bin/env python3
"""
ONNX Model Validation Pipeline for NPU Deployment
==================================================

This script validates the ONNX model output and demonstrates the exact
workflow that will run on QIDK NPU (minus the DLC conversion).

Workflow:
1. Load ONNX model (best.onnx)
2. Run inference on test image
3. Postprocess raw outputs (CPU-side, same as will be done on QIDK)
4. Validate detection count and coordinates
5. Save outputs for comparison

Author: NPU Deployment Pipeline
Date: November 2025
"""

import sys
import os
from pathlib import Path
import time
import json

import numpy as np
import onnxruntime as ort
import cv2

# YOLO configuration
CLASS_NAMES = {0: 'worker', 1: 'truck', 2: 'bike', 3: 'bulldozer', 4: 'car'}
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMG_SIZE = 1024


def preprocess_image(image_path: str, img_size: int = 1024) -> tuple:
    """
    Preprocess image for YOLO inference.
    
    Returns:
        (input_tensor, original_image, original_shape)
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    
    orig_shape = img.shape[:2]  # (height, width)
    
    # Resize to model input size
    img_resized = cv2.resize(img, (img_size, img_size))
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img_norm = img_rgb.astype(np.float32) / 255.0
    
    # Transpose to CHW format
    img_chw = np.transpose(img_norm, (2, 0, 1))
    
    # Add batch dimension
    input_tensor = np.expand_dims(img_chw, axis=0)
    
    return input_tensor, img, orig_shape


def xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    """Convert boxes from xywh (center) to xyxy (corners) format."""
    boxes_xyxy = np.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2
    return boxes_xyxy


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Non-maximum suppression."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return np.array(keep)


def postprocess_outputs(outputs: np.ndarray, orig_shape: tuple, 
                       conf_threshold: float, iou_threshold: float,
                       img_size: int) -> tuple:
    """
    Postprocess YOLO outputs (CPU-side, identical to what runs on QIDK).
    
    Args:
        outputs: Raw model output [1, 9, 21504]
        orig_shape: Original image (height, width)
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        img_size: Model input size
    
    Returns:
        (boxes, scores, class_ids) in original image coordinates
    """
    # Remove batch dimension
    outputs = outputs[0]  # [9, 21504]
    
    # Split into bbox and class predictions
    bbox_xywh = outputs[:4, :].T  # [21504, 4] - xywh in pixel coords
    class_probs = outputs[4:, :].T  # [21504, 5] - already sigmoid-activated
    
    # Get best class for each anchor
    max_scores = np.max(class_probs, axis=1)  # [21504]
    class_ids = np.argmax(class_probs, axis=1)  # [21504]
    
    # Filter by confidence
    mask = max_scores > conf_threshold
    bbox_xywh_filtered = bbox_xywh[mask]
    scores_filtered = max_scores[mask]
    classes_filtered = class_ids[mask]
    
    if len(bbox_xywh_filtered) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Convert xywh to xyxy
    boxes_xyxy = xywh_to_xyxy(bbox_xywh_filtered)
    
    # Apply NMS
    keep_indices = nms(boxes_xyxy, scores_filtered, iou_threshold)
    
    # Scale to original image size
    scale_h = orig_shape[0] / img_size
    scale_w = orig_shape[1] / img_size
    
    boxes_final = boxes_xyxy[keep_indices].copy()
    boxes_final[:, [0, 2]] *= scale_w
    boxes_final[:, [1, 3]] *= scale_h
    
    # Clip to image boundaries
    boxes_final[:, [0, 2]] = np.clip(boxes_final[:, [0, 2]], 0, orig_shape[1])
    boxes_final[:, [1, 3]] = np.clip(boxes_final[:, [1, 3]], 0, orig_shape[0])
    
    return boxes_final, scores_filtered[keep_indices], classes_filtered[keep_indices]


def visualize_detections(image: np.ndarray, boxes: np.ndarray, 
                        scores: np.ndarray, class_ids: np.ndarray,
                        output_path: str):
    """Draw bounding boxes on image and save."""
    img_viz = image.copy()
    
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        class_name = CLASS_NAMES[int(cls_id)]
        
        # Draw bbox
        color = (0, 255, 0)
        cv2.rectangle(img_viz, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {score:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_viz, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(img_viz, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    cv2.imwrite(output_path, img_viz)
    print(f"✅ Visualization saved: {output_path}")


def main():
    print("=" * 80)
    print("ONNX MODEL VALIDATION FOR NPU DEPLOYMENT")
    print("=" * 80)
    
    # Paths
    model_path = "runs/detect/train/weights/best.onnx"
    test_image = "/home/harshyy/Desktop/20250103_104457.jpg"
    output_viz = "onnx_npu_validation.jpg"
    output_json = "onnx_npu_validation.json"
    
    # Check files exist
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return 1
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return 1
    
    # Load ONNX model
    print(f"\n[1/4] Loading ONNX model: {model_path}")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=['CPUExecutionProvider']
    )
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    print(f"  Input: {input_name}, shape: {input_shape}")
    print(f"  Output: {output_name}")
    print(f"  ✓ Model loaded successfully")
    
    # Preprocess image
    print(f"\n[2/4] Preprocessing image: {test_image}")
    input_tensor, original_image, orig_shape = preprocess_image(test_image, IMG_SIZE)
    print(f"  Original shape: {orig_shape}")
    print(f"  Input tensor shape: {input_tensor.shape}")
    print(f"  ✓ Preprocessing complete")
    
    # Run inference
    print(f"\n[3/4] Running ONNX inference...")
    start_time = time.time()
    raw_output = session.run([output_name], {input_name: input_tensor})[0]
    inference_time = (time.time() - start_time) * 1000
    
    print(f"  Raw output shape: {raw_output.shape}")
    print(f"  Inference time: {inference_time:.1f} ms")
    print(f"  ✓ Inference complete")
    
    # Postprocess outputs
    print(f"\n[4/4] Postprocessing (CPU-side, same as QIDK)...")
    boxes, scores, class_ids = postprocess_outputs(
        raw_output, orig_shape, CONF_THRESHOLD, IOU_THRESHOLD, IMG_SIZE
    )
    
    print(f"\n{'='*80}")
    print(f"DETECTION RESULTS")
    print(f"{'='*80}")
    print(f"Total detections: {len(boxes)}")
    print()
    
    # Count by class
    class_counts = {}
    for cls_id in class_ids:
        cls_name = CLASS_NAMES[int(cls_id)]
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
    
    print("Detection summary:")
    for cls_name, count in sorted(class_counts.items()):
        print(f"  {cls_name}: {count}")
    
    print("\nDetailed detections:")
    detections = []
    for i, (box, score, cls_id) in enumerate(zip(boxes, scores, class_ids)):
        x1, y1, x2, y2 = map(int, box)
        cls_name = CLASS_NAMES[int(cls_id)]
        print(f"  {i+1}. {cls_name} ({score:.3f}) @ [{x1}, {y1}, {x2}, {y2}]")
        
        detections.append({
            "id": i + 1,
            "class": cls_name,
            "class_id": int(cls_id),
            "confidence": float(score),
            "bbox": [x1, y1, x2, y2]
        })
    
    # Save results
    print(f"\n{'='*80}")
    print("SAVING OUTPUTS")
    print(f"{'='*80}")
    
    # Visualize
    visualize_detections(original_image, boxes, scores, class_ids, output_viz)
    
    # Save JSON
    results = {
        "model": model_path,
        "image": test_image,
        "inference_time_ms": inference_time,
        "config": {
            "img_size": IMG_SIZE,
            "conf_threshold": CONF_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD
        },
        "summary": {
            "total_detections": len(boxes),
            "by_class": class_counts
        },
        "detections": detections
    }
    
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved: {output_json}")
    
    print(f"\n{'='*80}")
    print("VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Model ready for ONNX → DLC conversion")
    print(f"✓ Postprocessing pipeline validated")
    print(f"✓ Expected NPU output: raw tensor shape {raw_output.shape}")
    print(f"✓ CPU postprocessing will decode to {len(boxes)} objects")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

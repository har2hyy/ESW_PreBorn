#!/usr/bin/env python3
"""
NPU Output Decoder for YOLOv11 INT8 DLC
========================================

Decodes raw tensor output from SNPE/QNN NPU runtime and applies
same postprocessing as ONNX validation pipeline.

Usage:
    python decode_npu_output.py output0.raw original_image.jpg

Input:
    - output0.raw: Raw NPU output tensor (1, 9, 21504) in float32
    - original_image.jpg: Original image before preprocessing

Output:
    - npu_result.jpg: Visualized detections
    - Console output: Detection details
"""

import sys
import numpy as np
import cv2
import json
from pathlib import Path

# Class names (must match training)
CLASS_NAMES = ['helmet', 'mask', 'no-helmet', 'no-mask', 'worker']


def postprocess_outputs(raw_output, orig_shape, conf_threshold=0.25, iou_threshold=0.45, img_size=1024):
    """
    Postprocess YOLOv11 NPU outputs to get final detections.
    
    Args:
        raw_output: Numpy array of shape (1, 9, 21504)
        orig_shape: Tuple of (H, W) of original image
        conf_threshold: Confidence threshold for filtering
        iou_threshold: IoU threshold for NMS
        img_size: Input size used during inference (1024)
    
    Returns:
        boxes: Array of [x1, y1, x2, y2] in original image coordinates
        scores: Array of confidence scores
        class_ids: Array of class IDs
    """
    batch_size, num_channels, num_anchors = raw_output.shape
    
    # Extract components (absolute pixel coordinates)
    x_center = raw_output[0, 0, :]
    y_center = raw_output[0, 1, :]
    width = raw_output[0, 2, :]
    height = raw_output[0, 3, :]
    class_probs = raw_output[0, 4:, :]  # (5, 21504) - already sigmoid
    
    # Convert xywh to xyxy
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    # Get max class probability and class ID
    class_scores = np.max(class_probs, axis=0)  # (21504,)
    class_ids = np.argmax(class_probs, axis=0)  # (21504,)
    
    # Filter by confidence
    mask = class_scores > conf_threshold
    boxes_xyxy = np.stack([x1[mask], y1[mask], x2[mask], y2[mask]], axis=1)
    scores_filtered = class_scores[mask]
    class_ids_filtered = class_ids[mask]
    
    # NMS
    if len(boxes_xyxy) == 0:
        return np.array([]), np.array([]), np.array([])
    
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(),
        scores_filtered.tolist(),
        conf_threshold,
        iou_threshold
    )
    
    if len(indices) > 0:
        indices = indices.flatten()
        final_boxes = boxes_xyxy[indices]
        final_scores = scores_filtered[indices]
        final_class_ids = class_ids_filtered[indices]
        
        # Scale to original image size
        orig_h, orig_w = orig_shape
        scale_x = orig_w / img_size
        scale_y = orig_h / img_size
        
        final_boxes[:, [0, 2]] *= scale_x
        final_boxes[:, [1, 3]] *= scale_y
        
        return final_boxes, final_scores, final_class_ids
    else:
        return np.array([]), np.array([]), np.array([])


def visualize_detections(image, boxes, scores, class_ids, output_path='npu_result.jpg'):
    """Draw bounding boxes and save visualization."""
    img_vis = image.copy()
    
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f"{CLASS_NAMES[int(cls_id)]}: {score:.2f}"
        
        # Color coding
        color = (0, 255, 0)  # Default green
        if 'no-helmet' in CLASS_NAMES[int(cls_id)]:
            color = (0, 0, 255)  # Red for no-helmet
        elif 'helmet' in CLASS_NAMES[int(cls_id)]:
            color = (0, 255, 0)  # Green for helmet
        elif 'no-mask' in CLASS_NAMES[int(cls_id)]:
            color = (0, 165, 255)  # Orange for no-mask
        
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_vis, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imwrite(output_path, img_vis)
    print(f"✓ Visualization saved to: {output_path}")


def decode_npu_output(raw_file, image_path, output_viz='npu_result.jpg', output_json='npu_result.json'):
    """
    Main decoder function.
    
    Args:
        raw_file: Path to NPU output .raw file
        image_path: Path to original image
        output_viz: Output visualization path
        output_json: Output JSON results path
    """
    print(f"\n{'='*60}")
    print("NPU Output Decoder - YOLOv11 INT8 DLC")
    print(f"{'='*60}\n")
    
    # Load raw output tensor
    print(f"Loading NPU output: {raw_file}")
    raw_output = np.fromfile(raw_file, dtype=np.float32)
    
    # Reshape to (1, 9, 21504)
    try:
        raw_output = raw_output.reshape(1, 9, 21504)
        print(f"✓ Output shape: {raw_output.shape}")
    except ValueError:
        print(f"✗ Error: Expected {1*9*21504} values, got {len(raw_output)}")
        print(f"  File may be from wrong model or corrupted")
        return
    
    # Load original image
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"✗ Error: Could not load image {image_path}")
        return
    
    orig_shape = img.shape[:2]
    print(f"✓ Image shape: {orig_shape[0]}x{orig_shape[1]}\n")
    
    # Postprocess
    print("Running postprocessing (NMS + decoding)...")
    boxes, scores, class_ids = postprocess_outputs(
        raw_output, orig_shape,
        conf_threshold=0.25,
        iou_threshold=0.45,
        img_size=1024
    )
    
    # Display results
    print(f"\n{'='*60}")
    print(f"NPU Detections: {len(boxes)}")
    print(f"{'='*60}\n")
    
    detections = []
    for i, (box, score, cls_id) in enumerate(zip(boxes, scores, class_ids)):
        x1, y1, x2, y2 = map(int, box)
        class_name = CLASS_NAMES[int(cls_id)]
        print(f"  {i+1}. {class_name:12s} (conf: {score:.3f}) @ [{x1:4d}, {y1:4d}, {x2:4d}, {y2:4d}]")
        
        detections.append({
            'class': class_name,
            'confidence': float(score),
            'bbox': [x1, y1, x2, y2]
        })
    
    # Visualize
    print(f"\nGenerating visualization...")
    visualize_detections(img, boxes, scores, class_ids, output_viz)
    
    # Save JSON
    results = {
        'model': 'best_yolo_int8.dlc',
        'runtime': 'SNPE/QNN NPU',
        'image': str(Path(image_path).name),
        'image_size': f"{orig_shape[1]}x{orig_shape[0]}",
        'num_detections': len(boxes),
        'detections': detections
    }
    
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ JSON results saved to: {output_json}")
    
    print(f"\n{'='*60}")
    print("✓ NPU output decoding complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python decode_npu_output.py <output0.raw> <original_image.jpg> [output_viz.jpg]")
        print("\nExample:")
        print("  python decode_npu_output.py output0.raw /home/harshyy/Desktop/20250103_104457.jpg")
        sys.exit(1)
    
    raw_file = sys.argv[1]
    image_path = sys.argv[2]
    output_viz = sys.argv[3] if len(sys.argv) > 3 else 'npu_result.jpg'
    
    decode_npu_output(raw_file, image_path, output_viz)

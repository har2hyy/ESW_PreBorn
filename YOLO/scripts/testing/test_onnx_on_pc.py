#!/usr/bin/env python3
"""
ONNX Model PC Testing Script
Tests YOLO ONNX model on PC using ONNX Runtime
"""

import argparse
import cv2
import numpy as np
import onnxruntime as ort
import json
import time
from pathlib import Path

# YOLO class names
CLASSES = ['worker', 'truck', 'bike', 'bulldozer', 'car']
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def preprocess_image(image_path, input_size=(1024, 1024)):
    """Preprocess image for YOLO model"""
    print(f"📸 Loading image: {image_path}")
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    original_shape = img.shape[:2]
    print(f"   Original size: {original_shape}")
    
    # Resize
    img_resized = cv2.resize(img, input_size)
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Convert to CHW format
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    
    # Add batch dimension
    img_input = np.expand_dims(img_chw, axis=0)
    
    print(f"   Preprocessed shape: {img_input.shape}")
    return img_input, img, original_shape

def postprocess_outputs(outputs, conf_threshold=0.25, iou_threshold=0.45, input_size=(1024, 1024), original_size=None):
    """Post-process YOLO outputs"""
    print(f"\n🔍 Post-processing outputs...")
    
    # Output shape: [1, 9, 21504] where 9 = 4 (bbox) + 5 (classes)
    output = outputs[0]  # Shape: [1, 9, 21504]
    
    print(f"   Output shape: {output.shape}")
    
    # Reshape: [1, 9, 21504] -> [21504, 9]
    predictions = output[0].transpose()  # Shape: [21504, 9]
    
    # Extract bbox and class scores
    boxes = predictions[:, :4]  # [21504, 4] - x, y, w, h
    scores = predictions[:, 4:]  # [21504, 5] - class scores
    
    # Get max class score and class id for each detection
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)
    
    # Filter by confidence
    mask = confidences > conf_threshold
    boxes = boxes[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]
    
    print(f"   Detections after confidence filtering: {len(boxes)}")
    
    if len(boxes) == 0:
        return []
    
    # Convert from xywh to xyxy
    boxes_xyxy = np.copy(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(),
        confidences.tolist(),
        conf_threshold,
        iou_threshold
    )
    
    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes_xyxy[i]
            class_id = int(class_ids[i])
            confidence = float(confidences[i])
            
            # Scale back to original image size if needed
            if original_size:
                scale_x = original_size[1] / input_size[0]
                scale_y = original_size[0] / input_size[1]
                box = [
                    box[0] * scale_x,
                    box[1] * scale_y,
                    box[2] * scale_x,
                    box[3] * scale_y
                ]
            
            detections.append({
                'class': CLASSES[class_id],
                'class_id': class_id,
                'confidence': confidence,
                'bbox': [float(x) for x in box]
            })
    
    print(f"   Final detections after NMS: {len(detections)}")
    return detections

def visualize_results(image, detections, output_path):
    """Draw bounding boxes on image"""
    print(f"\n🎨 Visualizing results...")
    vis_img = image.copy()
    
    for det in detections:
        box = det['bbox']
        class_id = det['class_id']
        class_name = det['class']
        confidence = det['confidence']
        
        # Draw bbox
        x1, y1, x2, y2 = map(int, box)
        color = COLORS[class_id].tolist()
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis_img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imwrite(str(output_path), vis_img)
    print(f"   Saved visualization to: {output_path}")
    return vis_img

def test_onnx_model(onnx_path, image_path, output_dir="onnx_test_results"):
    """Test ONNX model with ONNX Runtime"""
    print("="*70)
    print("🚀 YOLO ONNX Model Testing on PC")
    print("="*70)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load ONNX model
    print(f"\n📦 Loading ONNX model: {onnx_path}")
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    
    # Get model info
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name
    output_shape = session.get_outputs()[0].shape
    
    print(f"   Input: {input_name} {input_shape}")
    print(f"   Output: {output_name} {output_shape}")
    
    # Preprocess
    img_input, original_img, original_shape = preprocess_image(image_path)
    
    # Run inference
    print(f"\n⚡ Running inference...")
    start_time = time.time()
    outputs = session.run([output_name], {input_name: img_input})
    inference_time = (time.time() - start_time) * 1000
    print(f"   Inference time: {inference_time:.2f} ms")
    
    # Postprocess
    detections = postprocess_outputs(outputs, original_size=original_shape)
    
    # Save results
    result_json_path = output_path / "results.json"
    result_img_path = output_path / "result.jpg"
    
    result_data = {
        'model': str(onnx_path),
        'image': str(image_path),
        'inference_time_ms': inference_time,
        'num_detections': len(detections),
        'detections': detections
    }
    
    with open(result_json_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"\n💾 Saved results to: {result_json_path}")
    
    # Visualize
    visualize_results(original_img, detections, result_img_path)
    
    # Print summary
    print("\n" + "="*70)
    print("📊 RESULTS SUMMARY")
    print("="*70)
    print(f"Model: {onnx_path}")
    print(f"Image: {image_path}")
    print(f"Inference Time: {inference_time:.2f} ms")
    print(f"Detections: {len(detections)}")
    print("\nDetected objects:")
    for i, det in enumerate(detections, 1):
        print(f"  {i}. {det['class']}: {det['confidence']:.3f}")
    print("="*70)
    
    return detections

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test YOLO ONNX model on PC")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--output", type=str, default="onnx_test_results", help="Output directory")
    
    args = parser.parse_args()
    
    test_onnx_model(args.model, args.image, args.output)

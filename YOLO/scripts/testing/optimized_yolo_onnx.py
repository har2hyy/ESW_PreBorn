#!/usr/bin/env python3
"""
Optimized YOLO detector for QIDK deployment using ONNX Runtime.
Designed for low-memory environments with imgsz=1024.

For QIDK deployment, ONNX Runtime is preferred over TFLite because:
- Better quantization support (int8)
- Lighter runtime footprint
- Easier to optimize for edge devices
- Better performance on ARM/embedded CPUs
"""

import numpy as np
import onnxruntime as ort
import cv2
import os
from typing import List, Tuple

class OptimizedYOLOv11ONNX:
    """
    Memory-optimized YOLO detector for QIDK using ONNX Runtime.
    Configured for imgsz=1024 (matching training resolution).
    """
    
    def __init__(self, model_path: str = 'runs/detect/train/weights/best.onnx'):
        """Initialize ONNX model with optimization for low-memory deployment"""
        print(f"Loading ONNX model: {model_path}")
        
        # Create ONNX Runtime session with optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 2  # Limit threads for low-memory systems
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']  # For QIDK/embedded
        )
        
        # Get model metadata
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        print(f"Input: {self.input_name}, shape: {self.input_shape}")
        print(f"Output: {self.output_name}")
        
        self.img_size = self.input_shape[2]  # Should be 1024
        
        # YOLO configuration for imgsz=1024
        self.strides = [8, 16, 32]
        self.num_classes = 5
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        
        # No need to pre-generate anchors for YOLOv11
        # The model outputs are already in absolute pixel coordinates
        print(f"Configured for imgsz={self.img_size}, conf={self.conf_threshold}, iou={self.iou_threshold}")
        
        # Calculate expected output size for validation
        expected_anchors = sum((self.img_size // s) ** 2 for s in self.strides)
        print(f"Expected anchors: {expected_anchors}")
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for YOLO inference"""
        # Resize to model input size
        img_resized = cv2.resize(image, (self.img_size, self.img_size))
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and convert to float32
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # Transpose to CHW format
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        
        # Add batch dimension
        img_batch = np.expand_dims(img_chw, axis=0)
        
        return img_batch
    
    def postprocess(self, outputs: np.ndarray, orig_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Postprocess YOLOv11 outputs.
        
        YOLOv11 output format: [1, 9, 21504]
        - 9 channels: [x_center, y_center, width, height, class0_prob, class1_prob, ..., class4_prob]
        - 21504 anchors: predictions at different scales
        - Bbox coords are in pixel coordinates relative to 1024x1024 input
        - Class probs are already sigmoid-activated (not logits)
        
        Args:
            outputs: Model output [1, 9, 21504]
            orig_shape: Original image shape (height, width)
        
        Returns:
            boxes: [N, 4] in xyxy format
            scores: [N] confidence scores
            class_ids: [N] class IDs
        """
        # outputs shape: [1, 9, 21504]
        outputs = outputs[0]  # Remove batch dimension -> [9, 21504]
        
        # Extract components
        bbox_xywh = outputs[:4, :].T  # [21504, 4] - bbox in xywh format (pixel coords)
        class_probs = outputs[4:, :].T  # [21504, 5] - class probabilities (already sigmoid)
        
        # Get max class probability and ID for each anchor
        max_scores = np.max(class_probs, axis=1)  # [21504]
        class_ids = np.argmax(class_probs, axis=1)  # [21504]
        
        # Filter by confidence threshold
        mask = max_scores > self.conf_threshold
        bbox_xywh_filtered = bbox_xywh[mask]  # [N, 4]
        scores_filtered = max_scores[mask]  # [N]
        classes_filtered = class_ids[mask]  # [N]
        
        if len(bbox_xywh_filtered) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Convert xywh to xyxy format
        boxes_xyxy = self._xywh_to_xyxy(bbox_xywh_filtered)
        
        # Apply NMS
        indices = self._nms(boxes_xyxy, scores_filtered, self.iou_threshold)
        
        # Scale boxes to original image size
        scale_h = orig_shape[0] / self.img_size
        scale_w = orig_shape[1] / self.img_size
        
        boxes_final = boxes_xyxy[indices].copy()
        boxes_final[:, [0, 2]] *= scale_w
        boxes_final[:, [1, 3]] *= scale_h
        
        # Clip to image boundaries
        boxes_final[:, [0, 2]] = np.clip(boxes_final[:, [0, 2]], 0, orig_shape[1])
        boxes_final[:, [1, 3]] = np.clip(boxes_final[:, [1, 3]], 0, orig_shape[0])
        
        return boxes_final, scores_filtered[indices], classes_filtered[indices]
    
    @staticmethod
    def _xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
        """
        Convert boxes from xywh (center format) to xyxy (corner format).
        
        Args:
            boxes_xywh: [N, 4] boxes in [x_center, y_center, width, height] format
        
        Returns:
            boxes_xyxy: [N, 4] boxes in [x1, y1, x2, y2] format
        """
        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2
        return boxes_xyxy
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        """Non-maximum suppression (optimized for low memory)"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
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
    
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run detection on image.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            boxes: [N, 4] bounding boxes in xyxy format
            scores: [N] confidence scores
            class_ids: [N] class IDs
        """
        orig_shape = image.shape[:2]
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        
        # Postprocess
        boxes, scores, class_ids = self.postprocess(outputs, orig_shape)
        
        return boxes, scores, class_ids


# Test the detector
if __name__ == '__main__':
    import sys
    
    # Test detection
    detector = OptimizedYOLOv11ONNX()
    
    # Load test image
    test_image_path = '/home/harshyy/Desktop/20250103_104457.jpg'
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        sys.exit(1)
    
    image = cv2.imread(test_image_path)
    print(f"\nTest image: {test_image_path}, shape: {image.shape}")
    
    # Run detection
    print("\nRunning detection...")
    boxes, scores, class_ids = detector.detect(image)
    
    print(f"\n✅ Detected {len(boxes)} objects:")
    
    class_names = {0: 'worker', 1: 'truck', 2: 'bike', 3: 'bulldozer', 4: 'car'}
    
    for i, (box, score, cls_id) in enumerate(zip(boxes, scores, class_ids)):
        print(f"  {i+1}. {class_names[cls_id]}: {score:.3f} @ [{int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])}]")
    
    # Count by class
    print("\nDetection summary:")
    for cls_id in range(5):
        count = np.sum(class_ids == cls_id)
        if count > 0:
            print(f"  {class_names[cls_id]}: {count}")
    
    # Visualize
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_names[cls_id]}: {score:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    output_path = 'onnx_detection_1024_test.jpg'
    cv2.imwrite(output_path, image)
    print(f"\n✅ Visualization saved to: {output_path}")

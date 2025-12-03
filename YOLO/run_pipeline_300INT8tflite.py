#!/usr/bin/env python3
"""
YOLO Pipeline Runner - INT8 TFLite Model (300 Images Trained)
Runs quantized detection on test images for QIDK NPU
Saves outputs to pipeline_output_300INT8tflite/
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
import json

# Import YOLO post-processing from existing scripts
sys.path.append('scripts/testing')

class YOLOInt8TFLite:
    """YOLO INT8 TFLite inference wrapper"""
    
    def __init__(self, model_path, classes_path):
        """Initialize TFLite interpreter"""
        self.model_path = Path(model_path)
        
        # Load class names
        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details[0]['shape']
        self.input_dtype = self.input_details[0]['dtype']
        
        print(f"✅ Loaded INT8 TFLite model")
        print(f"   Input: {self.input_shape}, dtype: {self.input_dtype}")
        print(f"   Output: {self.output_details[0]['shape']}")
    
    def preprocess(self, image):
        """Preprocess image for INT8 model"""
        h, w = self.input_shape[1:3]
        
        # Resize
        img_resized = cv2.resize(image, (w, h))
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # For INT8 models, input is uint8 [0, 255]
        if self.input_dtype == np.uint8:
            img_input = img_rgb.astype(np.uint8)  # ✅ FIXED: Use RGB not BGR
        else:
            # FP32 fallback
            img_input = (img_rgb / 255.0).astype(np.float32)
        
        # Add batch dimension
        img_input = np.expand_dims(img_input, axis=0)
        
        return img_input
    
    def postprocess(self, output, conf_threshold=0.25, iou_threshold=0.45):
        """Post-process YOLO output to get bounding boxes"""
        # Output shape: [1, 9, 21504] for YOLO11
        # 9 channels = 4 bbox coords + 5 class scores
        
        # Step 1: Transpose to get correct shape [1, 21504, 9]
        output = output.transpose(0, 2, 1)  # [1, 9, 21504] → [1, 21504, 9]
        
        # Step 2: Remove batch dimension
        output = output[0]  # [21504, 9]
        
        # Step 3: Dequantize if INT8
        if self.output_details[0]['dtype'] == np.uint8:
            scale, zero_point = self.output_details[0]['quantization']
            output = scale * (output.astype(np.float32) - zero_point)
        
        # Step 4: Apply sigmoid activation (CRITICAL FIX!)
        output = 1 / (1 + np.exp(-output))
        
        # Step 5: Extract bounding boxes and scores
        boxes = output[:, :4]  # x_center, y_center, width, height (normalized [0,1])
        class_scores = output[:, 4:]  # 5 class scores
        
        # Step 6: Get class with max score
        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)
        
        # Step 7: Filter by confidence
        mask = confidences > conf_threshold
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]
        
        if len(boxes) == 0:
            return []
        
        # Step 8: Convert from normalized center format to pixel corner format
        h, w = self.input_shape[1:3]
        
        detections = []
        for box, cls_id, conf in zip(boxes, class_ids, confidences):
            x_center, y_center, width, height = box
            
            # Clamp to [0, 1] range
            x_center = np.clip(x_center, 0, 1)
            y_center = np.clip(y_center, 0, 1)
            width = np.clip(width, 0, 1)
            height = np.clip(height, 0, 1)
            
            # Convert to corner coordinates (still normalized)
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # Clamp to valid range
            x1 = np.clip(x1, 0, 1)
            y1 = np.clip(y1, 0, 1)
            x2 = np.clip(x2, 0, 1)
            y2 = np.clip(y2, 0, 1)
            
            # Scale to pixel coordinates
            x1 *= w
            y1 *= h
            x2 *= w
            y2 *= h
            
            detections.append({
                'class': self.classes[cls_id],
                'class_id': int(cls_id),
                'confidence': float(conf),
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'center': [float((x1 + x2) / 2), float((y1 + y2) / 2)]
            })
        
        # Step 9: Apply NMS
        if len(detections) > 0:
            detections = self.apply_nms(detections, iou_threshold)
        
        return detections
    
    def apply_nms(self, detections, iou_threshold):
        """Apply Non-Maximum Suppression"""
        if len(detections) == 0:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        # NMS
        keep = []
        while len(detections) > 0:
            best = detections.pop(0)
            keep.append(best)
            
            # Remove overlapping boxes
            detections = [
                det for det in detections
                if self.iou(best['bbox'], det['bbox']) < iou_threshold
            ]
        
        return keep
    
    def iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
        
        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def predict(self, image):
        """Run inference on image"""
        # Preprocess
        img_input = self.preprocess(image)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], img_input)
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Postprocess
        detections = self.postprocess(output)
        
        return detections
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes on image"""
        img_draw = image.copy()
        h, w = image.shape[:2]
        
        # Scale detections to original image size
        input_h, input_w = self.input_shape[1:3]
        scale_x = w / input_w
        scale_y = h / input_h
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Scale to original size
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            # Draw box
            color = (0, 255, 0)  # Green
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.putText(img_draw, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img_draw

def main():
    print("="*80)
    print("  YOLO11 Detection Pipeline - INT8 TFLite (300 Images)")
    print("="*80)
    
    # Paths
    model_path = Path('models/tflite/best_yolo11_300_int8.tflite')
    
    # Fallback to existing INT8 model if new one doesn't exist
    if not model_path.exists():
        model_path = Path('models/tflite/best_yolo_int8.tflite')
    
    classes_path = Path('data/classes.txt')
    if not classes_path.exists():
        classes_path = Path('classes.txt')
    
    test_images_dir = Path('test_images')
    output_dir = Path('pipeline_output_300INT8tflite')
    
    # Verify paths
    if not model_path.exists():
        print(f"❌ INT8 TFLite model not found: {model_path}")
        print("   Run export_int8_300pt.py first to create the INT8 model")
        return 1
    
    if not classes_path.exists():
        print(f"❌ Classes file not found: {classes_path}")
        return 1
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")
    
    # Load model
    print(f"\n🔄 Loading INT8 TFLite model: {model_path}")
    print(f"   Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    yolo = YOLOInt8TFLite(model_path, classes_path)
    
    # Get test images
    test_images = sorted(test_images_dir.glob('*.jpg'))
    if not test_images:
        print(f"❌ No test images found in {test_images_dir}")
        return 1
    
    print(f"\n📸 Found {len(test_images)} test images")
    
    # Process each image
    results_summary = []
    
    for idx, img_path in enumerate(test_images, 1):
        print(f"\n[{idx}/{len(test_images)}] Processing: {img_path.name}")
        
        # Load image
        image = cv2.imread(str(img_path))
        
        # Run detection
        detections = yolo.predict(image)
        
        # Print detections
        for det in detections:
            print(f"   ✓ {det['class']}: {det['confidence']:.3f} at {[int(x) for x in det['bbox']]}")
        
        print(f"   Total detections: {len(detections)}")
        
        # Draw and save
        img_annotated = yolo.draw_detections(image, detections)
        output_path = output_dir / f"{img_path.stem}_detected.jpg"
        cv2.imwrite(str(output_path), img_annotated)
        print(f"   💾 Saved: {output_path.name}")
        
        # Save JSON
        json_path = output_dir / f"{img_path.stem}_detections.json"
        result_data = {
            'image': img_path.name,
            'model': 'best_yolo11_300_int8.tflite',
            'quantization': 'INT8 (uint8)',
            'total_detections': len(detections),
            'detections': detections
        }
        
        with open(json_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        results_summary.append(result_data)
    
    # Save summary
    summary_path = output_dir / 'pipeline_summary.json'
    summary = {
        'model': str(model_path),
        'model_type': 'INT8 Quantized TFLite',
        'training_dataset': '300 manually annotated images',
        'quantization': 'INT8 with 150 calibration images',
        'deployment_target': 'QIDK NPU',
        'total_images_processed': len(test_images),
        'results': results_summary
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ INT8 TFLite Pipeline Complete!")
    print(f"   Processed: {len(test_images)} images")
    print(f"   Output: {output_dir}")
    print(f"   Summary: {summary_path.name}")
    print(f"   Ready for QIDK NPU deployment")
    print("="*80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

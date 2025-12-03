#!/usr/bin/env python3
"""
Proper YOLO11 Detection Visualization
Creates professional-looking detection outputs with clean labels
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json

# Configuration
MODEL_PATH = 'models/pytorch/best.pt'
TEST_IMAGES_DIR = Path('test_images')
OUTPUT_DIR = Path('pipeline_output_proper')
OUTPUT_DIR.mkdir(exist_ok=True)

# Class colors (matching your second image style)
CLASS_COLORS = {
    'worker': (255, 200, 0),    # Blue-ish for workers
    'truck': (0, 255, 255),     # Cyan for trucks
    'bike': (255, 0, 255),      # Magenta for bikes
    'bulldozer': (0, 165, 255), # Orange for bulldozers
    'car': (255, 255, 0)        # Yellow for cars
}

def draw_professional_detection(image, detection, color):
    """Draw detection with professional styling"""
    x1, y1, x2, y2 = map(int, detection['bbox'])
    class_name = detection['class']
    confidence = detection['confidence']
    
    # Draw bounding box with thicker lines
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    
    # Create label text
    label = f"{class_name} {confidence:.2f}"
    
    # Get text size for background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )
    
    # Draw label background (slightly transparent)
    label_y = max(y1 - 10, text_height + 10)
    cv2.rectangle(
        image,
        (x1, label_y - text_height - 10),
        (x1 + text_width + 10, label_y + baseline),
        color,
        -1  # Filled
    )
    
    # Draw label text in white
    cv2.putText(
        image,
        label,
        (x1 + 5, label_y - 5),
        font,
        font_scale,
        (255, 255, 255),  # White text
        thickness,
        cv2.LINE_AA
    )

def main():
    print("=" * 80)
    print("  YOLO11 Proper Detection Visualization")
    print("=" * 80)
    
    # Load model
    print(f"\n🔄 Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded successfully")
    print(f"   Classes: {list(model.names.values())}")
    
    # Get test images
    test_images = sorted(TEST_IMAGES_DIR.glob('*.jpg'))
    print(f"\n📸 Found {len(test_images)} test images")
    
    total_detections = 0
    all_results = []
    
    for idx, img_path in enumerate(test_images, 1):
        print(f"\n[{idx}/{len(test_images)}] Processing: {img_path.name}")
        
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"   ⚠️  Could not read image: {img_path}")
            continue
        
        h, w = image.shape[:2]
        print(f"   Image size: {w}x{h}")
        
        # Run detection
        results = model(image, conf=0.25, iou=0.45, imgsz=1024, verbose=False)
        
        # Parse detections
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get class and confidence
                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                class_name = model.names[cls_id]
                
                detections.append({
                    'class': class_name,
                    'class_id': cls_id,
                    'confidence': conf,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'center': [float((x1 + x2) / 2), float((y1 + y2) / 2)]
                })
        
        print(f"   ✓ Detected {len(detections)} objects")
        
        # Draw detections
        output_image = image.copy()
        for detection in detections:
            class_name = detection['class']
            color = CLASS_COLORS.get(class_name, (255, 255, 255))
            draw_professional_detection(output_image, detection, color)
            print(f"     • {class_name}: {detection['confidence']:.3f}")
        
        # Save visualized image
        output_img_path = OUTPUT_DIR / f"{img_path.stem}_detected.jpg"
        cv2.imwrite(str(output_img_path), output_image, 
                   [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"   💾 Saved: {output_img_path.name}")
        
        # Save JSON
        output_json_path = OUTPUT_DIR / f"{img_path.stem}_detections.json"
        with open(output_json_path, 'w') as f:
            json.dump({
                'image': img_path.name,
                'size': {'width': w, 'height': h},
                'detections': detections
            }, f, indent=2)
        
        # Update statistics
        total_detections += len(detections)
        all_results.append({
            'image': img_path.name,
            'detections': len(detections),
            'objects': detections
        })
    
    # Save summary
    summary = {
        'model': MODEL_PATH,
        'total_images': len(test_images),
        'total_detections': total_detections,
        'results': all_results
    }
    
    summary_path = OUTPUT_DIR / 'detection_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✅ Detection Visualization Complete!")
    print(f"   Total images: {len(test_images)}")
    print(f"   Total detections: {total_detections}")
    print(f"   Output directory: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == '__main__':
    main()

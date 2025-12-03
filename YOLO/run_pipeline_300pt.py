#!/usr/bin/env python3
"""
YOLO Pipeline Runner - PyTorch Model (300 Images Trained)
Runs detection on test images using the best.pt model and saves outputs
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import json

def main():
    print("="*80)
    print("  YOLO11 Detection Pipeline - PyTorch Model (300 Images)")
    print("="*80)
    
    # Paths
    model_path = Path('models/pytorch/best.pt')
    test_images_dir = Path('test_images')
    output_dir = Path('pipeline_output_300pt')
    
    # Verify model exists
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Expected: YOLO/models/pytorch/best.pt")
        return 1
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")
    
    # Load model
    print(f"\n🔄 Loading PyTorch model: {model_path}")
    model = YOLO(str(model_path))
    print("✅ Model loaded successfully")
    
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
        
        # Run inference
        results = model.predict(
            source=str(img_path),
            imgsz=1024,
            conf=0.25,  # Confidence threshold
            iou=0.45,   # NMS IoU threshold
            verbose=False
        )[0]
        
        # Extract detections
        detections = []
        boxes = results.boxes
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                class_name = results.names[cls]
                
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'center': [float((x1 + x2) / 2), float((y1 + y2) / 2)]
                })
                
                print(f"   ✓ {class_name}: {conf:.3f} at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
        
        print(f"   Total detections: {len(detections)}")
        
        # Save annotated image
        annotated_img = results.plot()
        output_path = output_dir / f"{img_path.stem}_detected.jpg"
        cv2.imwrite(str(output_path), annotated_img)
        print(f"   💾 Saved: {output_path.name}")
        
        # Save detection data as JSON
        json_path = output_dir / f"{img_path.stem}_detections.json"
        result_data = {
            'image': img_path.name,
            'model': 'best.pt (300 images trained)',
            'total_detections': len(detections),
            'detections': detections
        }
        
        with open(json_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        results_summary.append(result_data)
    
    # Save overall summary
    summary_path = output_dir / 'pipeline_summary.json'
    summary = {
        'model': 'best.pt',
        'training_dataset': '300 manually annotated images',
        'total_images_processed': len(test_images),
        'results': results_summary
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ Pipeline Complete!")
    print(f"   Processed: {len(test_images)} images")
    print(f"   Output: {output_dir}")
    print(f"   Summary: {summary_path.name}")
    print("="*80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

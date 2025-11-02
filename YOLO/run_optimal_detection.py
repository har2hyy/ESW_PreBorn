#!/usr/bin/env python3
"""
Simple runner script with the optimal settings found from your successful results.
This will give you the same excellent detection results you showed in the images.
"""

from tflite_yolo import OptimizedYOLOv11TFLite
import os

def main():
    """Run detection with optimal settings (confidence=0.60)"""
    
    print("🎯 YOLOv11 Detection - Optimal Configuration")
    print("Based on your successful results!")
    print("=" * 60)
    
    # Configuration - CHANGE THESE AS NEEDED
    MODEL_PATH = 'runs/detect/train/weights/best_saved_model/best_float32.tflite'
    CLASSES_PATH = '../data+label/classes.txt'
    
    # CHANGE THIS to your desired image
    IMAGE_PATH = '/home/harshyy/Desktop/20250103_104457.jpg'
    
    # Optimal settings based on your results
    CONFIDENCE_THRESHOLD = 0.53  # Perfect for detecting workers at 0.60-0.70 range
    NMS_THRESHOLD = 0.45
    
    # Output filename
    OUTPUT_FILE = 'my_optimal_result.jpg'
    
    print(f"📁 Model: {MODEL_PATH}")
    print(f"🖼️  Processing: {os.path.basename(IMAGE_PATH)}")
    print(f"🎯 Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"💾 Result will be saved as: {OUTPUT_FILE}")
    print("=" * 60)
    
    # Check if image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Image not found: {IMAGE_PATH}")
        print("💡 Please update IMAGE_PATH in this script")
        return
    
    # Initialize detector
    detector = OptimizedYOLOv11TFLite(
        model_path=MODEL_PATH,
        classes_path=CLASSES_PATH,
        conf_threshold=CONFIDENCE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD
    )
    
    # Run detection
    print(f"\\n🚀 Running detection...")
    result_image, boxes, confidences, class_ids = detector.predict(
        IMAGE_PATH, 
        OUTPUT_FILE
    )
    
    print(f"\\n✅ SUCCESS!")
    print(f"📊 Total detections: {len(boxes)}")
    print(f"💾 Result saved as: {OUTPUT_FILE}")
    
    # Count detections by class
    if boxes:
        class_counts = {}
        for class_id in class_ids:
            class_name = detector.class_names[class_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"\\n📋 Detection Summary:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  • {class_name}: {count}")
    
    print(f"\\n🎉 Done! Open '{OUTPUT_FILE}' to see your results!")

if __name__ == "__main__":
    main()
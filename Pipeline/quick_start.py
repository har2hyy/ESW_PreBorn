#!/usr/bin/env python3
"""
Quick Start Example for Integrated Pipeline

This script demonstrates the simplest way to use the pipeline.
"""

import sys
import os

# Add Pipeline directory to path
sys.path.append(os.path.dirname(__file__))

from integrated_detection_depth_pipeline import IntegratedPipeline


def quick_demo():
    """Run a quick demo of the pipeline"""
    
    # Paths (update these as needed)
    YOLO_MODEL = '../YOLO/runs/detect/train/weights/best_saved_model/best_float32.tflite'
    YOLO_CLASSES = '../YOLO/classes.txt'
    IMAGE_PATH = '/home/harshyy/Desktop/20250103_104457.jpg'
    OUTPUT_DIR = '/home/harshyy/Desktop/pipeline_output'
    
    print("🚀 Quick Start - Integrated Pipeline Demo\n")
    
    # Check files exist
    if not os.path.exists(YOLO_MODEL):
        print(f"❌ YOLO model not found at: {YOLO_MODEL}")
        print("Please update the path or train your model first.")
        return
    
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Image not found at: {IMAGE_PATH}")
        print("Please update IMAGE_PATH in this script.")
        return
    
    # Initialize pipeline (loads both models)
    print("Initializing pipeline...\n")
    pipeline = IntegratedPipeline(
        yolo_model_path=YOLO_MODEL,
        yolo_classes_path=YOLO_CLASSES,
        depth_encoder='vitb',  # Change to 'vits' for faster processing
        yolo_conf_threshold=0.53,  # Balanced threshold for good detections
        yolo_nms_threshold=0.45
    )
    
    # Process image
    results = pipeline.process_image(IMAGE_PATH, OUTPUT_DIR)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n✅ Detected {results['total_detections']} objects:")
    for i, det in enumerate(results['detections'], 1):
        print(f"   {i}. {det['class']} "
              f"(confidence: {det['confidence']:.2f}, "
              f"depth: {det['depth_avg']:.1f}, "
              f"position: {det['center']})")
    
    if results['distances']:
        print(f"\n📏 Distance measurements ({len(results['distances'])} pairs):")
        # Show top 5 closest pairs
        for i, dist in enumerate(results['distances'][:5], 1):
            print(f"   {i}. {dist['obj1_class']} ↔ {dist['obj2_class']}")
            print(f"      • Pixel distance: {dist['euclidean']:.1f}px")
            print(f"      • Depth difference: {dist['depth_diff']:.1f}")
            print(f"      • Horizontal separation: {dist['horizontal']:.1f}px")
            print(f"      • Vertical separation: {dist['vertical']:.1f}px")
    
    print(f"\n💾 All outputs saved to: {OUTPUT_DIR}")
    print("   Check the following files:")
    print(f"   • {os.path.basename(results['outputs']['yolo'])}")
    print(f"   • {os.path.basename(results['outputs']['depth'])}")
    print(f"   • {os.path.basename(results['outputs']['combined'])}")
    
    print("\n" + "="*80)
    print("✨ Done! Open the output directory to see results.")
    print("="*80 + "\n")


if __name__ == "__main__":
    quick_demo()

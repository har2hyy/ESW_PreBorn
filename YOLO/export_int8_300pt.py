#!/usr/bin/env python3
"""
Export YOLO PyTorch Model to INT8 Quantized TFLite
Uses calibration data from data_1-300 for quantization
Optimized for QIDK NPU deployment
"""
import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
import cv2
from ultralytics import YOLO

def load_calibration_images(calibration_dir, num_images=150):
    """Load and preprocess calibration images"""
    print(f"\n📁 Loading calibration images from: {calibration_dir}")
    
    calib_dir = Path(calibration_dir)
    if not calib_dir.exists():
        # Try alternative paths
        calib_dir = Path('data/train/images')
        if not calib_dir.exists():
            calib_dir = Path('data_1-300')
    
    # Get image files
    image_files = list(calib_dir.glob('*.jpg'))[:num_images]
    
    if not image_files:
        raise FileNotFoundError(f"No calibration images found in {calib_dir}")
    
    print(f"   Found {len(image_files)} calibration images")
    
    def representative_dataset():
        """Generator for calibration data"""
        for idx, img_path in enumerate(image_files):
            if idx % 20 == 0:
                print(f"   Calibrating: {idx+1}/{len(image_files)}")
            
            # Load and preprocess image
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (1024, 1024))
            
            # Normalize to [0, 1] for TFLite
            img = img.astype(np.float32) / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            yield [img]
    
    return representative_dataset

def main():
    print("="*80)
    print("  Export YOLO11 to INT8 Quantized TFLite (300 Images Model)")
    print("="*80)
    
    # Paths
    pt_model_path = Path('models/pytorch/best.pt')
    output_dir = Path('models/tflite')
    output_int8_path = output_dir / 'best_yolo11_300_int8.tflite'
    
    # Verify model exists
    if not pt_model_path.exists():
        print(f"❌ PyTorch model not found: {pt_model_path}")
        return 1
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Export to TFLite (FP32) first
    print("\n" + "="*80)
    print("Step 1: Export PyTorch → TFLite (FP32)")
    print("="*80)
    
    model = YOLO(str(pt_model_path))
    print(f"✅ Loaded model: {pt_model_path}")
    
    # Export to TFLite
    temp_tflite = output_dir / 'temp_fp32.tflite'
    print(f"\n🔄 Exporting to TFLite (FP32)...")
    
    model.export(
        format='tflite',
        imgsz=1024,
        int8=False,  # First export as FP32
    )
    
    # The exported file will be in the model directory
    exported_path = pt_model_path.parent / 'best_saved_model' / 'best_float32.tflite'
    
    if not exported_path.exists():
        # Try alternative path
        exported_path = Path('best_saved_model/best_float32.tflite')
    
    if not exported_path.exists():
        print("❌ FP32 TFLite export failed")
        return 1
    
    print(f"✅ FP32 TFLite created: {exported_path}")
    
    # Step 2: Convert FP32 TFLite to INT8 with calibration
    print("\n" + "="*80)
    print("Step 2: Quantize FP32 TFLite → INT8 with Calibration")
    print("="*80)
    
    # Load the FP32 TFLite model
    with open(exported_path, 'rb') as f:
        tflite_model = f.read()
    
    # Setup converter
    converter = tf.lite.TFLiteConverter.from_saved_model(
        str(exported_path.parent.parent)
    )
    
    # Enable INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Set representative dataset for calibration
    print("\n🔄 Loading calibration data...")
    representative_dataset = load_calibration_images(
        'data_1-300',
        num_images=150
    )
    converter.representative_dataset = representative_dataset
    
    # Force INT8 for inputs and outputs
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    print("\n🔄 Quantizing model to INT8...")
    print("   This may take a few minutes...")
    
    try:
        tflite_quant_model = converter.convert()
        
        # Save INT8 model
        with open(output_int8_path, 'wb') as f:
            f.write(tflite_quant_model)
        
        print(f"\n✅ INT8 TFLite model created!")
        print(f"   Saved to: {output_int8_path}")
        print(f"   Size: {output_int8_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Verify the model
        print("\n🔍 Verifying INT8 model...")
        interpreter = tf.lite.Interpreter(model_path=str(output_int8_path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"   Input shape: {input_details[0]['shape']}")
        print(f"   Input dtype: {input_details[0]['dtype']}")
        print(f"   Output shape: {output_details[0]['shape']}")
        print(f"   Output dtype: {output_details[0]['dtype']}")
        
        if input_details[0]['dtype'] == np.uint8:
            print("   ✅ Model uses INT8 quantization (uint8 input)")
        else:
            print("   ⚠️  Warning: Model input is not uint8")
        
        print("\n" + "="*80)
        print("✅ Export Complete!")
        print(f"   INT8 TFLite: {output_int8_path}")
        print(f"   Ready for QIDK NPU deployment")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ INT8 quantization failed: {e}")
        print("\nℹ️  This can happen if the model architecture is not fully compatible")
        print("   with INT8 quantization. Try using the FP32 model instead.")
        return 1

if __name__ == '__main__':
    sys.exit(main())

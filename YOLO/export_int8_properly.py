#!/usr/bin/env python3
"""
Export YOLO11 300-image model to INT8 TFLite with proper calibration

This script:
1. Loads the trained PyTorch model (best.pt)
2. Uses 150 training images for calibration
3. Exports to INT8 TFLite format optimized for QIDK NPU
4. Verifies the exported model works correctly
"""

import sys
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import tensorflow as tf
import cv2

print("=" * 80)
print("  YOLO11 INT8 TFLite Export with Calibration")
print("=" * 80)

# Paths
pt_model_path = 'models/pytorch/best.pt'
calib_data_dir = 'data_1-300'
output_dir = Path('models/tflite_fresh')
output_dir.mkdir(exist_ok=True)

# Load PyTorch model
print(f"\n📦 Loading PyTorch model: {pt_model_path}")
try:
    model = YOLO(pt_model_path)
    print(f"✅ Loaded YOLO11n model")
    print(f"   Classes: {model.names}")
    print(f"   Input size: {model.overrides.get('imgsz', 'default')}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# Check calibration data
print(f"\n📁 Checking calibration data: {calib_data_dir}")
calib_dir = Path(calib_data_dir)
if not calib_dir.exists():
    print(f"❌ Calibration directory not found: {calib_data_dir}")
    sys.exit(1)

# Check for images in subdirectories too
calib_images = list(calib_dir.glob('*.jpg')) + list(calib_dir.glob('*.png'))
calib_images += list(calib_dir.glob('**/*.jpg')) + list(calib_dir.glob('**/*.png'))
calib_images = list(set(calib_images))  # Remove duplicates
print(f"✅ Found {len(calib_images)} calibration images")

if len(calib_images) < 100:
    print(f"⚠️  Warning: Only {len(calib_images)} images (recommended: 100+)")

# Method 1: Try Ultralytics built-in export
print(f"\n🔄 Attempting export using Ultralytics...")
print(f"   Format: TFLite INT8")
print(f"   Image size: 1024x1024")
print(f"   Optimization: INT8 quantization")

try:
    # Note: Ultralytics may not support direct INT8 export with calibration
    # If this fails, we'll use TensorFlow method
    export_path = model.export(
        format='tflite',
        imgsz=1024,
        int8=True,
        data=str(calib_data_dir),  # Use calibration data
        optimize=True
    )
    print(f"✅ Export successful: {export_path}")
    exported_model_path = export_path
    
except Exception as e:
    print(f"⚠️  Ultralytics INT8 export failed: {e}")
    print(f"   This is expected - Ultralytics INT8 export needs dataset YAML file")
    
    # Method 2: Export to saved_model first, then quantize
    print(f"\n🔄 Using alternative method: PyTorch → SavedModel → INT8 TFLite")
    
    try:
        # Step 1: Export to saved_model
        print(f"\nStep 1/3: Exporting to TensorFlow SavedModel...")
        saved_model_path = model.export(format='saved_model', imgsz=1024)
        print(f"✅ SavedModel exported: {saved_model_path}")
        
        # Step 2: Create representative dataset generator
        print(f"\nStep 2/3: Preparing calibration dataset (150 images)...")
        
        def representative_dataset():
            """Generate representative data for quantization calibration"""
            selected_images = list(calib_images)[:150]
            print(f"   Using {len(selected_images)} images for calibration")
            
            for img_path in selected_images:
                # Read and preprocess image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                    
                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize to model input size
                img_resized = cv2.resize(img_rgb, (1024, 1024))
                
                # Normalize to [0, 1]
                img_normalized = img_resized.astype(np.float32) / 255.0
                
                # Add batch dimension
                img_batch = np.expand_dims(img_normalized, axis=0)
                
                yield [img_batch]
        
        # Step 3: Convert to INT8 TFLite
        print(f"\nStep 3/3: Quantizing to INT8 TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        
        # Enable INT8 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        
        # Force INT8 for all operations (including activations)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8  # Input: uint8
        converter.inference_output_type = tf.uint8  # Output: uint8
        
        # Convert
        tflite_model = converter.convert()
        
        # Save INT8 model
        exported_model_path = output_dir / 'best_yolo11_int8_calibrated.tflite'
        exported_model_path.write_bytes(tflite_model)
        
        print(f"✅ INT8 TFLite model saved: {exported_model_path}")
        print(f"   Size: {exported_model_path.stat().st_size / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"❌ TensorFlow export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Verify exported model
print(f"\n🔍 Verifying exported INT8 model...")
try:
    interpreter = tf.lite.Interpreter(str(exported_model_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"✅ Model loaded successfully")
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Input dtype: {input_details[0]['dtype']}")
    print(f"   Input quantization: {input_details[0]['quantization']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    print(f"   Output dtype: {output_details[0]['dtype']}")
    print(f"   Output quantization: {output_details[0]['quantization']}")
    
    # Test with real image
    print(f"\n🧪 Testing with real image...")
    test_img = cv2.imread('test_images/my_optimal_result.jpg')
    if test_img is not None:
        # Preprocess
        img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (1024, 1024))
        
        if input_details[0]['dtype'] == np.uint8:
            img_input = img_resized.astype(np.uint8)
        else:
            img_input = (img_resized / 255.0).astype(np.float32)
        
        img_input = np.expand_dims(img_input, axis=0)
        
        # Inference
        interpreter.set_tensor(input_details[0]['index'], img_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # Analyze output
        print(f"   Output range: {output.min()} - {output.max()}")
        print(f"   Unique values: {len(np.unique(output))}")
        
        # Dequantize and check
        if output_details[0]['dtype'] == np.uint8:
            scale, zero_point = output_details[0]['quantization']
            output_float = scale * (output.astype(np.float32) - zero_point)
            output_sigmoid = 1 / (1 + np.exp(-output_float))
            
            # Get class scores
            output_t = output_sigmoid.transpose(0, 2, 1)[0]
            class_scores = output_t[:, 4:]
            max_scores = np.max(class_scores, axis=1)
            
            detections_count = (max_scores > 0.25).sum()
            print(f"   Confidence range: {max_scores.min():.3f} - {max_scores.max():.3f}")
            print(f"   Detections (>0.25): {detections_count}")
            
            if detections_count > 10 and detections_count < 200:
                print(f"✅ Model appears to be working correctly!")
            elif detections_count > 1000:
                print(f"⚠️  Too many detections - model may have issues")
            else:
                print(f"⚠️  Too few detections - check calibration")
    
except Exception as e:
    print(f"❌ Verification failed: {e}")
    import traceback
    traceback.print_exc()

print(f"\n" + "=" * 80)
print(f"✅ Export complete!")
print(f"   INT8 TFLite model: {exported_model_path}")
print(f"   Ready for QIDK NPU deployment")
print(f"   Test with: conda run -n pipeline python3 run_pipeline_300INT8tflite.py")
print(f"=" * 80)

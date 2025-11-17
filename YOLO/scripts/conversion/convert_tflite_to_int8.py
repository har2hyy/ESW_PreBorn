#!/usr/bin/env python3
"""
TFLite INT8 Quantization Script
Converts float32 TFLite model to INT8 using calibration data
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm

# Paths
ROOT_DIR = Path(__file__).parent.parent.parent
MODEL_DIR = ROOT_DIR / "models" / "tflite" / "best_yolo_tflite"
DATA_DIR = ROOT_DIR / "data" / "val" / "images"
OUTPUT_DIR = ROOT_DIR / "models" / "tflite"

# Model configuration
INPUT_SIZE = 1024
BATCH_SIZE = 1

def preprocess_image(image_path):
    """Preprocess image for YOLO model"""
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def representative_dataset_gen():
    """Generate representative dataset for calibration"""
    print("\n📊 Loading calibration images...")
    
    # Get all validation images
    image_paths = sorted(list(DATA_DIR.glob("*.jpg")))
    
    if not image_paths:
        print(f"❌ No images found in {DATA_DIR}")
        return
    
    print(f"✅ Found {len(image_paths)} calibration images")
    
    # Use all available images or limit to reasonable number
    num_samples = min(len(image_paths), 100)
    print(f"📸 Using {num_samples} images for calibration\n")
    
    for i, img_path in enumerate(tqdm(image_paths[:num_samples], desc="Processing calibration data")):
        try:
            img = preprocess_image(img_path)
            yield [img]
        except Exception as e:
            print(f"⚠️  Skipping {img_path.name}: {e}")
            continue

def convert_to_int8():
    """Convert float32 TFLite model to INT8"""
    
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║                                                               ║")
    print("║         TFLite INT8 Quantization Converter                   ║")
    print("║                                                               ║")
    print("╚═══════════════════════════════════════════════════════════════╝\n")
    
    # Check if SavedModel exists
    saved_model_path = MODEL_DIR / "saved_model.pb"
    if not saved_model_path.exists():
        print(f"❌ SavedModel not found at {MODEL_DIR}")
        return False
    
    print(f"📂 Input SavedModel: {MODEL_DIR}")
    print(f"📂 Output directory: {OUTPUT_DIR}\n")
    
    try:
        # Configure converter
        print("⚙️  Configuring TFLite converter...")
        converter = tf.lite.TFLiteConverter.from_saved_model(str(MODEL_DIR))
        
        # Enable INT8 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        
        # Full integer quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8  # or tf.int8
        converter.inference_output_type = tf.uint8  # or tf.int8
        
        print("✅ Converter configured for full INT8 quantization\n")
        
        # Convert model
        print("🔄 Converting to INT8 (this may take a few minutes)...")
        tflite_quant_model = converter.convert()
        
        # Save INT8 model
        output_path = OUTPUT_DIR / "best_yolo_int8.tflite"
        output_path.write_bytes(tflite_quant_model)
        
        # Get file sizes
        float32_path = MODEL_DIR / "best_simplified_float32.tflite"
        float16_path = MODEL_DIR / "best_simplified_float16.tflite"
        
        int8_size = output_path.stat().st_size / (1024 * 1024)
        float32_size = float32_path.stat().st_size / (1024 * 1024) if float32_path.exists() else 0
        float16_size = float16_path.stat().st_size / (1024 * 1024) if float16_path.exists() else 0
        
        print("\n╔═══════════════════════════════════════════════════════════════╗")
        print("║                                                               ║")
        print("║              ✅ CONVERSION SUCCESSFUL! ✅                     ║")
        print("║                                                               ║")
        print("╚═══════════════════════════════════════════════════════════════╝\n")
        
        print("📊 MODEL SIZE COMPARISON")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        if float32_size > 0:
            print(f"  Float32:  {float32_size:.2f} MB  (baseline)")
        if float16_size > 0:
            print(f"  Float16:  {float16_size:.2f} MB  ({(float16_size/float32_size)*100:.1f}% of float32)")
        print(f"  INT8:     {int8_size:.2f} MB  ({(int8_size/float32_size)*100:.1f}% of float32)")
        print(f"\n  💾 Size reduction: {float32_size - int8_size:.2f} MB saved!")
        print(f"  📉 Compression ratio: {float32_size/int8_size:.1f}x smaller\n")
        
        print("📁 OUTPUT LOCATION")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  {output_path}\n")
        
        print("🚀 NEXT STEPS")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("  1. Deploy to Android app")
        print("  2. Enable Hexagon Delegate for NPU acceleration")
        print("  3. Expected speedup: 5-8x faster on NPU")
        print("  4. See: YOLO/docs/FIX_TFLITE_NPU.md for deployment guide\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = convert_to_int8()
    exit(0 if success else 1)

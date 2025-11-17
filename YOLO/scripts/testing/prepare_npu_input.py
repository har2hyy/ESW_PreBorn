#!/usr/bin/env python3
"""
Prepare Raw Input for SNPE/QNN NPU
===================================

Converts JPEG/PNG images to raw binary format required by SNPE runtime.

Usage:
    python prepare_npu_input.py input_image.jpg output.raw

Output:
    - Binary file with shape (3, 1024, 1024) in float32
    - Format: CHW (Channels, Height, Width)
    - Normalization: [0, 255] → [0.0, 1.0]
"""

import sys
import cv2
import numpy as np
from pathlib import Path


def prepare_raw_input(image_path, output_raw, img_size=1024):
    """
    Convert image to raw binary format for NPU.
    
    Args:
        image_path: Path to input image (JPEG/PNG)
        output_raw: Path to output .raw file
        img_size: Target size (1024 for YOLOv11)
    """
    print(f"\n{'='*60}")
    print("Preparing NPU Input - Raw Binary Conversion")
    print(f"{'='*60}\n")
    
    # Load image
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"✗ Error: Could not load image {image_path}")
        return False
    
    orig_shape = img.shape[:2]
    print(f"✓ Original shape: {orig_shape[0]}x{orig_shape[1]}")
    
    # Resize to 1024x1024
    print(f"Resizing to {img_size}x{img_size}...")
    img_resized = cv2.resize(img, (img_size, img_size))
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    print(f"✓ Converted BGR → RGB")
    
    # Normalize to [0.0, 1.0]
    img_float = img_rgb.astype(np.float32) / 255.0
    print(f"✓ Normalized [0, 255] → [0.0, 1.0]")
    
    # Convert to CHW format (3, 1024, 1024)
    img_chw = np.transpose(img_float, (2, 0, 1))
    print(f"✓ Transposed HWC → CHW: {img_chw.shape}")
    
    # Save as raw binary
    img_chw.tofile(output_raw)
    file_size_mb = Path(output_raw).stat().st_size / (1024 * 1024)
    print(f"✓ Saved to: {output_raw} ({file_size_mb:.2f} MB)")
    
    # Verification
    expected_size = 3 * img_size * img_size * 4  # float32 = 4 bytes
    actual_size = Path(output_raw).stat().st_size
    
    print(f"\nVerification:")
    print(f"  Expected size: {expected_size} bytes")
    print(f"  Actual size:   {actual_size} bytes")
    print(f"  Match: {'✓' if expected_size == actual_size else '✗'}")
    
    print(f"\n{'='*60}")
    print("✓ Raw input preparation complete!")
    print(f"{'='*60}\n")
    
    print("Next steps:")
    print(f"  1. Push to device: adb push {output_raw} /data/local/tmp/yolo_npu/")
    print(f"  2. Run inference: snpe-net-run --container best_yolo_int8.dlc --input_raw {Path(output_raw).name} --use_dsp")
    print(f"  3. Decode output: python decode_npu_output.py output0.raw {image_path}\n")
    
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python prepare_npu_input.py <input_image.jpg> [output.raw]")
        print("\nExample:")
        print("  python prepare_npu_input.py /home/harshyy/Desktop/20250103_104457.jpg test_input.raw")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Auto-generate output name if not provided
    if len(sys.argv) > 2:
        output_raw = sys.argv[2]
    else:
        input_name = Path(image_path).stem
        output_raw = f"{input_name}_1024x1024.raw"
    
    prepare_raw_input(image_path, output_raw)
